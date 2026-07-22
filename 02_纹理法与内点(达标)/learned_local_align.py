# -*- coding: utf-8 -*-
"""
关键测试: 我们训练的局部描述子(V21/V25a的local head, 14x13网格)
跨场景能否建立几何一致对应? (SIFT手工描述子已证跨场景对应崩溃)
若学习描述子跨场景仍有对应 → JIPNet类pose-alignment有戏; 否则数据天花板。
分数=RANSAC内点数。单场景对照 + 跨场景。
"""
import os, glob, random, sys, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_curve
from torchvision.models import resnet18

def load_img(p):
    with open(p,'rb') as f: return cv2.imdecode(np.frombuffer(f.read(),np.uint8), cv2.IMREAD_GRAYSCALE)
def clahe(img): return cv2.createCLAHE(4.0,(8,8)).apply(img)
def up2(img): return cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
def keep_cc(m):
    n,l,s,_=cv2.connectedComponentsWithStats(m,8,cv2.CV_32S)
    if n<=1: return m
    return ((l==1+np.argmax(s[1:,cv2.CC_STAT_AREA]))*255).astype(np.uint8)
def fillh(m):
    bg=cv2.bitwise_not(m); n,l,s,_=cv2.connectedComponentsWithStats(bg,8,cv2.CV_32S); h,w=m.shape
    for i in range(1,n):
        L,T,W,H=s[i,cv2.CC_STAT_LEFT],s[i,cv2.CC_STAT_TOP],s[i,cv2.CC_STAT_WIDTH],s[i,cv2.CC_STAT_HEIGHT]
        if L>0 and L+W<w-1 and T>0 and T+H<h-1: m[l==i]=255
    return m
def gmask(img,sig=13/3.,pct=95,r=.2):
    dx=cv2.Sobel(img,cv2.CV_32F,1,0,3); dy=cv2.Sobel(img,cv2.CV_32F,0,1,3); mg=cv2.magnitude(dx,dy)
    gs=int(np.ceil(3*sig))*2+1; ma=cv2.GaussianBlur(mg,(gs,gs),sig)
    th=np.percentile(mg.flatten(),pct)*r; _,m=cv2.threshold(ma,th,255,0); m=m.astype(np.uint8)
    se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,se,iterations=6); m=keep_cc(m); m=fillh(m)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,se,iterations=2); return keep_cc(m)
def prep(img):
    enh=clahe(img); u=up2(enh); m=gmask(u); mk=u.copy(); mk[m==0]=0
    v=mk[m>0].astype(np.float32); mu,sd=(v.mean(),v.std()+1e-6) if len(v) else (0,1)
    norm=(mk.astype(np.float32)-mu)/sd; norm[m==0]=0
    # 下采样mask到14x13网格, 标记有效cell
    return norm
def _stem(s):
    b=resnet18(weights=None)
    s.conv1=nn.Conv2d(1,64,7,2,3,bias=False); s.bn1,s.relu,s.maxpool=b.bn1,b.relu,b.maxpool
    s.layer1,s.layer2,s.layer3,s.layer4=b.layer1,b.layer2,b.layer3,b.layer4
    s.avgpool=nn.AdaptiveAvgPool2d((1,1))
class HybridEnc(nn.Module):
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.attention=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(),nn.Conv2d(128,1,1),nn.Softplus())
        s.attn_projector=nn.Sequential(nn.Linear(256,e),nn.BatchNorm1d(e)); s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        ld=F.normalize(s.local_head(x),dim=1)
        return ld   # (B,64,H,W)
dev=torch.device('cuda')
def lf(d,pre):
    out=[]
    for sd in [42,123,777]:
        m=HybridEnc().to(dev); m.load_state_dict(torch.load(f'f:/1111/指纹/{d}/{pre}_seed{sd}.pth',map_location=dev,weights_only=True),strict=False); m.eval(); out.append(m)
    return out
MODELS={'V21':lf('models_v21','v21'),'V25a':lf('models_v25a','v25a')}
print("loaded"); sys.stdout.flush()

@torch.no_grad()
def local_desc(model, frame):
    """返回 (coords (N,2), desc (N,64)) 仅前景cell, 双极性平均。"""
    accum=None
    for pol in (1,-1):
        x=torch.tensor(pol*frame,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
        ld=model(x)[0]  # (64,H,W)
        accum=ld if accum is None else accum+ld
    ld=F.normalize(accum,dim=0).cpu().numpy()  # (64,H,W)
    C,H,W=ld.shape
    # 前景: 下采样frame的mask到HxW
    mfull=(frame!=0).astype(np.float32)
    msmall=cv2.resize(mfull,(W,H),interpolation=cv2.INTER_AREA)
    desc=[]; coords=[]
    for r in range(H):
        for c in range(W):
            if msmall[r,c]>0.3:
                desc.append(ld[:,r,c]); coords.append([c,r])
    if not desc: return np.zeros((0,2)), np.zeros((0,64))
    return np.array(coords,float), np.stack(desc)

def align_score(f1,f2,ratio=0.85,thr=1.5):
    c1,d1=f1; c2,d2=f2
    if len(d1)<4 or len(d2)<4: return 0
    S=d1@d2.T  # (N1,N2) 余弦相似
    order=np.argsort(-S,axis=1)
    good=[]
    for i in range(len(d1)):
        j1=order[i,0]; j2=order[i,1] if S.shape[1]>1 else j1
        # 比值检验: best余弦距离明显小于2nd (1-S 越小越近)
        if (1-S[i,j1]) < ratio*(1-S[i,j2]):
            good.append((c1[i],c2[j1]))
    if len(good)<4: return 0
    src=np.float32([g[0] for g in good]).reshape(-1,1,2); dst=np.float32([g[1] for g in good]).reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=thr,maxIters=2000,confidence=0.99)
    return int(mask.sum()) if mask is not None else 0

def eer_far(g,i):
    g,i=np.array(g,float),np.array(i,float)
    y=np.concatenate([np.ones(len(g)),np.zeros(len(i))]); s=np.concatenate([g,i])
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; k=np.nanargmin(np.abs(fnr-fpr)); e=(fpr[k]+fnr[k])/2
    out={'eer':e}
    for tf in [0.03,0.05]:
        idx=np.argmin(np.abs(fnr-tf)); out[f'far{int(tf*100)}']=fpr[idx]
    return out

b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
random.seed(42)
def build(model):
    tmpl={}; test={}
    for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
        for fg in sorted(os.listdir(base)):
            if 'xzc' in fg.lower(): continue
            rp=os.path.join(base,fg,reg)
            if not os.path.exists(rp): continue
            ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
            if len(ps)<50: continue
            k=f'{pre}_{fg}'
            tmpl[k]=[local_desc(model,prep(load_img(ps[i]))) for i in random.sample(range(70),6)]
            test[k]=[local_desc(model,prep(load_img(p))) for p in random.sample(ps[70:],min(6,len(ps)-70))]
    return tmpl,test
MERGE={'dy_R0':'wtp_dy_R0','dy_R1':'wtp_dy_R1','dy_R2':'wtp_dy_R2','lwh_R0':'wtp_lwh_R0','lwh_R1':'wtp_lwh_R1',
       'lwh_R2':'wtp_lwh_R2','zyh_R0':'btp_zyh_R0','zyh_R1':'btp_zyh_R1','zyh_R2':'btp_zyh_R2'}
nb='f:/1111/指纹/ysjz_raw/ysjz'
def build_cross(model):
    cross={}
    for folder,gt in MERGE.items():
        cross.setdefault(gt,[])
        for p in sorted(glob.glob(os.path.join(nb,folder,'*.bmp')))[70:][:12]:
            cross[gt].append(local_desc(model,prep(load_img(p))))
    return cross
def evaluate(name,tmpl,probes):
    ids=list(tmpl.keys()); g,i=[],[]
    for gt,plist in probes.items():
        for pe in plist:
            for cid in ids:
                sc=max(align_score(pe,t) for t in tmpl[cid])
                (g if cid==gt else i).append(sc)
    r=eer_far(g,i)
    print(f"  {name}: genuine inliers mean={np.mean(g):.1f} | impostor mean={np.mean(i):.1f} max={np.max(i):.0f} | "
          f"EER={r['eer']*100:.2f}% FAR@FFR3%={r['far3']*100:.2f}% FAR@FFR5%={r['far5']*100:.2f}%"); sys.stdout.flush()

for mname,mlist in MODELS.items():
    m=mlist[0]  # 单seed足够看趋势
    print(f"\n=== {mname} learned local descriptors + RANSAC alignment ==="); sys.stdout.flush()
    t0=time.time()
    tmpl,test=build(m); cross=build_cross(m)
    evaluate("SINGLE-SESSION", tmpl, test)
    evaluate("CROSS-SESSION", tmpl, cross)
    print(f"  (t={time.time()-t0:.0f}s)")
print("\nDone.")
