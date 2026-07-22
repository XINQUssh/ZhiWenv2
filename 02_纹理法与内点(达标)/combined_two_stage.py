# -*- coding: utf-8 -*-
"""
两级"全局+局部"流水:
  Stage1 全局预筛: V21+V18b融合嵌入余弦 -> 阈值Tg, 过滤明显不匹配
  Stage2 局部终判: 自训纹理描述子 + 变换约束几何验证 -> 内点数(FAR=0判定)
  final(probe,class) = local_inlier  if global_cos>=Tg  else 0
逐(probe,class)同时算两者, 扫Tg看能否比纯局部更好。
"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob, random, sys, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
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
    e=clahe(img); u=up2(e); m=gmask(u); mk=u.copy(); mk[m==0]=0
    v=mk[m>0].astype(np.float32); mu,sd=(v.mean(),v.std()+1e-6) if len(v) else (0,1)
    norm=(mk.astype(np.float32)-mu)/sd; norm[m==0]=0
    return norm,m
def _stem(s):
    b=resnet18(weights=None)
    s.conv1=nn.Conv2d(1,64,7,2,3,bias=False); s.bn1,s.relu,s.maxpool=b.bn1,b.relu,b.maxpool
    s.layer1,s.layer2,s.layer3,s.layer4=b.layer1,b.layer2,b.layer3,b.layer4; s.avgpool=nn.AdaptiveAvgPool2d((1,1))
class DenseEnc(nn.Module):
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s); s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e)); s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x))); return s.global_projector(s.avgpool(s.layer4(x)).flatten(1))
class HybridEnc(nn.Module):
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s); s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.attention=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(),nn.Conv2d(128,1,1),nn.Softplus()); s.attn_projector=nn.Sequential(nn.Linear(256,e),nn.BatchNorm1d(e)); s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x))); return s.global_projector(s.avgpool(s.layer4(x)).flatten(1))
class Desc(nn.Module):
    def __init__(s,d=128):
        super().__init__()
        def cbr(i,o,st=1): return [nn.Conv2d(i,o,3,st,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
        s.net=nn.Sequential(*cbr(1,32),*cbr(32,32),*cbr(32,64,2),*cbr(64,64),*cbr(64,128,2),*cbr(128,128),nn.AdaptiveAvgPool2d(1)); s.fc=nn.Linear(128,d)
    def forward(s,x): z=s.net(x).flatten(1); return F.normalize(s.fc(z),dim=1)
dev=torch.device('cuda')
def lf(cls,d,pre):
    out=[]
    for sd in [42,123,777]:
        m=cls().to(dev); m.load_state_dict(torch.load(f'f:/1111/指纹/{d}/{pre}_seed{sd}.pth',map_location=dev,weights_only=True),strict=False); m.eval(); out.append(m)
    return out
GMODELS=lf(HybridEnc,'models_v21','v21')+lf(DenseEnc,'models_v18b','v18b')
texnet=Desc().to(dev); texnet.load_state_dict(torch.load('f:/1111/指纹/models_texdesc/texdesc.pth',map_location=dev)); texnet.eval()
print("loaded global(V21+V18b)+texdesc"); sys.stdout.flush()
PS=32; HALF=16; STRIDE=6
@torch.no_grad()
def feat(img):
    norm,m=prep(img); H,W=norm.shape
    # global fused emb
    gs=[]
    for mm in GMODELS:
        ge=[]
        for pol in (1,-1):
            x=torch.tensor(pol*norm,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev); ge.append(F.normalize(mm(x),dim=1))
        gs.append(F.normalize(torch.mean(torch.stack(ge),dim=0),dim=1).cpu().numpy().flatten())
    gf=np.mean(gs,axis=0); gf/=np.linalg.norm(gf)+1e-8
    # texture dense desc
    pts=[]; patches=[]
    for y in range(HALF,H-HALF,STRIDE):
        for x in range(HALF,W-HALF,STRIDE):
            if m[y,x]>0: pts.append((x,y)); patches.append(norm[y-HALF:y+HALF,x-HALF:x+HALF])
    if len(patches)<4: return gf,(np.zeros((0,2),np.float32),None)
    X=torch.tensor(np.stack(patches)[:,None],dtype=torch.float32).to(dev); des=[]
    for i in range(0,len(X),1024): des.append(texnet(X[i:i+1024]).cpu().numpy())
    return gf,(np.float32(pts),np.concatenate(des).astype(np.float32))
bf=cv2.BFMatcher(cv2.NORM_L2)
def inl(f1,f2,ratio=0.92,reproj=6,slo=0.85,shi=1.18,rotmax=30):
    p1,d1=f1; p2,d2=f2
    if d1 is None or d2 is None or len(p1)<4 or len(p2)<4: return 0
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0
    src=p1[[m.queryIdx for m in g]].reshape(-1,1,2); dst=p2[[m.trainIdx for m in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=3000,confidence=0.99)
    if M is None or mask is None: return 0
    a,b=M[0,0],M[1,0]; sc=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a))); rot=min(rot,360-rot)
    if sc<slo or sc>shi or rot>rotmax: return 0
    return int(mask.sum())
b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'; b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
cp={}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if os.path.exists(rp):
            ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
            if len(ps)>=50: cp[f'{pre}_{fg}']=ps
ids=list(cp.keys()); t0=time.time(); NT=40; idx=np.linspace(0,69,NT).astype(int)
print("extract..."); sys.stdout.flush()
tg={}; tl={}; te={}
for k in ids:
    fs=[feat(load_img(cp[k][i])) for i in idx]; tg[k]=[f[0] for f in fs]; tl[k]=[f[1] for f in fs]
    te[k]=[feat(load_img(p)) for p in cp[k][70:]]
print(f"  done t={time.time()-t0:.0f}s"); sys.stdout.flush()
G_g,G_l,I_g,I_l=[],[],[],[]; gfinger=[]
for pc in ids:
    for gemb,tex in te[pc]:
        for cc in ids:
            gc=max(float(gemb@t) for t in tg[cc]); li=max(inl(tex,t) for t in tl[cc])
            if cc==pc: G_g.append(gc); G_l.append(li); gfinger.append(pc)
            else: I_g.append(gc); I_l.append(li)
    print(f"  scored {pc} t={time.time()-t0:.0f}s"); sys.stdout.flush()
G_g,G_l,I_g,I_l=map(np.array,[G_g,G_l,I_g,I_l])
np.savez('f:/1111/指纹/two_stage_scores.npz',G_g=G_g,G_l=G_l,I_g=I_g,I_l=I_l,gfinger=np.array(gfinger))
def far0_frr(gl,il): thr=int(il.max())+1; return thr,(gl<thr).mean()
thr,frr=far0_frr(G_l,I_l)
print(f"\n[纯局部] FAR=0 thr={thr} FRR={frr*100:.3f}% (imp_local_max={I_l.max()})")
print("[两级 全局预筛Tg + 局部] 扫Tg:")
best=None
for Tg in [-1,0.0,0.2,0.3,0.4,0.5,0.6]:
    gl=np.where(G_g>=Tg,G_l,0); il=np.where(I_g>=Tg,I_l,0)
    thr=int(il.max())+1; frr=(gl<thr).mean()
    print(f"  Tg={Tg}: imp_local_max(过滤后)={il.max()}, FAR=0 thr={thr}, FRR={frr*100:.3f}% {'达标' if frr<0.03 else ''}")
    if best is None or frr<best[1]: best=(Tg,frr,thr)
print(f"\n*** 最佳两级: Tg={best[0]} FAR=0 FRR={best[1]*100:.3f}% ***")
# per-finger at best
Tg=best[0]; gl=np.where(G_g>=Tg,G_l,0); il=np.where(I_g>=Tg,I_l,0); thr=int(il.max())+1
from collections import defaultdict
rej=defaultdict(int); tot=defaultdict(int)
for i,pc in enumerate(gfinger):
    tot[pc]+=1
    if gl[i]<thr: rej[pc]+=1
print("per-finger rej:")
for pc in sorted(ids,key=lambda c:-rej[c]):
    if rej[pc]>0: print(f"  {pc:14s}: {rej[pc]}/{tot[pc]}")
print(f"t={time.time()-t0:.0f}s\nDone.")
