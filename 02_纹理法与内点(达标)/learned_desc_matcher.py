# -*- coding: utf-8 -*-
"""
决定性尝试: SIFT关键点位置 + 我们训练的局部描述子(V21 local_head, 关键点处grid_sample)
+ 变换约束几何验证。= "训练局部描述子+几何验证"配方, 看能否救2根坏手指的genuine内点。
对比纯SIFT描述子(FAR=0@FRR6.4%)。
"""
import os, glob, random, sys, time
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
    return norm,u,m
def _stem(s):
    b=resnet18(weights=None)
    s.conv1=nn.Conv2d(1,64,7,2,3,bias=False); s.bn1,s.relu,s.maxpool=b.bn1,b.relu,b.maxpool
    s.layer1,s.layer2,s.layer3,s.layer4=b.layer1,b.layer2,b.layer3,b.layer4; s.avgpool=nn.AdaptiveAvgPool2d((1,1))
class HybridEnc(nn.Module):
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.attention=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(),nn.Conv2d(128,1,1),nn.Softplus())
        s.attn_projector=nn.Sequential(nn.Linear(256,e),nn.BatchNorm1d(e)); s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        return F.normalize(s.local_head(x),dim=1)  # (B,64,H,W)
dev=torch.device('cuda')
model=HybridEnc().to(dev); model.load_state_dict(torch.load('f:/1111/指纹/models_v21/v21_seed42.pth',map_location=dev,weights_only=True),strict=False); model.eval()
print("V21 local head loaded"); sys.stdout.flush()

sift=cv2.SIFT_create()
@torch.no_grad()
def feat(img):
    norm,u,m=prep(img)
    s=u.copy(); s[m==0]=0
    kp=sift.detect(s,(m>0).astype(np.uint8)*255)
    if len(kp)<4: return (np.zeros((0,2),np.float32),None)
    pts=np.float32([k.pt for k in kp])  # (N,2) in 220x200
    ld=model(torch.tensor(norm,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev))  # (1,64,H,W)
    H,Wd=ld.shape[2],ld.shape[3]
    gx=pts[:,0]/(2*norm.shape[1]/2)  # 占位
    # 归一化关键点坐标到[-1,1] (grid_sample: x=W维, y=H维)
    nx=pts[:,0]/(norm.shape[1]-1)*2-1; ny=pts[:,1]/(norm.shape[0]-1)*2-1
    grid=torch.tensor(np.stack([nx,ny],axis=1),dtype=torch.float32).view(1,1,-1,2).to(dev)
    samp=F.grid_sample(ld,grid,mode='bilinear',align_corners=True)  # (1,64,1,N)
    desc=F.normalize(samp[0,:,0,:].t(),dim=1).cpu().numpy()  # (N,64)
    return (pts,desc)

bfL=cv2.BFMatcher(cv2.NORM_L2)
def match(f1,f2,ratio=0.85,reproj=6,slo=0.85,shi=1.18,rotmax=30):
    p1,d1=f1; p2,d2=f2
    if d1 is None or d2 is None or len(p1)<4 or len(p2)<4: return 0
    knn=bfL.knnMatch(d1.astype(np.float32),d2.astype(np.float32),k=2)
    g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0
    src=p1[[m.queryIdx for m in g]].reshape(-1,1,2); dst=p2[[m.trainIdx for m in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=3000,confidence=0.99)
    if M is None or mask is None: return 0
    a,b=M[0,0],M[1,0]; scale=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a))); rot=min(rot,360-rot)
    if scale<slo or scale>shi or rot>rotmax: return 0
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
ids=list(cp.keys()); random.seed(42); NT=20; t0=time.time()
print("extract..."); sys.stdout.flush()
tmpl={k:[feat(load_img(cp[k][i])) for i in random.sample(range(70),NT)] for k in ids}
test={k:[feat(load_img(p)) for p in cp[k][70:]] for k in ids}
print(f"  done t={time.time()-t0:.0f}s"); sys.stdout.flush()
gen,imp=[],[]; gen_by={k:[] for k in ids}
for pc in ids:
    for pe in test[pc]:
        for cc in ids:
            sc=max(match(pe,t) for t in tmpl[cc])
            if cc==pc: gen.append(sc); gen_by[pc].append(sc)
            else: imp.append(sc)
gen,imp=np.array(gen),np.array(imp)
print(f"\n[V21 learned desc @SIFT kp] GENUINE n={len(gen)} mean={gen.mean():.1f} min={gen.min()} | IMPOSTOR max={imp.max()} mean={imp.mean():.2f}")
thr0=int(imp.max())+1
print(f"*** FAR=0: thr={thr0} FRR={(gen<thr0).mean()*100:.3f}% {'达标' if (gen<thr0).mean()<0.03 else ''} ***")
print("2根坏手指genuine内点:")
for k in ['btp_fys_R0','wtp_xyz_R1']:
    a=np.array(gen_by[k]); print(f"  {k}: mean={a.mean():.1f} min={a.min()} rej@{thr0}={sum(a<thr0)}/{len(a)}")
print(f"t={time.time()-t0:.0f}s\nDone.")
