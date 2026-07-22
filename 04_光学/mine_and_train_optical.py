# -*- coding: utf-8 -*-
"""光学域重训: 在 optical_center_DB1_A 上挖真实对应点 + warm-start重训。
env: DATADIR, CURM, OUTM, ITERS, WARM, LR"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob, random, sys, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
def load_img(p):
    with open(p,'rb') as f: return cv2.imdecode(np.frombuffer(f.read(),np.uint8), cv2.IMREAD_GRAYSCALE)
def clahe(img): return cv2.createCLAHE(4.0,(8,8)).apply(img)
def up2(img): return cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
def keep_cc(m):
    n,l,s,_=cv2.connectedComponentsWithStats(m,8,cv2.CV_32S)
    return m if n<=1 else ((l==1+np.argmax(s[1:,cv2.CC_STAT_AREA]))*255).astype(np.uint8)
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
class Desc(nn.Module):
    def __init__(s,d=128):
        super().__init__()
        def cbr(i,o,st=1): return [nn.Conv2d(i,o,3,st,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
        s.net=nn.Sequential(*cbr(1,32),*cbr(32,32),*cbr(32,64,2),*cbr(64,64),*cbr(64,128,2),*cbr(128,128),nn.AdaptiveAvgPool2d(1)); s.fc=nn.Linear(128,d)
    def forward(s,x): z=s.net(x).flatten(1); return F.normalize(s.fc(z),dim=1)
dev=torch.device('cuda')
DATADIR=os.environ.get('DATADIR','f:/1111/指纹/optical_center_DB1_A')
CURM=os.environ.get('CURM','f:/1111/指纹/deliver/best.pth')
OUTM=os.environ.get('OUTM','f:/1111/指纹/models_texdesc/optical_boot1.pth')
WARM=os.environ.get('WARM','1')=='1'
cur=Desc().to(dev); cur.load_state_dict(torch.load(CURM,map_location=dev)); cur.eval()
print(f"mining-with={os.path.basename(CURM)} data={os.path.basename(DATADIR)} -> {os.path.basename(OUTM)}"); sys.stdout.flush()
PS=32; HALF=16; STRIDE=8
@torch.no_grad()
def dense(norm,m):
    H,W=norm.shape; pts=[]; pat=[]
    for y in range(HALF,H-HALF,STRIDE):
        for x in range(HALF,W-HALF,STRIDE):
            if m[y,x]>0: pts.append((x,y)); pat.append(norm[y-HALF:y+HALF,x-HALF:x+HALF])
    if len(pat)<4: return np.zeros((0,2),np.float32),np.zeros((0,128),np.float32)
    X=torch.tensor(np.stack(pat)[:,None],dtype=torch.float32).to(dev); d=[]
    for i in range(0,len(X),2048): d.append(cur(X[i:i+2048]).cpu().numpy())
    return np.float32(pts),np.concatenate(d).astype(np.float32)
bf=cv2.BFMatcher(cv2.NORM_L2)
def align_corr(fa,fb):
    p1,d1=fa; p2,d2=fb
    if len(p1)<8 or len(p2)<8: return None
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<0.9*b.distance]
    if len(g)<8: return None
    src=p1[[mm.queryIdx for mm in g]].reshape(-1,1,2); dst=p2[[mm.trainIdx for mm in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=3000,confidence=0.99)
    if M is None or mask is None or mask.sum()<20: return None
    a,b=M[0,0],M[1,0]; sc=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a))); rot=min(rot,360-rot)
    if sc<0.85 or sc>1.18 or rot>30: return None
    mk=mask.ravel().astype(bool)
    return src[mk,0,:], dst[mk,0,:]
print("load + mine (光学center, 每指~30crop, all-pairs)..."); sys.stdout.flush(); t0=time.time()
frames=[]; corrs=[]; centers=[]; fcnt=0
for fg in sorted(os.listdir(DATADIR)):
    d=os.path.join(DATADIR,fg)
    if not os.path.isdir(d): continue
    ps=sorted(glob.glob(os.path.join(d,'*.png')))
    if len(ps)<6: continue
    sel=[ps[i] for i in np.linspace(0,len(ps)-1,min(30,len(ps))).astype(int)]
    loc=[]
    for p in sel:
        norm,m=prep(load_img(p)); fi=len(frames); frames.append(norm); loc.append((fi,dense(norm,m)))
        pts=loc[-1][1][0]
        if len(pts):
            for j in np.random.RandomState(fi).choice(len(pts),min(30,len(pts)),replace=False): centers.append((fi,float(pts[j,0]),float(pts[j,1])))
    for ri in range(len(loc)):
        for j in range(ri+1,len(loc)):
            r=align_corr(loc[ri][1], loc[j][1])
            if r is None: continue
            sa,db=r
            for k in range(len(sa)):
                corrs.append((loc[ri][0],float(sa[k,0]),float(sa[k,1]),loc[j][0],float(db[k,0]),float(db[k,1])))
    fcnt+=1
print(f"  {fcnt} fingers, {len(frames)} frames, {len(corrs)} real corr, {len(centers)} centers, t={time.time()-t0:.0f}s"); sys.stdout.flush()
H,W=frames[0].shape
def patch(fi,x,y,jit=2):
    x=int(round(x))+random.randint(-jit,jit); y=int(round(y))+random.randint(-jit,jit)
    x=max(HALF,min(W-HALF,x)); y=max(HALF,min(H-HALF,y))
    return frames[fi][y-HALF:y+HALF, x-HALF:x+HALF]
def photo(p):
    p=p.copy()
    if random.random()<0.5: p=-p
    if random.random()<0.5: p=p*random.uniform(0.7,1.3)
    if random.random()<0.3: p=np.sign(p)*(np.abs(p)**random.uniform(0.7,1.4))
    if random.random()<0.4:
        ang=random.uniform(-12,12); p=cv2.warpAffine(p,cv2.getRotationMatrix2D((HALF,HALF),ang,1.0),(PS,PS),borderValue=0)
    return p+np.random.randn(PS,PS).astype(np.float32)*0.04
net=Desc().to(dev)
if WARM: net.load_state_dict(torch.load(CURM,map_location=dev)); print("warm-start")
opt=torch.optim.AdamW(net.parameters(),lr=float(os.environ.get('LR','3e-4')),weight_decay=1e-4)
B=384; ITERS=int(os.environ.get('ITERS','5000')); random.seed(1); np.random.seed(1); torch.manual_seed(1)
print(f"train ITERS={ITERS}..."); sys.stdout.flush()
for it in range(ITERS):
    A=np.zeros((B,1,PS,PS),np.float32); P=np.zeros((B,1,PS,PS),np.float32)
    for k in range(B):
        if random.random()<0.6 and corrs:
            fa,xa,ya,fb,xb,yb=corrs[random.randint(0,len(corrs)-1)]
            A[k,0]=photo(patch(fa,xa,ya)); P[k,0]=photo(patch(fb,xb,yb))
        else:
            fi,x,y=centers[random.randint(0,len(centers)-1)]
            A[k,0]=patch(fi,x,y,0); P[k,0]=photo(patch(fi,x,y))
    A=torch.tensor(A).to(dev); P=torch.tensor(P).to(dev); da=net(A); dp=net(P)
    Dap=2-2*torch.mm(da,dp.t()); pos=torch.diag(Dap); eye=torch.eye(B,device=dev).bool()
    neg=torch.min(Dap.masked_fill(eye,1e4).min(1).values, Dap.masked_fill(eye,1e4).min(0).values)
    loss=F.relu(1.0+pos-neg).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (it+1)%1000==0:
        print(f"  it{it+1}/{ITERS} loss={loss.item():.4f} pos={pos.mean().item():.3f} neg={neg.mean().item():.3f} t={time.time()-t0:.0f}s"); sys.stdout.flush()
        torch.save(net.state_dict(),OUTM)
torch.save(net.state_dict(),OUTM)
print(f"saved {os.path.basename(OUTM)} t={time.time()-t0:.0f}s\nDone.")
