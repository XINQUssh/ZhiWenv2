# -*- coding: utf-8 -*-
"""统一全匹配评估(无预筛, 全25指, float描述子, 重叠归一化)。
MODELP=模型路径, STRIDE=采样步长. 用于公平对比 baseline vs 微调模型。
用法: python full_eval.py <stride>   env: MODELP, TAG"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import sys, glob, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
def load(p):
    with open(p,'rb') as f: return cv2.imdecode(np.frombuffer(f.read(),np.uint8),cv2.IMREAD_GRAYSCALE)
def _clahe(img): return cv2.createCLAHE(4.0,(8,8)).apply(img)
def _up2(img): return cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
def _keep(m):
    n,l,s,_=cv2.connectedComponentsWithStats(m,8,cv2.CV_32S)
    return m if n<=1 else ((l==1+np.argmax(s[1:,cv2.CC_STAT_AREA]))*255).astype(np.uint8)
def _fill(m):
    bg=cv2.bitwise_not(m); n,l,s,_=cv2.connectedComponentsWithStats(bg,8,cv2.CV_32S); h,w=m.shape
    for i in range(1,n):
        L,T,W,H=s[i,cv2.CC_STAT_LEFT],s[i,cv2.CC_STAT_TOP],s[i,cv2.CC_STAT_WIDTH],s[i,cv2.CC_STAT_HEIGHT]
        if L>0 and L+W<w-1 and T>0 and T+H<h-1: m[l==i]=255
    return m
def _gm(img,sig=13/3.,pct=95,r=.2):
    dx=cv2.Sobel(img,cv2.CV_32F,1,0,3); dy=cv2.Sobel(img,cv2.CV_32F,0,1,3); mg=cv2.magnitude(dx,dy)
    gs=int(np.ceil(3*sig))*2+1; ma=cv2.GaussianBlur(mg,(gs,gs),sig)
    th=np.percentile(mg.flatten(),pct)*r; _,m=cv2.threshold(ma,th,255,0); m=m.astype(np.uint8)
    se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,se,iterations=6); m=_keep(m); m=_fill(m)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,se,iterations=2); return _keep(m)
def prep(img):
    e=_clahe(img); u=_up2(e); m=_gm(u); mk=u.copy(); mk[m==0]=0
    v=mk[m>0].astype(np.float32); mu,sd=(v.mean(),v.std()+1e-6) if len(v) else (0,1)
    norm=(mk.astype(np.float32)-mu)/sd; norm[m==0]=0
    return norm,m
class Desc(nn.Module):
    def __init__(s,d=128):
        super().__init__()
        def cbr(i,o,st=1): return [nn.Conv2d(i,o,3,st,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
        s.net=nn.Sequential(*cbr(1,32),*cbr(32,32),*cbr(32,64,2),*cbr(64,64),*cbr(64,128,2),*cbr(128,128),nn.AdaptiveAvgPool2d(1)); s.fc=nn.Linear(128,d)
    def forward(s,x): z=s.net(x).flatten(1); return F.normalize(s.fc(z),dim=1)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELP=os.environ.get('MODELP','f:/1111/指纹/deliver/best.pth'); TAG=os.environ.get('TAG','model')
net=Desc().to(dev); net.load_state_dict(torch.load(MODELP,map_location=dev)); net.eval()
HALF=16; STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 8; HO=20000.0
bf=cv2.BFMatcher(cv2.NORM_L2)
@torch.no_grad()
def feat(img):
    norm,m=prep(img); H,W=norm.shape; pts=[]; pat=[]
    for y in range(HALF,H-HALF,STRIDE):
        for x in range(HALF,W-HALF,STRIDE):
            if m[y,x]>0: pts.append((x,y)); pat.append(norm[y-HALF:y+HALF,x-HALF:x+HALF])
    if len(pat)<4: return None
    X=torch.tensor(np.stack(pat)[:,None],dtype=torch.float32).to(dev); d=[]
    for i in range(0,len(X),2048): d.append(net(X[i:i+2048]).cpu().numpy())
    return (np.float32(pts), np.concatenate(d).astype(np.float32), m)
def _score(f1,f2,ratio=0.92,reproj=6):
    """返回 (raw_inlier, overlap_norm) 两种分数"""
    p1,d1,m1=f1; p2,d2,m2=f2
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0.0,0.0
    src=p1[[x.queryIdx for x in g]].reshape(-1,1,2); dst=p2[[x.trainIdx for x in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=2000,confidence=0.99)
    if M is None or mask is None: return 0.0,0.0
    a,b=M[0,0],M[1,0]; sc=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a))); rot=min(rot,360-rot)
    if sc<0.85 or sc>1.18 or rot>30: return 0.0,0.0
    inl=int(mask.sum())
    if inl==0: return 0.0,0.0
    h,w=m2.shape; w1=cv2.warpAffine(m1,M,(w,h),flags=cv2.INTER_NEAREST); ov=int(np.count_nonzero((w1>0)&(m2>0)))
    ovn=inl*float(np.sqrt(HO/ov)) if ov>=500 else float(inl)
    return float(inl), ovn
SEL='f:/1111/指纹/select_data/select'
fingers=[]
for fg in sorted(os.listdir(SEL)):
    d=os.path.join(SEL,fg); ps=sorted(glob.glob(os.path.join(d,'*.bmp')))
    if os.path.isdir(d) and len(ps)>=15: fingers.append((fg,ps))
ids=[f[0] for f in fingers]
RES=open(f'f:/1111/指纹/_full_{TAG}.txt','w',encoding='utf-8')
def out(*a):
    s=' '.join(str(x) for x in a); print(s); RES.write(s+'\n'); RES.flush(); sys.stdout.flush()
out(f"全匹配eval TAG={TAG} model={os.path.basename(MODELP)} stride={STRIDE}")
t0=time.time(); TM={}; PR={}
for fg,ps in fingers:
    ne=int(len(ps)*0.7); pool=ps[:ne]; pr=ps[ne:]
    idx=np.linspace(0,len(pool)-1,min(20,len(pool))).astype(int)
    TM[fg]=[f for f in (feat(load(pool[i])) for i in idx) if f]
    PR[fg]=[f for f in (feat(load(p)) for p in pr) if f]
out(f"feat done t={time.time()-t0:.0f}s")
# 同时收集 raw_inlier 与 overlap_norm 两套分数
genR,impR,genO,impO=[],[],[],[]
for pc in ids:
    for pf in PR[pc]:
        for cc in ids:
            br=bo=0.0
            for t in TM[cc]:
                r,o=_score(pf,t)
                if r>br: br=r
                if o>bo: bo=o
            if cc==pc: genR.append(br); genO.append(bo)
            else: impR.append(br); impO.append(bo)
    out(f"  scored {pc} t={time.time()-t0:.0f}s")
def report(name,gen,imp):
    gen,imp=np.array(gen),np.array(imp)
    allv=np.unique(np.concatenate([gen,imp])); far=np.array([(imp>=v).mean() for v in allv]); frr=np.array([(gen<v).mean() for v in allv])
    ei=np.nanargmin(np.abs(far-frr))
    out(f"\n[{TAG}|{name}] GEN n={len(gen)} mean={gen.mean():.1f} med={np.median(gen):.1f} | IMP mean={imp.mean():.2f} max={imp.max():.2f}")
    out(f"  EER={(far[ei]+frr[ei])/2*100:.3f}%  FAR=0 FRR={(gen<imp.max()+1e-6).mean()*100:.3f}%")
    for tgt in [0.002,0.01,0.1]:
        ok=np.where(far<=tgt/100)[0]
        if len(ok): out(f"  FAR<={tgt}%: FRR={(gen<allv[ok[0]]).mean()*100:.3f}%")
    return gen,imp
gR,iR=report('raw_inlier',genR,impR)
gO,iO=report('overlap_norm',genO,impO)
np.save(f'f:/1111/指纹/_genR_{TAG}.npy',gR); np.save(f'f:/1111/指纹/_impR_{TAG}.npy',iR)
np.save(f'f:/1111/指纹/_genO_{TAG}.npy',gO); np.save(f'f:/1111/指纹/_impO_{TAG}.npy',iO)
out(f"t={time.time()-t0:.0f}s\nDone.")
