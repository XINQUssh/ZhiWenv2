# -*- coding: utf-8 -*-
"""诊断: (A)随机split vs 时序split 的genuine质量; (B)最高impostor指对(标注/混淆排查)。
model=best.pth stride8 overlap_norm。"""
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
MODELP=os.environ.get('MODELP','f:/1111/指纹/deliver/best.pth')
net=Desc().to(dev); net.load_state_dict(torch.load(MODELP,map_location=dev)); net.eval()
HALF=16; STRIDE=8; HO=20000.0; bf=cv2.BFMatcher(cv2.NORM_L2)
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
def score(f1,f2,ratio=0.92,reproj=6):
    p1,d1,m1=f1; p2,d2,m2=f2
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0.0,0,0
    src=p1[[x.queryIdx for x in g]].reshape(-1,1,2); dst=p2[[x.trainIdx for x in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=2000,confidence=0.99)
    if M is None or mask is None: return 0.0,0,0
    a,b=M[0,0],M[1,0]; sc=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a))); rot=min(rot,360-rot)
    if sc<0.85 or sc>1.18 or rot>30: return 0.0,0,0
    inl=int(mask.sum())
    if inl==0: return 0.0,0,0
    h,w=m2.shape; w1=cv2.warpAffine(m1,M,(w,h),flags=cv2.INTER_NEAREST); ov=int(np.count_nonzero((w1>0)&(m2>0)))
    return (inl*float(np.sqrt(HO/ov)) if ov>=500 else float(inl)), inl, ov
SEL='f:/1111/指纹/select_data/select'
fingers=[]
for fg in sorted(os.listdir(SEL)):
    d=os.path.join(SEL,fg); ps=sorted(glob.glob(os.path.join(d,'*.bmp')))
    if os.path.isdir(d) and len(ps)>=15: fingers.append((fg,ps))
ids=[f[0] for f in fingers]
RES=open('f:/1111/指纹/_diag_split.txt','w',encoding='utf-8')
def out(*a):
    s=' '.join(str(x) for x in a); print(s); RES.write(s+'\n'); RES.flush()
out(f"diag model={os.path.basename(MODELP)} stride={STRIDE}")
t0=time.time()
ALL={}
for fg,ps in fingers:
    ALL[fg]=[f for f in (feat(load(p)) for p in ps) if f]
out(f"feat all done t={time.time()-t0:.0f}s")

# (A) 时序 vs 随机 split 的 genuine 质量
def gen_quality(split):
    rng=np.random.RandomState(0); g_t={};
    res={}
    for fg in ids:
        fs=ALL[fg]; n=len(fs)
        if split=='temporal':
            ne=int(n*0.7); poolidx=list(range(ne)); pridx=list(range(ne,n))
        else:
            perm=rng.permutation(n); ne=int(n*0.7); poolidx=list(perm[:ne]); pridx=list(perm[ne:])
        tidx=[poolidx[i] for i in np.linspace(0,len(poolidx)-1,min(20,len(poolidx))).astype(int)]
        tm=[fs[i] for i in tidx]; pr=[fs[i] for i in pridx]
        sc=[max((score(p,t)[0] for t in tm),default=0.0) for p in pr]
        res[fg]=np.mean(sc) if sc else 0
    return res
qt=gen_quality('temporal'); qr=gen_quality('random')
out("\n[A] genuine 平均分: 时序split vs 随机split (per finger)")
out(f"  {'finger':10s} temporal random")
for fg in ids: out(f"  {fg:10s} {qt[fg]:7.1f} {qr[fg]:7.1f}")
out(f"  >> 总体均值 temporal={np.mean(list(qt.values())):.1f}  random={np.mean(list(qr.values())):.1f}")

# (B) 最高 impostor 指对: 用每指中间帧作代表, 跨指匹配
out("\n[B] 最高 impostor 指对 (代表帧两两匹配, score, inl, overlap)")
rep={fg:ALL[fg][len(ALL[fg])//2] for fg in ids}  # 每指中间帧
pairs=[]
for i in range(len(ids)):
    for j in range(len(ids)):
        if i==j: continue
        s,inl,ov=score(rep[ids[i]],rep[ids[j]])
        pairs.append((s,inl,ov,ids[i],ids[j]))
pairs.sort(reverse=True)
for s,inl,ov,a,b in pairs[:15]:
    out(f"  {a:10s} vs {b:10s}  score={s:6.1f} inl={inl:3d} ov={ov}")
out(f"t={time.time()-t0:.0f}s\nDone.")
