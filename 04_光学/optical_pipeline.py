# -*- coding: utf-8 -*-
"""光学指纹 DB1_A: (1)按手指整理到子文件夹; (2)跨impression协议测FRR/FAR。
用法: python optical_pipeline.py center|random"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import sys, re, time, zipfile, collections
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
TYPE=sys.argv[1] if len(sys.argv)>1 else 'center'
ZP=f'f:/1111/指纹/cut_{TYPE}.zip'
OUTDIR=f'f:/1111/指纹/optical_{TYPE}_DB1_A'   # 整理后: <finger>/<file>
def load_bytes(b): return cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_GRAYSCALE)
# ---- (1) 整理到子文件夹 (按手指) ----
z=zipfile.ZipFile(ZP)
pat=re.compile(rf'(\d+)_(\d+)_{TYPE}_\((\d+)\)')
byfinger=collections.defaultdict(list)   # finger -> [(imp,crop,zipname)]
for n in z.namelist():
    nm=n.replace('\\','/')
    if nm.endswith('.png') and nm.split('/')[1]=='DB1_A':
        m=pat.search(nm.split('/')[-1])
        if m: byfinger[m.group(1)].append((int(m.group(2)),int(m.group(3)),n))
if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)
    for fg,items in byfinger.items():
        fd=os.path.join(OUTDIR,fg); os.makedirs(fd,exist_ok=True)
        for imp,cr,zn in items:
            with open(os.path.join(fd,os.path.basename(zn.replace('\\','/'))),'wb') as f: f.write(z.read(zn))
    print(f"整理完成 -> {OUTDIR}/<手指>/  ({len(byfinger)}指)")
else:
    print(f"已存在 {OUTDIR}")
# ---- 匹配器 (纹理法 best.pth) ----
def _clahe(i): return cv2.createCLAHE(4.0,(8,8)).apply(i)
def _up2(i): return cv2.resize(i,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
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
def _score(f1,f2,ratio=0.92,reproj=6):
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
    return float(inl), (inl*float(np.sqrt(HO/ov)) if ov>=500 else float(inl))
RES=open(f'f:/1111/指纹/_optical_{TYPE}.txt','w',encoding='utf-8')
def out(*a):
    s=' '.join(str(x) for x in a); print(s); RES.write(s+'\n'); RES.flush()
# ---- (2) 跨impression协议 ----
import glob
finger_imps={}  # finger -> {imp:[filepaths]}
for fg in sorted(os.listdir(OUTDIR)):
    fd=os.path.join(OUTDIR,fg)
    if not os.path.isdir(fd): continue
    imps=collections.defaultdict(list)
    for p in glob.glob(os.path.join(fd,'*.png')):
        m=pat.search(os.path.basename(p))
        if m: imps[int(m.group(2))].append(p)
    if len(imps)>=2: finger_imps[fg]=imps   # 需≥2次impression才能跨场景
ids=list(finger_imps.keys())
out(f"光学 cut_{TYPE} DB1_A: {len(ids)}指(>=2 impression) model={os.path.basename(MODELP)}")
t0=time.time(); TM={}; PR={}
rng=np.random.RandomState(0)
for fg in ids:
    imps=finger_imps[fg]; ord_imp=sorted(imps.keys())
    ne=max(1,int(len(ord_imp)*0.6)); timp=ord_imp[:ne]; pimp=ord_imp[ne:]
    tmpl=[]
    for im in timp:
        for p in imps[im][:2]: tmpl.append(p)   # 每模板impression取2crop
    tmpl=tmpl[:12]
    prob=[]
    for im in pimp:
        for p in imps[im][:2]: prob.append(p)
    prob=prob[:5]
    TM[fg]=[f for f in (feat(load_bytes(open(p,'rb').read())) for p in tmpl) if f]
    PR[fg]=[f for f in (feat(load_bytes(open(p,'rb').read())) for p in prob) if f]
out(f"feat done t={time.time()-t0:.0f}s  (模板均值{np.mean([len(TM[f]) for f in ids]):.1f}/指, 探针{np.mean([len(PR[f]) for f in ids]):.1f}/指)")
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
    if (ids.index(pc)+1)%20==0: out(f"  scored {ids.index(pc)+1}/{len(ids)} t={time.time()-t0:.0f}s")
def rep(name,gen,imp):
    gen,imp=np.array(gen),np.array(imp)
    allv=np.unique(np.concatenate([gen,imp])); far=np.array([(imp>=v).mean() for v in allv]); frr=np.array([(gen<v).mean() for v in allv])
    ei=np.nanargmin(np.abs(far-frr))
    out(f"  [{name}] GEN n={len(gen)} mean={gen.mean():.1f} | IMP n={len(imp)} mean={imp.mean():.2f} max={imp.max():.2f}  (可分辨最小FAR≈{100/len(imp):.4f}%)")
    out(f"    EER={(far[ei]+frr[ei])/2*100:.3f}%  FAR=0 FRR={(gen<imp.max()+1e-6).mean()*100:.2f}%")
    for tgt in [0.1,0.01,0.002]:
        ok=np.where(far<=tgt/100)[0]
        out(f"    FAR<={tgt}% -> FRR={(gen<allv[ok[0]]).mean()*100:.2f}%" if len(ok) else f"    FAR<={tgt}% -> 样本不足")
out(f"\n==== 光学 cut_{TYPE} DB1_A 跨impression ====")
rep('raw_inlier',genR,impR); rep('overlap_norm',genO,impO)
out(f"t={time.time()-t0:.0f}s\nDone.")
