# -*- coding: utf-8 -*-
"""
贴屏指纹数据 FRR/FAR 自测工具（纹理法 best.pth）
用法:
  python fp_test.py <数据.zip或文件夹> [--noclahe] [--stride N] [--merge N]
说明:
  数据结构: 每个手指一个子文件夹, 帧命名任意 .bmp/.png
      data/  ├─ 手指A/ 1.bmp 2.bmp ...   ├─ 手指B/ ...
  --noclahe : 若数据已做过 CLAHE 增强, 加此项避免重复增强(降噪/原图不要加)
  --stride N: 【默认协议】每 N 张取 1 张作模板(N=5 → 100张出20模板), 其余全部作探针(默认 N=5)
  --merge N : 【可选】改为把连续 N 帧合并成一个模板(合并关键点); 不与 --stride 同时用
协议(默认): 每指每5张取1张作模板(共~20), 其余~80张作探针; 探针对每指取最高分; genuine=同指, impostor=异指。
输出: EER 及 FAR=1%/0.5%/0.1%/0.05%/0 时的 FRR。
需要: python + torch + opencv-python + numpy; 环境变量 KMP_DUPLICATE_LIB_OK=TRUE; best.pth 放本脚本同目录。
"""
import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import sys, glob, zipfile, tempfile
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
args=sys.argv[1:]
if not args: print(__doc__); sys.exit(0)
SRC=args[0]; NOCLAHE='--noclahe' in args
STRIDE_T=int(args[args.index('--stride')+1]) if '--stride' in args else 5   # 每N张取1作模板
MERGE=int(args[args.index('--merge')+1]) if '--merge' in args else 0        # >0则改为合并N帧模式
# 数据源: zip 或 文件夹
if SRC.lower().endswith('.zip'):
    tmp=tempfile.mkdtemp(); z=zipfile.ZipFile(SRC)
    for n in z.namelist():
        nm=n.replace('\\','/')
        if '__MACOSX' in nm or nm.split('/')[-1].startswith('._'): continue
        if nm.lower().endswith(('.bmp','.png')):
            fg=nm.split('/')[-2]; fn=nm.split('/')[-1]; d=os.path.join(tmp,fg); os.makedirs(d,exist_ok=True)
            open(os.path.join(d,fn),'wb').write(z.read(n))
    DIR=tmp
else: DIR=SRC
def load(p):
    with open(p,'rb') as f: return cv2.imdecode(np.frombuffer(f.read(),np.uint8),cv2.IMREAD_GRAYSCALE)
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
    e=img if NOCLAHE else _clahe(img)
    u=_up2(e); m=_gm(u); mk=u.copy(); mk[m==0]=0
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
net=Desc().to(dev); net.load_state_dict(torch.load(os.path.join(HERE,'best.pth'),map_location=dev)); net.eval()
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
def merge(fs):
    fs=[f for f in fs if f]
    if not fs: return None
    if len(fs)==1: return fs[0]
    pts=np.concatenate([f[0] for f in fs],0); desc=np.concatenate([f[1] for f in fs],0)
    m=fs[0][2].copy()
    for f in fs[1:]: m=cv2.bitwise_or(m,f[2])
    return (pts,desc,m)
def score(f1,f2,ratio=0.92,reproj=6):
    p1,d1,m1=f1; p2,d2,m2=f2
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0.0
    src=p1[[x.queryIdx for x in g]].reshape(-1,1,2); dst=p2[[x.trainIdx for x in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=2000,confidence=0.99)
    if M is None or mask is None: return 0.0
    a,b=M[0,0],M[1,0]; sc=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a))); rot=min(rot,360-rot)
    if sc<0.85 or sc>1.18 or rot>30: return 0.0
    inl=int(mask.sum())
    if inl==0: return 0.0
    h,w=m2.shape; w1=cv2.warpAffine(m1,M,(w,h),flags=cv2.INTER_NEAREST); ov=int(np.count_nonzero((w1>0)&(m2>0)))
    return inl*float(np.sqrt(HO/ov)) if ov>=500 else float(inl)
fingers={}
for fg in sorted(os.listdir(DIR)):
    d=os.path.join(DIR,fg); ps=sorted(glob.glob(os.path.join(d,'*.bmp'))+glob.glob(os.path.join(d,'*.png')))
    if os.path.isdir(d) and len(ps)>=10: fingers[fg]=ps
ids=list(fingers.keys())
print(f"数据: {DIR}  手指数={len(ids)}  NOCLAHE={NOCLAHE}  每模板{MERGE}帧")
TM={}; PR={}
print(f"协议: {'合并每'+str(MERGE)+'帧为一模板' if MERGE>0 else '每'+str(STRIDE_T)+'张取1作模板(共~20), 其余作探针'}")
for fg in ids:
    ps=fingers[fg]
    if MERGE>0:   # 可选: 合并模式
        ne=int(len(ps)*0.7); pool=ps[:ne]; pr=ps[ne:]
        fs=[feat(load(p)) for p in pool]; tmpl=[]
        for i in range(0,len(fs),MERGE):
            mt=merge(fs[i:i+MERGE])
            if mt: tmpl.append(mt)
            if len(tmpl)>=20: break
        TM[fg]=tmpl; PR[fg]=[f for f in (feat(load(p)) for p in pr[:8]) if f]
    else:         # 默认: 每STRIDE_T张取1作模板, 其余作探针
        tidx=list(range(0,len(ps),STRIDE_T))[:20]; tset=set(tidx)
        pidx=[i for i in range(len(ps)) if i not in tset]
        TM[fg]=[f for f in (feat(load(ps[i])) for i in tidx) if f]
        PR[fg]=[f for f in (feat(load(ps[i])) for i in pidx) if f]
gen,imp=[],[]
for pc in ids:
    for pf in PR[pc]:
        for cc in ids:
            best=max((score(pf,t) for t in TM[cc]), default=0.0)
            (gen if cc==pc else imp).append(best)
gen,imp=np.array(gen),np.array(imp)
allv=np.unique(np.concatenate([gen,imp])); far=np.array([(imp>=v).mean() for v in allv]); frr=np.array([(gen<v).mean() for v in allv])
ei=np.nanargmin(np.abs(far-frr))
print(f"\ngenuine {len(gen)}对, impostor {len(imp)}对 (可分辨最小FAR≈{100/max(1,len(imp)):.3f}%)")
print(f"EER = {(far[ei]+frr[ei])/2*100:.2f}%")
for tgt in [1.0,0.5,0.1,0.05]:
    ok=np.where(far<=tgt/100)[0]
    print(f"FAR<={tgt}% -> FRR = {(gen<allv[ok[0]]).mean()*100:.2f}%" if len(ok) else f"FAR<={tgt}% -> 样本不足(手指/帧太少)")
print(f"FAR=0   -> FRR = {(gen<imp.max()+1e-6).mean()*100:.2f}%  (impostor最高分={imp.max():.1f})")
