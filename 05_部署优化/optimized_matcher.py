# -*- coding: utf-8 -*-
"""
Task2: 部署优化匹配器 = int8描述子量化 + stride调整 + 全局嵌入预筛。
同时测 部署指标(存储/注册/解锁) 与 精度(FAR/FRR) on 贴屏select数据。
用法: python optimized_matcher.py <stride> <topk>   默认 8 15
"""
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
net=Desc().to(dev); net.load_state_dict(torch.load('f:/1111/指纹/deliver/best.pth',map_location=dev)); net.eval()
PS=32; HALF=16
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 8
TOPK=int(sys.argv[2]) if len(sys.argv)>2 else 15
MDS=4  # mask下采样因子(紧凑存储)
HO=20000.0
bf=cv2.BFMatcher(cv2.NORM_L2)

@torch.no_grad()
def feat(img):
    """返回压缩模板: (pts_int16, desc_int8, scale, global_f16, mask_small)"""
    norm,m=prep(img); H,W=norm.shape
    pts=[]; pat=[]
    for y in range(HALF,H-HALF,STRIDE):
        for x in range(HALF,W-HALF,STRIDE):
            if m[y,x]>0: pts.append((x,y)); pat.append(norm[y-HALF:y+HALF,x-HALF:x+HALF])
    if len(pat)<4: return None
    X=torch.tensor(np.stack(pat)[:,None],dtype=torch.float32).to(dev); d=[]
    for i in range(0,len(X),2048): d.append(net(X[i:i+2048]).cpu().numpy())
    desc=np.concatenate(d).astype(np.float32)              # (N,128) L2归一化
    gvec=desc.mean(0); gvec/=np.linalg.norm(gvec)+1e-8     # 全局签名(预筛用)
    desc_i8=np.clip(np.round(desc*127),-127,127).astype(np.int8)   # int8量化
    pts_i16=np.round(np.array(pts)).astype(np.int16)
    msmall=cv2.resize(m,(W//MDS,H//MDS),interpolation=cv2.INTER_NEAREST)
    return (pts_i16, desc_i8, gvec.astype(np.float16), msmall)

def dequant(desc_i8):
    d=desc_i8.astype(np.float32)/127.0
    return d/(np.linalg.norm(d,axis=1,keepdims=True)+1e-8)
def _score(f1,f2,ratio=0.92,reproj=6,slo=0.85,shi=1.18,rotmax=30):
    p1,di1,g1,m1=f1; p2,di2,g2,m2=f2
    d1=dequant(di1); d2=dequant(di2)
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0.0
    src=p1[[x.queryIdx for x in g]].astype(np.float32).reshape(-1,1,2); dst=p2[[x.trainIdx for x in g]].astype(np.float32).reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=2000,confidence=0.99)
    if M is None or mask is None: return 0.0
    a,b=M[0,0],M[1,0]; sc=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a))); rot=min(rot,360-rot)
    if sc<slo or sc>shi or rot>rotmax: return 0.0
    inl=int(mask.sum())
    if inl==0: return 0.0
    Ms=M.copy(); Ms[:,2]/=MDS  # 变换缩放到小mask
    h,w=m2.shape; w1=cv2.warpAffine(m1,Ms,(w,h),flags=cv2.INTER_NEAREST); ov=int(np.count_nonzero((w1>0)&(m2>0)))*MDS*MDS
    return inl*float(np.sqrt(HO/ov)) if ov>=500 else float(inl)

def prefilter(pf, cand_feats, topk):
    """全局签名余弦预筛 -> 返回topk候选索引"""
    gq=pf[2].astype(np.float32)
    sims=[float(gq@cf[2].astype(np.float32)) for cf in cand_feats]
    return np.argsort(sims)[::-1][:topk]

# ---- 数据 ----
SEL='f:/1111/指纹/select_data/select'
fingers=[]
for fg in sorted(os.listdir(SEL)):
    d=os.path.join(SEL,fg)
    if os.path.isdir(d):
        ps=sorted(glob.glob(os.path.join(d,'*.bmp')))
        if len(ps)>=15: fingers.append((fg,ps))
ids=[f[0] for f in fingers]
print(f"优化匹配器: stride={STRIDE}, topk={TOPK}, int8量化, 全局预筛. {len(ids)}指"); sys.stdout.flush()
t0=time.time()
tmpl={}; probes={}; enroll_t=[]; store_bytes=0; npts=[]
for fg,ps in fingers:
    ne=int(len(ps)*0.7); pool=ps[:ne]; pr=ps[ne:]
    idx=np.linspace(0,len(pool)-1,min(20,len(pool))).astype(int)
    fs=[]
    for i in idx:
        t1=time.perf_counter(); f=feat(load(pool[i]));
        if torch.cuda.is_available(): torch.cuda.synchronize()
        enroll_t.append((time.perf_counter()-t1)*1000)
        if f is not None:
            fs.append(f); npts.append(len(f[0]))
            store_bytes += f[0].nbytes+f[1].nbytes+f[2].nbytes+f[3].nbytes
    tmpl[fg]=fs; probes[fg]=[feat(load(p)) for p in pr]
et=np.array(enroll_t)
n_tmpl=sum(len(tmpl[f]) for f in ids)
print(f"[空间] 平均kp/模板={np.mean(npts):.0f}; {n_tmpl}模板总存储(int8+紧凑mask+全局vec)={store_bytes/1e6:.2f}MB"
      f"  -> 折算100模板={store_bytes/n_tmpl*100/1e6:.2f}MB (预算<10MB)")
print(f"[注册] 单模板: 平均={et.mean():.1f}ms 最慢={et.max():.1f}ms (预算<200ms)"); sys.stdout.flush()

# ---- 精度 + 解锁耗时(带预筛) ----
all_tmpl=[(fg,f) for fg in ids for f in tmpl[fg]]   # (finger, feat)
gen,imp=[],[]; gen_by={f:[] for f in ids}; unlock_t=[]
for pc in ids:
    for pf in probes[pc]:
        if pf is None: continue
        t1=time.perf_counter()
        cand=prefilter(pf,[t[1] for t in all_tmpl],TOPK)   # 预筛topk模板
        cls_best={}
        for ci in cand:
            fg,tf=all_tmpl[ci]; s=_score(pf,tf)
            if s>cls_best.get(fg,0): cls_best[fg]=s
        if torch.cuda.is_available(): torch.cuda.synchronize()
        unlock_t.append((time.perf_counter()-t1)*1000)
        for cc in ids:
            sc=cls_best.get(cc,0.0)
            if cc==pc: gen.append(sc); gen_by[pc].append(sc)
            else: imp.append(sc)
gen,imp=np.array(gen),np.array(imp); ut=np.array(unlock_t)
print(f"[耗时] 5指解锁(预筛top{TOPK}+几何): 平均={ut.mean():.0f}ms 最快={ut.min():.0f}ms (预算 最快<50/平均<100ms)"); sys.stdout.flush()
allv=np.unique(np.concatenate([gen,imp])); far=np.array([(imp>=v).mean() for v in allv]); frr=np.array([(gen<v).mean() for v in allv])
ei=np.nanargmin(np.abs(far-frr)); thr0=imp.max()+1e-6
print(f"[精度 贴屏] EER={(far[ei]+frr[ei])/2*100:.3f}%  FAR=0 FRR={(gen<thr0).mean()*100:.3f}%  (gen mean={gen.mean():.0f}, imp max={imp.max():.1f})")
for tgt in [0.002,0.01]:
    ok=np.where(far<=tgt/100)[0]
    if len(ok): v=allv[ok[0]]; print(f"  FAR<={tgt}%: FRR={(gen<v).mean()*100:.3f}%")
print(f"t={time.time()-t0:.0f}s\nDone.")
