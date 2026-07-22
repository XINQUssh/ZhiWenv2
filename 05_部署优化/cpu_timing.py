# -*- coding: utf-8 -*-
"""CPU上测优化后管线的 注册耗时 + 5指解锁耗时(全局预筛top15+几何)。int8+stride8。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import sys, glob, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(os.cpu_count())
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
dev=torch.device('cpu')   # 强制CPU
net=Desc().to(dev); net.load_state_dict(torch.load('f:/1111/指纹/deliver/best.pth',map_location=dev)); net.eval()
HALF=16; STRIDE=8; MDS=4; HO=20000.0; TOPK=15; bf=cv2.BFMatcher(cv2.NORM_L2)
@torch.no_grad()
def feat(img):
    norm,m=prep(img); H,W=norm.shape; pts=[]; pat=[]
    for y in range(HALF,H-HALF,STRIDE):
        for x in range(HALF,W-HALF,STRIDE):
            if m[y,x]>0: pts.append((x,y)); pat.append(norm[y-HALF:y+HALF,x-HALF:x+HALF])
    if len(pat)<4: return None
    X=torch.tensor(np.stack(pat)[:,None],dtype=torch.float32).to(dev); d=[]
    for i in range(0,len(X),2048): d.append(net(X[i:i+2048]).cpu().numpy())
    desc=np.concatenate(d).astype(np.float32); gvec=desc.mean(0); gvec/=np.linalg.norm(gvec)+1e-8
    di8=np.clip(np.round(desc*127),-127,127).astype(np.int8)
    return (np.round(np.array(pts)).astype(np.int16), di8, gvec.astype(np.float16),
            cv2.resize(m,(W//MDS,H//MDS),interpolation=cv2.INTER_NEAREST))
def dequant(di8):
    d=di8.astype(np.float32)/127.0; return d/(np.linalg.norm(d,axis=1,keepdims=True)+1e-8)
def _score(f1,f2,ratio=0.92,reproj=6):
    p1,di1,_,m1=f1; p2,di2,_,m2=f2; d1=dequant(di1); d2=dequant(di2)
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0.0
    src=p1[[x.queryIdx for x in g]].astype(np.float32).reshape(-1,1,2); dst=p2[[x.trainIdx for x in g]].astype(np.float32).reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=2000,confidence=0.99)
    if M is None or mask is None: return 0.0
    return float(mask.sum())
RES=open('f:/1111/指纹/_cpu_timing.txt','w',encoding='utf-8')
def out(*a):
    s=' '.join(str(x) for x in a); print(s); RES.write(s+'\n'); RES.flush()
SEL='f:/1111/指纹/select_data/select'
fingers=sorted([d for d in os.listdir(SEL) if os.path.isdir(os.path.join(SEL,d))])[:5]
out(f"CPU={os.cpu_count()}线程. 5指={fingers}")
# warmup
_=feat(load(sorted(glob.glob(os.path.join(SEL,fingers[0],'*.bmp')))[0]))
# 注册: 每指20模板, 计时每次feat
gal=[]; et=[]
for fg in fingers:
    ps=sorted(glob.glob(os.path.join(SEL,fg,'*.bmp')))[:20]
    for p in ps:
        img=load(p); t1=time.perf_counter(); f=feat(img); et.append((time.perf_counter()-t1)*1000)
        if f: gal.append((fg,f))
et=np.array(et)
out(f"\n[注册-CPU] 单模板 平均={et.mean():.0f}ms 中位={np.median(et):.0f}ms 最慢={et.max():.0f}ms (预算<200ms)  共{len(gal)}模板")
# 解锁: 探针=各指第30帧附近; 全局预筛top15 + 几何
gmat=np.stack([f[2].astype(np.float32) for _,f in gal])  # (100,128)
ut=[]; probes=[]
for fg in fingers:
    ps=sorted(glob.glob(os.path.join(SEL,fg,'*.bmp')))[25:33]
    probes+= [(fg,p) for p in ps]
for fg,p in probes:
    img=load(p); t1=time.perf_counter()
    pf=feat(img)
    if pf is None: continue
    gq=pf[2].astype(np.float32); sims=gmat@gq; cand=np.argsort(sims)[::-1][:TOPK]
    best={}
    for ci in cand:
        gg,tf=gal[ci]; s=_score(pf,tf)
        if s>best.get(gg,0): best[gg]=s
    ut.append((time.perf_counter()-t1)*1000)
ut=np.array(ut)
out(f"[解锁-CPU] 5指(预筛top{TOPK}+几何) 平均={ut.mean():.0f}ms 中位={np.median(ut):.0f}ms 最快={ut.min():.0f}ms 最慢={ut.max():.0f}ms (预算 最快<50/平均<100)")
# 拆解: 单纯提特征 vs 匹配
t1=time.perf_counter(); _=feat(load(probes[0][1])); feat_ms=(time.perf_counter()-t1)*1000
out(f"[拆解] 提探针特征≈{feat_ms:.0f}ms(固定大头) + 预筛~1ms + 匹配{TOPK}个≈{TOPK*1.5:.0f}ms")
out("Done.")
