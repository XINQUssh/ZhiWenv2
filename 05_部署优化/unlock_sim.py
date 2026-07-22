# -*- coding: utf-8 -*-
"""100模板 早退解锁 最快/平均 匹配耗时(CPU, 仅匹配不含feat)。
模型: 全局预筛排序100模板 -> 按序几何匹配 -> 命中(内点>=THR)即停 -> 记比了几个。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8)
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
net=Desc(); net.load_state_dict(torch.load('f:/1111/指纹/deliver/best.pth',map_location='cpu')); net.eval()
HALF=16; STRIDE=8; bf=cv2.BFMatcher(cv2.NORM_L2)
@torch.no_grad()
def feat(img):
    norm,m=prep(img); H,W=norm.shape; pts=[]; pat=[]
    for y in range(HALF,H-HALF,STRIDE):
        for x in range(HALF,W-HALF,STRIDE):
            if m[y,x]>0: pts.append((x,y)); pat.append(norm[y-HALF:y+HALF,x-HALF:x+HALF])
    if len(pat)<4: return None
    X=torch.tensor(np.stack(pat)[:,None],dtype=torch.float32); d=[]
    for i in range(0,len(X),2048): d.append(net(X[i:i+2048]).numpy())
    desc=np.concatenate(d).astype(np.float32); gv=desc.mean(0); gv/=np.linalg.norm(gv)+1e-8
    di8=np.clip(np.round(desc*127),-127,127).astype(np.int8)  # 正确int8量化(×127)
    return (np.round(np.array(pts)).astype(np.int16), di8, gv, m)
def match_inl(f1,f2):
    p1,di1,_,m1=f1; p2,di2,_,m2=f2; d1=di1.astype(np.float32); d2=di2.astype(np.float32)
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<0.92*b.distance]
    if len(g)<4: return 0
    src=p1[[x.queryIdx for x in g]].astype(np.float32).reshape(-1,1,2); dst=p2[[x.trainIdx for x in g]].astype(np.float32).reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=6,maxIters=2000,confidence=0.99)
    return int(mask.sum()) if mask is not None else 0
TPF=int(os.environ.get('TPF','10'))   # 每指模板数
SEL='f:/1111/指纹/select_data/select'
allf=sorted([d for d in os.listdir(SEL) if os.path.isdir(os.path.join(SEL,d))])
# 预提所有需要的特征(不计时)
FT={}
for fg in allf:
    ps=sorted(glob.glob(os.path.join(SEL,fg,'*.bmp')))
    if len(ps)<30: continue
    idx=np.linspace(0,int(len(ps)*0.7)-1,min(TPF,int(len(ps)*0.7))).astype(int)
    FT[fg]={'tmpl':[feat(load(ps[i])) for i in idx],
            'probe':[feat(load(p)) for p in ps[int(len(ps)*0.7):int(len(ps)*0.7)+5]]}
    FT[fg]['tmpl']=[f for f in FT[fg]['tmpl'] if f]; FT[fg]['probe']=[f for f in FT[fg]['probe'] if f]
ids=[k for k in FT if len(FT[k]['tmpl'])>=TPF and len(FT[k]['probe'])>0]
# 每次匹配CPU耗时(实测)
import itertools
g0=FT[ids[0]]['tmpl']; pr0=FT[ids[0]]['probe'][0]
_=match_inl(pr0,g0[0])
ts=[]
for _ in range(50): t1=time.perf_counter(); match_inl(pr0,g0[np.random.randint(len(g0))]); ts.append((time.perf_counter()-t1)*1000)
per=np.median(ts); print(f"CPU单次几何匹配 中位={per:.2f}ms")
THR=15  # 命中阈(内点>=THR即解锁)
rng=np.random.RandomState(0)
sets=[list(rng.choice(len(ids),5,replace=False)) for _ in range(8)]
counts=[]; unlocked=0; total=0
for sset in sets:
    enr=[ids[i] for i in sset]
    gal=[(fg,f) for fg in enr for f in FT[fg]['tmpl']][:5*TPF]
    gmat=np.stack([f[2] for _,f in gal])
    for pc in enr:
        for pf in FT[pc]['probe']:
            gv=pf[2]; order=np.argsort(gmat@gv)[::-1]  # 预筛排序
            cnt=0; hit=False
            for oi in order:
                cnt+=1
                if match_inl(pf,gal[oi][1])>=THR: hit=True; break
            total+=1
            if hit: unlocked+=1; counts.append(cnt)
            else: counts.append(len(order))  # 没命中=扫完全部
counts=np.array(counts); NT=5*TPF
print(f"\n{NT}模板(5指x{TPF}) 早退解锁(THR={THR}内点, 全局预筛排序, {total}次genuine尝试, 解锁率{unlocked/total*100:.0f}%):")
print(f"  比对模板数: 最少={counts.min()} 平均={counts.mean():.1f} 中位={np.median(counts):.0f} 最多={counts.max()}")
print(f"  => 匹配耗时(CPU, 不含feat): 最快={counts.min()*per:.1f}ms  平均={counts.mean()*per:.1f}ms  中位={np.median(counts)*per:.1f}ms")
succ=counts[counts<NT]
print(f"  (仅统计成功解锁的: 平均比对={succ.mean() if len(succ) else 0:.1f}个 -> {succ.mean()*per if len(succ) else 0:.1f}ms)")
print(f"参考: 全量扫{NT} = {NT*per:.0f}ms; 最快(第1个即命中)={per:.1f}ms")
print("Done.")
