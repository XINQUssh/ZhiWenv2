# -*- coding: utf-8 -*-
"""
回答客户: 全局+局部(我们V18b/V21融合)能不能达 FAR<0.002%?
计算 V21+V18b 全局余弦 + V21 局部dense 的融合分(tnorm), 出分布图+FAR/FRR曲线,
直接看它在严格工作点的真实FRR(对比内点法的离散可分分布)。
"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob, random, sys, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

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
    return norm
def _stem(s):
    b=resnet18(weights=None)
    s.conv1=nn.Conv2d(1,64,7,2,3,bias=False); s.bn1,s.relu,s.maxpool=b.bn1,b.relu,b.maxpool
    s.layer1,s.layer2,s.layer3,s.layer4=b.layer1,b.layer2,b.layer3,b.layer4; s.avgpool=nn.AdaptiveAvgPool2d((1,1))
class DenseEnc(nn.Module):  # V18b (global only用)
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e)); s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        ld=F.normalize(s.local_head(x),dim=1); return s.global_projector(s.avgpool(s.layer4(x)).flatten(1)),ld
class HybridEnc(nn.Module):  # V21 (global+local)
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.attention=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(),nn.Conv2d(128,1,1),nn.Softplus())
        s.attn_projector=nn.Sequential(nn.Linear(256,e),nn.BatchNorm1d(e)); s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        ld=F.normalize(s.local_head(x),dim=1); return s.global_projector(s.avgpool(s.layer4(x)).flatten(1)),ld
dev=torch.device('cuda')
def lf(cls,d,pre):
    out=[]
    for sd in [42,123,777]:
        m=cls().to(dev); m.load_state_dict(torch.load(f'f:/1111/指纹/{d}/{pre}_seed{sd}.pth',map_location=dev,weights_only=True),strict=False); m.eval(); out.append(m)
    return out
V21=lf(HybridEnc,'models_v21','v21'); V18=lf(DenseEnc,'models_v18b','v18b')
print("loaded V21+V18b"); sys.stdout.flush()
@torch.no_grad()
def feat(img):
    norm=prep(img); gs=[]; locs=[]
    for m in V21+V18:
        ge=[]; le=[]
        for pol in (1,-1):
            x=torch.tensor(pol*norm,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
            g,ld=m(x); ge.append(F.normalize(g,dim=1))
            if m in V21: le.append(ld[0].view(64,-1).t())
        gs.append(F.normalize(torch.mean(torch.stack(ge),dim=0),dim=1).cpu().numpy().flatten())
        if m in V21: locs.append(F.normalize(torch.mean(torch.stack(le),dim=0),dim=1).cpu().numpy())
    gfused=np.mean(gs,axis=0); gfused/=np.linalg.norm(gfused)+1e-8
    return gfused, locs  # 全局融合嵌入(512,), V21三seed局部(各182,64)
b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'; b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
random.seed(42); cp={}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if os.path.exists(rp):
            ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
            if len(ps)>=50: cp[f'{pre}_{fg}']=ps
ids=list(cp.keys()); t0=time.time()
print("extract..."); sys.stdout.flush()
tmplG={}; tmplL={}; testF={}
for k in ids:
    idx=random.sample(range(70),20)
    fs=[feat(load_img(cp[k][i])) for i in idx]
    tmplG[k]=[f[0] for f in fs]
    tmplL[k]=[np.concatenate([f[1][s] for f in fs],axis=0) for s in range(3)]  # 每seed拼接所有模板局部
    testF[k]=[feat(load_img(p)) for p in cp[k][70:]]
print(f"  done t={time.time()-t0:.0f}s"); sys.stdout.flush()
def localscore(qlocs, k):  # V21三seed dense_all均值
    vals=[]
    for s in range(3):
        Q=qlocs[s]; T=tmplL[k][s]; S=Q@T.T; mx=S.max(axis=1); kk=max(1,int(0.7*len(mx))); vals.append(np.sort(mx)[-kk:].mean())
    return float(np.mean(vals))
# 每probe对每class: global=max余弦, local=max dense, fused=0.5l+0.5g, 再tnorm
gen,imp=[],[]
for pc in ids:
    for ge,locs in testF[pc]:
        raw={}
        for cc in ids:
            g=max(float(ge@t) for t in tmplG[cc]); l=localscore(locs,cc); raw[cc]=0.5*l+0.5*g
        for cc in ids:
            others=[raw[o] for o in ids if o!=cc]; v=(raw[cc]-np.mean(others))/(np.std(others)+1e-8)
            (gen if cc==pc else imp).append(v)
gen,imp=np.array(gen),np.array(imp)
def eerfar(gen,imp):
    lo,hi=min(gen.min(),imp.min()),max(gen.max(),imp.max()); ths=np.linspace(lo,hi,2000)
    far=np.array([(imp>=t).mean() for t in ths]); frr=np.array([(gen<t).mean() for t in ths])
    i=np.nanargmin(np.abs(far-frr)); return ths,far,frr,(far[i]+frr[i])/2
ths,far,frr,eer=eerfar(gen,imp)
print(f"\n[全局+局部融合 tnorm] EER={eer*100:.3f}%")
for tgt in [0.002,0.01,0.1,1.0]:
    idx=np.argmin(np.abs(far-tgt/100)); print(f"  FAR={tgt}% -> FRR={frr[idx]*100:.2f}%")
idx=np.where(far==0)[0]
if len(idx): print(f"  FAR=0 -> FRR={frr[idx[0]]*100:.2f}%")
# 出图
fig=plt.figure(figsize=(8.5,7.5))
ax1=fig.add_axes([0.10,0.58,0.84,0.36])
ax1.plot(ths,far,label='FAR',lw=2); ax1.plot(ths,frr,label='FRR',lw=2,color='orange'); ax1.legend(); ax1.set_xlabel('score threshold (fused tnorm)')
ax1.set_title('GLOBAL+LOCAL fusion (V21+V18b, our CNN method)  —  continuous score',fontsize=10.5)
ax2=fig.add_axes([0.10,0.10,0.84,0.36])
lo,hi=min(gen.min(),imp.min()),max(gen.max(),imp.max()); bins=np.linspace(lo,hi,80)
ax2.hist(imp,bins=bins,density=True,alpha=0.7,label='impostor'); ax2.hist(gen,bins=bins,density=True,alpha=0.7,label='genuine',color='orange')
ax2.legend(); ax2.set_xlabel('fused score');
fig.text(0.5,0.005,f'EER={eer*100:.2f}%  BUT  FAR=0.002% -> FRR={frr[np.argmin(np.abs(far-0.00002))]*100:.1f}%  (genuine/impostor tails OVERLAP -> cannot reach strict FAR)',
         ha='center',fontsize=10,weight='bold',color='#b00')
plt.savefig('f:/1111/指纹/全局加局部_严格工作点.png',dpi=115,bbox_inches='tight')
print("saved 全局加局部_严格工作点.png"); print(f"t={time.time()-t0:.0f}s")
