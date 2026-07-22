# -*- coding: utf-8 -*-
"""最终确认: 单场景 V21单独 vs V21+V18b融合, 各3次模板采样, 坐实是否真改善。"""
import os, glob, random, sys
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_curve
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
    enh=clahe(img); u=up2(enh); m=gmask(u); mk=u.copy(); mk[m==0]=0
    v=mk[m>0].astype(np.float32); mu,sd=(v.mean(),v.std()+1e-6) if len(v) else (0,1)
    norm=(mk.astype(np.float32)-mu)/sd; norm[m==0]=0
    return norm
def _stem(s):
    b=resnet18(weights=None)
    s.conv1=nn.Conv2d(1,64,7,2,3,bias=False); s.bn1,s.relu,s.maxpool=b.bn1,b.relu,b.maxpool
    s.layer1,s.layer2,s.layer3,s.layer4=b.layer1,b.layer2,b.layer3,b.layer4
    s.avgpool=nn.AdaptiveAvgPool2d((1,1))
class DenseEnc(nn.Module):
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e)); s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        return s.global_projector(s.avgpool(s.layer4(x)).flatten(1)),F.normalize(s.local_head(x),dim=1)
class HybridEnc(nn.Module):
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.attention=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(),nn.Conv2d(128,1,1),nn.Softplus())
        s.attn_projector=nn.Sequential(nn.Linear(256,e),nn.BatchNorm1d(e)); s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        return s.global_projector(s.avgpool(s.layer4(x)).flatten(1)),F.normalize(s.local_head(x),dim=1)
dev=torch.device('cuda')
def lf(cls,d,pre):
    return [ (lambda m: (m.load_state_dict(torch.load(f'f:/1111/指纹/{d}/{pre}_seed{sd}.pth',map_location=dev,weights_only=True)),m.eval(),m)[-1])(cls().to(dev)) for sd in [42,123,777]]
FAM={'V18b':lf(DenseEnc,'models_v18b','v18b'),'V21':lf(HybridEnc,'models_v21','v21')}
print("loaded"); sys.stdout.flush()
@torch.no_grad()
def feats(fams,frame):
    out={}
    for fam in fams:
        gs,lds=[],[]
        for m in FAM[fam]:
            ge,le=[],[]
            for pol in (1,-1):
                x=torch.tensor(pol*frame,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
                g,ld=m(x); ge.append(F.normalize(g,dim=1)); le.append(ld[0].view(64,-1).t())
            gs.append(F.normalize(torch.mean(torch.stack(ge),dim=0),dim=1).cpu().numpy().flatten())
            lds.append(F.normalize(torch.mean(torch.stack(le),dim=0),dim=1).cpu().numpy())
        out[fam]=(gs,lds)
    return out
def fuse_g(per,w):
    acc=None
    for fam,wt in w.items():
        for e in per[fam][0]: acc=e*wt if acc is None else acc+e*wt
    return acc/(np.linalg.norm(acc)+1e-8)
def eer(g,i):
    g,i=np.array(g),np.array(i); y=np.concatenate([np.ones(len(g)),np.zeros(len(i))]); s=np.concatenate([g,i])
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; k=np.nanargmin(np.abs(fnr-fpr)); return (fpr[k]+fnr[k])/2
b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
FAMS=['V18b','V21']; print("extract..."); sys.stdout.flush()
pool={}; test={}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if not os.path.exists(rp): continue
        ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
        if len(ps)<50: continue
        k=f'{pre}_{fg}'
        pool[k]=[feats(FAMS,prep(load_img(p))) for p in ps[:70]]; test[k]=[feats(FAMS,prep(load_img(p))) for p in ps[70:]]
ids=list(pool.keys()); print(f"  {len(ids)} classes"); sys.stdout.flush()
def pick(n,seed):
    random.seed(seed); return {k:[pool[k][j] for j in random.sample(range(len(pool[k])),min(n,len(pool[k])))] for k in ids}
def run(T,w,lfams,alpha,tnorm,raw_only=False):
    GT={cid:[fuse_g(e,w) for e in T[cid]] for cid in ids}; g,i=[],[]
    for gt,plist in test.items():
        for e in plist:
            pe=fuse_g(e,w); sc={}
            for cid in ids:
                gs=float(max(pe@t for t in GT[cid]))
                if alpha>0:
                    ls=[]
                    for fam in lfams:
                        for mi in range(3):
                            Q=e[fam][1][mi]; Tl=np.concatenate([t[fam][1][mi] for t in T[cid]],axis=0)
                            mx=(Q@Tl.T).max(axis=1); k=max(1,int(0.7*len(mx))); ls.append(np.sort(mx)[-k:].mean())
                    sc[cid]=alpha*float(np.mean(ls))+(1-alpha)*gs
                else: sc[cid]=gs
            for cid in ids:
                v=sc[cid]
                if tnorm:
                    imp=[sc[o] for o in ids if o!=cid]; v=(v-np.mean(imp))/(np.std(imp)+1e-8)
                (g if cid==gt else i).append(v)
    return eer(g,i)
print(f"\n{'='*60}\n单场景 3次模板采样 (seed 42/7/123)\n{'='*60}")
configs=[('V21 only raw',{'V21':1},[],0.0,False),
         ('V21 only tnorm',{'V21':1},[],0.0,True),
         ('V21+V18b raw',{'V21':1,'V18b':1},[],0.0,False),
         ('V21+V18b tnorm',{'V21':1,'V18b':1},[],0.0,True),
         ('V21+V18b local0.3 tnorm',{'V21':1,'V18b':1},['V21','V18b'],0.3,True)]
for name,w,lf_,a,tn in configs:
    vals=[run(pick(10,sd),w,lf_,a,tn) for sd in [42,7,123]]
    print(f"  {name:28s}: {np.mean(vals)*100:.3f}% ± {np.std(vals)*100:.3f}%  {[f'{v*100:.2f}' for v in vals]}"); sys.stdout.flush()
print("\nDone.")
