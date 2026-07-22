# -*- coding: utf-8 -*-
"""
迭代4(最后零训练): 模板数量 + 分数级vs嵌入级融合 + 跨场景多采样稳定读数。
模型: V18b,V21(单) / V25a,V26b,V21(跨)。无TTA。
"""
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
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        return s.global_projector(s.avgpool(s.layer4(x)).flatten(1)),F.normalize(s.local_head(x),dim=1)
class HybridEnc(nn.Module):
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.attention=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(),nn.Conv2d(128,1,1),nn.Softplus())
        s.attn_projector=nn.Sequential(nn.Linear(256,e),nn.BatchNorm1d(e))
        s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        return s.global_projector(s.avgpool(s.layer4(x)).flatten(1)),F.normalize(s.local_head(x),dim=1)
dev=torch.device('cuda')
def lf(cls,d,pre):
    out=[]
    for sd in [42,123,777]:
        m=cls().to(dev); m.load_state_dict(torch.load(f'f:/1111/指纹/{d}/{pre}_seed{sd}.pth',map_location=dev,weights_only=True)); m.eval(); out.append(m)
    return out
FAM={'V18b':lf(DenseEnc,'models_v18b','v18b'),'V21':lf(HybridEnc,'models_v21','v21'),
     'V25a':lf(HybridEnc,'models_v25a','v25a'),'V26b':lf(HybridEnc,'models_v26b','v26b')}
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
    return {fam:(gs,lds) for fam in fams}  # placeholder; rebuilt below
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
FAMS=['V18b','V21','V25a','V26b']
# 预提取所有老train帧(前70)与老test帧, 新cross帧
print("extract pool..."); sys.stdout.flush()
trainpool={}; test={}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if not os.path.exists(rp): continue
        ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
        if len(ps)<50: continue
        k=f'{pre}_{fg}'
        trainpool[k]=[feats(FAMS,prep(load_img(p))) for p in ps[:70]]
        test[k]=[feats(FAMS,prep(load_img(p))) for p in ps[70:]]
ids=list(trainpool.keys()); print(f"  {len(ids)} classes"); sys.stdout.flush()
MERGE={'dy_R0':'wtp_dy_R0','dy_R1':'wtp_dy_R1','dy_R2':'wtp_dy_R2','lwh_R0':'wtp_lwh_R0','lwh_R1':'wtp_lwh_R1',
       'lwh_R2':'wtp_lwh_R2','zyh_R0':'btp_zyh_R0','zyh_R1':'btp_zyh_R1','zyh_R2':'btp_zyh_R2'}
nb='f:/1111/指纹/ysjz_raw/ysjz'; cross={}
for folder,gt in MERGE.items():
    cross.setdefault(gt,[])
    for p in sorted(glob.glob(os.path.join(nb,folder,'*.bmp')))[70:]:
        cross[gt].append(feats(FAMS,prep(load_img(p))))
print("  cross done"); sys.stdout.flush()

def pick_tmpl(n,seed):
    random.seed(seed); T={}
    for k in ids:
        idx=random.sample(range(len(trainpool[k])),min(n,len(trainpool[k])))
        T[k]=[trainpool[k][j] for j in idx]
    return T

def run(probes,T,w,lfams,alpha,tnorm,score_level=False):
    GTg={cid:[fuse_g(e,w) for e in T[cid]] for cid in ids}
    # 分数级: 每模型单独max-cosine再平均
    g,i=[],[]
    for gt,plist in probes.items():
        for e in plist:
            sc={}
            for cid in ids:
                if score_level:
                    gss=[]
                    for fam,wt in w.items():
                        for mi in range(3):
                            pe=e[fam][0][mi]; gss+= [wt*max(float(pe@t[fam][0][mi]) for t in T[cid])]
                    gs=float(np.sum(gss)/sum(w.values())/3)
                else:
                    pe=fuse_g(e,w); gs=float(max(pe@t for t in GTg[cid]))
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

print(f"\n{'='*64}\nSINGLE V21+V18b (best=1.093%): 模板数 / 分数级 / 局部\n{'='*64}")
w={'V21':1,'V18b':1}
for n in [10,20,40]:
    T=pick_tmpl(n,42)
    e0=run(test,T,w,['V21','V18b'],0.3,True)
    es=run(test,T,w,['V21','V18b'],0.0,True,score_level=True)
    print(f"  n_tmpl={n}: embed+local0.3+tnorm={e0*100:.4f}%   score-level+tnorm={es*100:.4f}%"); sys.stdout.flush()

print(f"\n{'='*64}\nCROSS V25a+V26b+V21 2:1:1 +tnorm: 多采样稳定读数\n{'='*64}")
wc={'V25a':2,'V26b':1,'V21':1}
for n in [10,20]:
    vals=[run(cross,pick_tmpl(n,sd),wc,[],0,True) for sd in [42,7,123]]
    print(f"  n_tmpl={n}: tnorm EER = {np.mean(vals)*100:.3f}% ± {np.std(vals)*100:.3f}%  (samples {[f'{v*100:.2f}' for v in vals]})"); sys.stdout.flush()
print("\nDone.")
