# -*- coding: utf-8 -*-
"""
迭代1: 跨模型融合(ensemble) — 零训练, 看能否压低单场景/跨场景EER。
已训模型: V17(3) V18b(3) V21(3) V25a(3) V26b(3), 均输出512维全局嵌入。
嵌入级融合 = 选定模型集的归一化嵌入平均后重归一化, 再余弦匹配。
Eval A: 老27类单场景(模板20/类seed42, 老测试帧) → 对比 V21 的 1.19%
Eval B: 跨场景(老模板 x 新9指探针)         → 对比 V25a 的 5.71%
"""
import os, glob, random, sys, time
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
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

def _resnet_stem(s):
    b=resnet18(weights=None)
    s.conv1=nn.Conv2d(1,64,7,2,3,bias=False); s.bn1,s.relu,s.maxpool=b.bn1,b.relu,b.maxpool
    s.layer1,s.layer2,s.layer3,s.layer4=b.layer1,b.layer2,b.layer3,b.layer4
    s.avgpool=nn.AdaptiveAvgPool2d((1,1))

class FingerprintEncoder(nn.Module):       # V17
    def __init__(s,e=512):
        super().__init__(); _resnet_stem(s)
        s.projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer4(s.layer3(s.layer2(s.layer1(x))))
        return s.projector(s.avgpool(x).flatten(1))

class DenseDescriptorEncoder(nn.Module):   # V18b
    def __init__(s,e=512,l=64):
        super().__init__(); _resnet_stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        return s.global_projector(s.avgpool(s.layer4(x)).flatten(1))

class HybridDelfEncoder(nn.Module):        # V21/V25a/V26b
    def __init__(s,e=512,l=64):
        super().__init__(); _resnet_stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.attention=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(),nn.Conv2d(128,1,1),nn.Softplus())
        s.attn_projector=nn.Sequential(nn.Linear(256,e),nn.BatchNorm1d(e))
        s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        return s.global_projector(s.avgpool(s.layer4(x)).flatten(1))

dev=torch.device('cuda')
def load_family(cls, d, prefix):
    ms=[]
    for sd in [42,123,777]:
        p=f'f:/1111/指纹/{d}/{prefix}_seed{sd}.pth'
        m=cls().to(dev); m.load_state_dict(torch.load(p,map_location=dev,weights_only=True)); m.eval(); ms.append((f'{prefix}_{sd}',m))
    return ms

MODELS=[]
MODELS+= [('V17',n,m) for n,m in load_family(FingerprintEncoder,'models_v17','v17')]
MODELS+= [('V18b',n,m) for n,m in load_family(DenseDescriptorEncoder,'models_v18b','v18b')]
MODELS+= [('V21',n,m) for n,m in load_family(HybridDelfEncoder,'models_v21','v21')]
MODELS+= [('V25a',n,m) for n,m in load_family(HybridDelfEncoder,'models_v25a','v25a')]
MODELS+= [('V26b',n,m) for n,m in load_family(HybridDelfEncoder,'models_v26b','v26b')]
NAMES=[f"{fam}:{n}" for fam,n,_ in MODELS]
print(f"Loaded {len(MODELS)} models."); sys.stdout.flush()

@torch.no_grad()
def emb_all(frame):
    """返回 (n_models,512) 每模型双极性平均后归一化的嵌入。"""
    out=[]
    for _,_,m in MODELS:
        es=[]
        for pol in (1,-1):
            x=torch.tensor(pol*frame,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
            es.append(F.normalize(m(x),dim=1))
        out.append(F.normalize(torch.mean(torch.stack(es),dim=0),dim=1).cpu().numpy().flatten())
    return np.stack(out)  # (n_models,512)

def fuse(emb_stack, idxs):
    """选定模型集嵌入平均后重归一化 → (512,)"""
    v=emb_stack[idxs].mean(0); return v/(np.linalg.norm(v)+1e-8)

def eer(g,i):
    g,i=np.array(g),np.array(i); y=np.concatenate([np.ones(len(g)),np.zeros(len(i))]); s=np.concatenate([g,i])
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; k=np.nanargmin(np.abs(fnr-fpr)); return (fpr[k]+fnr[k])/2

# ---- 数据 ----
b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
random.seed(42)
print("Extracting old templates + test..."); sys.stdout.flush()
tmpl={}; test={}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if not os.path.exists(rp): continue
        ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
        if len(ps)<50: continue
        k=f'{pre}_{fg}'
        tmpl[k]=[emb_all(prep(load_img(ps[i]))) for i in random.sample(range(70),20)]
        test[k]=[emb_all(prep(load_img(p))) for p in ps[70:]]
ids=list(tmpl.keys())
print(f"  {len(ids)} classes."); sys.stdout.flush()

MERGE={'dy_R0':'wtp_dy_R0','dy_R1':'wtp_dy_R1','dy_R2':'wtp_dy_R2','lwh_R0':'wtp_lwh_R0','lwh_R1':'wtp_lwh_R1',
       'lwh_R2':'wtp_lwh_R2','zyh_R0':'btp_zyh_R0','zyh_R1':'btp_zyh_R1','zyh_R2':'btp_zyh_R2'}
nb='f:/1111/指纹/ysjz_raw/ysjz'; cross={}
for folder,gt in MERGE.items():
    cross.setdefault(gt,[])
    for p in sorted(glob.glob(os.path.join(nb,folder,'*.bmp')))[70:]:
        cross[gt].append(emb_all(prep(load_img(p))))
print("  cross probes extracted."); sys.stdout.flush()

def idx_of(*fams):
    return np.array([i for i,(fam,_,_) in enumerate(MODELS) if fam in fams])

def run(probes_by_gt, idxs, tnorm=False):
    # 预融合模板
    T={cid:[fuse(e,idxs) for e in tmpl[cid]] for cid in ids}
    g,i=[],[]
    for gt,plist in probes_by_gt.items():
        for e in plist:
            pe=fuse(e,idxs)
            sc={cid: float(max(pe@t for t in T[cid])) for cid in ids}
            for cid in ids:
                v=sc[cid]
                if tnorm:
                    imp=[sc[o] for o in ids if o!=cid]; v=(sc[cid]-np.mean(imp))/(np.std(imp)+1e-8)
                (g if cid==gt else i).append(v)
    return eer(g,i)

print(f"\n{'='*64}\nEVAL A (single-session old 27)  — V21 baseline raw=1.19%\n{'='*64}")
comboA=[('V21',idx_of('V21')),('V21+V18b',idx_of('V21','V18b')),
        ('V21+V18b+V17',idx_of('V21','V18b','V17')),
        ('V21+V25a+V26b',idx_of('V21','V25a','V26b')),
        ('ALL5',idx_of('V17','V18b','V21','V25a','V26b'))]
for name,idxs in comboA:
    r=run(test,idxs,tnorm=False); t=run(test,idxs,tnorm=True)
    print(f"  {name:16s} (n={len(idxs)}): raw={r*100:.4f}%  tnorm={t*100:.4f}%"); sys.stdout.flush()

print(f"\n{'='*64}\nEVAL B (cross-session)  — V25a baseline best=5.71%, global_raw=8.06%\n{'='*64}")
comboB=[('V25a',idx_of('V25a')),('V26b',idx_of('V26b')),('V25a+V26b',idx_of('V25a','V26b')),
        ('V25a+V26b+V21',idx_of('V25a','V26b','V21')),('ALL5',idx_of('V17','V18b','V21','V25a','V26b'))]
for name,idxs in comboB:
    r=run(cross,idxs,tnorm=False); t=run(cross,idxs,tnorm=True)
    print(f"  {name:16s} (n={len(idxs)}): raw={r*100:.4f}%  tnorm={t*100:.4f}%"); sys.stdout.flush()
print("\nDone.")
