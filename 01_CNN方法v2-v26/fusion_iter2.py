# -*- coding: utf-8 -*-
"""
迭代2: 在迭代1赢家(V21+V18b单场景tnorm=1.107%, V25a+V26b跨场景)上继续压低。
新增: (a)加权融合 (b)TTA旋转多视图 (c)分数级vs嵌入级融合 (d)局部描述子融合。
仅全局先跑(a)(b)(c); 局部融合(d)用V21/V18b/V25a/V26b的local_head。
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
def rotate(f, ang):
    h,w=f.shape
    return cv2.warpAffine(f, cv2.getRotationMatrix2D((w/2,h/2),ang,1.0),(w,h),borderValue=0)

def _stem(s):
    b=resnet18(weights=None)
    s.conv1=nn.Conv2d(1,64,7,2,3,bias=False); s.bn1,s.relu,s.maxpool=b.bn1,b.relu,b.maxpool
    s.layer1,s.layer2,s.layer3,s.layer4=b.layer1,b.layer2,b.layer3,b.layer4
    s.avgpool=nn.AdaptiveAvgPool2d((1,1))
class V17Enc(nn.Module):
    def __init__(s,e=512):
        super().__init__(); _stem(s); s.projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer4(s.layer3(s.layer2(s.layer1(x))))
        return s.projector(s.avgpool(x).flatten(1))
class DenseEnc(nn.Module):   # V18b
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x,local=False):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        ld=F.normalize(s.local_head(x),dim=1) if local else None
        g=s.global_projector(s.avgpool(s.layer4(x)).flatten(1))
        return (g,ld) if local else g
class HybridEnc(nn.Module):  # V21/V25a/V26b
    def __init__(s,e=512,l=64):
        super().__init__(); _stem(s)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.attention=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(),nn.Conv2d(128,1,1),nn.Softplus())
        s.attn_projector=nn.Sequential(nn.Linear(256,e),nn.BatchNorm1d(e))
        s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x,local=False):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        ld=F.normalize(s.local_head(x),dim=1) if local else None
        g=s.global_projector(s.avgpool(s.layer4(x)).flatten(1))
        return (g,ld) if local else g

dev=torch.device('cuda')
def load_family(cls,d,pre):
    out=[]
    for sd in [42,123,777]:
        m=cls().to(dev); m.load_state_dict(torch.load(f'f:/1111/指纹/{d}/{pre}_seed{sd}.pth',map_location=dev,weights_only=True)); m.eval(); out.append(m)
    return out
FAM={'V17':load_family(V17Enc,'models_v17','v17'),'V18b':load_family(DenseEnc,'models_v18b','v18b'),
     'V21':load_family(HybridEnc,'models_v21','v21'),'V25a':load_family(HybridEnc,'models_v25a','v25a'),
     'V26b':load_family(HybridEnc,'models_v26b','v26b')}
HAS_LOCAL={'V17':False,'V18b':True,'V21':True,'V25a':True,'V26b':True}
print("models loaded"); sys.stdout.flush()

@torch.no_grad()
def gemb(fams, frame, tta=False):
    """选定families的全局嵌入(每模型双极性[+TTA旋转]平均, 各模型归一化), 返回list按家族。"""
    angs=[0,5,-5] if tta else [0]
    per={}
    for fam in fams:
        vecs=[]
        for m in FAM[fam]:
            es=[]
            for a in angs:
                fr=rotate(frame,a) if a else frame
                for pol in (1,-1):
                    x=torch.tensor(pol*fr,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
                    es.append(F.normalize(m(x),dim=1))
            vecs.append(F.normalize(torch.mean(torch.stack(es),dim=0),dim=1).cpu().numpy().flatten())
        per[fam]=vecs   # list of 3 normalized embeddings
    return per

def fuse_weighted(per, weights):
    """per:{fam:[emb*3]}; weights:{fam:w} → 加权平均归一化(512,)"""
    acc=None
    for fam,w in weights.items():
        for e in per[fam]:
            acc = (e*w) if acc is None else acc + e*w
    return acc/(np.linalg.norm(acc)+1e-8)

def eer(g,i):
    g,i=np.array(g),np.array(i); y=np.concatenate([np.ones(len(g)),np.zeros(len(i))]); s=np.concatenate([g,i])
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; k=np.nanargmin(np.abs(fnr-fpr)); return (fpr[k]+fnr[k])/2

b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
ALLFAMS=['V17','V18b','V21','V25a','V26b']
random.seed(42)
print("extract templates/test (TTA)..."); sys.stdout.flush()
tmpl={}; test={}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if not os.path.exists(rp): continue
        ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
        if len(ps)<50: continue
        k=f'{pre}_{fg}'
        tmpl[k]=[gemb(ALLFAMS,prep(load_img(ps[i])),tta=True) for i in random.sample(range(70),20)]
        test[k]=[gemb(ALLFAMS,prep(load_img(p)),tta=True) for p in ps[70:]]
ids=list(tmpl.keys()); print(f"  {len(ids)} classes"); sys.stdout.flush()
MERGE={'dy_R0':'wtp_dy_R0','dy_R1':'wtp_dy_R1','dy_R2':'wtp_dy_R2','lwh_R0':'wtp_lwh_R0','lwh_R1':'wtp_lwh_R1',
       'lwh_R2':'wtp_lwh_R2','zyh_R0':'btp_zyh_R0','zyh_R1':'btp_zyh_R1','zyh_R2':'btp_zyh_R2'}
nb='f:/1111/指纹/ysjz_raw/ysjz'; cross={}
for folder,gt in MERGE.items():
    cross.setdefault(gt,[])
    for p in sorted(glob.glob(os.path.join(nb,folder,'*.bmp')))[70:]:
        cross[gt].append(gemb(ALLFAMS,prep(load_img(p)),tta=True))
print("  cross extracted"); sys.stdout.flush()

def run(probes, weights, tnorm=False):
    T={cid:[fuse_weighted(e,weights) for e in tmpl[cid]] for cid in ids}
    g,i=[],[]
    for gt,plist in probes.items():
        for e in plist:
            pe=fuse_weighted(e,weights)
            sc={cid: float(max(pe@t for t in T[cid])) for cid in ids}
            for cid in ids:
                v=sc[cid]
                if tnorm:
                    imp=[sc[o] for o in ids if o!=cid]; v=(v-np.mean(imp))/(np.std(imp)+1e-8)
                (g if cid==gt else i).append(v)
    return eer(g,i)

print(f"\n{'='*64}\nEVAL A single-session (TTA on)  — iter1 best V21+V18b tnorm=1.107%\n{'='*64}")
combosA=[('V21+V18b 1:1',{'V21':1,'V18b':1}),('V21+V18b 2:1',{'V21':2,'V18b':1}),
         ('V21+V18b 3:1',{'V21':3,'V18b':1}),('V21 only',{'V21':1}),
         ('V21+V18b+V25a 2:1:1',{'V21':2,'V18b':1,'V25a':1}),
         ('V21+V18b+V26b 2:1:1',{'V21':2,'V18b':1,'V26b':1})]
for name,w in combosA:
    print(f"  {name:22s}: raw={run(test,w)*100:.4f}%  tnorm={run(test,w,True)*100:.4f}%"); sys.stdout.flush()

print(f"\n{'='*64}\nEVAL B cross-session (TTA on)  — V25a best=5.71%\n{'='*64}")
combosB=[('V25a+V26b 1:1',{'V25a':1,'V26b':1}),('V25a+V26b 2:1',{'V25a':2,'V26b':1}),
         ('V25a only',{'V25a':1}),('V25a+V26b+V21 2:1:1',{'V25a':2,'V26b':1,'V21':1})]
for name,w in combosB:
    print(f"  {name:22s}: raw={run(cross,w)*100:.4f}%  tnorm={run(cross,w,True)*100:.4f}%"); sys.stdout.flush()
print("\nDone.")
