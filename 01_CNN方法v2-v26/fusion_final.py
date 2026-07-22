# -*- coding: utf-8 -*-
"""
最终融合评估: 支持 ResNet18 + ResNet34 混合集成。
家族: V17/V18b/V21/V25a/V26b(r18) + v21r34/v25r34(r34, 训练后存在则纳入)。
单场景: 在 V21+V18b(+v21r34) 全局+局部融合+tnorm 上找最优。
跨场景: 在 V25a+V26b+V21(+v25r34) 全局+TTA+tnorm 上找最优。
对缺失家族自动跳过(便于分阶段运行)。
"""
import os, glob, random, sys
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_curve
from torchvision.models import resnet18, resnet34

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
def rotate(f,a):
    h,w=f.shape; return cv2.warpAffine(f,cv2.getRotationMatrix2D((w/2,h/2),a,1.0),(w,h),borderValue=0)

def _stem(s, backbone):
    b=backbone(weights=None)
    s.conv1=nn.Conv2d(1,64,7,2,3,bias=False); s.bn1,s.relu,s.maxpool=b.bn1,b.relu,b.maxpool
    s.layer1,s.layer2,s.layer3,s.layer4=b.layer1,b.layer2,b.layer3,b.layer4
    s.avgpool=nn.AdaptiveAvgPool2d((1,1))
class V17Enc(nn.Module):
    def __init__(s,bb=resnet18,e=512):
        super().__init__(); _stem(s,bb); s.projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer4(s.layer3(s.layer2(s.layer1(x))))
        return s.projector(s.avgpool(x).flatten(1)), None
class DenseEnc(nn.Module):
    def __init__(s,bb=resnet18,e=512,l=64):
        super().__init__(); _stem(s,bb)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        ld=F.normalize(s.local_head(x),dim=1)
        return s.global_projector(s.avgpool(s.layer4(x)).flatten(1)), ld
class HybridEnc(nn.Module):
    def __init__(s,bb=resnet18,e=512,l=64):
        super().__init__(); _stem(s,bb)
        s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.attention=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(),nn.Conv2d(128,1,1),nn.Softplus())
        s.attn_projector=nn.Sequential(nn.Linear(256,e),nn.BatchNorm1d(e))
        s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        ld=F.normalize(s.local_head(x),dim=1)
        return s.global_projector(s.avgpool(s.layer4(x)).flatten(1)), ld

dev=torch.device('cuda')
# (家族名, 编码器类, backbone, 目录, 文件前缀, 是否有local)
SPECS=[('V17',V17Enc,resnet18,'models_v17','v17',False),
       ('V18b',DenseEnc,resnet18,'models_v18b','v18b',True),
       ('V21',HybridEnc,resnet18,'models_v21','v21',True),
       ('V25a',HybridEnc,resnet18,'models_v25a','v25a',True),
       ('V26b',HybridEnc,resnet18,'models_v26b','v26b',True),
       ('V21r34',HybridEnc,resnet34,'models_v21r34','v21r34',True),
       ('V25r34',HybridEnc,resnet34,'models_v25r34','v25r34',True)]
FAM={}; HAS_LOCAL={}
for name,cls,bb,d,pre,hl in SPECS:
    paths=[f'f:/1111/指纹/{d}/{pre}_seed{sd}.pth' for sd in [42,123,777]]
    if not all(os.path.exists(p) for p in paths):
        print(f"  skip {name} (not trained yet)"); continue
    ms=[]
    for p in paths:
        m=cls(bb=bb).to(dev); m.load_state_dict(torch.load(p,map_location=dev,weights_only=True)); m.eval(); ms.append(m)
    FAM[name]=ms; HAS_LOCAL[name]=hl
print(f"loaded families: {list(FAM.keys())}"); sys.stdout.flush()

@torch.no_grad()
def feats(fams, frame, tta=False):
    angs=[0,5,-5] if tta else [0]
    out={}
    for fam in fams:
        if fam not in FAM: continue
        gs,lds=[],[]
        for m in FAM[fam]:
            ge,le=[],[]
            for a in angs:
                fr=rotate(frame,a) if a else frame
                for pol in (1,-1):
                    x=torch.tensor(pol*fr,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
                    g,ld=m(x); ge.append(F.normalize(g,dim=1))
                    if ld is not None and a==0: le.append(ld[0].view(ld.shape[1],-1).t())
            gs.append(F.normalize(torch.mean(torch.stack(ge),dim=0),dim=1).cpu().numpy().flatten())
            lds.append(F.normalize(torch.mean(torch.stack(le),dim=0),dim=1).cpu().numpy() if le else None)
        out[fam]=(gs,lds)
    return out

def fuse_g(per,weights):
    acc=None
    for fam,w in weights.items():
        if fam not in per: continue
        for e in per[fam][0]: acc=e*w if acc is None else acc+e*w
    return acc/(np.linalg.norm(acc)+1e-8)
def eer(g,i):
    g,i=np.array(g),np.array(i); y=np.concatenate([np.ones(len(g)),np.zeros(len(i))]); s=np.concatenate([g,i])
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; k=np.nanargmin(np.abs(fnr-fpr)); return (fpr[k]+fnr[k])/2

b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
ALL=list(FAM.keys())
TTA=False  # 单场景默认关TTA(有害); 跨场景下方单独开
random.seed(42); print("extract..."); sys.stdout.flush()
tmpl={}; test={}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if not os.path.exists(rp): continue
        ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
        if len(ps)<50: continue
        k=f'{pre}_{fg}'
        tmpl[k]=[feats(ALL,prep(load_img(ps[i])),tta=False) for i in random.sample(range(70),10)]
        test[k]=[feats(ALL,prep(load_img(p)),tta=False) for p in ps[70:]]
ids=list(tmpl.keys()); print(f"  {len(ids)} classes"); sys.stdout.flush()
MERGE={'dy_R0':'wtp_dy_R0','dy_R1':'wtp_dy_R1','dy_R2':'wtp_dy_R2','lwh_R0':'wtp_lwh_R0','lwh_R1':'wtp_lwh_R1',
       'lwh_R2':'wtp_lwh_R2','zyh_R0':'btp_zyh_R0','zyh_R1':'btp_zyh_R1','zyh_R2':'btp_zyh_R2'}
nb='f:/1111/指纹/ysjz_raw/ysjz'; cross={}
for folder,gt in MERGE.items():
    cross.setdefault(gt,[])
    for p in sorted(glob.glob(os.path.join(nb,folder,'*.bmp')))[70:]:
        cross[gt].append(feats(ALL,prep(load_img(p)),tta=True))
print("  cross done"); sys.stdout.flush()

def build_LT(lfams):
    LT={}
    for cid in ids:
        LT[cid]={}
        for fam in lfams:
            if fam not in FAM or not HAS_LOCAL.get(fam): continue
            per=[]
            for mi in range(3):
                parts=[t[fam][1][mi] for t in tmpl[cid] if t[fam][1][mi] is not None]
                per.append(np.concatenate(parts,axis=0))
            LT[cid][fam]=per
    return LT

def run(probes,gw,lfams,alpha,tnorm,LT,use_tta):
    GT={cid:[fuse_g(e,gw) for e in tmpl[cid]] for cid in ids}
    g,i=[],[]
    for gt,plist in probes.items():
        for e in plist:
            pe=fuse_g(e,gw); sc={}
            for cid in ids:
                gs=float(max(pe@t for t in GT[cid]))
                if alpha>0 and lfams:
                    ls=[]
                    for fam in lfams:
                        if fam not in FAM or not HAS_LOCAL.get(fam): continue
                        for mi in range(3):
                            Q=e[fam][1][mi]
                            if Q is None: continue
                            T=LT[cid][fam][mi]; S=Q@T.T; mx=S.max(axis=1)
                            k=max(1,int(0.7*len(mx))); ls.append(np.sort(mx)[-k:].mean())
                    sc[cid]=alpha*float(np.mean(ls))+(1-alpha)*gs if ls else gs
                else: sc[cid]=gs
            for cid in ids:
                v=sc[cid]
                if tnorm:
                    imp=[sc[o] for o in ids if o!=cid]; v=(v-np.mean(imp))/(np.std(imp)+1e-8)
                (g if cid==gt else i).append(v)
    return eer(g,i)

print(f"\n{'='*64}\nSINGLE-SESSION (best=1.093%, +v21r34?)\n{'='*64}")
singles=[('V21+V18b',{'V21':1,'V18b':1},['V21','V18b'])]
if 'V21r34' in FAM:
    singles+=[('V21+V18b+V21r34',{'V21':1,'V18b':1,'V21r34':1},['V21','V18b','V21r34']),
              ('V21+V21r34',{'V21':1,'V21r34':1},['V21','V21r34'])]
for name,gw,lf in singles:
    for a in [0.0,0.3]:
        print(f"  {name:22s} a={a}: tnorm={run(test,gw,lf,a,True,build_LT(lf),False)*100:.4f}%"); sys.stdout.flush()

print(f"\n{'='*64}\nCROSS-SESSION (best=5.18%, +v25r34?)  [TTA on]\n{'='*64}")
crosses=[('V25a+V26b+V21 2:1:1',{'V25a':2,'V26b':1,'V21':1})]
if 'V25r34' in FAM:
    crosses+=[('+V25r34',{'V25a':2,'V26b':1,'V21':1,'V25r34':1}),
              ('V25a+V26b+V25r34',{'V25a':1,'V26b':1,'V25r34':1})]
for name,gw in crosses:
    print(f"  {name:26s}: raw={run(cross,gw,[],0,False,None,True)*100:.4f}%  "
          f"tnorm={run(cross,gw,[],0,True,None,True)*100:.4f}%"); sys.stdout.flush()
print("\nDone.")
