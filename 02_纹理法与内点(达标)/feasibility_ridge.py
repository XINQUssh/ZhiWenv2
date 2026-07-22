# -*- coding: utf-8 -*-
"""
脊线结构(ridge/脊骨)可行性实验 — 给客户"脊骨变换"方向最对口的公平验证。
测试: 灰度图 / Gabor增强图 / 二值脊线 / 脊线骨架 四种表征, 用平移搜索的归一化互相关匹配,
      在单场景与跨场景上比较, 并与V25a CNN融合, 看脊线结构是否比灰度外观跨场景更稳。
不训练。
"""
import os, glob, random, sys, time
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_curve
from torchvision.models import resnet18

HAS_THIN = hasattr(cv2, 'ximgproc')

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
    return norm,u,m

def pixel_orient(u,m):
    g=u.astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    Gxx=cv2.GaussianBlur(gx*gx,(0,0),3); Gyy=cv2.GaussianBlur(gy*gy,(0,0),3); Gxy=cv2.GaussianBlur(gx*gy,(0,0),3)
    theta=0.5*np.arctan2(2*Gxy,Gxx-Gyy)          # 梯度方向
    return (theta+np.pi/2)%np.pi                   # 脊线方向(垂直于梯度)

def gabor_enhance(u,m,wavelength=9.0):
    img=u.astype(np.float32); img=(img-img.mean())/(img.std()+1e-6)
    ridge=pixel_orient(u,m); n=8
    resp=np.stack([cv2.filter2D(img,cv2.CV_32F,
            cv2.getGaborKernel((21,21),4.0,o,wavelength,0.5,0)) for o in np.arange(n)*np.pi/n])
    idx=(np.round(ridge/(np.pi/n)).astype(int))%n
    H,W=img.shape; out=resp[idx,np.arange(H)[:,None],np.arange(W)[None,:]]
    out=out*(m>0); return out

def ridge_binary(u,m):
    e=gabor_enhance(u,m); e=cv2.normalize(e,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    b=cv2.adaptiveThreshold(e,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    b[m==0]=0; return b

def ridge_skeleton(u,m):
    b=ridge_binary(u,m)
    if HAS_THIN:
        sk=cv2.ximgproc.thinning(b)
    else:
        sk=cv2.morphologyEx(b,cv2.MORPH_GRADIENT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    sk[m==0]=0; return sk

def best_corr(probe,tmpl,margin=10,method=cv2.TM_CCOEFF_NORMED):
    t=tmpl[margin:-margin,margin:-margin]
    if t.shape[0]<8 or t.shape[1]<8: return 0.0
    res=cv2.matchTemplate(probe.astype(np.float32),t.astype(np.float32),method)
    return float(res.max())

# CNN
class Enc(nn.Module):
    def __init__(s,e=512,l=64):
        super().__init__(); b=resnet18(weights=None)
        s.conv1=nn.Conv2d(1,64,7,2,3,bias=False); s.bn1,s.relu,s.maxpool=b.bn1,b.relu,b.maxpool
        s.layer1,s.layer2,s.layer3,s.layer4=b.layer1,b.layer2,b.layer3,b.layer4
        s.avgpool=nn.AdaptiveAvgPool2d((1,1)); s.global_projector=nn.Sequential(nn.Linear(512,e),nn.BatchNorm1d(e))
        s.attention=nn.Sequential(nn.Conv2d(256,128,1),nn.ReLU(),nn.Conv2d(128,1,1),nn.Softplus())
        s.attn_projector=nn.Sequential(nn.Linear(256,e),nn.BatchNorm1d(e))
        s.local_head=nn.Sequential(nn.Conv2d(256,l,1,bias=False),nn.BatchNorm2d(l))
    def forward(s,x):
        x=s.maxpool(s.relu(s.bn1(s.conv1(x)))); x=s.layer3(s.layer2(s.layer1(x)))
        return s.global_projector(s.avgpool(s.layer4(x)).flatten(1))
dev=torch.device('cuda'); MM=[]
for sd in [42,123,777]:
    m=Enc().to(dev); m.load_state_dict(torch.load(f'f:/1111/指纹/models_v25a/v25a_seed{sd}.pth',map_location=dev,weights_only=True)); m.eval(); MM.append(m)
@torch.no_grad()
def embed(norm):
    es=[]
    for m in MM:
        for p in (1,-1):
            x=torch.tensor(p*norm,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev); es.append(F.normalize(m(x),dim=1))
    return F.normalize(torch.mean(torch.stack(es),dim=0),dim=1).cpu().numpy().flatten()
def eer(g,i):
    g,i=np.array(g),np.array(i); y=np.concatenate([np.ones(len(g)),np.zeros(len(i))]); s=np.concatenate([g,i])
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; k=np.nanargmin(np.abs(fnr-fpr)); return (fpr[k]+fnr[k])/2
print(f"V25a loaded. ximgproc thinning={HAS_THIN}"); sys.stdout.flush()

# ---- 模板(老数据): 每类3帧, 存 gray/gabor/skel/cnn ----
b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
random.seed(42)
TMPL={}; OLD_TEST={}
def feats(img):
    norm,u,m=prep(img)
    return {'gray':(norm*255).astype(np.float32),'gabor':gabor_enhance(u,m),
            'skel':ridge_skeleton(u,m).astype(np.float32),'cnn':embed(norm)}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if not os.path.exists(rp): continue
        ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
        if len(ps)<50: continue
        k=f'{pre}_{fg}'
        TMPL[k]=[feats(load_img(ps[i])) for i in random.sample(range(70),3)]
        OLD_TEST[k]=[feats(load_img(p)) for p in random.sample(ps[70:],min(8,len(ps)-70))]
ids=list(TMPL.keys()); print(f"{len(ids)} classes ready."); sys.stdout.flush()

def evaluate(name,probes):
    keys=['gray','gabor','skel','cnn']; G={k:[] for k in keys}; I={k:[] for k in keys}
    fuse={k:([],[]) for k in ['gabor','skel']}  # 与cnn融合(0.3权重)
    for gt,plist in probes.items():
        for pf in plist:
            for cid in ids:
                sc={}
                sc['cnn']=max(np.dot(pf['cnn'],t['cnn']) for t in TMPL[cid])
                for rep in ['gray','gabor','skel']:
                    sc[rep]=max(best_corr(pf[rep],t[rep]) for t in TMPL[cid])
                tag=0 if cid==gt else 1
                for k in keys: (G[k] if tag==0 else I[k]).append(sc[k])
                for rep in ['gabor','skel']:
                    fuse[rep][tag].append(0.3*sc[rep]+0.7*sc['cnn'])
    print(f"\n=== {name} ===")
    for k in keys: print(f"  {k:6s} EER = {eer(G[k],I[k])*100:.3f}%")
    for rep in ['gabor','skel']: print(f"  cnn+{rep}(0.3) EER = {eer(fuse[rep][0],fuse[rep][1])*100:.3f}%")

evaluate("SINGLE-SESSION (old)", OLD_TEST)

MERGE={'dy_R0':'wtp_dy_R0','dy_R1':'wtp_dy_R1','dy_R2':'wtp_dy_R2','lwh_R0':'wtp_lwh_R0','lwh_R1':'wtp_lwh_R1',
       'lwh_R2':'wtp_lwh_R2','zyh_R0':'btp_zyh_R0','zyh_R1':'btp_zyh_R1','zyh_R2':'btp_zyh_R2'}
nb='f:/1111/指纹/ysjz_raw/ysjz'; cross={}
for folder,gt in MERGE.items():
    ps=sorted(glob.glob(os.path.join(nb,folder,'*.bmp')))[70:]
    cross[gt]=[feats(load_img(p)) for p in random.sample(ps,min(15,len(ps)))]
evaluate("CROSS-SESSION (old tmpl x new raw probe)", cross)
print("\nDone."); sys.stdout.flush()
