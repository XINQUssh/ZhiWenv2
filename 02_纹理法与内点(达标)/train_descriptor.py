# -*- coding: utf-8 -*-
"""
步骤1: 训练 HardNet 风格密集纹理描述子(替代SIFT)。
自监督: 脊线patch为anchor, 经旋转/平移抖动/对比/形变/噪声增广为positive(模拟帧间变化),
        批内其他patch为negative, 用"批内最难负样本"三元组损失训练。
输出: models_texdesc/texdesc.pth (32x32 patch -> 128维L2归一化描述子)
"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob, random, sys, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

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
    return norm,m

PS=32; HALF=PS//2
b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'; b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
print("load train frames..."); sys.stdout.flush()
frames=[]  # (norm, mask)
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if not os.path.exists(rp): continue
        ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
        if len(ps)<50: continue
        for p in ps[:70]:
            n,m=prep(load_img(p)); frames.append((n,m))
print(f"  {len(frames)} frames"); sys.stdout.flush()
# 预采可用中心点(前景且离边界>HALF+6给抖动留量)
H,W=frames[0][0].shape
centers=[]
for fi,(n,m) in enumerate(frames):
    ys,xs=np.where(m>0)
    ok=(xs>HALF+6)&(xs<W-HALF-6)&(ys>HALF+6)&(ys<H-HALF-6)
    xs,ys=xs[ok],ys[ok]
    if len(xs)>0:
        sel=np.random.RandomState(fi).choice(len(xs),min(40,len(xs)),replace=False)
        for j in sel: centers.append((fi,int(xs[j]),int(ys[j])))
print(f"  {len(centers)} candidate patch centers"); sys.stdout.flush()

def get_patch(fi,x,y): return frames[fi][0][y-HALF:y+HALF, x-HALF:x+HALF]
def aug(fi,x,y):
    dx,dy=random.randint(-3,3),random.randint(-3,3)
    p=frames[fi][0][y+dy-HALF:y+dy+HALF, x+dx-HALF:x+dx+HALF].copy()
    if p.shape!=(PS,PS): p=get_patch(fi,x,y).copy()
    if random.random()<0.5: p=-p
    if random.random()<0.5:
        ang=random.uniform(-20,20); p=cv2.warpAffine(p,cv2.getRotationMatrix2D((HALF,HALF),ang,1.0),(PS,PS),borderValue=0)
    if random.random()<0.6: p=p*random.uniform(0.7,1.3)
    if random.random()<0.3: p=np.sign(p)*(np.abs(p)**random.uniform(0.7,1.4))
    if random.random()<0.3:
        a=8; sx=cv2.GaussianBlur(np.random.randn(PS,PS).astype(np.float32)*a,(0,0),4); sy=cv2.GaussianBlur(np.random.randn(PS,PS).astype(np.float32)*a,(0,0),4)
        gx,gy=np.meshgrid(np.arange(PS),np.arange(PS)); p=cv2.remap(p,(gx+sx).astype(np.float32),(gy+sy).astype(np.float32),cv2.INTER_LINEAR,borderValue=0)
    p=p+np.random.randn(PS,PS).astype(np.float32)*0.04
    return p

class Desc(nn.Module):
    def __init__(s,d=128):
        super().__init__()
        def cbr(i,o,st=1): return [nn.Conv2d(i,o,3,st,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
        s.net=nn.Sequential(*cbr(1,32),*cbr(32,32),*cbr(32,64,2),*cbr(64,64),*cbr(64,128,2),*cbr(128,128),
                            nn.AdaptiveAvgPool2d(1))
        s.fc=nn.Linear(128,d)
    def forward(s,x):
        z=s.net(x).flatten(1); z=s.fc(z); return F.normalize(z,dim=1)

dev=torch.device('cuda'); net=Desc().to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
B=384; ITERS=4000
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print("training..."); sys.stdout.flush(); t0=time.time()
for it in range(ITERS):
    idx=random.sample(range(len(centers)),B)
    A=np.zeros((B,1,PS,PS),np.float32); P=np.zeros((B,1,PS,PS),np.float32)
    for k,ci in enumerate(idx):
        fi,x,y=centers[ci]; A[k,0]=get_patch(fi,x,y); P[k,0]=aug(fi,x,y)
    A=torch.tensor(A).to(dev); P=torch.tensor(P).to(dev)
    da=net(A); dp=net(P)
    D=2-2*torch.mm(da,dp.t())  # (B,B) L2平方距离(单位向量)
    pos=torch.diag(D)
    eye=torch.eye(B,device=dev).bool()
    Dmask=D.masked_fill(eye,1e4)
    neg=torch.min(Dmask.min(1).values, Dmask.min(0).values)  # 批内最难负(双向)
    loss=F.relu(1.0+pos-neg).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (it+1)%200==0:
        print(f"  it{it+1}/{ITERS} loss={loss.item():.4f} pos={pos.mean().item():.3f} neg={neg.mean().item():.3f} t={time.time()-t0:.0f}s"); sys.stdout.flush()
os.makedirs('f:/1111/指纹/models_texdesc',exist_ok=True)
torch.save(net.state_dict(),'f:/1111/指纹/models_texdesc/texdesc.pth')
print(f"saved models_texdesc/texdesc.pth  t={time.time()-t0:.0f}s"); print("Done.")
