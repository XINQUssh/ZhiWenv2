# -*- coding: utf-8 -*-
"""CPU特征提取耗时 vs torch线程数(排查过订阅)。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob, time, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
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
HALF=16
def patches(img,STRIDE):
    norm,m=prep(img); H,W=norm.shape; pts=[]; pat=[]
    for y in range(HALF,H-HALF,STRIDE):
        for x in range(HALF,W-HALF,STRIDE):
            if m[y,x]>0: pts.append((x,y)); pat.append(norm[y-HALF:y+HALF,x-HALF:x+HALF])
    return np.stack(pat)[:,None].astype(np.float32) if pat else None, len(pat)
imgs=[load(p) for p in sorted(glob.glob('f:/1111/指纹/select_data/select/SSH_L0/*.bmp'))[:12]]
print("prep+patch once to get patch counts...")
P8=[patches(im,8) for im in imgs]; P10=[patches(im,10) for im in imgs]; P12=[patches(im,12) for im in imgs]
import numpy as np
print(f"avg patches: stride8={np.mean([p[1] for p in P8]):.0f}  stride10={np.mean([p[1] for p in P10]):.0f}  stride12={np.mean([p[1] for p in P12]):.0f}")
@torch.no_grad()
def run_batch(X,nt):
    torch.set_num_threads(nt); d=[]
    for i in range(0,len(X),2048): d.append(net(torch.tensor(X[i:i+2048])).numpy())
    return np.concatenate(d)
for nt in [1,2,4,6,8,16,24]:
    # warmup
    _=run_batch(P8[0][0],nt)
    ts=[]
    for X,_n in P8:
        t1=time.perf_counter(); _=run_batch(X,nt); ts.append((time.perf_counter()-t1)*1000)
    print(f"  threads={nt:2d}: CNN forward(stride8, ~{P8[0][1]}patch) 平均={np.mean(ts):.0f}ms 中位={np.median(ts):.0f}ms")
# 也测 完整prep+feat 在最优线程
print("full prep also timed separately:")
for nt in [4,8]:
    torch.set_num_threads(nt); ts=[]
    for im in imgs:
        t1=time.perf_counter(); X,_n=patches(im,8); _=run_batch(X,nt); ts.append((time.perf_counter()-t1)*1000)
    print(f"  threads={nt}: prep+patch+CNN(完整feat) 平均={np.mean(ts):.0f}ms")
print("Done.")
