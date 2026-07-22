# -*- coding: utf-8 -*-
"""#2 CPU上"仅匹配"耗时(不含特征提取) vs 模板数; #3 描述子长度(局部一起)。"""
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
    X=torch.tensor(np.stack(pat)[:,None],dtype=torch.float32); d=[]
    for i in range(0,len(X),2048): d.append(net(X[i:i+2048]).numpy())
    desc=np.concatenate(d).astype(np.float32)
    di8=np.clip(np.round(desc*127),-127,127).astype(np.int8)  # 正确int8量化
    return (np.round(np.array(pts)).astype(np.int16), di8, m)
def match(f1,f2):  # 仅匹配(几何验证), int8反量化
    p1,di1,m1=f1; p2,di2,m2=f2
    d1=di1.astype(np.float32); d2=di2.astype(np.float32)
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<0.92*b.distance]
    if len(g)<4: return 0
    src=p1[[x.queryIdx for x in g]].astype(np.float32).reshape(-1,1,2); dst=p2[[x.trainIdx for x in g]].astype(np.float32).reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=6,maxIters=2000,confidence=0.99)
    return int(mask.sum()) if mask is not None else 0
SEL='f:/1111/指纹/select_data/select'
fs=sorted([d for d in os.listdir(SEL) if os.path.isdir(os.path.join(SEL,d))])[:5]
gal=[]
for fg in fs:
    for p in sorted(glob.glob(os.path.join(SEL,fg,'*.bmp')))[:20]: gal.append(feat(load(p)))
probe=feat(load(sorted(glob.glob(os.path.join(SEL,fs[0],'*.bmp')))[25]))
print(f"CPU线程=8. 模板池={len(gal)}")
# warmup
_=match(probe,gal[0])
# #3 描述子长度
nkp=[len(g[0]) for g in gal]; avgkp=np.mean(nkp)
print(f"\n#3 描述子长度(局部):")
print(f"  每关键点局部描述子 = 128 维 → int8 {128*8}bit(128字节) / 二值化 {128}bit")
print(f"  每模板(帧) 平均 {avgkp:.0f} 个关键点 → 局部描述子共 {avgkp*128:.0f}字节 ≈ {avgkp*128*8/1000:.0f} kbit ({avgkp*128/1024:.0f}KB)")
print(f"  (纹理法为纯局部, 无单独全局描述子; 若加CNN全局: 512维 int8=4096bit/二值=512bit)")
# #2 仅匹配耗时 vs 模板数 (CPU, 不含feat)
print(f"\n#2 CPU 仅匹配耗时(不含特征提取), 探针 vs N模板:")
for N in [100,50,30,20,15,10]:
    ts=[]
    for rep in range(3):
        t1=time.perf_counter()
        for t in gal[:N]: match(probe,t)
        ts.append((time.perf_counter()-t1)*1000)
    print(f"  N={N:3d} 模板: 匹配耗时 {np.min(ts):.0f}ms (单模板≈{np.min(ts)/N:.2f}ms)")
print("Done.")
