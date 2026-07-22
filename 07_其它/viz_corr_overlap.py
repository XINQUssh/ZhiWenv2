# -*- coding: utf-8 -*-
"""示意图: (1)同指两帧的真实对应点(RANSAC内点=重训用的patch对); (2)重叠区=把A的前景按位姿变换warp到B后与B前景求交。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
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
    return norm,m,u
class Desc(nn.Module):
    def __init__(s,d=128):
        super().__init__()
        def cbr(i,o,st=1): return [nn.Conv2d(i,o,3,st,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
        s.net=nn.Sequential(*cbr(1,32),*cbr(32,32),*cbr(32,64,2),*cbr(64,64),*cbr(64,128,2),*cbr(128,128),nn.AdaptiveAvgPool2d(1)); s.fc=nn.Linear(128,d)
    def forward(s,x): z=s.net(x).flatten(1); return F.normalize(s.fc(z),dim=1)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=Desc().to(dev); net.load_state_dict(torch.load('f:/1111/指纹/deliver/best.pth',map_location=dev)); net.eval()
HALF=16; STRIDE=8; bf=cv2.BFMatcher(cv2.NORM_L2)
@torch.no_grad()
def feat(img):
    norm,m,u=prep(img); H,W=norm.shape; pts=[]; pat=[]
    for y in range(HALF,H-HALF,STRIDE):
        for x in range(HALF,W-HALF,STRIDE):
            if m[y,x]>0: pts.append((x,y)); pat.append(norm[y-HALF:y+HALF,x-HALF:x+HALF])
    X=torch.tensor(np.stack(pat)[:,None],dtype=torch.float32).to(dev); d=[]
    for i in range(0,len(X),2048): d.append(net(X[i:i+2048]).cpu().numpy())
    return np.float32(pts), np.concatenate(d).astype(np.float32), m, u
def match(fa,fb,ratio=0.92):
    p1,d1,m1,_=fa; p2,d2,m2,_=fb
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    src=p1[[x.queryIdx for x in g]].reshape(-1,1,2); dst=p2[[x.trainIdx for x in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=6,maxIters=2000,confidence=0.99)
    inl=mask.ravel().astype(bool)
    return M, src[inl,0,:], dst[inl,0,:]
# 选SSH_L0里能对齐上的两帧
SEL='f:/1111/指纹/select_data/select/SSH_L0'; ps=sorted(glob.glob(SEL+'/*.bmp'))
fa=feat(load(ps[0])); best=None
for j in range(1,min(12,len(ps))):
    fb=feat(load(ps[j])); M,s,d=match(fa,fb)
    if M is not None and len(s)> (0 if best is None else len(best[1])): best=(j,s,d,M,fb)
j,src,dst,M,fb=best
p1,d1,m1,u1=fa; p2,d2,m2,u2=fb
h,w=m2.shape; w1=cv2.warpAffine(m1,M,(w,h),flags=cv2.INTER_NEAREST)
inter=((w1>0)&(m2>0)); ov=int(inter.sum())
print(f"pair (0,{j}): {len(src)} 内点对(真实对应), 重叠面积={ov}px")
# 画图
fig=plt.figure(figsize=(13,5.6))
# 左: 对应点连线
ax1=fig.add_subplot(1,2,1)
canvas=np.zeros((h,w*2+20),np.uint8); canvas[:, :w]=u1; canvas[:, w+20:]=u2
ax1.imshow(canvas,cmap='gray')
rng=np.random.RandomState(0); idx=rng.choice(len(src),min(40,len(src)),replace=False)
for i in idx:
    x1,y1=src[i]; x2,y2=dst[i]
    ax1.plot([x1,x2+w+20],[y1,y2],'-',color=plt.cm.hsv(i/len(src)),lw=0.6,alpha=0.8)
    ax1.plot(x1,y1,'o',ms=2,color='lime'); ax1.plot(x2+w+20,y2,'o',ms=2,color='lime')
ax1.set_title(f'① 真实对应点(RANSAC内点={len(src)}对)\n每条线两端=同一物理脊线点在两帧的位置\n→取其32×32 patch作正样本对重训描述子',fontsize=10.5)
ax1.axis('off'); ax1.text(w/2,h+8,'帧A',ha='center'); ax1.text(w+20+w/2,h+8,'帧B',ha='center')
# 右: 重叠区
ax2=fig.add_subplot(1,2,2)
rgb=cv2.cvtColor(u2,cv2.COLOR_GRAY2RGB)
rgb[m2>0]=(0.6*rgb[m2>0]+np.array([0,60,0])).clip(0,255).astype(np.uint8)   # B前景淡绿
rgb[w1>0]=(0.6*rgb[w1>0]+np.array([60,0,0])).clip(0,255).astype(np.uint8)   # A(warp后)前景淡红
rgb[inter]=(0.4*rgb[inter]+np.array([120,120,0])).clip(0,255).astype(np.uint8)  # 交集=黄
ax2.imshow(rgb)
ax2.set_title(f'② 重叠区 = 把A前景按位姿M变换到B坐标 ∩ B前景\n红=A(变换后)前景  绿=B前景  黄=重叠({ov}px)\n分数 = 内点数 × √(20000/重叠面积)',fontsize=10.5)
ax2.axis('off')
plt.tight_layout()
open('f:/1111/指纹/_tmp_co.png','wb').write(b''); plt.savefig('f:/1111/指纹/_tmp_co.png',dpi=120,bbox_inches='tight')
open('f:/1111/指纹/对应点与重叠区_示意.png','wb').write(open('f:/1111/指纹/_tmp_co.png','rb').read()); os.remove('f:/1111/指纹/_tmp_co.png')
print('saved 对应点与重叠区_示意.png')
