# -*- coding: utf-8 -*-
"""域差诊断: 贴屏(select) vs 老数据(无贴屏/不贴屏) 原图+mask+归一化 对比图。CPU only。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob, numpy as np, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
import importlib.util
spec=importlib.util.spec_from_file_location("mt","f:/1111/指纹/mine_and_train.py")
# 直接复制prep相关(避免import触发训练)。简单重写:
def load(p):
    with open(p,'rb') as f: return cv2.imdecode(np.frombuffer(f.read(),np.uint8),cv2.IMREAD_GRAYSCALE)
def clahe(img): return cv2.createCLAHE(4.0,(8,8)).apply(img)
def up2(img): return cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
def keep(m):
    n,l,s,_=cv2.connectedComponentsWithStats(m,8,cv2.CV_32S)
    return m if n<=1 else ((l==1+np.argmax(s[1:,cv2.CC_STAT_AREA]))*255).astype(np.uint8)
def fill(m):
    bg=cv2.bitwise_not(m); n,l,s,_=cv2.connectedComponentsWithStats(bg,8,cv2.CV_32S); h,w=m.shape
    for i in range(1,n):
        L,T,W,H=s[i,cv2.CC_STAT_LEFT],s[i,cv2.CC_STAT_TOP],s[i,cv2.CC_STAT_WIDTH],s[i,cv2.CC_STAT_HEIGHT]
        if L>0 and L+W<w-1 and T>0 and T+H<h-1: m[l==i]=255
    return m
def gm(img,sig=13/3.,pct=95,r=.2):
    dx=cv2.Sobel(img,cv2.CV_32F,1,0,3); dy=cv2.Sobel(img,cv2.CV_32F,0,1,3); mg=cv2.magnitude(dx,dy)
    gs=int(np.ceil(3*sig))*2+1; ma=cv2.GaussianBlur(mg,(gs,gs),sig)
    th=np.percentile(mg.flatten(),pct)*r; _,m=cv2.threshold(ma,th,255,0); m=m.astype(np.uint8)
    se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,se,iterations=6); m=keep(m); m=fill(m)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,se,iterations=2); return keep(m)
def prep(img):
    e=clahe(img); u=up2(e); m=gm(u); mk=u.copy(); mk[m==0]=0
    v=mk[m>0].astype(np.float32); mu,sd=(v.mean(),v.std()+1e-6) if len(v) else (0,1)
    norm=(mk.astype(np.float32)-mu)/sd; norm[m==0]=0
    return e,u,m,norm

SEL='f:/1111/指纹/select_data/select'
OLD1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
samples=[]
# 3 贴屏
for fg in ['dy_L0','SSH_L0','zyh_R0']:
    p=sorted(glob.glob(os.path.join(SEL,fg,'*.bmp')))[0]; samples.append((f'贴屏 {fg}',p))
# 2 老数据
for fg in sorted(os.listdir(OLD1))[:2]:
    rp=os.path.join(OLD1,fg,'Rgd1245')
    ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
    if ps: samples.append((f'老(无贴屏) {fg}',ps[0]))
n=len(samples)
fig,ax=plt.subplots(n,3,figsize=(9,3*n))
for r,(name,p) in enumerate(samples):
    img=load(p); e,u,m,norm=prep(img)
    cov=100*np.count_nonzero(m)/m.size
    # 前景内对比度(梯度均值)估计ridge清晰度
    g=cv2.magnitude(cv2.Sobel(u,cv2.CV_32F,1,0,3),cv2.Sobel(u,cv2.CV_32F,0,1,3))
    sharp=g[m>0].mean() if np.count_nonzero(m) else 0
    ax[r,0].imshow(img,cmap='gray'); ax[r,0].set_title(f'{name}\n原图 {img.shape}',fontsize=9)
    ax[r,1].imshow(u,cmap='gray'); ax[r,1].imshow(m,cmap='Reds',alpha=0.25); ax[r,1].set_title(f'CLAHE+上采样+前景\ncov={cov:.0f}% sharp={sharp:.0f}',fontsize=9)
    nn=norm.copy(); nn[m==0]=np.nan
    ax[r,2].imshow(nn,cmap='gray'); ax[r,2].set_title('归一化(送描述子)',fontsize=9)
    for c in range(3): ax[r,c].axis('off')
plt.suptitle('域差诊断: 贴屏 select vs 老数据 (原图/前景/归一化)',fontsize=12)
plt.tight_layout()
buf=cv2.imencode('.png',np.zeros((1,1)))  # dummy
plt.savefig('f:/1111/指纹/_tmp_domain.png',dpi=110,bbox_inches='tight')
# 中文路径保存
import shutil
data=open('f:/1111/指纹/_tmp_domain.png','rb').read()
open('f:/1111/指纹/域差诊断_贴屏vs老.png','wb').write(data); os.remove('f:/1111/指纹/_tmp_domain.png')
print('saved 域差诊断_贴屏vs老.png')
for name,p in samples:
    img=load(p); print(f'  {name}: shape={img.shape} mean={img.mean():.0f} std={img.std():.0f}')
