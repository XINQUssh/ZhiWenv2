# -*- coding: utf-8 -*-
"""结合指纹脊线特征的多级融合匹配 (btp不贴屏)。
Level1 学习纹理描述子 + Level2 脊线方向场一致性 + Level3 脊线频率一致性。
对比: 基线(仅几何内点) vs 融合(方向+频率一致的内点)。看能否压冲突尾巴、提FAR=0。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
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
def orient_field(u,blk=16):
    # 脊线方向场(结构张量), 返回每像素 theta(mod pi)
    gx=cv2.Sobel(u,cv2.CV_32F,1,0,3); gy=cv2.Sobel(u,cv2.CV_32F,0,1,3)
    Gxx=cv2.boxFilter(gx*gx,-1,(blk,blk)); Gyy=cv2.boxFilter(gy*gy,-1,(blk,blk)); Gxy=cv2.boxFilter(gx*gy,-1,(blk,blk))
    theta=0.5*np.arctan2(2*Gxy, Gxx-Gyy+1e-6)   # 梯度主方向; 脊线方向=+pi/2, 相对一致即可
    return theta.astype(np.float32)
class Desc(nn.Module):
    def __init__(s,d=128):
        super().__init__()
        def cbr(i,o,st=1): return [nn.Conv2d(i,o,3,st,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
        s.net=nn.Sequential(*cbr(1,32),*cbr(32,32),*cbr(32,64,2),*cbr(64,64),*cbr(64,128,2),*cbr(128,128),nn.AdaptiveAvgPool2d(1)); s.fc=nn.Linear(128,d)
    def forward(s,x): z=s.net(x).flatten(1); return F.normalize(s.fc(z),dim=1)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=Desc().to(dev); net.load_state_dict(torch.load('f:/1111/指纹/deliver/best.pth',map_location=dev)); net.eval()
HALF=16; STRIDE=6; HO=20000.0; bf=cv2.BFMatcher(cv2.NORM_L2)
def patch_freq(p):
    # 32x32 patch 的主脊线频率(FFT径向峰), 用于频率一致性
    f=np.fft.fftshift(np.abs(np.fft.fft2(p*np.hanning(32)[:,None]*np.hanning(32)[None,:])))
    f[14:19,14:19]=0  # 去DC附近
    idx=np.unravel_index(np.argmax(f),f.shape); return float(np.hypot(idx[0]-16,idx[1]-16))
@torch.no_grad()
def feat(img):
    norm,m=prep(img); u2=norm; th=orient_field(u2); H,W=norm.shape; pts=[]; pat=[]; ori=[]
    for y in range(HALF,H-HALF,STRIDE):
        for x in range(HALF,W-HALF,STRIDE):
            if m[y,x]>0: pts.append((x,y)); pat.append(norm[y-HALF:y+HALF,x-HALF:x+HALF]); ori.append(th[y,x])
    if len(pat)<4: return None
    P=np.stack(pat); X=torch.tensor(P[:,None],dtype=torch.float32).to(dev); d=[]
    for i in range(0,len(X),2048): d.append(net(X[i:i+2048]).cpu().numpy())
    frq=np.array([patch_freq(pp) for pp in P],np.float32)
    return dict(pts=np.float32(pts),desc=np.concatenate(d).astype(np.float32),m=m,ori=np.array(ori,np.float32),frq=frq)
def angdiff_pi(a,b):
    d=np.abs(a-b)%np.pi; return np.minimum(d,np.pi-d)
def match(f1,f2,ratio=0.92,reproj=6,ori_tol=np.deg2rad(22),frq_tol=0.35):
    p1,d1=f1['pts'],f1['desc']; p2,d2=f2['pts'],f2['desc']
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0,0,0,0
    qi=np.array([x.queryIdx for x in g]); ti=np.array([x.trainIdx for x in g])
    src=p1[qi].reshape(-1,1,2); dst=p2[ti].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=2000,confidence=0.99)
    if M is None or mask is None: return 0,0,0,0
    a,b=M[0,0],M[1,0]; sc=np.sqrt(a*a+b*b); rot=np.arctan2(b,a); rotd=abs(np.degrees(rot)); rotd=min(rotd,360-rotd)
    if sc<0.85 or sc>1.18 or rotd>30: return 0,0,0,0
    inl=mask.ravel().astype(bool); base=int(inl.sum())
    if base==0: return 0,0,0,0
    # Level2 方向一致: probe方向+全局旋转 ≈ template方向 (mod pi)
    o1=f1['ori'][qi[inl]]+rot; o2=f2['ori'][ti[inl]]
    ok_ori=angdiff_pi(o1,o2)<ori_tol
    # Level3 频率一致: 频率比在容差内(尺度校正)
    fr1=f1['frq'][qi[inl]]*sc; fr2=f2['frq'][ti[inl]]
    ok_frq=(np.abs(fr1-fr2)/(fr2+1e-3))<frq_tol
    fp_ori=int(ok_ori.sum()); fp_both=int((ok_ori&ok_frq).sum())
    # 重叠归一化因子
    h,w=f2['m'].shape; w1=cv2.warpAffine(f1['m'],M,(w,h),flags=cv2.INTER_NEAREST); ov=int(np.count_nonzero((w1>0)&(f2['m']>0)))
    nf=float(np.sqrt(HO/ov)) if ov>=500 else 1.0
    return base*nf, fp_ori*nf, fp_both*nf, base

# ---- 数据: DATA=btp(不贴屏) 或 select(贴屏) ----
DATA=os.environ.get('DATA','btp')
fingers={}
if DATA=='select':
    SEL='f:/1111/指纹/select_data/select'
    for fg in sorted(os.listdir(SEL)):
        ps=sorted(glob.glob(os.path.join(SEL,fg,'*.bmp')))
        if os.path.isdir(os.path.join(SEL,fg)) and len(ps)>=15: fingers[fg]=ps
else:
    BTP='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
    for fg in sorted(os.listdir(BTP)):
        if 'xzc' in fg: continue
        for rgd in ['Rgd1237','Rgd1239','Rgd1241']:
            ps=sorted(glob.glob(os.path.join(BTP,fg,rgd,'*.bmp')))
            if len(ps)>=50: fingers[fg]=ps; break
ids=list(fingers.keys())
RES=open('f:/1111/指纹/_fpfusion.txt','w',encoding='utf-8')
def out(*a):
    s=' '.join(str(x) for x in a); print(s); RES.write(s+'\n'); RES.flush()
out(f"btp 指纹脊线多级融合: {len(ids)}指 {ids}")
t0=time.time(); TM={}; PR={}
for fg in ids:
    ps=fingers[fg]; ne=int(len(ps)*0.7); pool=ps[:ne]; pr=ps[ne:]
    idx=np.linspace(0,len(pool)-1,min(20,len(pool))).astype(int)
    TM[fg]=[f for f in (feat(load(pool[i])) for i in idx) if f]
    PR[fg]=[f for f in (feat(load(p)) for p in pr[:10]) if f]
out(f"feat done t={time.time()-t0:.0f}s")
# 三种分数: 基线 / +方向 / +方向+频率
G={'基线(几何内点)':[[],[]],'融合(+方向场)':[[],[]],'融合(+方向+频率)':[[],[]]}
for pc in ids:
    for pf in PR[pc]:
        bb={'基线(几何内点)':0.0,'融合(+方向场)':0.0,'融合(+方向+频率)':0.0} if False else None
        for cc in ids:
            mb=mo=mf=0.0
            for t in TM[cc]:
                b,o,fb,_=match(pf,t)
                if b>mb: mb=b
                if o>mo: mo=o
                if fb>mf: mf=fb
            gi=0 if cc==pc else 1
            G['基线(几何内点)'][gi].append(mb); G['融合(+方向场)'][gi].append(mo); G['融合(+方向+频率)'][gi].append(mf)
def rep(name,gen,imp):
    gen,imp=np.array(gen),np.array(imp)
    allv=np.unique(np.concatenate([gen,imp])); far=np.array([(imp>=v).mean() for v in allv]); frr=np.array([(gen<v).mean() for v in allv])
    ei=np.nanargmin(np.abs(far-frr))
    out(f"  [{name}] gen均值{gen.mean():.1f} imp均值{imp.mean():.2f} imp_max{imp.max():.1f}  EER={(far[ei]+frr[ei])/2*100:.2f}%  FAR=0 FRR={(gen<imp.max()+1e-6).mean()*100:.1f}%")
    for tgt in [0.1,0.01]:
        ok=np.where(far<=tgt/100)[0]
        if len(ok): out(f"    FAR<={tgt}% -> FRR={(gen<allv[ok[0]]).mean()*100:.1f}%")
out(f"\n== btp 结果 (n_gen={len(G['基线(几何内点)'][0])}, n_imp={len(G['基线(几何内点)'][1])}) ==")
for k in G: rep(k,G[k][0],G[k][1])
out(f"t={time.time()-t0:.0f}s Done.")
