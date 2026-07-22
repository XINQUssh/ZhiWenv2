# -*- coding: utf-8 -*-
"""
可行性: SIFT关键点 + 几何验证(RANSAC估变换=自带位姿对齐) 能否救跨场景?
直接验证客户"SIFT达到FAR≈0"的说法, 并测试pose-alignment范式。
分数 = RANSAC内点数(几何一致匹配数)。跨场景(主) + 单场景(对照)。
对比全局基线: 单场景~1.1%, 跨场景~5.7%。
"""
import os, glob, random, sys, time
import numpy as np, cv2
from sklearn.metrics import roc_curve

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
def enhance_mask(img):
    enh=clahe(img); u=up2(enh); m=gmask(u); e=u.copy(); e[m==0]=0
    return e, m

sift=cv2.SIFT_create()
bf=cv2.BFMatcher(cv2.NORM_L2)
def sift_feat(img):
    e,m=enhance_mask(img)
    kp,des=sift.detectAndCompute(e, (m>0).astype(np.uint8)*255)
    return kp,des
def match_score(f1,f2,ratio=0.75):
    kp1,des1=f1; kp2,des2=f2
    if des1 is None or des2 is None or len(kp1)<4 or len(kp2)<4: return 0
    knn=bf.knnMatch(des1,des2,k=2)
    good=[a for a,b in (p for p in knn if len(p)==2) if a.distance<ratio*b.distance]
    if len(good)<4: return len(good)*0  # 太少, 0内点
    src=np.float32([kp1[g.queryIdx].pt for g in good]).reshape(-1,1,2)
    dst=np.float32([kp2[g.trainIdx].pt for g in good]).reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=8,maxIters=2000,confidence=0.99)
    if mask is None: return 0
    return int(mask.sum())   # 几何一致内点数

def eer_far(g,i):
    g,i=np.array(g,float),np.array(i,float)
    y=np.concatenate([np.ones(len(g)),np.zeros(len(i))]); s=np.concatenate([g,i])
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; k=np.nanargmin(np.abs(fnr-fpr)); e=(fpr[k]+fnr[k])/2
    out={'eer':e}
    for tf in [0.01,0.03,0.05]:
        idx=np.argmin(np.abs(fnr-tf)); out[f'far@ffr{int(tf*100)}']=fpr[idx]
    return out

b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
random.seed(42)
print("SIFT extract old templates+test..."); sys.stdout.flush()
NT=6; tmpl={}; test={}
t0=time.time()
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if not os.path.exists(rp): continue
        ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
        if len(ps)<50: continue
        k=f'{pre}_{fg}'
        tmpl[k]=[sift_feat(load_img(ps[i])) for i in random.sample(range(70),NT)]
        test[k]=[sift_feat(load_img(p)) for p in random.sample(ps[70:],min(6,len(ps)-70))]  # 单场景对照取6
ids=list(tmpl.keys())
nkp=np.mean([len(kp) for k in ids for kp,_ in tmpl[k]])
print(f"  {len(ids)} classes, avg keypoints/frame={nkp:.0f}, t={time.time()-t0:.0f}s"); sys.stdout.flush()

MERGE={'dy_R0':'wtp_dy_R0','dy_R1':'wtp_dy_R1','dy_R2':'wtp_dy_R2','lwh_R0':'wtp_lwh_R0','lwh_R1':'wtp_lwh_R1',
       'lwh_R2':'wtp_lwh_R2','zyh_R0':'btp_zyh_R0','zyh_R1':'btp_zyh_R1','zyh_R2':'btp_zyh_R2'}
nb='f:/1111/指纹/ysjz_raw/ysjz'; cross={}
for folder,gt in MERGE.items():
    cross.setdefault(gt,[])
    for p in sorted(glob.glob(os.path.join(nb,folder,'*.bmp')))[70:][:15]:  # 每指15探针
        cross[gt].append(sift_feat(load_img(p)))
print(f"  cross extracted, t={time.time()-t0:.0f}s"); sys.stdout.flush()

def evaluate(name, probes, per_finger=False):
    g,i=[],[]; pf={}
    for gt,plist in probes.items():
        pfs=[]
        for pe in plist:
            for cid in ids:
                sc=max(match_score(pe,t) for t in tmpl[cid])
                if cid==gt: g.append(sc); pfs.append(sc)
                else: i.append(sc)
        if pfs: pf[gt]=(np.mean(pfs),np.min(pfs))
    r=eer_far(g,i)
    print(f"\n=== {name} (SIFT inliers) ===")
    print(f"  genuine: mean={np.mean(g):.1f} min={np.min(g):.0f}  impostor: mean={np.mean(i):.1f} max={np.max(i):.0f}")
    print(f"  EER={r['eer']*100:.3f}%  FAR@FFR1%={r['far@ffr1']*100:.3f}%  FAR@FFR3%={r['far@ffr3']*100:.3f}%  FAR@FFR5%={r['far@ffr5']*100:.3f}%")
    if per_finger:
        print("  per-finger genuine inliers (mean/min):")
        for gt,(mn,mi) in pf.items(): print(f"    {gt:14s}: {mn:.1f} / {mi:.0f}")
    sys.stdout.flush()

evaluate("SINGLE-SESSION (old, baseline global~1.1%)", test)
evaluate("CROSS-SESSION (new probe x old tmpl, baseline global~5.7%)", cross, per_finger=True)
print(f"\nTotal t={time.time()-t0:.0f}s\nDone.")
