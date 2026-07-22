# -*- coding: utf-8 -*-
"""
达标交付: 内点匹配器(SIFT+变换约束) + 注册端质量门。
质量门 = 每根手指模板帧的"自一致性"(组内互匹配内点中位数); 低于阈值→标记重采(FTE)。
报告: 全部手指 vs 质量门通过手指 的 FAR/FRR。目标 FAR<0.002%时FFR<3%。
"""
import os, glob, random, sys, time, itertools
import numpy as np, cv2

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

sift=cv2.SIFT_create()
bf=cv2.BFMatcher(cv2.NORM_L2)
def feat(img):
    e=clahe(img); u=up2(e); m=gmask(u); s=u.copy(); s[m==0]=0
    kp,des=sift.detectAndCompute(s,(m>0).astype(np.uint8)*255)
    if des is None or len(kp)<4: return (np.zeros((0,2),np.float32),None)
    return (np.float32([k.pt for k in kp]),des)
def match(f1,f2,ratio=0.75,reproj=6,slo=0.85,shi=1.18,rotmax=30):
    p1,d1=f1; p2,d2=f2
    if d1 is None or d2 is None or len(p1)<4 or len(p2)<4: return 0
    knn=bf.knnMatch(d1,d2,k=2)
    g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0
    src=p1[[m.queryIdx for m in g]].reshape(-1,1,2); dst=p2[[m.trainIdx for m in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=3000,confidence=0.99)
    if M is None or mask is None: return 0
    a,b=M[0,0],M[1,0]; scale=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a))); rot=min(rot,360-rot)
    if scale<slo or scale>shi or rot>rotmax: return 0
    return int(mask.sum())

b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
cp={}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if os.path.exists(rp):
            ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
            if len(ps)>=50: cp[f'{pre}_{fg}']=ps
ids=list(cp.keys()); random.seed(42); NT=20; t0=time.time()
print(f"extract feats (NT={NT})..."); sys.stdout.flush()
tmpl={k:[feat(load_img(cp[k][i])) for i in random.sample(range(70),NT)] for k in ids}
test={k:[feat(load_img(p)) for p in cp[k][70:]] for k in ids}
print(f"  done t={time.time()-t0:.0f}s"); sys.stdout.flush()

# ---- 注册质量门: 每根手指模板帧组内自一致性(中位内点) ----
qual={}
for k in ids:
    ss=[match(tmpl[k][a],tmpl[k][b]) for a,b in itertools.combinations(range(NT),2)]
    qual[k]=np.median(ss)
print(f"\n--- 注册质量(模板自一致性中位内点) ---")
for k in sorted(qual,key=lambda c:qual[c]):
    print(f"  {k:14s}: {qual[k]:.0f}")
QGATE=40  # 质量阈值(中位自匹配内点); 低于→标记重采
passed=[k for k in ids if qual[k]>=QGATE]; flagged=[k for k in ids if qual[k]<QGATE]
print(f"\n质量门QGATE={QGATE}: 通过{len(passed)}根, 标记重采{len(flagged)}根: {flagged} (FTE={len(flagged)/len(ids)*100:.1f}%)")

# ---- 打分 ----
print("scoring..."); sys.stdout.flush()
gen=[]; imp=[]  # (probe_finger, score)
for pc in ids:
    for pe in test[pc]:
        for cc in ids:
            sc=max(match(pe,t) for t in tmpl[cc])
            (gen if cc==pc else imp).append((pc,cc,sc))
print(f"  done t={time.time()-t0:.0f}s")

def report(name, allowset):
    g=np.array([s for pc,cc,s in gen if pc in allowset])
    i=np.array([s for pc,cc,s in imp if pc in allowset and cc in allowset])
    ths=np.arange(0,int(max(g.max(),i.max()))+2)
    fars=np.array([(i>=t).mean() for t in ths]);
    thr0=int(i.max())+1; frr0=(g<thr0).mean()
    print(f"\n=== {name} (n_finger={len(allowset)}) ===")
    print(f"  genuine n={len(g)} mean={g.mean():.0f} | impostor n={len(i)} max={i.max()}")
    print(f"  *** FAR=0: thr={thr0}, FRR={frr0*100:.3f}% {'✓达标' if frr0<0.03 else ''} ***")
    for tgt in [0.002,0.01]:
        ok=np.where(fars<=tgt/100)[0]
        if len(ok): t=ths[ok[0]]; f=(g<t).mean(); print(f"  FAR<={tgt}%: thr={t} FRR={f*100:.3f}% {'✓达标' if (tgt==0.002 and f<0.03) else ''}")

report("全部手指 (无质量门)", set(ids))
report("质量门通过手指", set(passed))
print(f"\nTotal t={time.time()-t0:.0f}s\nDone.")
