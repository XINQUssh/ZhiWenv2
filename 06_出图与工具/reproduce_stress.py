# -*- coding: utf-8 -*-
"""
复现 + 压测 内点匹配方法(SIFT+几何验证)的鲁棒性。
1) 复现: genuine/impostor内点分布, FAR/FRR-阈值曲线, FAR=0工作点+FRR (对标客户 FAR=0@FRR0.96%)
2) 压测A 阈值泛化: 标定组手指选FAR=0阈值 → 未见测试组手指测真实FAR (3次随机劈分)
3) 压测B 冒充尾巴: 内点max随冒充对数量增长 → 外推1/50000风险
4) 被拒genuine的per-finger归因
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

sift=cv2.SIFT_create()
bf=cv2.BFMatcher(cv2.NORM_L2)
def sift_feat(img):
    e=clahe(img); u=up2(e); m=gmask(u); e2=u.copy(); e2[m==0]=0
    kp,des=sift.detectAndCompute(e2,(m>0).astype(np.uint8)*255)
    if des is None or len(kp)<4: return (np.zeros((0,2),np.float32),None)
    return (np.float32([k.pt for k in kp]), des)
def inliers(f1,f2,ratio=0.75,thr=8):
    p1,d1=f1; p2,d2=f2
    if d1 is None or d2 is None or len(p1)<4 or len(p2)<4: return 0
    knn=bf.knnMatch(d1,d2,k=2)
    g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0
    src=p1[[m.queryIdx for m in g]].reshape(-1,1,2); dst=p2[[m.trainIdx for m in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=thr,maxIters=2000,confidence=0.99)
    return int(mask.sum()) if mask is not None else 0

b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
random.seed(42)
NT=12  # 模板数(抬高genuine下限)
print("SIFT extract (templates+all test probes)..."); sys.stdout.flush()
t0=time.time()
tmpl={}; test={}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if not os.path.exists(rp): continue
        ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
        if len(ps)<50: continue
        k=f'{pre}_{fg}'
        tmpl[k]=[sift_feat(load_img(ps[i])) for i in random.sample(range(70),NT)]
        test[k]=[sift_feat(load_img(p)) for p in ps[70:]]
ids=list(tmpl.keys())
print(f"  {len(ids)} classes, {sum(len(v) for v in test.values())} probes, t={time.time()-t0:.0f}s"); sys.stdout.flush()

# ---- 打分: 每probe对每class取max内点(over templates) ----
print("scoring (probe x class)..."); sys.stdout.flush()
# rows: (probe_class, probe_idx, class, score)
G=[]  # (probe_class, score)
I=[]  # (probe_class, claimed_class, score)
for pc in ids:
    for pe in test[pc]:
        for cc in ids:
            sc=max(inliers(pe,t) for t in tmpl[cc])
            if cc==pc: G.append((pc,sc))
            else: I.append((pc,cc,sc))
    print(f"  done {pc}, t={time.time()-t0:.0f}s"); sys.stdout.flush()
gen=np.array([s for _,s in G]); imp=np.array([s for _,_,s in I])
print(f"\nGENUINE n={len(gen)} mean={gen.mean():.1f} min={gen.min()} | IMPOSTOR n={len(imp)} mean={imp.mean():.1f} max={imp.max()}")

def far_frr(thr): return (imp>=thr).mean(), (gen<thr).mean()
# EER + FAR=0工作点
ths=np.arange(0,int(max(gen.max(),imp.max()))+2)
fars=np.array([(imp>=t).mean() for t in ths]); frrs=np.array([(gen<t).mean() for t in ths])
ei=np.nanargmin(np.abs(fars-frrs)); eer=(fars[ei]+frrs[ei])/2
thr0=ths[np.argmax(fars==0)]  # 第一个FAR=0阈值
print(f"EER={eer*100:.3f}% @thr={ths[ei]}")
print(f"*** FAR=0 工作点: thr={thr0}, FRR={(gen<thr0).mean()*100:.3f}%, impostor_max={imp.max()} (margin={thr0-imp.max()}) ***")
for tgt in [0.002,0.01]:  # FAR目标(%)
    ok=np.where(fars<=tgt/100)[0]
    if len(ok): t=ths[ok[0]]; print(f"  FAR<={tgt}%: thr={t}, FRR={(gen<t).mean()*100:.3f}%")

# ---- 压测A: 阈值跨手指泛化 ----
print(f"\n{'='*60}\n压测A: 阈值跨手指泛化 (标定组选FAR=0阈值 → 未见手指测FAR)\n{'='*60}")
Iarr=np.array([(idp,idc,s) for (idp,_,s),(idp_,idc,_) in zip(I,I)],dtype=object) if False else I
for trial in range(3):
    rng=random.Random(100+trial); cls=ids[:]; rng.shuffle(cls)
    val=set(cls[:len(cls)//2]); tst=set(cls[len(cls)//2:])
    imp_val=[s for pc,cc,s in I if pc in val and cc in val]
    imp_tst=[s for pc,cc,s in I if pc in tst and cc in tst]
    gen_tst=[s for pc,s in G if pc in tst]
    thr_v=int(max(imp_val))+1   # 标定组FAR=0阈值
    far_t=np.mean(np.array(imp_tst)>=thr_v); frr_t=np.mean(np.array(gen_tst)<thr_v)
    print(f"  trial{trial}: thr(val FAR=0)={thr_v} | val_imp_max={max(imp_val)} tst_imp_max={max(imp_tst)} "
          f"| 未见手指 FAR={far_t*100:.4f}% FRR={frr_t*100:.3f}%")

# ---- 压测B: 冒充尾巴随样本量 ----
print(f"\n{'='*60}\n压测B: 冒充内点尾巴随样本量 (max / 99.9% / 99.99%)\n{'='*60}")
rng=np.random.RandomState(0); impv=imp.copy()
for n in [1000,5000,10000,len(impv)]:
    n=min(n,len(impv)); sub=rng.choice(impv,n,replace=False)
    print(f"  n={n:6d}: max={sub.max()}  p99.9={np.percentile(sub,99.9):.1f}  p99.99={np.percentile(sub,99.99):.1f}")
print(f"  (genuine分位 p1={np.percentile(gen,1):.0f} p5={np.percentile(gen,5):.0f} median={np.percentile(gen,50):.0f})")

# ---- 被拒genuine归因 ----
print(f"\n{'='*60}\n被拒genuine (thr={thr0}) per-finger\n{'='*60}")
from collections import defaultdict
rej=defaultdict(int); tot=defaultdict(int)
for pc,s in G:
    tot[pc]+=1
    if s<thr0: rej[pc]+=1
for pc in sorted(ids,key=lambda c:-rej[c]/max(1,tot[c])):
    if rej[pc]>0: print(f"  {pc:14s}: rejected {rej[pc]}/{tot[pc]} (genuine inlier示例不足)")
print(f"\nTotal t={time.time()-t0:.0f}s\nDone.")
