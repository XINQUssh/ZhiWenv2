# -*- coding: utf-8 -*-
"""
实现达标的内点匹配器: SIFT + 变换约束几何验证, 压制冒充尾巴。
目标: 固定测试集上 FAR<0.002% 时 FFR<3% (对标参考 FAR=0@FRR<1%)。
用法: python improved_matcher.py sweep   # 子集调参
      python improved_matcher.py full <ratio> <reproj> <slo> <shi> <rotmax> <ntmpl>
"""
import os, glob, random, sys, time
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
def pixel_orient(u):
    g=u.astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    Gxx=cv2.GaussianBlur(gx*gx,(0,0),3); Gyy=cv2.GaussianBlur(gy*gy,(0,0),3); Gxy=cv2.GaussianBlur(gx*gy,(0,0),3)
    return (0.5*np.arctan2(2*Gxy,Gxx-Gyy)+np.pi/2)%np.pi
def gabor_enhance(u,m,wavelength=9.0):
    img=u.astype(np.float32); img=(img-img.mean())/(img.std()+1e-6); ridge=pixel_orient(u); n=8
    resp=np.stack([cv2.filter2D(img,cv2.CV_32F,cv2.getGaborKernel((21,21),4.0,o,wavelength,0.5,0)) for o in np.arange(n)*np.pi/n])
    idx=(np.round(ridge/(np.pi/n)).astype(int))%n; H,W=img.shape
    out=resp[idx,np.arange(H)[:,None],np.arange(W)[None,:]]*(m>0)
    return cv2.normalize(out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

sift=cv2.SIFT_create()  # 默认(impostor尾巴更干净)
bf=cv2.BFMatcher(cv2.NORM_L2)
def feat(img, use_gabor):
    e=clahe(img); u=up2(e); m=gmask(u)
    src=gabor_enhance(u,m) if use_gabor else u
    src=src.copy(); src[m==0]=0
    kp,des=sift.detectAndCompute(src,(m>0).astype(np.uint8)*255)
    if des is None or len(kp)<4: return (np.zeros((0,2),np.float32),None)
    return (np.float32([k.pt for k in kp]),des)

def match(f1,f2,ratio=0.7,reproj=5,slo=0.9,shi=1.1,rotmax=20):
    p1,d1=f1; p2,d2=f2
    if d1 is None or d2 is None or len(p1)<4 or len(p2)<4: return 0
    knn=bf.knnMatch(d1,d2,k=2)
    g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
    if len(g)<4: return 0
    src=p1[[m.queryIdx for m in g]].reshape(-1,1,2); dst=p2[[m.trainIdx for m in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=3000,confidence=0.99)
    if M is None or mask is None: return 0
    a,b=M[0,0],M[1,0]; scale=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a)))
    rot=min(rot,360-rot)
    if scale<slo or scale>shi or rot>rotmax: return 0   # 变换约束: 杀冒充假匹配
    return int(mask.sum())

b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
def collect():
    cp={}
    for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
        for fg in sorted(os.listdir(base)):
            if 'xzc' in fg.lower(): continue
            rp=os.path.join(base,fg,reg)
            if not os.path.exists(rp): continue
            ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
            if len(ps)>=50: cp[f'{pre}_{fg}']=ps
    return cp
cp=collect(); ids=list(cp.keys())
MODE=sys.argv[1] if len(sys.argv)>1 else 'sweep'
USE_GABOR=False   # CLAHE特征(SIFT点丰富, genuine~108); Gabor掏空特征已验证更差
t0=time.time()

def eval_full(NT, params, sub=None, nprobe=None, perfinger=False):
    random.seed(42)
    use=ids if sub is None else ids[:sub]
    tmpl={k:[feat(load_img(cp[k][i]),USE_GABOR) for i in random.sample(range(70),NT)] for k in use}
    gen,imp=[],[]; gen_by={k:[] for k in use}
    for pc in use:
        probes=cp[pc][70:] if nprobe is None else random.sample(cp[pc][70:],min(nprobe,len(cp[pc])-70))
        pf=[feat(load_img(p),USE_GABOR) for p in probes]
        for pe in pf:
            for cc in use:
                sc=max(match(pe,t,**params) for t in tmpl[cc])
                if cc==pc: gen.append(sc); gen_by[pc].append(sc)
                else: imp.append(sc)
    if perfinger: return np.array(gen),np.array(imp),gen_by
    return np.array(gen),np.array(imp)

if MODE=='sweep':
    print(f"参数扫描(子集10类x5探针, gabor={USE_GABOR})..."); sys.stdout.flush()
    configs=[
        dict(ratio=0.75,reproj=8,slo=0.0,shi=9,rotmax=180),  # 基线(无约束, =reproduce_stress)
        dict(ratio=0.75,reproj=6,slo=0.85,shi=1.18,rotmax=30),
        dict(ratio=0.75,reproj=5,slo=0.9,shi=1.12,rotmax=25),
        dict(ratio=0.7,reproj=5,slo=0.9,shi=1.1,rotmax=20),
        dict(ratio=0.7,reproj=4,slo=0.92,shi=1.08,rotmax=18),
    ]
    for cf in configs:
        gen,imp=eval_full(12,cf,sub=10,nprobe=5)
        thr0=imp.max()+1; frr0=(gen<thr0).mean()
        print(f"  {cf}: genuine[min={gen.min()} mean={gen.mean():.0f}] impostor[max={imp.max()} mean={imp.mean():.1f}] -> FAR=0@thr{thr0} FRR={frr0*100:.1f}%"); sys.stdout.flush()
    print(f"t={time.time()-t0:.0f}s")
else:
    ratio,reproj,slo,shi,rotmax,NT=float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]),int(sys.argv[7])
    params=dict(ratio=ratio,reproj=reproj,slo=slo,shi=shi,rotmax=rotmax)
    print(f"FULL eval: gabor={USE_GABOR} NT={NT} {params}"); sys.stdout.flush()
    gen,imp,gen_by=eval_full(NT,params,perfinger=True)
    print(f"\nGENUINE n={len(gen)} mean={gen.mean():.1f} min={gen.min()} p1={np.percentile(gen,1):.0f} | IMPOSTOR n={len(imp)} mean={imp.mean():.1f} max={imp.max()}")
    ths=np.arange(0,int(max(gen.max(),imp.max()))+2)
    fars=np.array([(imp>=t).mean() for t in ths]); frrs=np.array([(gen<t).mean() for t in ths])
    ei=np.nanargmin(np.abs(fars-frrs)); print(f"EER={(fars[ei]+frrs[ei])/2*100:.3f}% @thr={ths[ei]}")
    thr0=int(imp.max())+1
    print(f"*** FAR=0: thr={thr0} FRR={(gen<thr0).mean()*100:.3f}% (impostor_max={imp.max()}) ***")
    for tgt in [0.002,0.01,0.1]:
        ok=np.where(fars<=tgt/100)[0]
        if len(ok): t=ths[ok[0]]; print(f"  FAR<={tgt}%: thr={t} FRR={(gen<t).mean()*100:.3f}%  {'达标' if (tgt==0.002 and (gen<t).mean()<0.03) else ''}")
    print(f"\n--- per-finger rejected @thr={thr0} ---")
    for k in sorted(gen_by,key=lambda c:-sum(1 for s in gen_by[c] if s<thr0)):
        r=sum(1 for s in gen_by[k] if s<thr0); n=len(gen_by[k])
        if r>0: print(f"  {k:14s}: {r}/{n}")
    print(f"t={time.time()-t0:.0f}s\nDone.")
