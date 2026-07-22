# -*- coding: utf-8 -*-
"""
步骤2(客户方法): SIFT关键点检测 + HardNet描述子(texdesc_kp) + 变换约束几何验证(=姿态对齐)。
用法: python texture_matcher_kp.py <ratio> <rotmax>
对比密集网格版(texture_matcher.py)。
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
    return norm,u,m
class Desc(nn.Module):
    def __init__(s,d=128):
        super().__init__()
        def cbr(i,o,st=1): return [nn.Conv2d(i,o,3,st,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
        s.net=nn.Sequential(*cbr(1,32),*cbr(32,32),*cbr(32,64,2),*cbr(64,64),*cbr(64,128,2),*cbr(128,128),nn.AdaptiveAvgPool2d(1)); s.fc=nn.Linear(128,d)
    def forward(s,x): z=s.net(x).flatten(1); return F.normalize(s.fc(z),dim=1)
dev=torch.device('cuda'); net=Desc().to(dev)
net.load_state_dict(torch.load('f:/1111/指纹/models_texdesc/texdesc_kp.pth',map_location=dev)); net.eval()
sift=cv2.SIFT_create(); PS=32; HALF=16
print("texdesc_kp + SIFT keypoints loaded"); sys.stdout.flush()
@torch.no_grad()
def feat(img):
    norm,u,m=prep(img); H,W=norm.shape
    us=u.copy(); us[m==0]=0; kp=sift.detect(us,(m>0).astype(np.uint8)*255)
    pts=[]; patches=[]
    for k in kp:
        x,y=int(round(k.pt[0])),int(round(k.pt[1]))
        if HALF<=x<W-HALF and HALF<=y<H-HALF:
            pts.append((k.pt[0],k.pt[1])); patches.append(norm[y-HALF:y+HALF,x-HALF:x+HALF])
    if len(patches)<4: return (np.zeros((0,2),np.float32),None)
    X=torch.tensor(np.stack(patches)[:,None],dtype=torch.float32).to(dev); des=[]
    for i in range(0,len(X),1024): des.append(net(X[i:i+1024]).cpu().numpy())
    return (np.float32(pts), np.concatenate(des).astype(np.float32))
bf=cv2.BFMatcher(cv2.NORM_L2)
RATIO=float(sys.argv[1]) if len(sys.argv)>1 else 0.85
ROTMAX=float(sys.argv[2]) if len(sys.argv)>2 else 30
def match(f1,f2,reproj=6,slo=0.85,shi=1.18):
    p1,d1=f1; p2,d2=f2
    if d1 is None or d2 is None or len(p1)<4 or len(p2)<4: return 0
    knn=bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<RATIO*b.distance]
    if len(g)<4: return 0
    src=p1[[m.queryIdx for m in g]].reshape(-1,1,2); dst=p2[[m.trainIdx for m in g]].reshape(-1,1,2)
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=3000,confidence=0.99)
    if M is None or mask is None: return 0
    a,b=M[0,0],M[1,0]; sc=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a))); rot=min(rot,360-rot)
    if sc<slo or sc>shi or rot>ROTMAX: return 0
    return int(mask.sum())
b1='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'; b2='f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
cp={}
for base,pre,reg in [(b1,'wtp','Rgd1245'),(b2,'btp','Rgd1237')]:
    for fg in sorted(os.listdir(base)):
        if 'xzc' in fg.lower(): continue
        rp=os.path.join(base,fg,reg)
        if os.path.exists(rp):
            ps=sorted(glob.glob(os.path.join(rp,'*.bmp')))
            if len(ps)>=50: cp[f'{pre}_{fg}']=ps
ids=list(cp.keys()); t0=time.time(); NT=40; idx=np.linspace(0,69,NT).astype(int)
print(f"extract (SIFT kp, ratio={RATIO}, rotmax={ROTMAX})..."); sys.stdout.flush()
tmpl={k:[feat(load_img(cp[k][i])) for i in idx] for k in ids}
test={k:[feat(load_img(p)) for p in cp[k][70:]] for k in ids}
npts=np.mean([len(t[0]) for k in ids for t in tmpl[k]])
print(f"  avg kp/frame={npts:.0f}, t={time.time()-t0:.0f}s"); sys.stdout.flush()
gen,imp=[],[]; gen_by={k:[] for k in ids}
for pc in ids:
    for pe in test[pc]:
        for cc in ids:
            sc=max(match(pe,t) for t in tmpl[cc])
            if cc==pc: gen.append(sc); gen_by[pc].append(sc)
            else: imp.append(sc)
    print(f"  scored {pc} t={time.time()-t0:.0f}s"); sys.stdout.flush()
gen,imp=np.array(gen),np.array(imp)
np.save('f:/1111/指纹/texkp_gen.npy',gen); np.save('f:/1111/指纹/texkp_imp.npy',imp)
print(f"\n[SIFT-kp + HardNet描述子 + 几何] GENUINE n={len(gen)} mean={gen.mean():.1f} min={gen.min()} | IMPOSTOR n={len(imp)} mean={imp.mean():.2f} max={imp.max()}")
thr0=int(imp.max())+1; frr0=(gen<thr0).mean()
print(f"*** FAR=0: thr={thr0} FRR={frr0*100:.3f}% {'✓达标' if frr0<0.03 else '✗'} ***")
print("--- per-finger rej ---")
for k in sorted(gen_by,key=lambda c:-sum(1 for s in gen_by[c] if s<thr0)):
    r=sum(1 for s in gen_by[k] if s<thr0)
    if r>0: print(f"  {k:14s}: {r}/{len(gen_by[k])} (mean={np.mean(gen_by[k]):.0f})")
# 排除2根损坏手指
bad={'btp_fys_R0','wtp_xyz_R1'}
g2=[s for pc in ids if pc not in bad for s in gen_by[pc]]; i2=[s for s in imp]  # impostor不变(近似)
frr25=np.mean(np.array(g2)<thr0)
print(f"\n排除2根损坏手指(25根): FRR≈{frr25*100:.2f}%")
print(f"t={time.time()-t0:.0f}s\nDone.")
