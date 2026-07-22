# -*- coding: utf-8 -*-
"""出图(同客户款): FAR/FRR-阈值曲线 + genuine/impostor内点分布直方图 + 工作点表格。
用变换约束内点匹配器, 全段均匀40模板(FRR5.2%最优配置)。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import glob, random, sys, time
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
sift=cv2.SIFT_create(); bf=cv2.BFMatcher(cv2.NORM_L2)
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
print("extract..."); sys.stdout.flush()
tmpl={k:[feat(load_img(cp[k][i])) for i in idx] for k in ids}
test={k:[feat(load_img(p)) for p in cp[k][70:]] for k in ids}
print(f"done t={time.time()-t0:.0f}s"); sys.stdout.flush()
gen,imp=[],[]
for pc in ids:
    for pe in test[pc]:
        for cc in ids:
            sc=max(match(pe,t) for t in tmpl[cc])
            (gen if cc==pc else imp).append(sc)
gen,imp=np.array(gen),np.array(imp)
np.save('f:/1111/指纹/scores_gen.npy',gen); np.save('f:/1111/指纹/scores_imp.npy',imp)
ng,ni=len(gen),len(imp)
maxs=int(max(gen.max(),imp.max()))
ths=np.arange(0,maxs+2)
far=np.array([(imp>=t).mean() for t in ths]); frr=np.array([(gen<t).mean() for t in ths])
thr0=int(imp.max())+1; frr0=(gen<thr0).mean()

# ---- 三面板图 ----
fig=plt.figure(figsize=(8,11))
ax1=fig.add_axes([0.1,0.70,0.78,0.25])
ax1.plot(ths,far,label='FAR',lw=2); ax1.plot(ths,frr,label='FRR',lw=2,color='orange')
ax1.axvline(thr0,ls='--',color='gray'); ax1.set_xlabel('unlock threshold (inliers)'); ax1.legend(); ax1.set_ylim(-0.02,1.02)
ax1.set_title('Matching Performance (transform-constrained inlier matcher)')

ax2=fig.add_axes([0.1,0.40,0.78,0.23])
bins=np.arange(0,maxs+3)
ax2.hist(imp,bins=bins,density=True,alpha=0.7,label='impostor')
ax2.hist(gen,bins=bins,density=True,alpha=0.7,label='genuine',color='orange')
ax2.axvline(thr0,ls='--',color='gray'); ax2.set_xlabel('unique inlier score'); ax2.legend()

ax3=fig.add_axes([0.05,0.03,0.9,0.30]); ax3.axis('off')
rows=[]; cols=['thr','FAR','FRR','TAR','gen_acc','gen_rej','imp_acc','imp_rej']
for t in range(max(1,thr0-7),thr0+5):
    ga=int((gen>=t).sum()); gr=ng-ga; ia=int((imp>=t).sum()); ir=ni-ia
    rows.append([t,f'{ia/ni:.2e}',f'{gr/ng:.4f}',f'{ga/ng:.4f}',ga,gr,ia,ir])
tb=ax3.table(cellText=rows,colLabels=cols,loc='center',cellLoc='center')
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1,1.3)
ax3.set_title(f'Operating points  (genuine_total={ng}, impostor_total={ni})\n'
              f'*** FAR=0 @ thr={thr0}: FRR={frr0*100:.2f}%, TAR={(1-frr0)*100:.2f}% (all 27 fingers) ***',
              fontsize=10,pad=18)
plt.savefig('f:/wk_match_result.png',dpi=110,bbox_inches='tight')
print(f"saved f:/wk_match_result.png  FAR=0@thr{thr0} FRR={frr0*100:.2f}%  imp_max={imp.max()} gen_mean={gen.mean():.0f}")
print(f"t={time.time()-t0:.0f}s")
