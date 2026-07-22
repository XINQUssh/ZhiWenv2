# -*- coding: utf-8 -*-
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
gen=np.load('f:/1111/指纹/tex_gen.npy'); imp=np.load('f:/1111/指纹/tex_imp.npy')
ng,ni=len(gen),len(imp); maxs=int(max(gen.max(),imp.max()))
ths=np.arange(0,maxs+2)
far=np.array([(imp>=t).mean() for t in ths]); frr=np.array([(gen<t).mean() for t in ths])
thr0=int(imp.max())+1; frr0=(gen<thr0).mean()
fig=plt.figure(figsize=(8.5,12))
ax1=fig.add_axes([0.10,0.74,0.84,0.20])
ax1.plot(ths,far,label='FAR',lw=2); ax1.plot(ths,frr,label='FRR',lw=2,color='orange')
ax1.axvline(thr0,ls='--',color='gray'); ax1.set_xlabel('unlock threshold (inliers)'); ax1.legend(loc='center right'); ax1.set_ylim(-0.02,1.02)
ax1.set_title('Texture-descriptor matcher (self-trained, NO SIFT) + geometric verification',fontsize=10.5)
ax2=fig.add_axes([0.10,0.46,0.84,0.20])
bins=np.arange(0,maxs+3)
ax2.hist(imp,bins=bins,density=True,alpha=0.75,label='impostor')
ax2.hist(gen,bins=bins,density=True,alpha=0.75,label='genuine',color='orange')
ax2.axvline(thr0,ls='--',color='gray'); ax2.set_xlabel('unique inlier score'); ax2.legend(loc='center right')
ax3=fig.add_axes([0.04,0.02,0.92,0.30]); ax3.axis('off')
cols=['thr','FAR','FRR','TAR','gen_acc','gen_rej','imp_acc','imp_rej']; rows=[]
for t in range(max(1,thr0-7),thr0+5):
    ga=int((gen>=t).sum()); gr=ng-ga; ia=int((imp>=t).sum()); ir=ni-ia
    rows.append([t,f'{ia/ni:.2e}',f'{gr/ng:.4f}',f'{ga/ng:.4f}',ga,gr,ia,ir])
tb=ax3.table(cellText=rows,colLabels=cols,loc='center',cellLoc='center'); tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1,1.35)
ax3.text(0.5,1.02,f'Operating points   (genuine_total={ng}, impostor_total={ni})',ha='center',fontsize=10,transform=ax3.transAxes)
fig.text(0.5,0.355,f'*** FAR=0 @ thr={thr0}: FRR={frr0*100:.2f}% (all 27)   |   ~2.4% on 25 fingers (excl. 2 corrupted-frame fingers) ***',
         ha='center',fontsize=10,weight='bold',color='#b00')
plt.savefig('f:/1111/指纹/匹配效果_纹理方法.png',dpi=115,bbox_inches='tight')
print(f'saved 匹配效果_纹理方法.png FAR=0@thr{thr0} FRR={frr0*100:.2f}% imp_max={imp.max()} gen_mean={gen.mean():.0f}')
