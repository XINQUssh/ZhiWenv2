# -*- coding: utf-8 -*-
"""贴屏精度对比图: 读取各eval保存的 gen/imp npy, 画分数分布 + EER/FAR=0 对比柱状。
用法: python plot_screen_compare.py   (自动读 _genX_TAG.npy)"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
B='f:/1111/指纹'
def metrics(gen,imp):
    allv=np.unique(np.concatenate([gen,imp])); far=np.array([(imp>=v).mean() for v in allv]); frr=np.array([(gen<v).mean() for v in allv])
    ei=np.nanargmin(np.abs(far-frr)); eer=(far[ei]+frr[ei])/2*100
    frr0=(gen<imp.max()+1e-6).mean()*100
    return eer,frr0
# 收集所有可用配置
cfgs=[]  # (label, genfile, impfile)
for tag in ['v3s8','screen']:
    for sc in ['R','O']:
        gf=f'{B}/_gen{sc}_{tag}.npy'; ig=f'{B}/_imp{sc}_{tag}.npy'
        if os.path.exists(gf) and os.path.exists(ig):
            cfgs.append((f'{tag}/{"raw" if sc=="R" else "ovn"}',gf,ig))
if not cfgs:
    print("no npy yet"); raise SystemExit
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
labels=[]; eers=[]; frrs=[]
for lab,gf,ig in cfgs:
    gen=np.load(gf); imp=np.load(ig); eer,frr0=metrics(gen,imp)
    labels.append(lab); eers.append(eer); frrs.append(frr0)
    axes[0].hist(gen,bins=40,alpha=0.4,density=True,label=f'{lab} genuine')
ax=axes[1]; x=np.arange(len(labels)); w=0.38
ax.bar(x-w/2,eers,w,label='EER %',color='#e67'); ax.bar(x+w/2,frrs,w,label='FAR=0 FRR %',color='#69c')
ax.axhline(3,color='g',ls='--',label='目标FRR<3%')
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,fontsize=8); ax.legend(); ax.set_title('贴屏精度: 各配置 EER / FAR=0 FRR')
for i,(e,f) in enumerate(zip(eers,frrs)):
    ax.text(i-w/2,e+0.5,f'{e:.1f}',ha='center',fontsize=7); ax.text(i+w/2,f+0.5,f'{f:.0f}',ha='center',fontsize=7)
axes[0].set_title('genuine 分数分布(各配置)'); axes[0].legend(fontsize=7)
plt.tight_layout()
open(f'{B}/_tmp_cmp.png','wb').write(b''); plt.savefig(f'{B}/_tmp_cmp.png',dpi=120,bbox_inches='tight')
data=open(f'{B}/_tmp_cmp.png','rb').read(); open(f'{B}/贴屏精度对比.png','wb').write(data); os.remove(f'{B}/_tmp_cmp.png')
print('saved 贴屏精度对比.png'); print('configs:',list(zip(labels,[f"{e:.1f}" for e in eers],[f"{f:.0f}" for f in frrs])))
