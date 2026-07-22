# -*- coding: utf-8 -*-
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
fig,(axL,axR)=plt.subplots(1,2,figsize=(13,4.8))
# 左: 模板数 sweep
K=[3,5,10,20]; eer=[33.2,26.7,22.8,17.2]; f0=[81.7,70.6,68.3,58.0]; f01=[76.0,64.5,57.3,50.8]
axL.plot(K,eer,'o-',label='EER %',lw=2); axL.plot(K,f0,'s-',label='FAR=0 FRR %',lw=2); axL.plot(K,f01,'^-',label='FAR=0.1% FRR %',lw=2)
axL.set_xlabel('每指模板数 K'); axL.set_ylabel('%'); axL.set_xticks(K)
axL.set_title('模板数越多越好 (贴屏低重叠→需覆盖)',fontsize=12); axL.legend()
for x,y in zip(K,eer): axL.text(x,y-3,f'{y}',ha='center',fontsize=8)
axL.annotate('模板砍到3-5张\n真匹配崩→FRR暴涨',xy=(3,81.7),xytext=(6,88),fontsize=9,color='#b00',
             arrowprops=dict(arrowstyle='->',color='#b00'))
# 右: CPU vs GPU 耗时
labels=['单模板注册','5指解锁\n(平均)','5指解锁\n(最快)']; budget=[200,100,50]; cpu=[310,346,270]; gpu=[21,32,23]
x=np.arange(len(labels)); w=0.26
axR.bar(x-w,budget,w,label='预算上限',color='#5b8',alpha=0.85)
axR.bar(x,cpu,w,label='CPU实测',color='#e67')
axR.bar(x+w,gpu,w,label='GPU实测',color='#69c')
axR.set_yscale('log'); axR.set_xticks(x); axR.set_xticklabels(labels,fontsize=9); axR.set_ylabel('ms'); axR.legend()
axR.set_title('耗时: CPU超预算 / GPU达标 (大头=提特征)',fontsize=12)
for i in range(len(labels)):
    axR.text(i-w,budget[i]*1.05,f'{budget[i]}',ha='center',fontsize=7)
    axR.text(i,cpu[i]*1.05,f'{cpu[i]}',ha='center',fontsize=8,color='#900',weight='bold')
    axR.text(i+w,gpu[i]*1.05,f'{gpu[i]}',ha='center',fontsize=8,color='#036',weight='bold')
plt.tight_layout()
open('f:/1111/指纹/_tmp_ct.png','wb').write(b''); plt.savefig('f:/1111/指纹/_tmp_ct.png',dpi=120,bbox_inches='tight')
open('f:/1111/指纹/CPU耗时与模板数.png','wb').write(open('f:/1111/指纹/_tmp_ct.png','rb').read()); os.remove('f:/1111/指纹/_tmp_ct.png')
print('saved CPU耗时与模板数.png')
