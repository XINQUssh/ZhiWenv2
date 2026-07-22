# -*- coding: utf-8 -*-
"""贴屏精度汇总图(纯精度): 左=各方法EER, 右=模板数vs精度。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
fig,(axL,axR)=plt.subplots(1,2,figsize=(13,5))
# 左: 各方法精度
methods=['纹理法\n(交付)','纹理法\n贴屏微调','V21\nglobal','V21\nlocal','V21\nfused','V25a fused\n(泄漏上限)']
eer=[20.5,18.0,16.0,12.4,13.5,9.3]
bars=axL.bar(range(len(methods)),eer,color=['#e67','#e88','#69c','#5ac','#7bd','#fa3']); bars[-1].set_hatch('//')
axL.axhline(3,color='g',ls='--',lw=2,label='目标 EER<3%')
axL.axhline(1.19,color='#080',ls=':',lw=1.5,label='老数据单场景1.19%(参考)')
axL.set_xticks(range(len(methods))); axL.set_xticklabels(methods,fontsize=9); axL.set_ylabel('EER %')
axL.set_title('贴屏 各方法精度 — 均远超目标',fontsize=12); axL.legend(fontsize=9)
for i,v in enumerate(eer): axL.text(i,v+0.4,f'{v}%',ha='center',fontsize=9,weight='bold')
axL.text(0.5,0.93,'纹理/CNN两大类都卡12-20%',transform=axL.transAxes,ha='center',color='#b00',fontsize=10,weight='bold')
# 右: 模板数 vs 精度
K=[3,5,10,20]; f0=[81.7,70.6,68.3,58.0]; eerk=[33.2,26.7,22.8,17.2]; f01=[76.0,64.5,57.3,50.8]
axR.plot(K,eerk,'o-',label='EER %',lw=2); axR.plot(K,f0,'s-',label='FAR=0 FRR %',lw=2); axR.plot(K,f01,'^-',label='FAR=0.1% FRR %',lw=2)
axR.set_xlabel('每指模板数 K'); axR.set_ylabel('%'); axR.set_xticks(K); axR.legend()
axR.set_title('模板数越多越好(低重叠→需覆盖)',fontsize=12)
axR.annotate('模板砍到3-5张\n真匹配崩',xy=(3,81.7),xytext=(6,86),fontsize=9,color='#b00',arrowprops=dict(arrowstyle='->',color='#b00'))
plt.tight_layout()
open('f:/1111/指纹/_tmp_acc.png','wb').write(b''); plt.savefig('f:/1111/指纹/_tmp_acc.png',dpi=120,bbox_inches='tight')
open('f:/1111/指纹/贴屏精度_汇总.png','wb').write(open('f:/1111/指纹/_tmp_acc.png','rb').read()); os.remove('f:/1111/指纹/_tmp_acc.png')
print('saved 贴屏精度_汇总.png')
