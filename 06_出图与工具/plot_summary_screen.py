# -*- coding: utf-8 -*-
"""贴屏汇总图: 左=各方法精度EER, 右=部署指标优化前后vs预算。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
fig,(axL,axR)=plt.subplots(1,2,figsize=(14,5.2))
# 左: 精度
methods=['纹理法\n(交付)','纹理法\n贴屏微调','V21\nglobal','V21\nlocal','V21\nfused','V25a fused\n(泄漏上限)']
eer=[20.5,18.0,16.0,12.4,13.5,9.3]
bars=axL.bar(range(len(methods)),eer,color=['#e67','#e88','#69c','#5ac','#7bd','#fa3'])
bars[-1].set_hatch('//')
axL.axhline(3,color='g',ls='--',lw=2,label='目标 EER<3%')
axL.axhline(1.19,color='#080',ls=':',lw=1.5,label='老数据单场景 1.19%(参考)')
axL.set_xticks(range(len(methods))); axL.set_xticklabels(methods,fontsize=9)
axL.set_ylabel('EER %'); axL.set_title('贴屏数据 各方法精度(越低越好) — 均远超目标',fontsize=12); axL.legend(fontsize=9)
for i,v in enumerate(eer): axL.text(i,v+0.4,f'{v}%',ha='center',fontsize=9,weight='bold')
axL.text(0.5,0.93,'两大类方法(纹理/CNN)都卡在12-20%',transform=axL.transAxes,ha='center',color='#b00',fontsize=10,weight='bold')
# 右: 部署空间(仅存储, 时间待定稿模型后再测)
labels=['模板存储\n(MB,100模板)','模型大小\n(MB)']
budget=[10,30]; before=[51.0,1.23]; after=[6.95,1.23]
x=np.arange(len(labels)); w=0.26
axR.bar(x-w,budget,w,label='预算上限',color='#5b8',alpha=0.85)
axR.bar(x,before,w,label='优化前',color='#e67')
axR.bar(x+w,after,w,label='优化后',color='#69c')
axR.set_yscale('log'); axR.set_xticks(x); axR.set_xticklabels(labels,fontsize=9); axR.legend(fontsize=9)
axR.set_title('部署空间: 优化前→后 vs 预算 — 存储达标',fontsize=12)
for i in range(len(labels)):
    axR.text(i+w,after[i]*1.1,f'{after[i]:g}',ha='center',fontsize=8,weight='bold',color='#036')
    axR.text(i,before[i]*1.1,f'{before[i]:g}',ha='center',fontsize=7,color='#900')
plt.tight_layout()
open('f:/1111/指纹/_tmp_sum.png','wb').write(b''); plt.savefig('f:/1111/指纹/_tmp_sum.png',dpi=120,bbox_inches='tight')
open('f:/1111/指纹/贴屏精度与部署_汇总.png','wb').write(open('f:/1111/指纹/_tmp_sum.png','rb').read()); os.remove('f:/1111/指纹/_tmp_sum.png')
print('saved 贴屏精度与部署_汇总.png')
