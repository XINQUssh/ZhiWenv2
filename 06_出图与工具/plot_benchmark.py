# -*- coding: utf-8 -*-
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
# 指标: (名称, 预算, CPU实测, GPU实测, 单位)
M=[('单模板注册','<200','149','16','ms'),
   ('5指模板存储','<10','51.0','51.0','MB'),
   ('5指解锁(平均)','<100','265','136','ms'),
   ('AI模型','<30','1.2','1.2','MB')]
labels=[m[0] for m in M]; budget=[float(m[1].strip('<')) for m in M]; cpu=[float(m[2]) for m in M]; gpu=[float(m[3]) for m in M]
x=np.arange(len(M)); w=0.25
fig,ax=plt.subplots(figsize=(9,4.6))
b1=ax.bar(x-w,budget,w,label='预算上限',color='#5b8',alpha=0.9)
b2=ax.bar(x,cpu,w,label='CPU实测',color='#e67')
b3=ax.bar(x+w,gpu,w,label='GPU实测',color='#69c')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([f'{m[0]}\n({m[4]})' for m in M]); ax.legend()
ax.set_title('部署指标: 实测 vs 预算 (对数轴, 贴屏数据, 当前模型texdesc_v3 stride6)',fontsize=11)
for bars in (b1,b2,b3):
    for r in bars:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()*1.05, f'{r.get_height():g}', ha='center',fontsize=8)
# 标注达标/超标
for i,(lab,bud,c,g,u) in enumerate(M):
    bv=float(bud.strip('<')); ok = float(c)<=bv
    ax.text(i, max(float(c),float(g),bv)*1.6, '达标' if ok else f'超{float(c)/bv:.1f}×', ha='center', color=('#070' if ok else '#b00'), fontsize=9, weight='bold')
plt.tight_layout(); plt.savefig('f:/1111/指纹/部署指标_实测vs预算.png',dpi=120,bbox_inches='tight')
print('saved 部署指标_实测vs预算.png')
