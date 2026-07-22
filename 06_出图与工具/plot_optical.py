# -*- coding: utf-8 -*-
"""光学 center vs random 对照: 重叠决定一切。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
fig,(axL,axR)=plt.subplots(1,2,figsize=(13,5))
# 左: 性能指标
labels=['EER','FAR=0.1%\nFRR','FAR=0\nFRR']; center=[13.7,30.5,41.5]; random=[39.6,86.4,91.4]
x=np.arange(len(labels)); w=0.35
axL.bar(x-w/2,center,w,label='center裁剪(高重叠)',color='#4a90d9')
axL.bar(x+w/2,random,w,label='random裁剪(低重叠)',color='#e07b39')
axL.set_xticks(x); axL.set_xticklabels(labels); axL.set_ylabel('%'); axL.legend()
axL.set_title('光学DB1_A: 同手指同模型, 只改裁剪位置(=重叠)',fontsize=12)
for i,(c,r) in enumerate(zip(center,random)):
    axL.text(i-w/2,c+1,f'{c}',ha='center',fontsize=9,weight='bold'); axL.text(i+w/2,r+1,f'{r}',ha='center',fontsize=9,weight='bold')
# 右: genuine均值 vs 冲突max — 真匹配随重叠崩, 冲突不变
labels2=['genuine\n均值内点','impostor\nmax内点']; cen=[28.8,16]; ran=[4.5,17]
x2=np.arange(len(labels2))
axR.bar(x2-w/2,cen,w,label='center(高重叠)',color='#4a90d9')
axR.bar(x2+w/2,ran,w,label='random(低重叠)',color='#e07b39')
axR.set_xticks(x2); axR.set_xticklabels(labels2); axR.set_ylabel('内点数'); axR.legend()
axR.set_title('真匹配随重叠塌陷(28.8→4.5), 冲突几乎不变',fontsize=12)
for i,(c,r) in enumerate(zip(cen,ran)):
    axR.text(i-w/2,c+0.5,f'{c}',ha='center',fontsize=9,weight='bold'); axR.text(i+w/2,r+0.5,f'{r}',ha='center',fontsize=9,weight='bold')
axR.annotate('重叠低→真匹配崩',xy=(0+w/2,4.5),xytext=(0.5,20),fontsize=10,color='#b00',arrowprops=dict(arrowstyle='->',color='#b00'))
plt.suptitle('重叠(capture overlap)是决定性因素 — 光学受控实验',fontsize=13,weight='bold')
plt.tight_layout()
open('f:/1111/指纹/_tmp_op.png','wb').write(b''); plt.savefig('f:/1111/指纹/_tmp_op.png',dpi=120,bbox_inches='tight')
open('f:/1111/指纹/光学_重叠决定一切.png','wb').write(open('f:/1111/指纹/_tmp_op.png','rb').read()); os.remove('f:/1111/指纹/_tmp_op.png')
print('saved 光学_重叠决定一切.png')
