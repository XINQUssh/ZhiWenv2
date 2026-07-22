# -*- coding: utf-8 -*-
"""27组完整指标: EER / FAR=0 FRR / FAR<=0.002% FRR / FAR<=0.01% FRR。两级与纯纹理, 含/排除损坏手指。"""
import numpy as np
def metrics(gen,imp,name):
    gen=np.asarray(gen,float); imp=np.asarray(imp,float)
    lo,hi=min(gen.min(),imp.min()),max(gen.max(),imp.max())
    ths=np.linspace(lo,hi,4000)
    far=np.array([(imp>=t).mean() for t in ths]); frr=np.array([(gen<t).mean() for t in ths])
    i=np.nanargmin(np.abs(far-frr)); eer=(far[i]+frr[i])/2
    def frr_at(target):
        ok=np.where(far<=target)[0]
        return frr[ok[0]] if len(ok) else 1.0
    print(f"  {name}: EER={eer*100:.3f}%  FAR=0→FRR={frr_at(0)*100:.3f}%  "
          f"FAR<=0.002%→FRR={frr_at(0.00002)*100:.3f}%  FAR<=0.01%→FRR={frr_at(0.0001)*100:.3f}%")

print("=== 27组 两级(全局预筛Tg=0.2 + 局部几何) ===")
d=np.load('f:/1111/指纹/two_stage_scores.npz',allow_pickle=True)
Tg=0.2; G_l=np.where(d['G_g']>=Tg,d['G_l'],0); I_l=np.where(d['I_g']>=Tg,d['I_l'],0); gf=d['gfinger']
metrics(G_l,I_l,"全部27组")
bad={'btp_fys_R0','wtp_xyz_R1'}
keep=np.array([f not in bad for f in gf])
metrics(G_l[keep],I_l,"排除2根损坏手指(25组)")

print("\n=== 27组 纯密集纹理 + 几何 ===")
tg=np.load('f:/1111/指纹/tex_gen.npy'); ti=np.load('f:/1111/指纹/tex_imp.npy')
metrics(tg,ti,"全部27组")
