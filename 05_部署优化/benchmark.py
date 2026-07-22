# -*- coding: utf-8 -*-
"""
部署指标基准: 当前模型(texdesc_v3 + 密集匹配)在贴屏数据(select)上的 空间 & 时间。
预算: 5指模板(100模板)RAM<10MB; 含AI模型总内存<30MB; 单模板注册<200ms; 5指解锁 最快<50ms/平均<100ms。
在 CPU(部署相关) 和 GPU(参考) 各测一遍。
"""
import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import sys, glob, time, random
import numpy as np, cv2, torch
sys.path.insert(0,'f:/1111/指纹/deliver')
from predict import FingerprintMatcher, STRIDE

SEL='f:/1111/指纹/select_data/select'
MODELP='f:/1111/指纹/deliver/best.pth'
random.seed(0)

# 选5指(帧数>=25), 每指注册20模板, 探针用其余帧
fingers=[d for d in sorted(os.listdir(SEL)) if os.path.isdir(os.path.join(SEL,d))]
cand=[]
for fg in fingers:
    ps=sorted(glob.glob(os.path.join(SEL,fg,'*.bmp')))
    if len(ps)>=25: cand.append((fg,ps))
pick=cand[:5]
print(f"选用5指: {[p[0] for p in pick]}"); sys.stdout.flush()

def load(p):
    with open(p,'rb') as f: return cv2.imdecode(np.frombuffer(f.read(),np.uint8),cv2.IMREAD_GRAYSCALE)

model_bytes=os.path.getsize(MODELP)
nparams=sum(p.numel() for p in FingerprintMatcher.__init__.__defaults__ and []) if False else None

for device in ['cpu','cuda']:
    if device=='cuda' and not torch.cuda.is_available(): continue
    print(f"\n{'='*60}\nDEVICE = {device}  (stride={STRIDE})\n{'='*60}"); sys.stdout.flush()
    m=FingerprintMatcher(model_path=MODELP, device=device)
    pcount=sum(p.numel() for p in m.net.parameters())

    # ---- 注册: 5指 x 20模板, 计时+存储 ----
    gallery={}; enroll_times=[]; tmpl_bytes=0; npts_list=[]
    for fg,ps in pick:
        idx=random.sample(range(len(ps)),20)
        feats=[]
        for i in idx:
            img=load(ps[i])
            t0=time.perf_counter(); f=m.feat(img);
            if device=='cuda': torch.cuda.synchronize()
            enroll_times.append((time.perf_counter()-t0)*1000)
            feats.append(f)
            pts,desc,msk=f
            if desc is not None:
                npts_list.append(len(pts))
                tmpl_bytes += desc.nbytes + pts.nbytes + msk.nbytes   # 当前存储(float32 desc + pts + 完整mask)
        gallery[fg]=feats
    et=np.array(enroll_times)
    print(f"[注册] 单模板注册: 平均={et.mean():.1f}ms 最慢={et.max():.1f}ms (预算<200ms)"); sys.stdout.flush()
    print(f"[空间] 平均关键点/模板={np.mean(npts_list):.0f}; 100模板总存储(float32 desc+pts+mask)={tmpl_bytes/1e6:.1f}MB (预算<10MB)")
    # 压缩选项估算
    avg_n=np.mean(npts_list)
    f16=sum(len(g[1])*128*2 for fg in gallery for g in gallery[fg] if g[1] is not None)
    i8 =sum(len(g[1])*128*1 for fg in gallery for g in gallery[fg] if g[1] is not None)
    print(f"        压缩估算(仅desc, 去mask/pts): float32={i8*4/1e6:.1f}MB float16={f16/1e6:.1f}MB int8={i8/1e6:.1f}MB")
    print(f"[空间] AI模型: best.pth={model_bytes/1e6:.2f}MB, 参数量={pcount/1e6:.2f}M (预算 总<30MB)"); sys.stdout.flush()

    # ---- 解锁: 探针 vs 5指x20模板=100模板, 计时 ----
    all_feats=[f for fg in gallery for f in gallery[fg]]   # 100模板特征
    unlock_times=[]
    for fg,ps in pick:
        probe_idx=[i for i in range(len(ps))][70:75] if len(ps)>72 else [len(ps)-1]
        for pi in probe_idx[:3]:
            img=load(ps[pi])
            t0=time.perf_counter()
            pf=m.feat(img)                                  # 探针特征
            best=0.0
            for tf in all_feats:
                s=m._score(pf,tf)
                if s>best: best=s
            if device=='cuda': torch.cuda.synchronize()
            unlock_times.append((time.perf_counter()-t0)*1000)
    ut=np.array(unlock_times)
    print(f"[耗时] 5指解锁(探针特征+匹配100模板): 平均={ut.mean():.0f}ms 最快={ut.min():.0f}ms (预算 最快<50/平均<100ms)")
    print(f"        其中: 探针特征提取/匹配占比需拆分(见下)"); sys.stdout.flush()
    # 拆分: 特征提取 vs 100次匹配
    img=load(pick[0][1][72] if len(pick[0][1])>72 else pick[0][1][-1])
    t0=time.perf_counter(); pf=m.feat(img);
    if device=='cuda': torch.cuda.synchronize()
    t_feat=(time.perf_counter()-t0)*1000
    t0=time.perf_counter()
    for tf in all_feats: m._score(pf,tf)
    t_match=(time.perf_counter()-t0)*1000
    print(f"        探针特征提取={t_feat:.0f}ms, 匹配100模板={t_match:.0f}ms (单次匹配≈{t_match/100:.2f}ms)")
print("\nDone.")
