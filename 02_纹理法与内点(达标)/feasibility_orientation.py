# -*- coding: utf-8 -*-
"""
方向场(orientation field)可行性实验 — 在投入V26训练前验证客户方向是否有效。
回答3个问题:
  Q1 方向场单独能否区分身份? (老数据单场景 OF EER)
  Q2 [关键] 方向场跨场景是否比CNN更稳? 融合后能否改善V25a的跨场景8%?
  Q3 方向场相干性能否当质量门, 解释客户的拒识表?
方法: 块状结构张量方向场 + 相干性加权相似度(带±2块平移搜索, 处理小位移).
不训练, 纯几何特征. 用V25a模型做CNN对照与融合.
"""
import os, glob, random, sys, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torchvision.models import resnet18

def load_img(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

def clahe_enhance(img):
    return cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(img)

def upsample_2x(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

def keep_largest_cc(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    if n <= 1: return mask
    return ((lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) * 255).astype(np.uint8)

def fill_internal_holes(mask):
    bg = cv2.bitwise_not(mask)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, 8, cv2.CV_32S)
    h, w = mask.shape
    for i in range(1, n):
        l, t, ww, hh = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if l > 0 and (l+ww < w-1) and t > 0 and (t+hh < h-1):
            mask[lab == i] = 255
    return mask

def gmfs_mask(img, sigma=13.0/3, percentile=95, ratio=0.2):
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3); dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(dx, dy)
    gs = int(np.ceil(3*sigma))*2 + 1
    m_a = cv2.GaussianBlur(m, (gs, gs), sigma)
    thr = np.percentile(m.flatten(), percentile) * ratio
    _, mask = cv2.threshold(m_a, thr, 255, cv2.THRESH_BINARY); mask = mask.astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=6)
    mask = keep_largest_cc(mask); mask = fill_internal_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=2)
    return keep_largest_cc(mask)

def preprocess_full(img):
    """返回 (CNN用归一化float, 增强uint8, mask)。"""
    enh = clahe_enhance(img); up = upsample_2x(enh); mask = gmfs_mask(up)
    masked = up.copy(); masked[mask == 0] = 0
    valid = masked[mask > 0].astype(np.float32)
    mu, std = (valid.mean(), valid.std()+1e-6) if len(valid) else (0, 1)
    norm = (masked.astype(np.float32) - mu) / std; norm[mask == 0] = 0
    return norm, up, mask

# ---------- 方向场 ----------
B = 10  # 块大小(px), 220x200 -> 22x20 网格
def orientation_grid(enh, mask):
    enh = enh.astype(np.float32)
    gx = cv2.Sobel(enh, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(enh, cv2.CV_32F, 0, 1, ksize=3)
    Gxx, Gyy, Gxy = gx*gx, gy*gy, gx*gy
    fg = (mask > 0).astype(np.float32)
    H, W = enh.shape; gh, gw = H//B, W//B
    def blocksum(a):
        return a[:gh*B, :gw*B].reshape(gh, B, gw, B).sum(axis=(1, 3))
    sXX, sYY, sXY, cnt = blocksum(Gxx), blocksum(Gyy), blocksum(Gxy), blocksum(fg)
    R = np.sqrt((sXX - sYY)**2 + (2*sXY)**2) + 1e-8
    c2 = (sXX - sYY) / R           # cos(2theta) 单位向量分量
    s2 = (2*sXY) / R               # sin(2theta)
    coh = R / (sXX + sYY + 1e-8)   # 相干性 [0,1]
    valid = (cnt >= 0.5*B*B).astype(np.float32)
    coh *= valid
    return c2, s2, coh, valid

def of_template(grids):
    """多帧方向场平均(相干性加权的2theta向量平均)。"""
    acc_c = np.mean([g[0]*g[2] for g in grids], axis=0)
    acc_s = np.mean([g[1]*g[2] for g in grids], axis=0)
    mag = np.sqrt(acc_c**2 + acc_s**2) + 1e-8
    c2, s2 = acc_c/mag, acc_s/mag
    coh = mag
    valid = (np.mean([g[3] for g in grids], axis=0) > 0.3).astype(np.float32)
    coh *= valid
    return c2, s2, coh, valid

SHIFTS = [(dy, dx) for dy in (-2,-1,0,1,2) for dx in (-2,-1,0,1,2)]
def of_sim(a, b):
    """带±2块平移搜索的相干性加权方向相似度 ∈[-1,1]。"""
    c2a, s2a, coha, va = a; c2b, s2b, cohb, vb = b
    H, W = c2a.shape; best = -1.0
    for dy, dx in SHIFTS:
        y0a, y1a = max(0, dy), min(H, H+dy); x0a, x1a = max(0, dx), min(W, W+dx)
        y0b, y1b = max(0, -dy), min(H, H-dy); x0b, x1b = max(0, -dx), min(W, W-dx)
        wa = coha[y0a:y1a, x0a:x1a]; wb = cohb[y0b:y1b, x0b:x1b]
        w = wa * wb
        if w.sum() < 1e-3: continue
        dot = (c2a[y0a:y1a, x0a:x1a]*c2b[y0b:y1b, x0b:x1b] +
               s2a[y0a:y1a, x0a:x1a]*s2b[y0b:y1b, x0b:x1b])
        sim = float((w*dot).sum() / (w.sum()+1e-8))
        if sim > best: best = sim
    return best

# ---------- CNN (V25a) ----------
class HybridDelfEncoder(nn.Module):
    def __init__(self, embed_dim=512, local_dim=64):
        super().__init__()
        base = resnet18(weights=None)
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1, self.relu, self.maxpool = base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_projector = nn.Sequential(nn.Linear(512, embed_dim), nn.BatchNorm1d(embed_dim))
        self.attention = nn.Sequential(nn.Conv2d(256,128,1), nn.ReLU(), nn.Conv2d(128,1,1), nn.Softplus())
        self.attn_projector = nn.Sequential(nn.Linear(256, embed_dim), nn.BatchNorm1d(embed_dim))
        self.local_head = nn.Sequential(nn.Conv2d(256, local_dim, 1, bias=False), nn.BatchNorm2d(local_dim))
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.avgpool(self.layer4(x)).flatten(1)
        return self.global_projector(x)

device = torch.device('cuda')
models = []
for seed in [42, 123, 777]:
    m = HybridDelfEncoder().to(device)
    m.load_state_dict(torch.load(f'f:/1111/指纹/models_v25a/v25a_seed{seed}.pth', map_location=device, weights_only=True))
    m.eval(); models.append(m)
print("V25a models loaded."); sys.stdout.flush()

@torch.no_grad()
def embed(norm):
    es = []
    for m in models:
        for pol in (1, -1):
            x = torch.tensor(pol*norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            es.append(F.normalize(m(x), dim=1))
    return F.normalize(torch.mean(torch.stack(es), dim=0), dim=1).cpu().numpy().flatten()

def eer_of(gen, imp):
    gen, imp = np.array(gen), np.array(imp)
    y = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    s = np.concatenate([gen, imp])
    fpr, tpr, _ = roc_curve(y, s); fnr = 1-tpr
    i = np.nanargmin(np.abs(fnr-fpr))
    return (fpr[i]+fnr[i])/2

# ---------- 数据 ----------
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
random.seed(42)
print("Building old templates (CNN + OF)..."); sys.stdout.flush()
tmpl_cnn, tmpl_of, old_test = {}, {}, {}
for base, prefix, reg in [(base1,'wtp','Rgd1245'), (base2,'btp','Rgd1237')]:
    for finger in sorted(os.listdir(base)):
        if 'xzc' in finger.lower(): continue
        rp = os.path.join(base, finger, reg)
        if not os.path.exists(rp): continue
        paths = sorted(glob.glob(os.path.join(rp, '*.bmp')))
        if len(paths) < 50: continue
        key = f'{prefix}_{finger}'
        tr = paths[:70]
        cnn_list, of_grids = [], []
        sel = random.sample(range(len(tr)), 20)
        for i in sel:
            norm, enh, mask = preprocess_full(load_img(tr[i]))
            cnn_list.append(embed(norm))
        for i in random.sample(range(len(tr)), 12):
            norm, enh, mask = preprocess_full(load_img(tr[i]))
            of_grids.append(orientation_grid(enh, mask))
        tmpl_cnn[key] = cnn_list
        tmpl_of[key] = of_template(of_grids)
        # 老测试帧(单场景): 取10帧
        te = paths[70:]
        tt = []
        for p in random.sample(te, min(10, len(te))):
            norm, enh, mask = preprocess_full(load_img(p))
            tt.append((embed(norm), orientation_grid(enh, mask)))
        old_test[key] = tt
ids = list(tmpl_cnn.keys())
print(f"  {len(ids)} classes ready."); sys.stdout.flush()

def run_eval(name, probes_by_finger):
    """probes_by_finger: {gt_class: [(cnn_emb, of_grid), ...]}"""
    cnn_g, cnn_i, of_g, of_i = [], [], [], []
    fused = {a: ([], []) for a in (0.2,0.3,0.4,0.5)}
    pf = {}
    for gt, probes in probes_by_finger.items():
        pf_of, pf_cnn = [], []
        for cnn_e, of_grid in probes:
            for cid in ids:
                cs = float(np.max([np.dot(cnn_e, t) for t in tmpl_cnn[cid]]))
                os_ = of_sim(of_grid, tmpl_of[cid])
                if cid == gt:
                    cnn_g.append(cs); of_g.append(os_); pf_of.append(os_); pf_cnn.append(cs)
                else:
                    cnn_i.append(cs); of_i.append(os_)
                for a in fused:
                    fv = a*os_ + (1-a)*cs
                    fused[a][0 if cid == gt else 1].append(fv)
        if pf_of:
            pf[gt] = (np.mean(pf_cnn), np.mean(pf_of))
    print(f"\n=== {name} ===")
    print(f"  CNN(V25a) EER = {eer_of(cnn_g, cnn_i)*100:.3f}%")
    print(f"  OF        EER = {eer_of(of_g, of_i)*100:.3f}%")
    for a in sorted(fused):
        print(f"  Fuse a={a} (OF权重) EER = {eer_of(fused[a][0], fused[a][1])*100:.3f}%")
    return pf

# Q1 单场景(老数据)
ss = {k: v for k, v in old_test.items()}
run_eval("Q1 OLD SINGLE-SESSION (sanity)", ss)

# Q2 跨场景(老模板 x 新raw探针)
MERGE = {'dy_R0':'wtp_dy_R0','dy_R1':'wtp_dy_R1','dy_R2':'wtp_dy_R2',
         'lwh_R0':'wtp_lwh_R0','lwh_R1':'wtp_lwh_R1','lwh_R2':'wtp_lwh_R2',
         'zyh_R0':'btp_zyh_R0','zyh_R1':'btp_zyh_R1','zyh_R2':'btp_zyh_R2'}
newbase = 'f:/1111/指纹/ysjz_raw/ysjz'
cross = {}
for folder, gt in MERGE.items():
    probes = []
    for p in sorted(glob.glob(os.path.join(newbase, folder, '*.bmp')))[70:]:
        norm, enh, mask = preprocess_full(load_img(p))
        probes.append((embed(norm), orientation_grid(enh, mask)))
    cross[gt] = probes
pf_cross = run_eval("Q2 CROSS-SESSION (old tmpl x new raw probe)", cross)
print("\n  --- per-finger genuine (cnn / of) ---")
for gt, (c, o) in pf_cross.items():
    print(f"    {gt:14s}: CNN={c:.3f}  OF={o:.3f}")

# Q3 质量门: 所有新手指平均相干性 vs 客户拒识数
print(f"\n{'='*60}\nQ3 QUALITY GATE: OF coherence vs customer refuse\n{'='*60}")
refuse = {'dy_L0':0,'dy_L1':1,'dy_L2':0,'dy_R0':73,'dy_R1':2,'dy_R2':1,
          'lwh_R0':80,'lwh_R1':80,'lwh_R2':80,'SSH_L0':0,'SSH_L1':0,'SSH_L2-half':41,
          'yjx_R0':4,'yjx_R1':4,'yjx_R2':3,'zyh_L0':1,'zyh_L1':1,'zyh_L2':7,
          'zyh_R0':80,'zyh_R1':3,'zyh_R2':80}
rows = []
for folder in sorted(os.listdir(newbase)):
    fp = os.path.join(newbase, folder)
    if not os.path.isdir(fp): continue
    paths = sorted(glob.glob(os.path.join(fp, '*.bmp')))
    if len(paths) < 10: continue
    cohs = []
    for p in random.sample(paths, min(30, len(paths))):
        norm, enh, mask = preprocess_full(load_img(p))
        _, _, coh, valid = orientation_grid(enh, mask)
        if valid.sum() > 0:
            cohs.append(coh[valid > 0].mean())
    rows.append((folder, np.mean(cohs), refuse.get(folder, -1), len(paths)))
rows.sort(key=lambda r: r[1])
print(f"  {'finger':14s} {'mean_coh':>8s} {'refuse':>7s}")
for f, c, r, n in rows:
    flag = ' <- 客户大量拒识' if r >= 40 else ''
    print(f"  {f:14s} {c:8.4f} {str(r) if r>=0 else 'NA':>7s}{flag}")
print("\nDone."); sys.stdout.flush()
