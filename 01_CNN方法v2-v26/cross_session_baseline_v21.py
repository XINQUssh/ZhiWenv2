# -*- coding: utf-8 -*-
"""
跨场景基线评估（V21冻结模型，无训练）:
  客户确认: ysjz新数据中 dy_R*/lwh_R*/zyh_R* 与老数据同名类是同一根手指(贴屏再采集)。
  协议(Eval B): 注册=老数据train前70帧中20模板(seed42, 与V21协议一致);
               探针=新数据同名手指的后30帧(paths[70:], 跨场景genuine) vs 其他26类(impostor)。
  对 raw 和 denoised 两个变体分别评估。输出 global_raw / global_tnorm 的 EER 与 per-finger。
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
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

def clahe_enhance(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def upsample_2x(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

def keep_largest_cc(mask):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    if n_labels <= 1:
        return mask
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ((labels == max_label) * 255).astype(np.uint8)

def fill_internal_holes(mask):
    bg_mask = cv2.bitwise_not(mask)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg_mask, 8, cv2.CV_32S)
    h, w = mask.shape
    for i in range(1, n_labels):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if left > 0 and (left + width < w - 1) and top > 0 and (top + height < h - 1):
            mask[labels == i] = 255
    return mask

def generate_gmfs_mask(img, sigma=13.0/3, percentile=95, threshold_ratio=0.2):
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(dx, dy)
    gs = int(np.ceil(3 * sigma)) * 2 + 1
    m_a = cv2.GaussianBlur(m, (gs, gs), sigma)
    p_val = np.percentile(m.flatten(), percentile)
    thresh = p_val * threshold_ratio
    _, mask = cv2.threshold(m_a, thresh, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=6)
    mask = keep_largest_cc(mask)
    mask = fill_internal_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=2)
    mask = keep_largest_cc(mask)
    return mask

def preprocess_frame(img):
    enhanced = clahe_enhance(img)
    upsampled = upsample_2x(enhanced)
    mask = generate_gmfs_mask(upsampled)
    masked = upsampled.copy()
    masked[mask == 0] = 0
    valid = masked[mask > 0].astype(np.float32)
    if len(valid) > 0:
        mu, std = valid.mean(), valid.std() + 1e-6
    else:
        mu, std = 0, 1
    result = (masked.astype(np.float32) - mu) / std
    result[mask == 0] = 0
    return result

class HybridDelfEncoder(nn.Module):
    def __init__(self, embed_dim=512, local_dim=64):
        super().__init__()
        base = resnet18(weights=None)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_projector = nn.Sequential(nn.Linear(512, embed_dim), nn.BatchNorm1d(embed_dim))
        self.attention = nn.Sequential(nn.Conv2d(256, 128, 1), nn.ReLU(), nn.Conv2d(128, 1, 1), nn.Softplus())
        self.attn_projector = nn.Sequential(nn.Linear(256, embed_dim), nn.BatchNorm1d(embed_dim))
        self.local_head = nn.Sequential(nn.Conv2d(256, local_dim, kernel_size=1, bias=False), nn.BatchNorm2d(local_dim))

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x4 = self.layer4(x)
        x4 = self.avgpool(x4).flatten(1)
        return self.global_projector(x4)

device = torch.device('cuda')
models = []
for seed in [42, 123, 777]:
    m = HybridDelfEncoder().to(device)
    sd = torch.load(f'f:/1111/指纹/models_v21/v21_seed{seed}.pth', map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)
print("V21 models loaded."); sys.stdout.flush()

@torch.no_grad()
def embed(frame):
    embs = []
    for m in models:
        for pol in [1, -1]:
            x = torch.tensor(pol * frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            embs.append(F.normalize(m(x), dim=1))
    return F.normalize(torch.mean(torch.stack(embs), dim=0), dim=1).cpu().numpy().flatten()

# ---- 老数据train帧 + V21式模板选择 (seed42, 类顺序与V21一致) ----
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
finger_train_frames = {}
for base, prefix, reg in [(base1, 'wtp', 'Rgd1245'), (base2, 'btp', 'Rgd1237')]:
    for finger in sorted(os.listdir(base)):
        if 'xzc' in finger.lower():
            continue
        rpath = os.path.join(base, finger, reg)
        if not os.path.exists(rpath):
            continue
        paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
        if len(paths) < 50:
            continue
        frames = []
        for p in paths[:70]:
            img = load_img(p)
            if img is not None:
                frames.append(preprocess_frame(img))
        finger_train_frames[f'{prefix}_{finger}'] = frames
        sys.stdout.flush()
print(f"Old train frames loaded: {len(finger_train_frames)} classes"); sys.stdout.flush()

random.seed(42)
templates = {}
for key, frames in finger_train_frames.items():   # 插入顺序 = V21的fingers顺序
    idxs = random.sample(range(len(frames)), min(20, len(frames)))
    templates[key] = np.stack([embed(frames[i]) for i in idxs])
all_ids = list(templates.keys())
print("Templates ready."); sys.stdout.flush()

MERGE_MAP = {
    'dy_R0': 'wtp_dy_R0', 'dy_R1': 'wtp_dy_R1', 'dy_R2': 'wtp_dy_R2',
    'lwh_R0': 'wtp_lwh_R0', 'lwh_R1': 'wtp_lwh_R1', 'lwh_R2': 'wtp_lwh_R2',
    'zyh_R0': 'btp_zyh_R0', 'zyh_R1': 'btp_zyh_R1', 'zyh_R2': 'btp_zyh_R2',
}

def eer_of(gen, imp):
    y = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    s = np.concatenate([gen, imp])
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2, fpr, fnr

for variant, newbase in [('raw', 'f:/1111/指纹/ysjz_raw/ysjz'),
                         ('denoised', 'f:/1111/指纹/ysjz_denoised/ysjz_denoised')]:
    print(f"\n{'='*72}\nCROSS-SESSION EVAL (V21 frozen) — probe variant: {variant}\n{'='*72}")
    gen_raw, imp_raw, gen_t, imp_t = [], [], [], []
    pf = {}
    t0 = time.time()
    for folder, old_key in MERGE_MAP.items():
        paths = sorted(glob.glob(os.path.join(newbase, folder, '*.bmp')))
        probes = []
        for p in paths[70:]:
            img = load_img(p)
            if img is not None:
                probes.append(embed(preprocess_frame(img)))
        pf_scores = []
        for e in probes:
            scores = {cid: float(np.max(templates[cid] @ e)) for cid in all_ids}
            mu = {cid: scores[cid] for cid in all_ids}
            for cid in all_ids:
                others = [scores[o] for o in all_ids if o != cid]
                tn = (scores[cid] - np.mean(others)) / (np.std(others) + 1e-8)
                if cid == old_key:
                    gen_raw.append(scores[cid]); gen_t.append(tn)
                else:
                    imp_raw.append(scores[cid]); imp_t.append(tn)
            pf_scores.append(scores[old_key])
        pf[folder] = (np.mean(pf_scores), np.min(pf_scores), len(pf_scores))
        print(f"  {folder:8s} -> {old_key:12s}: genuine mean={pf[folder][0]:.4f}, "
              f"min={pf[folder][1]:.4f}, n={pf[folder][2]}, elapsed={time.time()-t0:.0f}s")
        sys.stdout.flush()

    for name, gen, imp in [('global_raw', gen_raw, imp_raw), ('global_tnorm', gen_t, imp_t)]:
        gen, imp = np.array(gen), np.array(imp)
        eer, fpr, fnr = eer_of(gen, imp)
        print(f"\n  [{name}] cross-session EER = {eer*100:.4f}%  "
              f"(genuine n={len(gen)} mean={gen.mean():.4f} | impostor n={len(imp)} mean={imp.mean():.4f})")
        for tf in [0.01, 0.03, 0.05, 0.10, 0.20]:
            i = np.argmin(np.abs(fnr - tf))
            print(f"    FFR={tf*100:.0f}% -> FAR={fpr[i]*100:.4f}%")
    sys.stdout.flush()
print("\nDone.")
