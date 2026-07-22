# -*- coding: utf-8 -*-
"""
用已训练的V21模型检验：新数据(ysjz)中同名手指是否与老数据中的同一手指匹配。
对每个新文件夹: 取10帧 -> 全局嵌入 -> 与老数据27类各自20个模板比对 -> 看最像哪一类。
同时对比 raw vs denoised 版本的相似度水平。
"""
import os, glob, random, sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
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
print("V21 models loaded.")

@torch.no_grad()
def embed(frame):
    embs = []
    for m in models:
        for pol in [1, -1]:
            x = torch.tensor(pol * frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            embs.append(F.normalize(m(x), dim=1))
    return F.normalize(torch.mean(torch.stack(embs), dim=0), dim=1).cpu().numpy().flatten()

# ---- 老数据模板 (与V21评估相同: train前70帧中随机20帧) ----
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
random.seed(42)
templates = {}
for base, prefix, reg in [(base1, 'wtp', 'Rgd1245'), (base2, 'btp', 'Rgd1237')]:
    for finger in sorted(os.listdir(base)):
        if 'xzc' in finger.lower():
            continue
        rpath = os.path.join(base, finger, reg)
        if not os.path.exists(rpath):
            continue
        paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))[:70]
        if len(paths) < 50:
            continue
        sel = random.sample(paths, 20)
        embs = []
        for p in sel:
            img = load_img(p)
            if img is None: continue
            embs.append(embed(preprocess_frame(img)))
        templates[f'{prefix}_{finger}'] = np.stack(embs)
        print(f"  template {prefix}_{finger}: {len(embs)}")
        sys.stdout.flush()

# ---- 新数据探针 ----
for variant, newbase in [('raw', 'f:/1111/指纹/ysjz_raw/ysjz'),
                         ('denoised', 'f:/1111/指纹/ysjz_denoised/ysjz_denoised')]:
    print(f"\n{'='*72}\nNEW DATA VARIANT: {variant}\n{'='*72}")
    print(f"{'new folder':22s} {'best_match':16s} {'score':>7s}   {'2nd_match':16s} {'score':>7s}")
    for folder in sorted(os.listdir(newbase)):
        fpath = os.path.join(newbase, folder)
        if not os.path.isdir(fpath):
            continue
        paths = sorted(glob.glob(os.path.join(fpath, '*.bmp')))
        if not paths:
            continue
        idx = np.linspace(0, len(paths)-1, min(10, len(paths))).astype(int)
        probe_embs = []
        for i in idx:
            img = load_img(paths[i])
            if img is None: continue
            probe_embs.append(embed(preprocess_frame(img)))
        # 每个探针对每类取max(模板相似度), 再对探针取均值
        cls_scores = {}
        for cls, T in templates.items():
            s = np.mean([np.max(T @ e) for e in probe_embs])
            cls_scores[cls] = s
        ranked = sorted(cls_scores.items(), key=lambda kv: -kv[1])
        (c1, s1), (c2, s2) = ranked[0], ranked[1]
        print(f"{folder:22s} {c1:16s} {s1:7.4f}   {c2:16s} {s2:7.4f}")
        sys.stdout.flush()
print("\nDone.")
