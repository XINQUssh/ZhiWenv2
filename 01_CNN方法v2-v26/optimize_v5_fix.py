"""
SimCLR预训练 + ArcFace微调 — 30类验证 (修复版)
修复: SimCLR预训练严格只用前70帧(训练集), 消除数据泄露
"""
import os, glob, random, math, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 数据加载
# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'

finger_data_train = {}  # 严格分离: 训练集
finger_data_test = {}   # 严格分离: 测试集
finger_labels = {}
finger_source = {}
fi = 0

print("Loading 无贴屏 (Rgd1245)...")
for finger in sorted(os.listdir(base1)):
    rpath = os.path.join(base1, finger, 'Rgd1245')
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    loaded = []
    for p in imgs_paths:
        img = load_img(p).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)
        loaded.append(img)
    key = f"wtp_{finger}"
    finger_data_train[key] = loaded[:70]   # 训练: 前70帧
    finger_data_test[key] = loaded[70:]    # 测试: 后30帧
    finger_labels[key] = fi
    finger_source[key] = "无贴屏"
    fi += 1

n_wtp = fi
print(f"  无贴屏: {n_wtp} classes")

print("Loading 不贴屏 (Rgd1237)...")
for finger in sorted(os.listdir(base2)):
    rpath = os.path.join(base2, finger, 'Rgd1237')
    if not os.path.exists(rpath):
        print(f"  SKIP {finger} (no Rgd1237)")
        continue
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    if len(imgs_paths) < 50:
        print(f"  SKIP {finger} (only {len(imgs_paths)} imgs)")
        continue
    loaded = []
    for p in imgs_paths:
        img = load_img(p).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)
        loaded.append(img)
    key = f"btp_{finger}"
    finger_data_train[key] = loaded[:70]
    finger_data_test[key] = loaded[70:]
    finger_labels[key] = fi
    finger_source[key] = "不贴屏"
    fi += 1

n_classes = fi
fingers = list(finger_data_train.keys())
print(f"Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes - n_wtp} 不贴屏)")
for f in fingers:
    print(f"  {f}: train={len(finger_data_train[f])}, test={len(finger_data_test[f])} [{finger_source[f]}]")

# ============================================================
# 模型定义
# ============================================================
class FrameEncoder(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Linear(256 * 4, feat_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SetAttentionAggregator(nn.Module):
    def __init__(self, feat_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)

    def forward(self, frame_feats):
        B = frame_feats.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, frame_feats], dim=1)
        x = self.transformer(x)
        return x[:, 0, :]

class SetFingerNet(nn.Module):
    def __init__(self, feat_dim=128, embed_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.frame_encoder = FrameEncoder(feat_dim)
        self.aggregator = SetAttentionAggregator(feat_dim, n_heads, n_layers)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, frames):
        B, N, H, W = frames.shape
        x = frames.view(B * N, 1, H, W)
        frame_feats = self.frame_encoder(x)
        frame_feats = frame_feats.view(B, N, -1)
        agg = self.aggregator(frame_feats)
        emb = self.projector(agg)
        return emb

class ArcFace(nn.Module):
    def __init__(self, embed_dim, n_classes, s=64.0, m=0.5):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels):
        emb_n = F.normalize(emb, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        cos = torch.mm(emb_n, w_n.t()).clamp(-1+1e-7, 1-1e-7)
        theta = torch.acos(cos)
        oh = torch.zeros_like(cos)
        oh.scatter_(1, labels.view(-1, 1), 1)
        return F.cross_entropy(torch.cos(theta + oh * self.m) * self.s, labels)

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)
        return F.cross_entropy(sim, labels)

# ============================================================
# 数据集 — 修复版: 预训练也只用训练集
# ============================================================
class ContrastiveFrameSetDataset(Dataset):
    """SimCLR预训练: 严格只用训练帧"""
    def __init__(self, finger_data_train, n_frames=20, n_samples=200):
        self.finger_data = finger_data_train  # ← 只用训练集!
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.fingers = list(finger_data_train.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        imgs = self.finger_data[finger]  # ← 只有前70帧

        idx1 = random.sample(range(len(imgs)), min(self.n_frames, len(imgs)))
        idx2 = random.sample(range(len(imgs)), min(self.n_frames, len(imgs)))

        f1 = np.stack([imgs[i] for i in idx1]).astype(np.float32)
        f2 = np.stack([imgs[i] for i in idx2]).astype(np.float32)

        for i in range(len(f1)):
            if random.random() < 0.3: f1[i] = -f1[i]
        for i in range(len(f2)):
            if random.random() < 0.3: f2[i] = -f2[i]

        f1 += np.random.randn(*f1.shape).astype(np.float32) * 0.02
        f2 += np.random.randn(*f2.shape).astype(np.float32) * 0.02

        return torch.tensor(f1), torch.tensor(f2)

class FrameSetDataset(Dataset):
    """ArcFace微调: 只用训练帧"""
    def __init__(self, finger_data_train, labels, n_frames=20, n_samples=120):
        self.finger_data = finger_data_train  # ← 只用训练集
        self.labels = labels
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.fingers = list(finger_data_train.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        imgs = self.finger_data[finger]

        chosen = random.sample(range(len(imgs)), min(self.n_frames, len(imgs)))
        frames = np.stack([imgs[i] for i in chosen], axis=0)

        # 数据增强
        for i in range(len(frames)):
            if random.random() < 0.3:
                frames[i] = -frames[i]
        frames += np.random.randn(*frames.shape).astype(np.float32) * 0.02

        return torch.tensor(frames, dtype=torch.float32), self.labels[finger]

# ============================================================
# 训练
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
print("NOTE: 本次实验修复了数据泄露, SimCLR预训练严格只用训练集(前70帧)")

N_FRAMES = 25

# ---------- Phase 1: SimCLR 预训练 (只用训练集!) ----------
print("\n" + "="*60)
print("Phase 1: SimCLR Pre-training (TRAIN DATA ONLY)")
print("="*60)

model = SetFingerNet(feat_dim=128, embed_dim=256, n_heads=4, n_layers=2).to(device)
simclr_loss = SimCLRLoss(temperature=0.1)
opt1 = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=80)

pretrain_ds = ContrastiveFrameSetDataset(finger_data_train, n_frames=N_FRAMES, n_samples=200)
pretrain_loader = DataLoader(pretrain_ds, batch_size=16, shuffle=True, num_workers=0)

t0 = time.time()
print(f"Pre-training... ({len(pretrain_ds)} samples, train frames only)")
for epoch in range(80):
    model.train()
    tl = 0
    for f1, f2 in pretrain_loader:
        f1, f2 = f1.to(device), f2.to(device)
        z1 = F.normalize(model(f1), dim=1)
        z2 = F.normalize(model(f2), dim=1)
        loss = simclr_loss(z1, z2)
        opt1.zero_grad(); loss.backward(); opt1.step()
        tl += loss.item()
    sch1.step()
    if (epoch+1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  Pre-train epoch {epoch+1}: loss={tl/len(pretrain_loader):.4f}, time={elapsed:.0f}s")

# ---------- Phase 2: ArcFace 微调 (只用训练集!) ----------
print("\n" + "="*60)
print("Phase 2: ArcFace Fine-tuning (TRAIN DATA ONLY)")
print("="*60)

arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)
opt2 = optim.AdamW(
    list(model.parameters()) + list(arcface.parameters()),
    lr=0.0002, weight_decay=1e-3
)
sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=80)

ft_ds = FrameSetDataset(finger_data_train, finger_labels, n_frames=N_FRAMES, n_samples=150)
ft_loader = DataLoader(ft_ds, batch_size=16, shuffle=True, num_workers=0)

t1 = time.time()
print(f"Fine-tuning... ({len(ft_ds)} samples, train frames only)")
for epoch in range(80):
    model.train(); arcface.train()
    tl, cor, tot = 0, 0, 0
    for X, Y in ft_loader:
        X, Y = X.to(device), Y.to(device)
        emb = model(X)
        loss = arcface(emb, Y)
        opt2.zero_grad(); loss.backward(); opt2.step()
        tl += loss.item()
        with torch.no_grad():
            en = F.normalize(emb, dim=1)
            wn = F.normalize(arcface.weight, dim=1)
            _, p = torch.mm(en, wn.t()).max(1)
            cor += p.eq(Y).sum().item(); tot += Y.size(0)
    sch2.step()
    if (epoch+1) % 10 == 0:
        elapsed = time.time() - t1
        print(f"  Fine-tune epoch {epoch+1}: loss={tl/len(ft_loader):.4f}, acc={cor/tot*100:.1f}%, time={elapsed:.0f}s")

# ============================================================
# 测试 — 注册用训练集, 验证用测试集 (完全隔离)
# ============================================================
model.eval()
print("\n" + "="*60)
print("Testing (30 classes, STRICT train/test split)")
print("  Enroll: train frames (前70帧)")
print("  Verify: test frames (后30帧, 模型从未见过)")
print("="*60)

def get_set_emb(imgs, n_frames=25):
    chosen = random.sample(range(len(imgs)), min(n_frames, len(imgs)))
    frames = np.stack([imgs[i] for i in chosen], axis=0)
    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)
    return F.normalize(emb, dim=1).cpu().numpy().flatten()

def topk_avg(e, v, k=5):
    sims = sorted([np.dot(a, b) for a in e for b in v], reverse=True)
    return np.mean(sims[:k])

for test_n_frames in [15, 20, 25]:
    K = 20
    enroll_embs, verify_embs = {}, {}
    for finger in fingers:
        # 注册: 用训练集帧
        enroll_embs[finger] = [get_set_emb(finger_data_train[finger], test_n_frames) for _ in range(K)]
        # 验证: 用测试集帧 (模型从未见过!)
        verify_embs[finger] = [get_set_emb(finger_data_test[finger], test_n_frames) for _ in range(K)]

    for k_val in [1, 3, 5, 10]:
        genuine, impostor = [], []
        for f in fingers:
            genuine.append(topk_avg(enroll_embs[f], verify_embs[f], k_val))
        for i in range(len(fingers)):
            for j in range(i+1, len(fingers)):
                impostor.append(topk_avg(enroll_embs[fingers[i]], verify_embs[fingers[j]], k_val))

        im, ist = np.mean(impostor), np.std(impostor)
        gz = [(g - im)/(ist+1e-8) for g in genuine]
        iz = [(x - im)/(ist+1e-8) for x in impostor]

        fpr, tpr, _ = roc_curve([1]*len(gz)+[0]*len(iz), gz+iz)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        gmin, imax = np.min(gz), np.max(iz)

        sep = "*** PERFECT ***" if gmin > imax else f"overlap={sum(1 for x in iz if x > gmin)}/{len(iz)}"
        print(f"\n  frames={test_n_frames}, top-{k_val}: EER={eer*100:.2f}%, gen_min={gmin:.3f}, imp_max={imax:.3f}, {sep}")

        if k_val in [3, 5]:
            for tf in [0.0, 0.03, 0.05, 0.10, 0.20]:
                idx = np.argmin(np.abs(fnr - tf))
                print(f"    FFR={tf*100:.0f}% -> FAR={fpr[idx]*100:.4f}%")
            idx_far = np.argmin(np.abs(fpr - 0.00002))
            print(f"    FAR=0.002% -> FFR={fnr[idx_far]*100:.2f}%")

# 分组分析
print("\n" + "="*60)
print("分组分析: 无贴屏 vs 不贴屏")
print("="*60)

wtp_fingers = [f for f in fingers if f.startswith("wtp_")]
btp_fingers = [f for f in fingers if f.startswith("btp_")]

# 用最后一轮的embs (test_n_frames=25)
for group_name, group_fingers in [("无贴屏", wtp_fingers), ("不贴屏", btp_fingers)]:
    if not group_fingers:
        continue
    genuine_g, impostor_g = [], []
    for f in group_fingers:
        genuine_g.append(topk_avg(enroll_embs[f], verify_embs[f], 5))
    for i in range(len(group_fingers)):
        for j in range(i+1, len(group_fingers)):
            impostor_g.append(topk_avg(enroll_embs[group_fingers[i]], verify_embs[group_fingers[j]], 5))

    im_g, ist_g = np.mean(impostor_g), np.std(impostor_g)
    gz_g = [(g - im_g)/(ist_g+1e-8) for g in genuine_g]
    iz_g = [(x - im_g)/(ist_g+1e-8) for x in impostor_g]

    fpr_g, tpr_g, _ = roc_curve([1]*len(gz_g)+[0]*len(iz_g), gz_g+iz_g)
    fnr_g = 1 - tpr_g
    eer_idx_g = np.nanargmin(np.abs(fnr_g - fpr_g))
    eer_g = (fpr_g[eer_idx_g] + fnr_g[eer_idx_g]) / 2
    gmin_g, imax_g = np.min(gz_g), np.max(iz_g)

    sep_g = "PERFECT" if gmin_g > imax_g else f"overlap={sum(1 for x in iz_g if x > gmin_g)}/{len(iz_g)}"
    print(f"\n  {group_name} ({len(group_fingers)}类): EER={eer_g*100:.2f}%, gen_min={gmin_g:.3f}, imp_max={imax_g:.3f}, {sep_g}")
    for tf in [0.0, 0.03, 0.05, 0.10]:
        idx = np.argmin(np.abs(fnr_g - tf))
        print(f"    FFR={tf*100:.0f}% -> FAR={fpr_g[idx]*100:.4f}%")

# ============================================================
# 额外对比: 不用SimCLR预训练, 直接Set Transformer + ArcFace (作为baseline)
# ============================================================
print("\n" + "="*60)
print("Baseline对比: 无SimCLR预训练, 直接ArcFace训练")
print("="*60)

model_base = SetFingerNet(feat_dim=128, embed_dim=256, n_heads=4, n_layers=2).to(device)
arcface_base = ArcFace(256, n_classes, s=64, m=0.5).to(device)
opt_base = optim.AdamW(
    list(model_base.parameters()) + list(arcface_base.parameters()),
    lr=0.0005, weight_decay=1e-3
)
sch_base = optim.lr_scheduler.CosineAnnealingLR(opt_base, T_max=120)

base_ds = FrameSetDataset(finger_data_train, finger_labels, n_frames=N_FRAMES, n_samples=150)
base_loader = DataLoader(base_ds, batch_size=16, shuffle=True, num_workers=0)

t2 = time.time()
print(f"Training baseline... ({len(base_ds)} samples)")
for epoch in range(120):
    model_base.train(); arcface_base.train()
    tl, cor, tot = 0, 0, 0
    for X, Y in base_loader:
        X, Y = X.to(device), Y.to(device)
        emb = model_base(X)
        loss = arcface_base(emb, Y)
        opt_base.zero_grad(); loss.backward(); opt_base.step()
        tl += loss.item()
        with torch.no_grad():
            en = F.normalize(emb, dim=1)
            wn = F.normalize(arcface_base.weight, dim=1)
            _, p = torch.mm(en, wn.t()).max(1)
            cor += p.eq(Y).sum().item(); tot += Y.size(0)
    sch_base.step()
    if (epoch+1) % 30 == 0:
        elapsed = time.time() - t2
        print(f"  Epoch {epoch+1}: loss={tl/len(base_loader):.4f}, acc={cor/tot*100:.1f}%, time={elapsed:.0f}s")

model_base.eval()
print("\nBaseline Test:")

def get_base_emb(imgs, n_frames=25):
    chosen = random.sample(range(len(imgs)), min(n_frames, len(imgs)))
    frames = np.stack([imgs[i] for i in chosen], axis=0)
    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model_base(x)
    return F.normalize(emb, dim=1).cpu().numpy().flatten()

K = 20
en_b, ve_b = {}, {}
for finger in fingers:
    en_b[finger] = [get_base_emb(finger_data_train[finger], 25) for _ in range(K)]
    ve_b[finger] = [get_base_emb(finger_data_test[finger], 25) for _ in range(K)]

for k_val in [5]:
    genuine_b, impostor_b = [], []
    for f in fingers:
        genuine_b.append(topk_avg(en_b[f], ve_b[f], k_val))
    for i in range(len(fingers)):
        for j in range(i+1, len(fingers)):
            impostor_b.append(topk_avg(en_b[fingers[i]], ve_b[fingers[j]], k_val))

    im_b, ist_b = np.mean(impostor_b), np.std(impostor_b)
    gz_b = [(g - im_b)/(ist_b+1e-8) for g in genuine_b]
    iz_b = [(x - im_b)/(ist_b+1e-8) for x in impostor_b]

    fpr_b, tpr_b, _ = roc_curve([1]*len(gz_b)+[0]*len(iz_b), gz_b+iz_b)
    fnr_b = 1 - tpr_b
    eer_idx_b = np.nanargmin(np.abs(fnr_b - fpr_b))
    eer_b = (fpr_b[eer_idx_b] + fnr_b[eer_idx_b]) / 2
    gmin_b, imax_b = np.min(gz_b), np.max(iz_b)

    sep_b = "PERFECT" if gmin_b > imax_b else f"overlap={sum(1 for x in iz_b if x > gmin_b)}/{len(iz_b)}"
    print(f"  Baseline top-{k_val}: EER={eer_b*100:.2f}%, gen_min={gmin_b:.3f}, imp_max={imax_b:.3f}, {sep_b}")
    for tf in [0.0, 0.03, 0.05, 0.10]:
        idx = np.argmin(np.abs(fnr_b - tf))
        print(f"    FFR={tf*100:.0f}% -> FAR={fpr_b[idx]*100:.4f}%")

print("\n" + "="*60)
print("All done. (Fixed version - no data leakage)")
print("="*60)
