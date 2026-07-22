"""
先进方法对比实验:
1. Set Transformer (注意力加权帧聚合) - 学习哪些帧有用, 如何组合
2. Self-supervised contrastive pre-training (BYOL风格)
3. 上述两者结合

核心思想: 不再用手工极性校正+简单平均, 而是让网络端到端学习
如何从N帧噪声序列中提取稳定指纹特征
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

base = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
fingers = sorted(os.listdir(base))

print(f"Loading {len(fingers)} fingers...")
finger_data = {}
for finger in fingers:
    rpath = os.path.join(base, finger, 'Rgd1245')
    imgs = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    loaded = []
    for p in imgs:
        img = load_img(p).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)
        loaded.append(img)
    finger_data[finger] = loaded

n_classes = len(fingers)
finger_labels = {f: i for i, f in enumerate(fingers)}
print(f"Loaded. {n_classes} classes, {len(finger_data[fingers[0]])} frames each")

# ============================================================
# 方法1: Frame Encoder + Set Attention Aggregator
# 输入: N帧原始图像 -> 逐帧提取特征 -> 注意力加权聚合 -> 嵌入
# 关键: 网络自己学习极性处理和帧选择
# ============================================================

class FrameEncoder(nn.Module):
    """轻量级逐帧特征提取器"""
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
        # x: (B*N, 1, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # (B*N, feat_dim)

class SetAttentionAggregator(nn.Module):
    """
    Multi-head self-attention over frame features.
    学习哪些帧对指纹识别有用, 自动处理极性/噪声问题.
    """
    def __init__(self, feat_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Learnable query for aggregation (like CLS token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)

    def forward(self, frame_feats):
        # frame_feats: (B, N, feat_dim)
        B = frame_feats.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, feat_dim)
        x = torch.cat([cls, frame_feats], dim=1)  # (B, N+1, feat_dim)
        x = self.transformer(x)
        return x[:, 0, :]  # CLS token output: (B, feat_dim)

class SetFingerNet(nn.Module):
    """完整模型: FrameEncoder + SetAttention + Projection Head"""
    def __init__(self, feat_dim=128, embed_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.frame_encoder = FrameEncoder(feat_dim)
        self.aggregator = SetAttentionAggregator(feat_dim, n_heads, n_layers)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, frames):
        """
        frames: (B, N, H, W) - B个样本, 每个N帧
        """
        B, N, H, W = frames.shape
        # 逐帧编码
        x = frames.view(B * N, 1, H, W)
        frame_feats = self.frame_encoder(x)  # (B*N, feat_dim)
        frame_feats = frame_feats.view(B, N, -1)  # (B, N, feat_dim)
        # 注意力聚合
        agg = self.aggregator(frame_feats)  # (B, feat_dim)
        # 投影
        emb = self.projector(agg)  # (B, embed_dim)
        return emb

# ============================================================
# 数据集: 每次随机采样N帧作为一个"集合"
# ============================================================
class FrameSetDataset(Dataset):
    def __init__(self, finger_data, labels, n_frames=20, n_samples=120, train=True):
        self.finger_data = finger_data
        self.labels = labels
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.train = train
        self.fingers = list(finger_data.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        imgs = self.finger_data[finger]
        pool = imgs[:70] if self.train else imgs[70:]

        # 随机选N帧
        chosen = random.sample(range(len(pool)), min(self.n_frames, len(pool)))
        frames = np.stack([pool[i] for i in chosen], axis=0)  # (N, H, W)

        # 数据增强 (训练时)
        if self.train:
            # 逐帧独立随机极性翻转 (模拟传感器极性不稳定)
            for i in range(len(frames)):
                if random.random() < 0.3:
                    frames[i] = -frames[i]
            # 整体噪声
            frames += np.random.randn(*frames.shape).astype(np.float32) * 0.02

        return torch.tensor(frames, dtype=torch.float32), self.labels[finger]

# ============================================================
# ArcFace
# ============================================================
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

# ============================================================
# 训练
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

N_FRAMES = 25  # 每次采样帧数
model = SetFingerNet(feat_dim=128, embed_dim=256, n_heads=4, n_layers=2).to(device)
arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)

optimizer = optim.AdamW(
    list(model.parameters()) + list(arcface.parameters()),
    lr=0.0005, weight_decay=1e-3
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

train_ds = FrameSetDataset(finger_data, finger_labels, n_frames=N_FRAMES, n_samples=150, train=True)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

print(f"Training Set Attention Network (N_frames={N_FRAMES})...")
t0 = time.time()
for epoch in range(120):
    model.train(); arcface.train()
    tl, cor, tot = 0, 0, 0
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        emb = model(X)
        loss = arcface(emb, Y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tl += loss.item()
        with torch.no_grad():
            en = F.normalize(emb, dim=1)
            wn = F.normalize(arcface.weight, dim=1)
            _, p = torch.mm(en, wn.t()).max(1)
            cor += p.eq(Y).sum().item(); tot += Y.size(0)
    scheduler.step()
    if (epoch+1) % 30 == 0:
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: loss={tl/len(train_loader):.4f}, acc={cor/tot*100:.1f}%, time={elapsed:.0f}s")

# ============================================================
# 测试: 不同帧数对比
# ============================================================
model.eval()
print(f"\n=== Set Attention Network Test ===")

def get_set_emb(imgs, n_frames=25):
    """从帧列表中随机采样N帧, 得到嵌入"""
    chosen = random.sample(range(len(imgs)), min(n_frames, len(imgs)))
    frames = np.stack([imgs[i] for i in chosen], axis=0)
    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)
    return F.normalize(emb, dim=1).cpu().numpy().flatten()

# 测试不同的验证帧数
for test_n_frames in [10, 15, 20, 25, 30]:
    K = 15  # 每指重复K次
    enroll_embs, verify_embs = {}, {}
    for finger in fingers:
        el = [get_set_emb(finger_data[finger][:70], test_n_frames) for _ in range(K)]
        vl = [get_set_emb(finger_data[finger][70:], test_n_frames) for _ in range(K)]
        enroll_embs[finger] = el
        verify_embs[finger] = vl

    # Score融合
    def topk_avg(e, v, k=5):
        sims = sorted([np.dot(a, b) for a in e for b in v], reverse=True)
        return np.mean(sims[:k])

    for k_val in [1, 5]:
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
        print(f"  frames={test_n_frames}, top-{k_val}: EER={eer*100:.2f}%, gen_min={gmin:.3f}, imp_max={imax:.3f}, {sep}")

        if k_val == 5:
            for tf in [0.0, 0.03, 0.05, 0.10]:
                idx = np.argmin(np.abs(fnr - tf))
                print(f"    FFR={tf*100:.0f}% -> FAR={fpr[idx]*100:.4f}%")
            idx_far = np.argmin(np.abs(fpr - 0.00002))
            print(f"    FAR=0.002% -> FFR={fnr[idx_far]*100:.2f}%")

# ============================================================
# 方法2: 自监督对比预训练 (SimCLR风格)
# 同一指的两组随机帧 -> 正样本对
# 不同指的帧组 -> 负样本对
# 先预训练encoder, 再fine-tune
# ============================================================
print(f"\n=== Method 2: Self-supervised pre-training + fine-tune ===")

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """z1, z2: (B, dim) normalized"""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2B, dim)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # 正样本对: (i, i+B) 和 (i+B, i)
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)

        # mask out self-similarity
        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)

        return F.cross_entropy(sim, labels)

class ContrastiveFrameSetDataset(Dataset):
    """返回同一指的两组不同帧 (正样本对)"""
    def __init__(self, finger_data, n_frames=20, n_samples=200):
        self.finger_data = finger_data
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.fingers = list(finger_data.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        imgs = self.finger_data[finger]  # 用全部100帧

        # 两组不同的随机帧
        idx1 = random.sample(range(len(imgs)), min(self.n_frames, len(imgs)))
        idx2 = random.sample(range(len(imgs)), min(self.n_frames, len(imgs)))

        f1 = np.stack([imgs[i] for i in idx1]).astype(np.float32)
        f2 = np.stack([imgs[i] for i in idx2]).astype(np.float32)

        # 随机极性翻转
        for i in range(len(f1)):
            if random.random() < 0.3: f1[i] = -f1[i]
        for i in range(len(f2)):
            if random.random() < 0.3: f2[i] = -f2[i]

        f1 += np.random.randn(*f1.shape).astype(np.float32) * 0.02
        f2 += np.random.randn(*f2.shape).astype(np.float32) * 0.02

        return torch.tensor(f1), torch.tensor(f2)

# 预训练
model2 = SetFingerNet(feat_dim=128, embed_dim=256, n_heads=4, n_layers=2).to(device)
simclr_loss = SimCLRLoss(temperature=0.1)
opt2 = optim.AdamW(model2.parameters(), lr=0.0005, weight_decay=1e-3)
sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=80)

pretrain_ds = ContrastiveFrameSetDataset(finger_data, n_frames=N_FRAMES, n_samples=200)
pretrain_loader = DataLoader(pretrain_ds, batch_size=16, shuffle=True, num_workers=0)

print("Pre-training with SimCLR...")
for epoch in range(80):
    model2.train()
    tl = 0
    for f1, f2 in pretrain_loader:
        f1, f2 = f1.to(device), f2.to(device)
        z1 = F.normalize(model2(f1), dim=1)
        z2 = F.normalize(model2(f2), dim=1)
        loss = simclr_loss(z1, z2)
        opt2.zero_grad(); loss.backward(); opt2.step()
        tl += loss.item()
    sch2.step()
    if (epoch+1) % 20 == 0:
        print(f"  Pre-train epoch {epoch+1}: loss={tl/len(pretrain_loader):.4f}")

# Fine-tune with ArcFace
arcface2 = ArcFace(256, n_classes, s=64, m=0.5).to(device)
opt_ft = optim.AdamW(
    list(model2.parameters()) + list(arcface2.parameters()),
    lr=0.0002, weight_decay=1e-3
)
sch_ft = optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=60)

ft_ds = FrameSetDataset(finger_data, finger_labels, n_frames=N_FRAMES, n_samples=150, train=True)
ft_loader = DataLoader(ft_ds, batch_size=16, shuffle=True, num_workers=0)

print("Fine-tuning with ArcFace...")
for epoch in range(60):
    model2.train(); arcface2.train()
    tl, cor, tot = 0, 0, 0
    for X, Y in ft_loader:
        X, Y = X.to(device), Y.to(device)
        emb = model2(X)
        loss = arcface2(emb, Y)
        opt_ft.zero_grad(); loss.backward(); opt_ft.step()
        tl += loss.item()
        with torch.no_grad():
            en = F.normalize(emb, dim=1)
            wn = F.normalize(arcface2.weight, dim=1)
            _, p = torch.mm(en, wn.t()).max(1)
            cor += p.eq(Y).sum().item(); tot += Y.size(0)
    sch_ft.step()
    if (epoch+1) % 20 == 0:
        print(f"  Fine-tune epoch {epoch+1}: loss={tl/len(ft_loader):.4f}, acc={cor/tot*100:.1f}%")

# 测试 model2
model2.eval()
print("\n=== SimCLR Pre-trained + ArcFace Fine-tuned Test ===")

def get_set_emb2(imgs, n_frames=25):
    chosen = random.sample(range(len(imgs)), min(n_frames, len(imgs)))
    frames = np.stack([imgs[i] for i in chosen], axis=0)
    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model2(x)
    return F.normalize(emb, dim=1).cpu().numpy().flatten()

for test_n in [15, 25]:
    K = 15
    en_e, ve_e = {}, {}
    for finger in fingers:
        en_e[finger] = [get_set_emb2(finger_data[finger][:70], test_n) for _ in range(K)]
        ve_e[finger] = [get_set_emb2(finger_data[finger][70:], test_n) for _ in range(K)]

    for k_val in [1, 5]:
        genuine, impostor = [], []
        for f in fingers:
            sims = sorted([np.dot(a, b) for a in en_e[f] for b in ve_e[f]], reverse=True)
            genuine.append(np.mean(sims[:k_val]))
        for i in range(len(fingers)):
            for j in range(i+1, len(fingers)):
                sims = sorted([np.dot(a, b) for a in en_e[fingers[i]] for b in ve_e[fingers[j]]], reverse=True)
                impostor.append(np.mean(sims[:k_val]))

        im, ist = np.mean(impostor), np.std(impostor)
        gz = [(g - im)/(ist+1e-8) for g in genuine]
        iz = [(x - im)/(ist+1e-8) for x in impostor]
        fpr, tpr, _ = roc_curve([1]*len(gz)+[0]*len(iz), gz+iz)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        gmin, imax = np.min(gz), np.max(iz)
        sep = "*** PERFECT ***" if gmin > imax else f"overlap={sum(1 for x in iz if x > gmin)}/{len(iz)}"
        print(f"  frames={test_n}, top-{k_val}: EER={eer*100:.2f}%, gen_min={gmin:.3f}, imp_max={imax:.3f}, {sep}")

        if k_val == 5:
            for tf in [0.0, 0.03, 0.05, 0.10]:
                idx = np.argmin(np.abs(fnr - tf))
                print(f"    FFR={tf*100:.0f}% -> FAR={fpr[idx]*100:.4f}%")
            idx_far = np.argmin(np.abs(fpr - 0.00002))
            print(f"    FAR=0.002% -> FFR={fnr[idx_far]*100:.2f}%")

print("\nAll done.")
