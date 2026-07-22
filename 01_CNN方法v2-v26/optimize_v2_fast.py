"""
简化版多寄存器融合: 去掉耗时的quality_weighted_template, 用简单极性校正平均
"""
import os, glob, random
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
regs_list = ['Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245', 'Rgd1247']

print(f"Loading {len(fingers)} fingers x {len(regs_list)} registers...")
finger_data = {}
for finger in fingers:
    finger_data[finger] = {}
    for reg in regs_list:
        rpath = os.path.join(base, finger, reg)
        imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
        imgs = []
        for p in imgs_paths:
            img = load_img(p).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            imgs.append(img)
        finger_data[finger][reg] = imgs

n_classes = len(fingers)
finger_labels = {f: i for i, f in enumerate(fingers)}
print(f"Loaded. Classes={n_classes}")

def fast_template(imgs, n=30):
    """快速模板: 随机选帧 -> 极性校正 -> 平均"""
    chosen = random.sample(range(len(imgs)), min(n, len(imgs)))
    selected = [imgs[i].copy() for i in chosen]
    ref = selected[0].flatten()
    aligned = [selected[0]]
    for img in selected[1:]:
        corr = np.corrcoef(ref, img.flatten())[0, 1]
        aligned.append(-img if corr < 0 else img)
    return np.mean(aligned, axis=0).astype(np.float32)

def build_multi_reg(finger, img_dict, n=25, train=True):
    """5寄存器各建模板, 堆叠为5通道"""
    channels = []
    for reg in regs_list:
        imgs = img_dict[reg][:70] if train else img_dict[reg][70:]
        channels.append(fast_template(imgs, n))
    return np.stack(channels, axis=0)  # (5, H, W)

class MultiRegDataset(Dataset):
    def __init__(self, finger_data, labels, n_template=25, n_samples=120):
        self.finger_data = finger_data
        self.labels = labels
        self.n_template = n_template
        self.n_samples = n_samples
        self.fingers = list(finger_data.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        mt = build_multi_reg(finger, self.finger_data[finger], self.n_template, train=True)
        if random.random() < 0.5:
            mt = -mt
        mt += np.random.randn(*mt.shape).astype(np.float32) * 0.02
        dx, dy = random.randint(-1, 1), random.randint(-1, 1)
        mt = np.roll(np.roll(mt, dx, axis=2), dy, axis=1)
        return torch.tensor(mt, dtype=torch.float32), self.labels[finger]

class MultiRegCNN(nn.Module):
    def __init__(self, in_ch=5, embed_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.embed = nn.Linear(512*4, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.bn(self.embed(x))

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = MultiRegCNN(in_ch=5, embed_dim=256).to(device)
arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)
optimizer = optim.AdamW(list(model.parameters()) + list(arcface.parameters()), lr=0.0005, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

train_ds = MultiRegDataset(finger_data, finger_labels, n_template=25, n_samples=120)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

print("Training...")
for epoch in range(100):
    model.train(); arcface.train()
    tl, cor, tot = 0, 0, 0
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        emb = model(X)
        loss = arcface(emb, Y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tl += loss.item()
        with torch.no_grad():
            _, p = torch.mm(F.normalize(emb), F.normalize(arcface.weight).t()).max(1)
            cor += p.eq(Y).sum().item(); tot += Y.size(0)
    scheduler.step()
    if (epoch+1) % 25 == 0:
        print(f"  Epoch {epoch+1}: loss={tl/len(train_loader):.4f}, acc={cor/tot*100:.1f}%")

# === 测试 ===
model.eval()
print("\n=== Testing ===")

def get_emb(mt_np):
    t = torch.tensor(mt_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return F.normalize(model(t)).cpu().numpy().flatten()

K = 15
enroll_embs, verify_embs = {}, {}
print("Building templates...")
for finger in fingers:
    el, vl = [], []
    for _ in range(K):
        el.append(get_emb(build_multi_reg(finger, finger_data[finger], 25, train=True)))
        vl.append(get_emb(build_multi_reg(finger, finger_data[finger], 20, train=False)))
    enroll_embs[finger] = el
    verify_embs[finger] = vl
print("Done building templates.")

def topk_avg(e, v, k=5):
    sims = sorted([np.dot(a, b) for a in e for b in v], reverse=True)
    return np.mean(sims[:k])

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

    print(f"\n--- Top-{k_val} (Z-norm) ---")
    print(f"Genuine: mean={np.mean(gz):.3f}, min={np.min(gz):.3f}")
    print(f"Impostor: mean={np.mean(iz):.3f}, max={np.max(iz):.3f}")
    print(f"EER: {eer*100:.2f}%")
    if np.min(gz) > np.max(iz):
        print("*** PERFECT SEPARATION ***")
    else:
        ov = sum(1 for x in iz if x > np.min(gz))
        print(f"Overlap: {ov}/{len(iz)}")
    for tf in [0.0, 0.03, 0.05, 0.10, 0.20]:
        idx = np.argmin(np.abs(fnr - tf))
        print(f"  FFR={tf*100:.0f}% -> FAR={fpr[idx]*100:.4f}%")
    idx_far = np.argmin(np.abs(fpr - 0.00002))
    print(f"  FAR=0.002% -> FFR={fnr[idx_far]*100:.2f}%")

print("\nDone.")
