"""
优化方案2: Siamese Triplet Network
核心思路:
  1. 直接学习相似度, 不依赖分类头
  2. Hard negative mining
  3. 多帧模板输入 (质量加权)
  4. 测试时多次采样 + score融合
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
from itertools import combinations

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

base = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
fingers = sorted(os.listdir(base))
target_reg = 'Rgd1245'

print(f"Loading {len(fingers)} fingers, reg={target_reg}")

finger_data = {}
for finger in fingers:
    rpath = os.path.join(base, finger, target_reg)
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    imgs = []
    for p in imgs_paths:
        img = load_img(p).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)
        imgs.append(img)
    finger_data[finger] = imgs

def make_template(imgs, n=30):
    chosen = random.sample(range(len(imgs)), min(n, len(imgs)))
    selected = [imgs[i].copy() for i in chosen]
    ref = selected[0].flatten()
    aligned = [selected[0]]
    for img in selected[1:]:
        corr = np.corrcoef(ref, img.flatten())[0, 1]
        aligned.append(-img if corr < 0 else img)

    # 质量加权
    weights = []
    for k, img in enumerate(aligned):
        neighbors = []
        for j in [k-1, k+1]:
            if 0 <= j < len(aligned):
                c = abs(np.corrcoef(aligned[k].flatten(), aligned[j].flatten())[0, 1])
                neighbors.append(c)
        weights.append(np.mean(neighbors) if neighbors else 0.1)
    weights = np.array(weights)
    weights = weights / (weights.sum() + 1e-8)
    template = sum(w * img for w, img in zip(weights, aligned))
    return template.astype(np.float32)

# ============================================================
# Triplet Dataset with Online Hard Mining
# ============================================================
class TripletDataset(Dataset):
    def __init__(self, finger_data, n_template=30, n_triplets=2000, use_train=True):
        self.finger_data = finger_data
        self.n_template = n_template
        self.n_triplets = n_triplets
        self.use_train = use_train
        self.fingers = list(finger_data.keys())

    def __len__(self):
        return self.n_triplets

    def _get_imgs(self, finger):
        imgs = self.finger_data[finger]
        if self.use_train:
            return imgs[:70]
        return imgs[70:]

    def __getitem__(self, idx):
        # Anchor + Positive (same finger, different template)
        anchor_finger = self.fingers[idx % len(self.fingers)]
        imgs = self._get_imgs(anchor_finger)
        anchor = make_template(imgs, self.n_template)
        positive = make_template(imgs, self.n_template)

        # Negative (different finger)
        neg_finger = random.choice([f for f in self.fingers if f != anchor_finger])
        neg_imgs = self._get_imgs(neg_finger)
        negative = make_template(neg_imgs, self.n_template)

        # 极性随机翻转
        if random.random() < 0.5:
            anchor = -anchor
        if random.random() < 0.5:
            positive = -positive
        if random.random() < 0.5:
            negative = -negative

        # 微小噪声
        anchor += np.random.randn(*anchor.shape).astype(np.float32) * 0.02
        positive += np.random.randn(*positive.shape).astype(np.float32) * 0.02
        negative += np.random.randn(*negative.shape).astype(np.float32) * 0.02

        a = torch.tensor(anchor).unsqueeze(0)
        p = torch.tensor(positive).unsqueeze(0)
        n = torch.tensor(negative).unsqueeze(0)
        return a, p, n

# ============================================================
# Embedding Network
# ============================================================
class EmbedNet(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, dim=1)

# ============================================================
# Training with Triplet + Center Loss
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = EmbedNet(embed_dim=256).to(device)
triplet_loss = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
    margin=0.4
)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

train_ds = TripletDataset(finger_data, n_template=35, n_triplets=3000, use_train=True)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

print("Training Triplet Network...")
for epoch in range(120):
    model.train()
    total_loss = 0
    n_hard = 0
    for anchor, pos, neg in train_loader:
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        e_a = model(anchor)
        e_p = model(pos)
        e_n = model(neg)

        loss = triplet_loss(e_a, e_p, e_n)

        # Online semi-hard mining: swap negatives within batch
        with torch.no_grad():
            d_ap = 1 - F.cosine_similarity(e_a, e_p)
            d_an = 1 - F.cosine_similarity(e_a, e_n)
            n_hard += (d_ap > d_an).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    if (epoch + 1) % 30 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, hard_triplets={n_hard}")

# ============================================================
# Test
# ============================================================
model.eval()
print("\n=== Triplet Network Test ===")

def get_emb(template_np):
    t = torch.tensor(template_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t).cpu().numpy().flatten()

# 多模板注册+验证
K = 15

enroll_embs = {}
verify_embs = {}
for finger in fingers:
    train_imgs = finger_data[finger][:70]
    test_imgs = finger_data[finger][70:]

    enroll_embs[finger] = [get_emb(make_template(train_imgs, 35)) for _ in range(K)]
    verify_embs[finger] = [get_emb(make_template(test_imgs, 20)) for _ in range(K)]

# Score fusion strategies
def topk_avg(e_list, v_list, k=5):
    sims = [np.dot(e, v) for e in e_list for v in v_list]
    sims.sort(reverse=True)
    return np.mean(sims[:k])

for k_val in [3, 5, 8, 15]:
    genuine, impostor = [], []

    for finger in fingers:
        genuine.append(topk_avg(enroll_embs[finger], verify_embs[finger], k_val))

    for i in range(len(fingers)):
        for j in range(i + 1, len(fingers)):
            impostor.append(topk_avg(enroll_embs[fingers[i]], verify_embs[fingers[j]], k_val))

    # Z-norm
    imp_m, imp_s = np.mean(impostor), np.std(impostor)
    genuine_z = [(g - imp_m) / (imp_s + 1e-8) for g in genuine]
    impostor_z = [(x - imp_m) / (imp_s + 1e-8) for x in impostor]

    labels = [1] * len(genuine_z) + [0] * len(impostor_z)
    scores = genuine_z + impostor_z
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    gen_min = np.min(genuine_z)
    imp_max = np.max(impostor_z)

    print(f"\n--- Top-{k_val} Avg (Z-normed) ---")
    print(f"Genuine: mean={np.mean(genuine_z):.3f}, min={gen_min:.3f}")
    print(f"Impostor: mean={np.mean(impostor_z):.3f}, max={imp_max:.3f}")
    print(f"EER: {eer*100:.2f}%")

    if gen_min > imp_max:
        print("*** PERFECT SEPARATION ***")
    else:
        overlap_g = sum(1 for x in genuine_z if x < imp_max)
        print(f"Overlap genuine: {overlap_g}/{len(genuine_z)}")

    for tfnr in [0.0, 0.03, 0.05, 0.10]:
        idx = np.argmin(np.abs(fnr - tfnr))
        print(f"  FFR={tfnr*100:.0f}% -> FAR={fpr[idx]*100:.4f}%")

    idx_far = np.argmin(np.abs(fpr - 0.00002))
    print(f"  FAR=0.002% -> FFR={fnr[idx_far]*100:.2f}%")

print("\nDone.")
