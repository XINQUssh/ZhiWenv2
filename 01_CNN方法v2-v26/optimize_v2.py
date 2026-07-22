"""
综合优化方案: 在数据质量不变的前提下, 尽可能提升 FFR/FAR
核心思路:
  1. 多寄存器融合 (5个Rgd作为5通道)
  2. 质量加权模板 (按帧间一致性加权)
  3. 多模板注册 + Score融合
  4. ArcFace + 大margin
  5. 频域特征通道
"""
import os, glob, random, math, sys
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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 数据加载: 无贴屏 全部5个寄存器
# ============================================================
base = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
fingers = sorted(os.listdir(base))
regs_list = ['Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245', 'Rgd1247']

print(f"Loading {len(fingers)} fingers x {len(regs_list)} registers...")

# finger_data[finger][reg] = list of (H,W) float32 images
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

# ============================================================
# 质量加权模板构建
# ============================================================
def quality_weighted_template(imgs, n_select=40):
    """
    1. 计算每帧与相邻帧的相关性作为质量分
    2. 极性校正
    3. 按质量分加权平均
    """
    if len(imgs) < 5:
        return np.mean(imgs, axis=0)

    # 质量分
    scores = np.zeros(len(imgs))
    for i in range(len(imgs)):
        neighbors = []
        for j in [i-2, i-1, i+1, i+2]:
            if 0 <= j < len(imgs):
                c = abs(np.corrcoef(imgs[i].flatten(), imgs[j].flatten())[0, 1])
                neighbors.append(c)
        scores[i] = np.mean(neighbors) if neighbors else 0

    # 选top n_select帧
    top_idx = np.argsort(scores)[-n_select:]
    selected = [imgs[i].copy() for i in top_idx]
    weights = scores[top_idx]

    # 极性校正
    ref = selected[0].flatten()
    aligned = [selected[0]]
    aligned_w = [weights[0]]
    for k in range(1, len(selected)):
        corr = np.corrcoef(ref, selected[k].flatten())[0, 1]
        if corr < 0:
            aligned.append(-selected[k])
        else:
            aligned.append(selected[k])
        aligned_w.append(weights[k])

    # 加权平均
    aligned_w = np.array(aligned_w)
    aligned_w = aligned_w / (aligned_w.sum() + 1e-8)
    template = np.zeros_like(aligned[0])
    for k in range(len(aligned)):
        template += aligned_w[k] * aligned[k]

    return template

# ============================================================
# 多寄存器模板: 5个寄存器各自建模板, 堆叠为5通道
# ============================================================
def build_multi_reg_template(finger, imgs_dict, n_select=30, rand_subset=True):
    """返回 (5, H, W) 的多通道模板"""
    channels = []
    for reg in regs_list:
        reg_imgs = imgs_dict[reg]
        if rand_subset and len(reg_imgs) > n_select:
            chosen_idx = random.sample(range(len(reg_imgs)), n_select)
            subset = [reg_imgs[i] for i in chosen_idx]
        else:
            subset = reg_imgs
        template = quality_weighted_template(subset, n_select=min(n_select, len(subset)))
        channels.append(template)
    return np.stack(channels, axis=0)  # (5, H, W)

# ============================================================
# 加入频域特征通道
# ============================================================
def add_freq_channels(multi_template):
    """在5通道基础上加入频域特征, 输出 (10, H, W)"""
    freq_channels = []
    for ch in range(multi_template.shape[0]):
        img = multi_template[ch]
        # 功率谱
        fft = np.fft.fft2(img)
        power = np.log1p(np.abs(np.fft.fftshift(fft)))
        power = (power - power.mean()) / (power.std() + 1e-6)
        freq_channels.append(power)
    return np.concatenate([multi_template, np.stack(freq_channels, axis=0)], axis=0)

# ============================================================
# Dataset
# ============================================================
class MultiRegDataset(Dataset):
    def __init__(self, finger_data, finger_labels, regs,
                 n_template=30, n_samples=120, augment=True, use_freq=True):
        self.finger_data = finger_data
        self.labels = finger_labels
        self.regs = regs
        self.n_template = n_template
        self.n_samples = n_samples
        self.augment = augment
        self.use_freq = use_freq
        self.fingers = list(finger_data.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]

        # 构建多寄存器模板
        train_imgs = {}
        for reg in self.regs:
            all_imgs = self.finger_data[finger][reg]
            # 训练时用前70帧
            train_imgs[reg] = all_imgs[:70]

        mt = build_multi_reg_template(finger, train_imgs,
                                       n_select=self.n_template, rand_subset=True)

        if self.use_freq:
            mt = add_freq_channels(mt)  # (10, H, W)

        if self.augment:
            # 极性翻转
            if random.random() < 0.5:
                mt = -mt
            # 噪声
            mt += np.random.randn(*mt.shape).astype(np.float32) * 0.02
            # 平移
            dx, dy = random.randint(-2, 2), random.randint(-2, 2)
            mt = np.roll(np.roll(mt, dx, axis=2), dy, axis=1)

        return torch.tensor(mt, dtype=torch.float32), self.labels[finger]

# ============================================================
# CNN: 多通道输入
# ============================================================
class MultiRegCNN(nn.Module):
    def __init__(self, in_channels=10, embed_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
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
        self.embed = nn.Linear(512 * 4, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.bn(self.embed(x))

# ============================================================
# ArcFace
# ============================================================
class ArcFace(nn.Module):
    def __init__(self, embed_dim, n_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels):
        emb_norm = F.normalize(emb, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cos_theta = torch.mm(emb_norm, w_norm.t()).clamp(-1+1e-7, 1-1e-7)
        theta = torch.acos(cos_theta)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = torch.cos(theta + one_hot * self.m) * self.s
        return F.cross_entropy(output, labels)

# ============================================================
# 训练
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = MultiRegCNN(in_channels=10, embed_dim=256).to(device)
arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)

optimizer = optim.AdamW(
    list(model.parameters()) + list(arcface.parameters()),
    lr=0.0005, weight_decay=1e-3
)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)

train_ds = MultiRegDataset(finger_data, finger_labels, regs_list,
                           n_template=30, n_samples=150, augment=True, use_freq=True)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

print("Training multi-register ArcFace CNN...")
for epoch in range(150):
    model.train()
    arcface.train()
    total_loss, correct, total = 0, 0, 0
    for X_b, Y_b in train_loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        emb = model(X_b)
        loss = arcface(emb, Y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            emb_n = F.normalize(emb, dim=1)
            w_n = F.normalize(arcface.weight, dim=1)
            _, pred = torch.mm(emb_n, w_n.t()).max(1)
            correct += pred.eq(Y_b).sum().item()
            total += Y_b.size(0)
    scheduler.step()
    if (epoch + 1) % 30 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, acc={correct/total*100:.1f}%")

# ============================================================
# 测试: 多模板Score融合 + Z-norm
# ============================================================
model.eval()
print("\n=== Testing with multi-template score fusion ===")

def make_test_template(finger, n_select=20, use_freq=True):
    """从测试帧(后30帧)构建模板"""
    test_imgs = {}
    for reg in regs_list:
        test_imgs[reg] = finger_data[finger][reg][70:]  # 后30帧
    mt = build_multi_reg_template(finger, test_imgs, n_select=min(n_select, 25), rand_subset=True)
    if use_freq:
        mt = add_freq_channels(mt)
    return mt

def get_embedding(template_np):
    t = torch.tensor(template_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(t)
    return F.normalize(emb, dim=1).cpu().numpy().flatten()

# 每指生成 K 个模板嵌入
K_ENROLL = 10  # 注册模板数
K_VERIFY = 10  # 验证模板数

enroll_embs = {}
verify_embs = {}

for finger in fingers:
    # 注册: 从训练帧建模板
    e_list = []
    for _ in range(K_ENROLL):
        train_imgs = {reg: finger_data[finger][reg][:70] for reg in regs_list}
        mt = build_multi_reg_template(finger, train_imgs, n_select=30, rand_subset=True)
        mt = add_freq_channels(mt)
        e_list.append(get_embedding(mt))
    enroll_embs[finger] = e_list

    # 验证: 从测试帧建模板
    v_list = []
    for _ in range(K_VERIFY):
        mt = make_test_template(finger, n_select=20, use_freq=True)
        v_list.append(get_embedding(mt))
    verify_embs[finger] = v_list

# Score融合策略: 注册K个模板, 验证时取与所有注册模板相似度的最大值
def fused_score(enroll_list, verify_list):
    """多对多匹配, 取最大相似度"""
    max_sim = -1
    for e in enroll_list:
        for v in verify_list:
            sim = np.dot(e, v)
            if sim > max_sim:
                max_sim = sim
    return max_sim

def avg_fused_score(enroll_list, verify_list):
    """多对多匹配, 取平均相似度"""
    sims = []
    for e in enroll_list:
        for v in verify_list:
            sims.append(np.dot(e, v))
    return np.mean(sims)

def topk_avg_score(enroll_list, verify_list, k=3):
    """多对多匹配, 取top-k平均"""
    sims = []
    for e in enroll_list:
        for v in verify_list:
            sims.append(np.dot(e, v))
    sims.sort(reverse=True)
    return np.mean(sims[:k])

# 对比不同融合策略
for strategy_name, score_fn in [
    ("Max fusion", fused_score),
    ("Avg fusion", avg_fused_score),
    ("Top3 avg", lambda e, v: topk_avg_score(e, v, 3)),
    ("Top5 avg", lambda e, v: topk_avg_score(e, v, 5)),
]:
    genuine, impostor = [], []

    # Genuine: 同指 enroll vs verify
    for finger in fingers:
        s = score_fn(enroll_embs[finger], verify_embs[finger])
        genuine.append(s)

    # Impostor: 不同指
    for i in range(len(fingers)):
        for j in range(i + 1, len(fingers)):
            s = score_fn(enroll_embs[fingers[i]], verify_embs[fingers[j]])
            impostor.append(s)

    # Z-norm
    imp_mean = np.mean(impostor)
    imp_std = np.std(impostor)
    genuine_z = [(g - imp_mean) / (imp_std + 1e-8) for g in genuine]
    impostor_z = [(x - imp_mean) / (imp_std + 1e-8) for x in impostor]

    labels = [1] * len(genuine_z) + [0] * len(impostor_z)
    scores = genuine_z + impostor_z
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    gen_min = np.min(genuine_z)
    imp_max = np.max(impostor_z)

    print(f"\n--- {strategy_name} (Z-normed) ---")
    print(f"Genuine: mean={np.mean(genuine_z):.3f}, min={gen_min:.3f}")
    print(f"Impostor: mean={np.mean(impostor_z):.3f}, max={imp_max:.3f}")
    print(f"EER: {eer*100:.2f}%")

    if gen_min > imp_max:
        print("*** PERFECT SEPARATION ***")
    else:
        overlap_g = sum(1 for x in genuine_z if x < imp_max)
        overlap_i = sum(1 for x in impostor_z if x > gen_min)
        print(f"Overlap: genuine<imp_max: {overlap_g}/{len(genuine_z)}, impostor>gen_min: {overlap_i}/{len(impostor_z)}")

    for tfnr in [0.0, 0.03, 0.05, 0.10, 0.20]:
        idx = np.argmin(np.abs(fnr - tfnr))
        print(f"  FFR={tfnr*100:.0f}% -> FAR={fpr[idx]*100:.4f}%")

    idx_far = np.argmin(np.abs(fpr - 0.00002))
    print(f"  FAR=0.002% -> FFR={fnr[idx_far]*100:.2f}%")

# ============================================================
# 额外: 单通道模板CNN对比 (只用Rgd1245)
# ============================================================
print("\n\n=== Comparison: Single-reg (Rgd1245) with same fusion ===")
genuine_single, impostor_single = [], []

# 用单寄存器重新建模板
single_enroll = {}
single_verify = {}
for finger in fingers:
    e_list = []
    for _ in range(K_ENROLL):
        imgs = finger_data[finger]['Rgd1245'][:70]
        chosen = random.sample(range(len(imgs)), 30)
        selected = [imgs[i].copy() for i in chosen]
        ref = selected[0].flatten()
        aligned = [selected[0]]
        for img in selected[1:]:
            corr = np.corrcoef(ref, img.flatten())[0, 1]
            aligned.append(-img if corr < 0 else img)
        template = np.mean(aligned, axis=0).astype(np.float32)
        # 扩展为10通道 (复制 + FFT)
        mt = np.stack([template] * 5, axis=0)
        mt = add_freq_channels(mt)
        e_list.append(get_embedding(mt))
    single_enroll[finger] = e_list

    v_list = []
    for _ in range(K_VERIFY):
        imgs = finger_data[finger]['Rgd1245'][70:]
        chosen = random.sample(range(len(imgs)), min(20, len(imgs)))
        selected = [imgs[i].copy() for i in chosen]
        ref = selected[0].flatten()
        aligned = [selected[0]]
        for img in selected[1:]:
            corr = np.corrcoef(ref, img.flatten())[0, 1]
            aligned.append(-img if corr < 0 else img)
        template = np.mean(aligned, axis=0).astype(np.float32)
        mt = np.stack([template] * 5, axis=0)
        mt = add_freq_channels(mt)
        v_list.append(get_embedding(mt))
    single_verify[finger] = v_list

for finger in fingers:
    genuine_single.append(topk_avg_score(single_enroll[finger], single_verify[finger], 5))
for i in range(len(fingers)):
    for j in range(i + 1, len(fingers)):
        impostor_single.append(topk_avg_score(single_enroll[fingers[i]], single_verify[fingers[j]], 5))

imp_m = np.mean(impostor_single)
imp_s = np.std(impostor_single)
genuine_sz = [(g - imp_m) / (imp_s + 1e-8) for g in genuine_single]
impostor_sz = [(x - imp_m) / (imp_s + 1e-8) for x in impostor_single]

labels2 = [1] * len(genuine_sz) + [0] * len(impostor_sz)
scores2 = genuine_sz + impostor_sz
fpr2, tpr2, _ = roc_curve(labels2, scores2)
fnr2 = 1 - tpr2
eer2_idx = np.nanargmin(np.abs(fnr2 - fpr2))
eer2 = (fpr2[eer2_idx] + fnr2[eer2_idx]) / 2
print(f"Single-reg EER: {eer2*100:.2f}%")
for tfnr in [0.0, 0.03, 0.05, 0.10]:
    idx = np.argmin(np.abs(fnr2 - tfnr))
    print(f"  FFR={tfnr*100:.0f}% -> FAR={fpr2[idx]*100:.4f}%")

print("\nDone.")
