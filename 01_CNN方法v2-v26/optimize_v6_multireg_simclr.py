"""
多寄存器 + SimCLR预训练 + ArcFace微调 (修复版, 无数据泄露)
核心改进:
1. 每指使用所有可用寄存器, 从每个寄存器采帧组成多通道输入
2. 跨寄存器对比学习: 同指不同寄存器组合 = 正样本对
3. 严格训练/测试分离
"""
import os, glob, random, time
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
# 数据加载: 每指加载所有可用寄存器
# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'

# 无贴屏寄存器列表
wtp_regs = ['Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245', 'Rgd1247']
# 不贴屏寄存器列表
btp_regs = ['Rgd1237', 'Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245']

N_REGS = 5  # 每指5个寄存器通道

finger_train = {}  # {finger_key: {reg: [frames]}}
finger_test = {}
finger_labels = {}
finger_source = {}
finger_regs = {}  # 每指使用的寄存器列表
fi = 0

print("Loading 无贴屏 (5 registers)...")
for finger in sorted(os.listdir(base1)):
    key = f"wtp_{finger}"
    finger_train[key] = {}
    finger_test[key] = {}
    for reg in wtp_regs:
        rpath = os.path.join(base1, finger, reg)
        imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
        loaded = []
        for p in imgs_paths:
            img = load_img(p).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            loaded.append(img)
        finger_train[key][reg] = loaded[:70]
        finger_test[key][reg] = loaded[70:]
    finger_labels[key] = fi
    finger_source[key] = "无贴屏"
    finger_regs[key] = wtp_regs
    fi += 1

n_wtp = fi
print(f"  无贴屏: {n_wtp} classes x {len(wtp_regs)} registers")

print("Loading 不贴屏 (5 registers)...")
for finger in sorted(os.listdir(base2)):
    key = f"btp_{finger}"
    # 检查是否至少有主寄存器
    primary_path = os.path.join(base2, finger, 'Rgd1237')
    if not os.path.exists(primary_path):
        print(f"  SKIP {finger}")
        continue
    imgs_check = sorted(glob.glob(os.path.join(primary_path, '*.bmp')))
    if len(imgs_check) < 50:
        print(f"  SKIP {finger} (only {len(imgs_check)} imgs)")
        continue

    finger_train[key] = {}
    finger_test[key] = {}
    for reg in btp_regs:
        rpath = os.path.join(base2, finger, reg)
        if not os.path.exists(rpath):
            # 如果某个寄存器不存在, 用零填充
            finger_train[key][reg] = []
            finger_test[key][reg] = []
            continue
        imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
        loaded = []
        for p in imgs_paths:
            img = load_img(p).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            loaded.append(img)
        finger_train[key][reg] = loaded[:70]
        finger_test[key][reg] = loaded[70:]
    finger_labels[key] = fi
    finger_source[key] = "不贴屏"
    finger_regs[key] = btp_regs
    fi += 1

n_classes = fi
fingers = list(finger_train.keys())
print(f"Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes - n_wtp} 不贴屏)")

# ============================================================
# 工具函数: 从多寄存器采帧构建多通道输入
# ============================================================
def sample_multireg_frames(finger_reg_data, regs, n_frames_per_reg=5):
    """
    从每个寄存器采n_frames帧, 对每个寄存器帧组做极性校正+平均生成单通道
    最终输出: (N_REGS, H, W) = 5通道模板
    """
    channels = []
    for reg in regs:
        imgs = finger_reg_data.get(reg, [])
        if len(imgs) < 3:
            # 寄存器数据不足, 用零通道
            if channels:
                channels.append(np.zeros_like(channels[0]))
            else:
                channels.append(None)  # 需要后续处理
            continue

        chosen = random.sample(range(len(imgs)), min(n_frames_per_reg, len(imgs)))
        selected = [imgs[i].copy() for i in chosen]

        # 极性校正
        ref = selected[0].flatten()
        aligned = [selected[0]]
        for img in selected[1:]:
            corr = np.corrcoef(ref, img.flatten())[0, 1]
            aligned.append(-img if corr < 0 else img)

        template = np.mean(aligned, axis=0).astype(np.float32)
        channels.append(template)

    # 处理None通道
    valid = [c for c in channels if c is not None]
    if valid:
        shape = valid[0].shape
        channels = [c if c is not None else np.zeros(shape, dtype=np.float32) for c in channels]

    return np.stack(channels, axis=0)  # (5, H, W)

def sample_raw_frames(finger_reg_data, regs, n_frames_per_reg=5):
    """
    采帧但不做平均, 返回所有帧用于Set Transformer
    输出: (N_REGS * n_frames_per_reg, H, W)
    """
    all_frames = []
    for reg in regs:
        imgs = finger_reg_data.get(reg, [])
        if len(imgs) < 3:
            continue
        chosen = random.sample(range(len(imgs)), min(n_frames_per_reg, len(imgs)))
        for i in chosen:
            all_frames.append(imgs[i].copy())
    return np.stack(all_frames, axis=0) if all_frames else None

# ============================================================
# 方案A: 多寄存器模板 + SimCLR + ArcFace (5通道CNN)
# ============================================================
print("\n" + "="*60)
print("方案A: 多寄存器5通道模板 + SimCLR + ArcFace")
print("="*60)

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
        self.embed = nn.Linear(512 * 4, embed_dim)
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

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.1):
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

# --- SimCLR预训练数据集 (跨寄存器对比) ---
class MultiRegContrastiveDataset(Dataset):
    """
    正样本对: 同一指的两次不同采样(不同帧组 + 不同寄存器采样组合)
    """
    def __init__(self, finger_train, finger_regs, n_frames_per_reg=5, n_samples=200):
        self.finger_train = finger_train
        self.finger_regs = finger_regs
        self.n_fpr = n_frames_per_reg
        self.n_samples = n_samples
        self.fingers = list(finger_train.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        regs = self.finger_regs[finger]

        # 两次独立采样 -> 两个5通道模板
        t1 = sample_multireg_frames(self.finger_train[finger], regs, self.n_fpr)
        t2 = sample_multireg_frames(self.finger_train[finger], regs, self.n_fpr)

        # 随机极性翻转 (整体)
        if random.random() < 0.5: t1 = -t1
        if random.random() < 0.5: t2 = -t2
        # 噪声
        t1 += np.random.randn(*t1.shape).astype(np.float32) * 0.02
        t2 += np.random.randn(*t2.shape).astype(np.float32) * 0.02
        # 随机小位移
        dx, dy = random.randint(-1, 1), random.randint(-1, 1)
        t1 = np.roll(np.roll(t1, dx, axis=2), dy, axis=1)
        dx, dy = random.randint(-1, 1), random.randint(-1, 1)
        t2 = np.roll(np.roll(t2, dx, axis=2), dy, axis=1)

        return torch.tensor(t1, dtype=torch.float32), torch.tensor(t2, dtype=torch.float32)

# --- ArcFace微调数据集 ---
class MultiRegTrainDataset(Dataset):
    def __init__(self, finger_train, finger_regs, labels, n_fpr=5, n_samples=150):
        self.finger_train = finger_train
        self.finger_regs = finger_regs
        self.labels = labels
        self.n_fpr = n_fpr
        self.n_samples = n_samples
        self.fingers = list(finger_train.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        regs = self.finger_regs[finger]
        mt = sample_multireg_frames(self.finger_train[finger], regs, self.n_fpr)
        if random.random() < 0.5: mt = -mt
        mt += np.random.randn(*mt.shape).astype(np.float32) * 0.02
        dx, dy = random.randint(-1, 1), random.randint(-1, 1)
        mt = np.roll(np.roll(mt, dx, axis=2), dy, axis=1)
        return torch.tensor(mt, dtype=torch.float32), self.labels[finger]

# ============================================================
# 训练
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# --- Phase 1: SimCLR 预训练 ---
print("\nPhase 1: SimCLR Pre-training (multi-register, train only)...")
model = MultiRegCNN(in_ch=5, embed_dim=256).to(device)
simclr_loss = SimCLRLoss(temperature=0.1)
opt1 = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=100)

pretrain_ds = MultiRegContrastiveDataset(finger_train, finger_regs, n_frames_per_reg=8, n_samples=200)
pretrain_loader = DataLoader(pretrain_ds, batch_size=32, shuffle=True, num_workers=0)

t0 = time.time()
for epoch in range(100):
    model.train()
    tl = 0
    for t1, t2 in pretrain_loader:
        t1, t2 = t1.to(device), t2.to(device)
        z1 = F.normalize(model(t1), dim=1)
        z2 = F.normalize(model(t2), dim=1)
        loss = simclr_loss(z1, z2)
        opt1.zero_grad(); loss.backward(); opt1.step()
        tl += loss.item()
    sch1.step()
    if (epoch+1) % 10 == 0:
        print(f"  Pre-train epoch {epoch+1}: loss={tl/len(pretrain_loader):.4f}, time={time.time()-t0:.0f}s")

# --- Phase 2: ArcFace 微调 ---
print("\nPhase 2: ArcFace Fine-tuning...")
arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)
opt2 = optim.AdamW(
    list(model.parameters()) + list(arcface.parameters()),
    lr=0.0003, weight_decay=1e-3
)
sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100)

ft_ds = MultiRegTrainDataset(finger_train, finger_regs, finger_labels, n_fpr=8, n_samples=150)
ft_loader = DataLoader(ft_ds, batch_size=32, shuffle=True, num_workers=0)

t1_start = time.time()
for epoch in range(100):
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
        print(f"  Fine-tune epoch {epoch+1}: loss={tl/len(ft_loader):.4f}, acc={cor/tot*100:.1f}%, time={time.time()-t1_start:.0f}s")

# ============================================================
# 测试
# ============================================================
model.eval()
print("\n" + "="*60)
print("Testing (30 classes, multi-register, strict split)")
print("="*60)

def get_multireg_emb(finger_reg_data, regs, n_fpr=8):
    mt = sample_multireg_frames(finger_reg_data, regs, n_fpr)
    t = torch.tensor(mt, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return F.normalize(model(t)).cpu().numpy().flatten()

def topk_avg(e, v, k=5):
    sims = sorted([np.dot(a, b) for a in e for b in v], reverse=True)
    return np.mean(sims[:k])

K = 20
for n_fpr in [5, 8, 12, 15]:
    enroll_embs, verify_embs = {}, {}
    for finger in fingers:
        regs = finger_regs[finger]
        enroll_embs[finger] = [get_multireg_emb(finger_train[finger], regs, n_fpr) for _ in range(K)]
        verify_embs[finger] = [get_multireg_emb(finger_test[finger], regs, n_fpr) for _ in range(K)]

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
        print(f"\n  fpr={n_fpr}, top-{k_val}: EER={eer*100:.2f}%, gen_min={gmin:.3f}, imp_max={imax:.3f}, {sep}")

        if k_val in [3, 5]:
            for tf in [0.0, 0.03, 0.05, 0.10, 0.20]:
                idx = np.argmin(np.abs(fnr - tf))
                print(f"    FFR={tf*100:.0f}% -> FAR={fpr[idx]*100:.4f}%")
            idx_far = np.argmin(np.abs(fpr - 0.00002))
            print(f"    FAR=0.002% -> FFR={fnr[idx_far]*100:.2f}%")

# 分组分析
print("\n" + "="*60)
print("分组分析")
print("="*60)
wtp_fingers = [f for f in fingers if f.startswith("wtp_")]
btp_fingers = [f for f in fingers if f.startswith("btp_")]

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
    gmin_g = np.min(gz_g)
    imax_g = np.max(iz_g)
    sep_g = "PERFECT" if gmin_g > imax_g else f"overlap={sum(1 for x in iz_g if x > gmin_g)}/{len(iz_g)}"
    print(f"\n  {group_name} ({len(group_fingers)}类): EER={eer_g*100:.2f}%, gen_min={gmin_g:.3f}, imp_max={imax_g:.3f}, {sep_g}")
    for tf in [0.0, 0.03, 0.05, 0.10]:
        idx = np.argmin(np.abs(fnr_g - tf))
        print(f"    FFR={tf*100:.0f}% -> FAR={fpr_g[idx]*100:.4f}%")

print("\n" + "="*60)
print("All done. (Multi-register + SimCLR, no data leakage)")
print("="*60)
