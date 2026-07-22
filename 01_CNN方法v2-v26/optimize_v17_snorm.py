"""
V17: Score Normalization + Quality Templates + SWA
核心改进 (相对V14):
  1. SWA (V15验证有效) → 更好泛化
  2. Label Smoothing (0.1) → 减少过拟合
  3. 质量模板选择 → 选最具代表性的20帧模板 (替代随机)
  4. 扩展TTA (4视角: 2极性 × 2位移) → 降低方差
  5. 分数归一化 (Z-norm, T-norm, S-norm) → 补偿探针/模板质量变异
架构: 与V14一致 (ResNet-18 + GAP + Linear+BN → 512维)
不做: Gabor/GeM/MultiScale/ConvNeXt (V16证明在小数据集上过拟合)
27类 (排除xzc)
"""
import os, glob, random, time, sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_curve
from torchvision.models import resnet18, ResNet18_Weights
from scipy import stats as sp_stats

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 预处理 (与V14一致)
# ============================================================
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
    mask_ratio = np.sum(mask > 0) / mask.size
    return result, mask_ratio

# ============================================================
# 数据加载 (27类, 排除xzc)
# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
WTP_REG = 'Rgd1245'
BTP_REG = 'Rgd1237'
SKIP_BTP = ['xzc']

finger_train = {}
finger_test = {}
finger_labels = {}
finger_source = {}
finger_train_quality = {}
finger_test_quality = {}
fi = 0

print(f"{'='*70}")
print(f"V17: Score Normalization + Quality Templates + SWA")
print(f"{'='*70}")
print(f"Loading data..."); sys.stdout.flush()

t_load = time.time()

for finger in sorted(os.listdir(base1)):
    key = f"wtp_{finger}"
    rpath = os.path.join(base1, finger, WTP_REG)
    if not os.path.exists(rpath):
        continue
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    train_frames, test_frames = [], []
    train_q, test_q = [], []
    for p in imgs_paths[:70]:
        img = load_img(p)
        if img is not None:
            frame, qr = preprocess_frame(img)
            train_frames.append(frame)
            train_q.append(qr)
    for p in imgs_paths[70:]:
        img = load_img(p)
        if img is not None:
            frame, qr = preprocess_frame(img)
            test_frames.append(frame)
            test_q.append(qr)
    if train_frames and test_frames:
        finger_train[key] = train_frames
        finger_test[key] = test_frames
        finger_labels[key] = fi
        finger_source[key] = "无贴屏"
        finger_train_quality[key] = train_q
        finger_test_quality[key] = test_q
        fi += 1

n_wtp = fi
print(f"  无贴屏: {n_wtp} classes, time={time.time()-t_load:.0f}s"); sys.stdout.flush()

for finger in sorted(os.listdir(base2)):
    if any(s in finger.lower() for s in SKIP_BTP):
        print(f"  SKIP {finger}")
        continue
    rpath = os.path.join(base2, finger, BTP_REG)
    if not os.path.exists(rpath):
        continue
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    if len(imgs_paths) < 50:
        continue
    key = f"btp_{finger}"
    train_frames, test_frames = [], []
    train_q, test_q = [], []
    for p in imgs_paths[:70]:
        img = load_img(p)
        if img is not None:
            frame, qr = preprocess_frame(img)
            train_frames.append(frame)
            train_q.append(qr)
    for p in imgs_paths[70:]:
        img = load_img(p)
        if img is not None:
            frame, qr = preprocess_frame(img)
            test_frames.append(frame)
            test_q.append(qr)
    if train_frames and test_frames:
        finger_train[key] = train_frames
        finger_test[key] = test_frames
        finger_labels[key] = fi
        finger_source[key] = "不贴屏"
        finger_train_quality[key] = train_q
        finger_test_quality[key] = test_q
        fi += 1

n_classes = fi
fingers = list(finger_train.keys())
print(f"  Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes-n_wtp} 不贴屏)")
print(f"Data loading done in {time.time()-t_load:.0f}s"); sys.stdout.flush()

# ============================================================
# 模型: ResNet-18 (与V14一致, 简单架构最适合小数据集)
# ============================================================
class FingerprintEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        orig_w = base.conv1.weight.data
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = orig_w.mean(dim=1, keepdim=True)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.projector(x)

# ============================================================
# Sub-center ArcFace (with label smoothing)
# ============================================================
class SubCenterArcFace(nn.Module):
    def __init__(self, embed_dim, n_classes, K=3, s=64.0, m=0.5, label_smoothing=0.1):
        super().__init__()
        self.s, self.m, self.K = s, m, K
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.FloatTensor(n_classes * K, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels):
        emb_n = F.normalize(emb, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        cos_all = torch.mm(emb_n, w_n.t()).clamp(-1+1e-7, 1-1e-7)
        cos_all = cos_all.view(-1, self.n_classes, self.K)
        cos, _ = cos_all.max(dim=2)
        theta = torch.acos(cos)
        oh = torch.zeros_like(cos)
        oh.scatter_(1, labels.view(-1, 1), 1)
        logits = torch.cos(theta + oh * self.m) * self.s
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

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
        sim = sim.masked_fill(mask, -1e4)
        return F.cross_entropy(sim, labels)

# ============================================================
# 数据增强 (V14增强 + 轻度Random Erasing)
# ============================================================
def elastic_transform_light(image, alpha=10, sigma=3):
    h, w = image.shape
    dx = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma)
    dy = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)

def augment_frame(frame):
    f = frame.copy()
    if random.random() < 0.5:
        f = -f
    f += np.random.randn(*f.shape).astype(np.float32) * 0.03
    if random.random() < 0.3:
        f = f[:, ::-1].copy()
    if random.random() < 0.5:
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        f = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), borderValue=0)
    if random.random() < 0.3:
        angle = random.uniform(-3, 3)
        h, w = f.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        f = cv2.warpAffine(f, M, (w, h), borderValue=0)
    if random.random() < 0.25:
        f = elastic_transform_light(f)
    # Random Erasing (15%, 2-6% area) - lighter than V16
    if random.random() < 0.15:
        h, w = f.shape
        area = h * w
        erase_area = random.uniform(0.02, 0.06) * area
        aspect = random.uniform(0.5, 2.0)
        eh = int(np.sqrt(erase_area * aspect))
        ew = int(np.sqrt(erase_area / aspect))
        eh = min(eh, h - 1); ew = min(ew, w - 1)
        if eh > 0 and ew > 0:
            top = random.randint(0, h - eh)
            left = random.randint(0, w - ew)
            f[top:top+eh, left:left+ew] = 0
    return f

# ============================================================
# 数据集
# ============================================================
class ContrastiveDS(Dataset):
    def __init__(self, finger_data, n_samples=120):
        self.finger_data = finger_data
        self.n_samples = n_samples
        self.fingers = list(finger_data.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        frames = self.finger_data[finger]
        i1, i2 = random.sample(range(len(frames)), 2)
        f1 = augment_frame(frames[i1])
        f2 = augment_frame(frames[i2])
        return (torch.tensor(f1, dtype=torch.float32).unsqueeze(0),
                torch.tensor(f2, dtype=torch.float32).unsqueeze(0))

class ClassifyDS(Dataset):
    def __init__(self, finger_data, finger_labels, n_samples=100):
        self.finger_data = finger_data
        self.finger_labels = finger_labels
        self.n_samples = n_samples
        self.fingers = list(finger_data.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        frames = self.finger_data[finger]
        f = augment_frame(frames[random.randint(0, len(frames)-1)])
        return (torch.tensor(f, dtype=torch.float32).unsqueeze(0),
                self.finger_labels[finger])

# ============================================================
# 训练: SimCLR → ArcFace+SWA (2阶段, 简洁有效)
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}"); sys.stdout.flush()

EMBED_DIM = 512
SEEDS = [42, 123, 777]
USE_AMP = True
N_TEMPLATES = 20
MODEL_SAVE_DIR = 'f:/1111/指纹/models_v17/'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

N_PRETRAIN = 80     # SimCLR
N_FINETUNE = 140    # ArcFace
SWA_START = 110     # SWA from epoch 110 (30 averages)

def train_single_model(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    # Check for saved model
    save_path = os.path.join(MODEL_SAVE_DIR, f'v17_seed{seed}.pth')
    if os.path.exists(save_path):
        print(f"\n  Found saved model: {save_path}")
        model = FingerprintEncoder(embed_dim=EMBED_DIM).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model.eval()
        print(f"  Loaded successfully."); sys.stdout.flush()
        return model

    print(f"\n{'='*60}")
    print(f"Training seed={seed}")
    print(f"  Phase1: SimCLR {N_PRETRAIN}ep | Phase2: ArcFace {N_FINETUNE}ep (SWA@{SWA_START})")
    print(f"{'='*60}"); sys.stdout.flush()

    model = FingerprintEncoder(embed_dim=EMBED_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}"); sys.stdout.flush()

    simclr_loss = SimCLRLoss(temperature=0.07)

    # ---- Phase 1: SimCLR (backbone frozen) ----
    print(f"  Phase 1: SimCLR ({N_PRETRAIN} epochs, backbone frozen)")
    for name, param in model.named_parameters():
        if 'projector' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable: {trainable:,}"); sys.stdout.flush()

    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=0.001, weight_decay=1e-3)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=N_PRETRAIN)
    scaler = GradScaler('cuda') if USE_AMP else None

    pre_ds = ContrastiveDS(finger_train, n_samples=120)
    pre_loader = DataLoader(pre_ds, batch_size=32, shuffle=True, num_workers=0)

    t0 = time.time()
    for epoch in range(N_PRETRAIN):
        model.train(); tl = 0
        for f1, f2 in pre_loader:
            f1, f2 = f1.to(device), f2.to(device)
            opt1.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    z1 = F.normalize(model(f1), dim=1)
                    z2 = F.normalize(model(f2), dim=1)
                    loss = simclr_loss(z1, z2)
                scaler.scale(loss).backward(); scaler.step(opt1); scaler.update()
            else:
                z1 = F.normalize(model(f1), dim=1)
                z2 = F.normalize(model(f2), dim=1)
                loss = simclr_loss(z1, z2)
                loss.backward(); opt1.step()
            tl += loss.item()
        sch1.step()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_PRETRAIN}: loss={tl/len(pre_loader):.4f}, "
                  f"time={time.time()-t0:.0f}s"); sys.stdout.flush()

    # ---- Phase 2: Sub-center ArcFace + SWA ----
    print(f"  Phase 2: ArcFace ({N_FINETUNE} epochs, SWA@{SWA_START}, label_smoothing=0.1)")
    sys.stdout.flush()

    for param in model.parameters():
        param.requires_grad = True

    arcface = SubCenterArcFace(EMBED_DIM, n_classes, K=3, s=64, m=0.5,
                               label_smoothing=0.1).to(device)

    param_groups = [
        {'params': list(model.conv1.parameters()) + list(model.bn1.parameters()) +
                   list(model.layer1.parameters()) + list(model.layer2.parameters()),
         'lr': 1e-5},
        {'params': list(model.layer3.parameters()), 'lr': 5e-5},
        {'params': list(model.layer4.parameters()), 'lr': 1e-4},
        {'params': list(model.projector.parameters()) + list(arcface.parameters()),
         'lr': 3e-4},
    ]

    opt2 = optim.AdamW(param_groups, weight_decay=1e-3)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=N_FINETUNE)
    scaler2 = GradScaler('cuda') if USE_AMP else None

    ft_ds = ClassifyDS(finger_train, finger_labels, n_samples=100)
    ft_loader = DataLoader(ft_ds, batch_size=32, shuffle=True, num_workers=0)

    from torch.optim.swa_utils import AveragedModel, update_bn
    swa_model = AveragedModel(model)
    swa_n = 0

    t1 = time.time()
    for epoch in range(N_FINETUNE):
        model.train(); arcface.train()
        tl, cor, tot = 0, 0, 0
        for X, Y in ft_loader:
            X, Y = X.to(device), Y.to(device)
            opt2.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    emb = model(X)
                    loss = arcface(emb, Y)
                scaler2.scale(loss).backward(); scaler2.step(opt2); scaler2.update()
            else:
                emb = model(X)
                loss = arcface(emb, Y)
                loss.backward(); opt2.step()
            tl += loss.item()
            with torch.no_grad():
                en = F.normalize(emb.float(), dim=1)
                wn = F.normalize(arcface.weight.float(), dim=1)
                cos_all = torch.mm(en, wn.t()).view(-1, n_classes, 3)
                cos, _ = cos_all.max(dim=2)
                _, p = cos.max(1)
                cor += p.eq(Y).sum().item(); tot += Y.size(0)
        sch2.step()

        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_n += 1

        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_FINETUNE}: loss={tl/len(ft_loader):.4f}, "
                  f"acc={cor/tot*100:.1f}%, time={time.time()-t1:.0f}s"
                  f"{' [SWA]' if epoch >= SWA_START else ''}"); sys.stdout.flush()

    if swa_n > 0:
        print(f"  Updating SWA BN ({swa_n} averages)..."); sys.stdout.flush()
        bn_ds = ClassifyDS(finger_train, finger_labels, n_samples=20)
        bn_loader = DataLoader(bn_ds, batch_size=32, shuffle=True, num_workers=0)
        update_bn(bn_loader, swa_model, device=device)
        model.load_state_dict(swa_model.module.state_dict())
        print(f"  SWA applied."); sys.stdout.flush()

    model.eval()
    total_time = time.time() - t0
    print(f"  Model done. Total: {total_time:.0f}s ({total_time/60:.1f}min)")

    torch.save(model.state_dict(), save_path)
    print(f"  Saved: {save_path}"); sys.stdout.flush()
    return model

# Train all models
models = []
t_total = time.time()
for seed in SEEDS:
    m = train_single_model(seed)
    models.append(m)
print(f"\nAll models ready in {time.time()-t_total:.0f}s"); sys.stdout.flush()

# ============================================================
# 嵌入提取 (4-view TTA: 2极性 × 2位移)
# ============================================================
@torch.no_grad()
def get_single_emb(models_list, frame):
    """Single frame embedding (no TTA), ensemble across models."""
    x = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    embs = []
    for model in models_list:
        emb = F.normalize(model(x), dim=1)
        embs.append(emb)
    avg = torch.mean(torch.stack(embs), dim=0)
    return F.normalize(avg, dim=1).cpu().numpy().flatten()

@torch.no_grad()
def get_tta_emb(models_list, frame):
    """4-view TTA: original + negated + shifted(+2,+2) + negated+shifted(+2,+2)."""
    h, w = frame.shape
    # Create shifted version (+2, +2 pixels)
    shifted = np.zeros_like(frame)
    shifted[2:, 2:] = frame[:h-2, :w-2]

    views = [frame, -frame, shifted, -shifted]
    embs = []
    for v in views:
        x = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        for model in models_list:
            emb = F.normalize(model(x), dim=1)
            embs.append(emb)
    avg = torch.mean(torch.stack(embs), dim=0)
    return F.normalize(avg, dim=1).cpu().numpy().flatten()

@torch.no_grad()
def get_dual_polarity_emb(models_list, frame):
    """2-view TTA: original + negated (V14 style)."""
    x_orig = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x_flip = torch.tensor(-frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    embs = []
    for model in models_list:
        embs.append(F.normalize(model(x_orig), dim=1))
        embs.append(F.normalize(model(x_flip), dim=1))
    avg = torch.mean(torch.stack(embs), dim=0)
    return F.normalize(avg, dim=1).cpu().numpy().flatten()

# ============================================================
# 质量模板选择 (替代随机选择)
# ============================================================
def select_quality_templates(models_list, frames, n_templates=20):
    """Select most representative frames by intra-class consistency."""
    if len(frames) <= n_templates:
        return list(range(len(frames)))

    # Compute embeddings for all frames
    embs = []
    for f in frames:
        embs.append(get_dual_polarity_emb(models_list, f))
    embs = np.array(embs)  # (n_frames, embed_dim)

    # Pairwise cosine similarity
    sim_matrix = embs @ embs.T  # Already L2-normalized

    # Average similarity to others (exclude self)
    n = len(embs)
    np.fill_diagonal(sim_matrix, 0)
    avg_sim = sim_matrix.sum(axis=1) / (n - 1)

    # Select top n_templates
    best_indices = np.argsort(avg_sim)[-n_templates:]
    return sorted(best_indices.tolist())

# ============================================================
# 评估框架 (with score normalization)
# ============================================================
print(f"\n{'='*70}")
print(f"Evaluating V17")
print(f"{'='*70}"); sys.stdout.flush()

# Step 1: Select quality templates
print("Step 1: Quality template selection..."); sys.stdout.flush()
t_sel = time.time()
quality_template_indices = {}
random_template_indices = {}
for finger in fingers:
    frames = finger_train[finger]
    # Quality selection
    qi = select_quality_templates(models, frames, N_TEMPLATES)
    quality_template_indices[finger] = qi
    # Random selection (for comparison)
    random.seed(42)
    ri = random.sample(range(len(frames)), min(N_TEMPLATES, len(frames)))
    random_template_indices[finger] = ri
print(f"  Template selection done in {time.time()-t_sel:.0f}s"); sys.stdout.flush()

# Step 2: Extract template embeddings (both quality and random, using TTA)
print("Step 2: Extracting template embeddings..."); sys.stdout.flush()
t_emb = time.time()

# Quality templates
quality_templates_emb = {}  # finger -> list of embeddings
for finger in fingers:
    frames = finger_train[finger]
    indices = quality_template_indices[finger]
    quality_templates_emb[finger] = [get_tta_emb(models, frames[i]) for i in indices]

# Random templates (for comparison)
random_templates_emb = {}
for finger in fingers:
    frames = finger_train[finger]
    indices = random_template_indices[finger]
    random_templates_emb[finger] = [get_tta_emb(models, frames[i]) for i in indices]

print(f"  Template embeddings done in {time.time()-t_emb:.0f}s"); sys.stdout.flush()

# Step 3: Z-norm parameters (from template-vs-template impostor scores)
print("Step 3: Computing Z-norm parameters..."); sys.stdout.flush()

def compute_znorm_params(templates_emb):
    """Compute Z-norm parameters for each identity using template impostor scores."""
    z_params = {}
    for identity_j in fingers:
        imp_scores = []
        for identity_k in fingers:
            if identity_k == identity_j:
                continue
            # Each template of identity_k acts as a "probe" against identity_j's templates
            for t_k in templates_emb[identity_k]:
                max_sim = max(np.dot(t_k, t_j) for t_j in templates_emb[identity_j])
                imp_scores.append(max_sim)
        imp_scores = np.array(imp_scores)
        z_params[identity_j] = (imp_scores.mean(), imp_scores.std() + 1e-8)
    return z_params

z_params_quality = compute_znorm_params(quality_templates_emb)
z_params_random = compute_znorm_params(random_templates_emb)
print(f"  Z-norm params computed."); sys.stdout.flush()

# Step 4: Extract test embeddings and compute ALL raw scores
print("Step 4: Computing raw match scores..."); sys.stdout.flush()
t_match = time.time()

# Store ALL raw scores for normalization
# Structure: list of (probe_finger, probe_idx, {identity: raw_score})
all_probe_scores = []

for fi_idx, probe_finger in enumerate(fingers):
    test_frames = finger_test[probe_finger]
    for probe_idx, frame in enumerate(test_frames):
        # Get probe embedding (with TTA)
        probe_emb = get_tta_emb(models, frame)

        # Compute score against ALL identities (quality templates)
        scores_q = {}
        for identity in fingers:
            max_sim = max(np.dot(probe_emb, t) for t in quality_templates_emb[identity])
            scores_q[identity] = max_sim

        # Also compute with random templates
        scores_r = {}
        for identity in fingers:
            max_sim = max(np.dot(probe_emb, t) for t in random_templates_emb[identity])
            scores_r[identity] = max_sim

        all_probe_scores.append({
            'probe_finger': probe_finger,
            'probe_idx': probe_idx,
            'scores_quality': scores_q,
            'scores_random': scores_r,
        })

    if (fi_idx+1) % 5 == 0 or fi_idx == 0:
        print(f"  [{fi_idx+1}/{n_classes}] elapsed={time.time()-t_match:.0f}s"); sys.stdout.flush()

print(f"  Matching done in {time.time()-t_match:.0f}s"); sys.stdout.flush()

# ============================================================
# Score Normalization Functions
# ============================================================
def apply_znorm(raw_score, identity, z_params):
    """Z-norm: normalize by template identity's impostor distribution."""
    mu, std = z_params[identity]
    return (raw_score - mu) / std

def apply_tnorm(raw_score, claimed_identity, probe_scores_dict):
    """T-norm: normalize by probe's impostor score distribution."""
    imp_scores = [probe_scores_dict[k] for k in probe_scores_dict if k != claimed_identity]
    t_mu = np.mean(imp_scores)
    t_std = np.std(imp_scores) + 1e-8
    return (raw_score - t_mu) / t_std

def apply_snorm(raw_score, claimed_identity, probe_scores_dict, z_params):
    """S-norm: average of Z-norm and T-norm."""
    z_score = apply_znorm(raw_score, claimed_identity, z_params)
    t_score = apply_tnorm(raw_score, claimed_identity, probe_scores_dict)
    return (z_score + t_score) / 2

# ============================================================
# Collect scores for all normalization variants
# ============================================================
print("\nStep 5: Applying score normalization variants..."); sys.stdout.flush()

def collect_scores(all_probes, template_key, z_params):
    """Collect genuine/impostor scores with all normalization variants."""
    results = {
        'raw': {'gen': [], 'imp': []},
        'znorm': {'gen': [], 'imp': []},
        'tnorm': {'gen': [], 'imp': []},
        'snorm': {'gen': [], 'imp': []},
    }
    per_finger_genuine = {finger: [] for finger in fingers}

    for probe in all_probes:
        probe_finger = probe['probe_finger']
        scores_dict = probe[template_key]

        for identity in fingers:
            raw_score = scores_dict[identity]
            z_score = apply_znorm(raw_score, identity, z_params)
            t_score = apply_tnorm(raw_score, identity, scores_dict)
            s_score = apply_snorm(raw_score, identity, scores_dict, z_params)

            if identity == probe_finger:
                # Genuine
                results['raw']['gen'].append(raw_score)
                results['znorm']['gen'].append(z_score)
                results['tnorm']['gen'].append(t_score)
                results['snorm']['gen'].append(s_score)
                per_finger_genuine[probe_finger].append(raw_score)
            else:
                # Impostor
                results['raw']['imp'].append(raw_score)
                results['znorm']['imp'].append(z_score)
                results['tnorm']['imp'].append(t_score)
                results['snorm']['imp'].append(s_score)

    for key in results:
        results[key]['gen'] = np.array(results[key]['gen'])
        results[key]['imp'] = np.array(results[key]['imp'])

    return results, per_finger_genuine

# Quality templates
results_quality, per_finger_gen_quality = collect_scores(
    all_probe_scores, 'scores_quality', z_params_quality)

# Random templates
results_random, per_finger_gen_random = collect_scores(
    all_probe_scores, 'scores_random', z_params_random)

# ============================================================
# Analysis functions
# ============================================================
def full_analysis(name, gen, imp):
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    print(f"\nGenuine:  n={len(gen)}, mean={gen.mean():.4f}, std={gen.std():.4f}, min={gen.min():.4f}")
    print(f"Impostor: n={len(imp)}, mean={imp.mean():.4f}, std={imp.std():.4f}, max={imp.max():.4f}")

    gen_min, imp_max = gen.min(), imp.max()
    if gen_min > imp_max:
        print(f"\n*** PERFECT SEPARATION *** gen_min={gen_min:.4f} > imp_max={imp_max:.4f}")
    else:
        overlap = np.sum(imp > gen_min)
        print(f"overlap: {overlap}/{len(imp)}")

    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    y_scores = np.concatenate([gen, imp])
    fpr_arr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
    eer = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2
    eer_thresh = thresholds[eer_idx]
    print(f"\nEER = {eer*100:.4f}% (threshold={eer_thresh:.4f})")

    print(f"\n--- FFR -> FAR ---")
    for target_ffr in [0.0, 0.01, 0.03, 0.05, 0.10]:
        idx = np.argmin(np.abs(fnr - target_ffr))
        print(f"  FFR={target_ffr*100:.0f}% -> FAR={fpr_arr[idx]*100:.4f}% (threshold={thresholds[idx]:.4f})")

    print(f"\n--- FAR -> FFR ---")
    for target_far in [0.002, 0.01, 0.1, 1.0]:
        target_frac = target_far / 100
        idx = np.argmin(np.abs(fpr_arr - target_frac))
        print(f"  FAR={target_far}% -> FFR={fnr[idx]*100:.2f}%")

    target_far_frac = 0.00002
    idx = np.argmin(np.abs(fpr_arr - target_far_frac))
    print(f"\n  *** FAR=0.002% (1/50000) -> FFR={fnr[idx]*100:.2f}% ***")

    d_prime = (gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.std()**2 + imp.std()**2))
    print(f"  d-prime = {d_prime:.2f}")

    # Parametric extrapolation
    gen_mu, gen_std = gen.mean(), gen.std()
    imp_mu, imp_std = imp.mean(), imp.std()
    print(f"\n--- Parametric extrapolation (Gaussian) ---")
    for target_ffr in [0.01, 0.03, 0.05]:
        thresh = gen_mu - sp_stats.norm.ppf(1 - target_ffr) * gen_std
        far_theory = sp_stats.norm.cdf(thresh, loc=imp_mu, scale=imp_std)
        print(f"  FFR={target_ffr*100:.0f}% -> 理论FAR={far_theory*100:.6f}%")
    for target_far in [0.00002, 0.0001]:
        thresh = sp_stats.norm.ppf(target_far, loc=imp_mu, scale=imp_std)
        ffr_theory = 1 - sp_stats.norm.cdf(thresh, loc=gen_mu, scale=gen_std)
        print(f"  FAR={target_far*100:.4f}% -> 理论FFR={ffr_theory*100:.2f}%")

    sys.stdout.flush()
    return eer

# ============================================================
# Run all evaluations
# ============================================================
print(f"\n{'#'*70}")
print(f"# EVALUATION RESULTS")
print(f"{'#'*70}"); sys.stdout.flush()

eer_results = {}

# --- Quality templates ---
print(f"\n{'='*70}")
print(f"Quality Templates (selected by intra-class consistency)")
print(f"{'='*70}")

for norm_name in ['raw', 'znorm', 'tnorm', 'snorm']:
    gen = results_quality[norm_name]['gen']
    imp = results_quality[norm_name]['imp']
    label = f"Quality Templates + {norm_name.upper()}"
    eer = full_analysis(label, gen, imp)
    eer_results[f"quality_{norm_name}"] = eer

# --- Random templates ---
print(f"\n{'='*70}")
print(f"Random Templates (V14-style random selection)")
print(f"{'='*70}")

for norm_name in ['raw', 'znorm', 'tnorm', 'snorm']:
    gen = results_random[norm_name]['gen']
    imp = results_random[norm_name]['imp']
    label = f"Random Templates + {norm_name.upper()}"
    eer = full_analysis(label, gen, imp)
    eer_results[f"random_{norm_name}"] = eer

# ============================================================
# Group Analysis (best normalization variant)
# ============================================================
# Find best variant
best_key = min(eer_results, key=eer_results.get)
best_eer = eer_results[best_key]
print(f"\n{'='*70}")
print(f"Best variant: {best_key} (EER={best_eer*100:.4f}%)")
print(f"{'='*70}"); sys.stdout.flush()

# Group analysis for best variant using quality templates with raw scores
print(f"\n--- Group Analysis (Quality Templates + RAW) ---"); sys.stdout.flush()

def group_analysis_from_probes(group_name, group_fingers, all_probes, templates_emb):
    g_gen, g_imp = [], []
    for probe in all_probes:
        if probe['probe_finger'] not in group_fingers:
            continue
        for identity in group_fingers:
            raw_score = probe['scores_quality'][identity]
            if identity == probe['probe_finger']:
                g_gen.append(raw_score)
            else:
                g_imp.append(raw_score)

    g_gen = np.array(g_gen)
    g_imp = np.array(g_imp)
    if len(g_gen) == 0 or len(g_imp) == 0:
        print(f"  [{group_name}] insufficient data"); return

    y_t = np.concatenate([np.ones(len(g_gen)), np.zeros(len(g_imp))])
    y_s = np.concatenate([g_gen, g_imp])
    fpr_g, tpr_g, th_g = roc_curve(y_t, y_s)
    fnr_g = 1 - tpr_g
    ei = np.nanargmin(np.abs(fnr_g - fpr_g))
    eer_g = (fpr_g[ei] + fnr_g[ei]) / 2

    print(f"\n  [{group_name}] gen={len(g_gen)}, imp={len(g_imp)}, EER={eer_g*100:.4f}%")
    print(f"    gen: mean={g_gen.mean():.4f}, min={g_gen.min():.4f}")
    print(f"    imp: mean={g_imp.mean():.4f}, max={g_imp.max():.4f}")
    idx3 = np.argmin(np.abs(fnr_g - 0.03))
    idx5 = np.argmin(np.abs(fnr_g - 0.05))
    print(f"    FFR=3% -> FAR={fpr_g[idx3]*100:.4f}%")
    print(f"    FFR=5% -> FAR={fpr_g[idx5]*100:.4f}%")
    sys.stdout.flush()

wtp = [f for f in fingers if f.startswith("wtp_")]
btp = [f for f in fingers if f.startswith("btp_")]
group_analysis_from_probes("无贴屏", wtp, all_probe_scores, quality_templates_emb)
group_analysis_from_probes("不贴屏", btp, all_probe_scores, quality_templates_emb)

# ============================================================
# Per-finger Analysis
# ============================================================
print(f"\n--- Per-finger Genuine (worst to best) ---"); sys.stdout.flush()
per_finger_stats = []
for finger in fingers:
    scores = per_finger_gen_quality[finger]
    if scores:
        scores = np.array(scores)
        src = finger_source.get(finger, "?")
        per_finger_stats.append((finger, src, scores.mean(), scores.min(), len(scores)))

per_finger_stats.sort(key=lambda x: x[2])
for i, (fn, src, gm, gmin, n) in enumerate(per_finger_stats):
    status = "*** WORST ***" if i < 3 else ""
    print(f"  {i+1}. {fn} [{src}]: mean={gm:.4f}, min={gmin:.4f}, n={n} {status}")
sys.stdout.flush()

# ============================================================
# Also evaluate with dual-polarity only (2-view TTA, no shift)
# to see if 4-view TTA helps
# ============================================================
print(f"\n{'='*70}")
print(f"Ablation: 2-view TTA (dual polarity only) with quality templates")
print(f"{'='*70}"); sys.stdout.flush()

t_ablation = time.time()

# Re-extract templates with dual-polarity only
dp_templates = {}
for finger in fingers:
    frames = finger_train[finger]
    indices = quality_template_indices[finger]
    dp_templates[finger] = [get_dual_polarity_emb(models, frames[i]) for i in indices]

# Re-compute Z-norm params
z_params_dp = compute_znorm_params(dp_templates)

# Match with dual-polarity probes
dp_gen_raw, dp_imp_raw = [], []
dp_gen_snorm, dp_imp_snorm = [], []

for fi_idx, probe_finger in enumerate(fingers):
    test_frames = finger_test[probe_finger]
    for frame in test_frames:
        probe_emb = get_dual_polarity_emb(models, frame)
        scores = {}
        for identity in fingers:
            max_sim = max(np.dot(probe_emb, t) for t in dp_templates[identity])
            scores[identity] = max_sim

        for identity in fingers:
            raw = scores[identity]
            s_score = apply_snorm(raw, identity, scores, z_params_dp)
            if identity == probe_finger:
                dp_gen_raw.append(raw)
                dp_gen_snorm.append(s_score)
            else:
                dp_imp_raw.append(raw)
                dp_imp_snorm.append(s_score)

    if (fi_idx+1) % 5 == 0 or fi_idx == 0:
        print(f"  [{fi_idx+1}/{n_classes}] elapsed={time.time()-t_ablation:.0f}s"); sys.stdout.flush()

dp_gen_raw = np.array(dp_gen_raw); dp_imp_raw = np.array(dp_imp_raw)
dp_gen_snorm = np.array(dp_gen_snorm); dp_imp_snorm = np.array(dp_imp_snorm)

eer_dp_raw = full_analysis("2-view TTA + Quality Templates + RAW", dp_gen_raw, dp_imp_raw)
eer_dp_snorm = full_analysis("2-view TTA + Quality Templates + S-NORM", dp_gen_snorm, dp_imp_snorm)
eer_results["dp_quality_raw"] = eer_dp_raw
eer_results["dp_quality_snorm"] = eer_dp_snorm

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"V17 FINAL SUMMARY")
print(f"{'='*70}")

print(f"\n  --- V17 All Variants ---")
for key in sorted(eer_results.keys()):
    print(f"  {key:30s}: EER = {eer_results[key]*100:.4f}%")

best_key = min(eer_results, key=eer_results.get)
best_eer = eer_results[best_key]

print(f"\n  *** Best: {best_key} with EER = {best_eer*100:.4f}% ***")

print(f"\n  --- Comparison with Previous ---")
print(f"  V14 (27 classes, no SWA):      EER = 2.6776%")
print(f"  V15 CNN component (27 classes): EER = 2.40%")
print(f"  V16 ConvNeXt-Tiny:              EER = 2.5144%")
print(f"  V16 ResNet-18:                  EER = 3.7870%")
print(f"  V17 best ({best_key}):  EER = {best_eer*100:.4f}%")

improvement = 2.6776 - best_eer * 100
print(f"\n  Improvement over V14: {improvement:.4f}% absolute")
improvement_v15 = 2.40 - best_eer * 100
print(f"  Improvement over V15 CNN: {improvement_v15:.4f}% absolute")

print(f"\nTotal time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")
print(f"{'='*70}"); sys.stdout.flush()
