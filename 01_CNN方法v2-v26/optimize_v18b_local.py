"""
V18b: Trained Local Descriptors + Dense Matching + T-norm
核心改进 (相对V18):
  V18证明: 直接复用全局训练的layer3特征做局部匹配效果差(EER=8.89%)
  原因: 特征没有针对局部匹配优化, 缺少局部对比损失

  V18b解决方案:
  1. 在ResNet-18 layer3后加局部描述符头: Conv1x1(256→64) → BN → L2-norm
  2. SimCLR阶段增加position-wise contrastive loss (局部对比损失)
     - 同一帧两个增强视图, 对应位置为正对, 不同位置为负对
  3. ArcFace阶段联合训练: 全局分类损失 + 局部对比损失
  4. 推理: dense local matching + global matching + T-norm fusion

借鉴FLARE/DMD: 每个空间位置输出独立可匹配的描述符 (64维)
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
# 预处理 (与V14/V17一致)
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
# 数据加载
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
fi = 0

print(f"{'='*70}")
print(f"V18b: Trained Local Descriptors + Dense Matching + T-norm")
print(f"{'='*70}")
print(f"Loading data..."); sys.stdout.flush()
t_start = time.time()
t_load = time.time()

for finger in sorted(os.listdir(base1)):
    key = f"wtp_{finger}"
    rpath = os.path.join(base1, finger, WTP_REG)
    if not os.path.exists(rpath):
        continue
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    train_frames, test_frames = [], []
    for p in imgs_paths[:70]:
        img = load_img(p)
        if img is not None:
            frame, _ = preprocess_frame(img)
            train_frames.append(frame)
    for p in imgs_paths[70:]:
        img = load_img(p)
        if img is not None:
            frame, _ = preprocess_frame(img)
            test_frames.append(frame)
    if train_frames and test_frames:
        finger_train[key] = train_frames
        finger_test[key] = test_frames
        finger_labels[key] = fi
        finger_source[key] = "无贴屏"
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
    for p in imgs_paths[:70]:
        img = load_img(p)
        if img is not None:
            frame, _ = preprocess_frame(img)
            train_frames.append(frame)
    for p in imgs_paths[70:]:
        img = load_img(p)
        if img is not None:
            frame, _ = preprocess_frame(img)
            test_frames.append(frame)
    if train_frames and test_frames:
        finger_train[key] = train_frames
        finger_test[key] = test_frames
        finger_labels[key] = fi
        finger_source[key] = "不贴屏"
        fi += 1

n_classes = fi
fingers = list(finger_train.keys())
print(f"  Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes-n_wtp} 不贴屏)")
print(f"Data loading done in {time.time()-t_load:.0f}s"); sys.stdout.flush()

# ============================================================
# 模型: ResNet-18 + 局部描述符头 + 全局嵌入头
# ============================================================
class DenseDescriptorEncoder(nn.Module):
    """ResNet-18 with both global embedding head and local descriptor head.
    Local head: Conv1x1(256→64) on layer3 features → L2-normalized per position
    Global head: layer4 → GAP → Linear(512→512) → BN
    """
    def __init__(self, embed_dim=512, local_dim=64):
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

        # Global embedding head (same as V17)
        self.global_projector = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # Local descriptor head (NEW: trained for position-wise matching)
        self.local_head = nn.Sequential(
            nn.Conv2d(256, local_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(local_dim),
        )

    def forward(self, x, return_local=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # (B, 256, 14, 13)

        local_desc = None
        if return_local:
            local_desc = self.local_head(x)  # (B, local_dim, 14, 13)
            # L2 normalize per position
            local_desc = F.normalize(local_desc, dim=1)

        x = self.layer4(x)  # (B, 512, 7, 7)
        x = self.avgpool(x).flatten(1)  # (B, 512)
        global_emb = self.global_projector(x)  # (B, embed_dim)

        if return_local:
            return global_emb, local_desc
        return global_emb

# ============================================================
# 损失函数
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

class PositionContrastiveLoss(nn.Module):
    """Position-wise contrastive loss for local descriptors.
    For two augmented views of the same image:
    - Corresponding positions (same spatial location) are positive pairs
    - Different positions (different spatial locations) are negative pairs
    Uses InfoNCE-style loss at each spatial position.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, desc1, desc2):
        """
        desc1, desc2: (B, D, H, W) L2-normalized local descriptors from two views
        Treats each position as an anchor, corresponding position in other view as positive,
        other positions in same view as negatives.
        """
        B, D, H, W = desc1.shape
        N = H * W  # number of positions

        # Reshape to (B, N, D)
        d1 = desc1.view(B, D, N).permute(0, 2, 1)  # (B, N, D)
        d2 = desc2.view(B, D, N).permute(0, 2, 1)  # (B, N, D)

        total_loss = 0
        for b in range(B):
            # (N, D) @ (D, N) = (N, N) cosine similarity
            sim = torch.mm(d1[b], d2[b].t()) / self.temperature
            # Labels: position i in view1 should match position i in view2
            labels = torch.arange(N, device=sim.device)
            total_loss += F.cross_entropy(sim, labels)

        return total_loss / B

# ============================================================
# 数据增强 (与V17一致, 但避免大位移以保持空间对应关系)
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

def augment_frame_mild(frame):
    """Mild augmentation for local contrastive loss (preserves spatial correspondence)."""
    f = frame.copy()
    if random.random() < 0.5:
        f = -f
    f += np.random.randn(*f.shape).astype(np.float32) * 0.02
    # Small translate only (max 2px → stays within same receptive field at layer3)
    if random.random() < 0.5:
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        f = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), borderValue=0)
    return f

# ============================================================
# 数据集
# ============================================================
class ContrastiveDSWithLocal(Dataset):
    """Returns two augmented views: one with mild aug (for local loss), one with normal aug."""
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
        # View 1: mild augmentation (preserves spatial correspondence)
        f1 = augment_frame_mild(frames[i1])
        # View 2: mild augmentation of SAME frame (for local contrastive)
        f2 = augment_frame_mild(frames[i1])
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
        frame = frames[random.randint(0, len(frames)-1)]
        f1 = augment_frame(frame)
        f2 = augment_frame_mild(frame)
        return (torch.tensor(f1, dtype=torch.float32).unsqueeze(0),
                torch.tensor(f2, dtype=torch.float32).unsqueeze(0),
                self.finger_labels[finger])

# ============================================================
# 训练
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}"); sys.stdout.flush()

EMBED_DIM = 512
LOCAL_DIM = 64
SEEDS = [42, 123, 777]
USE_AMP = True
N_TEMPLATES = 20
MODEL_SAVE_DIR = 'f:/1111/指纹/models_v18b/'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

N_PRETRAIN = 60     # SimCLR + local contrastive
N_FINETUNE = 120    # ArcFace + local contrastive
SWA_START = 100

def train_single_model(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    save_path = os.path.join(MODEL_SAVE_DIR, f'v18b_seed{seed}.pth')
    if os.path.exists(save_path):
        print(f"\n  Found saved model: {save_path}")
        model = DenseDescriptorEncoder(embed_dim=EMBED_DIM, local_dim=LOCAL_DIM).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model.eval()
        print(f"  Loaded."); sys.stdout.flush()
        return model

    print(f"\n{'='*60}")
    print(f"Training seed={seed}")
    print(f"  Phase1: SimCLR+LocalCL {N_PRETRAIN}ep | Phase2: ArcFace+LocalCL {N_FINETUNE}ep (SWA@{SWA_START})")
    print(f"{'='*60}"); sys.stdout.flush()

    model = DenseDescriptorEncoder(embed_dim=EMBED_DIM, local_dim=LOCAL_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}"); sys.stdout.flush()

    simclr_loss_fn = SimCLRLoss(temperature=0.07)
    local_loss_fn = PositionContrastiveLoss(temperature=0.1)

    # ---- Phase 1: SimCLR + Local Contrastive (projector + local_head trainable) ----
    print(f"  Phase 1: SimCLR + Local CL ({N_PRETRAIN} epochs)")
    for name, param in model.named_parameters():
        if 'global_projector' in name or 'local_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable: {trainable:,}"); sys.stdout.flush()

    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=0.001, weight_decay=1e-3)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=N_PRETRAIN)
    scaler = GradScaler('cuda') if USE_AMP else None

    pre_ds = ContrastiveDSWithLocal(finger_train, n_samples=120)
    pre_loader = DataLoader(pre_ds, batch_size=32, shuffle=True, num_workers=0)

    t0 = time.time()
    for epoch in range(N_PRETRAIN):
        model.train(); tl_global, tl_local = 0, 0
        for f1, f2 in pre_loader:
            f1, f2 = f1.to(device), f2.to(device)
            opt1.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    g1, d1 = model(f1, return_local=True)
                    g2, d2 = model(f2, return_local=True)
                    z1 = F.normalize(g1, dim=1)
                    z2 = F.normalize(g2, dim=1)
                    loss_global = simclr_loss_fn(z1, z2)
                    loss_local = local_loss_fn(d1, d2)
                    loss = loss_global + 0.5 * loss_local
                scaler.scale(loss).backward(); scaler.step(opt1); scaler.update()
            else:
                g1, d1 = model(f1, return_local=True)
                g2, d2 = model(f2, return_local=True)
                z1 = F.normalize(g1, dim=1)
                z2 = F.normalize(g2, dim=1)
                loss_global = simclr_loss_fn(z1, z2)
                loss_local = local_loss_fn(d1, d2)
                loss = loss_global + 0.5 * loss_local
                loss.backward(); opt1.step()
            tl_global += loss_global.item()
            tl_local += loss_local.item()
        sch1.step()
        if (epoch+1) % 10 == 0:
            n_batches = len(pre_loader)
            print(f"    Epoch {epoch+1}/{N_PRETRAIN}: global={tl_global/n_batches:.4f}, "
                  f"local={tl_local/n_batches:.4f}, time={time.time()-t0:.0f}s")
            sys.stdout.flush()

    # ---- Phase 2: ArcFace + Local Contrastive + SWA ----
    print(f"  Phase 2: ArcFace + Local CL ({N_FINETUNE} epochs, SWA@{SWA_START})")
    sys.stdout.flush()

    for param in model.parameters():
        param.requires_grad = True

    arcface = SubCenterArcFace(EMBED_DIM, n_classes, K=3, s=64, m=0.5,
                               label_smoothing=0.1).to(device)

    param_groups = [
        {'params': list(model.conv1.parameters()) + list(model.bn1.parameters()) +
                   list(model.layer1.parameters()) + list(model.layer2.parameters()),
         'lr': 1e-5},
        {'params': list(model.layer3.parameters()) + list(model.local_head.parameters()),
         'lr': 5e-5},
        {'params': list(model.layer4.parameters()), 'lr': 1e-4},
        {'params': list(model.global_projector.parameters()) + list(arcface.parameters()),
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
        tl_arc, tl_loc, cor, tot = 0, 0, 0, 0
        for X, X_mild, Y in ft_loader:
            X, X_mild, Y = X.to(device), X_mild.to(device), Y.to(device)
            opt2.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    emb = model(X)
                    loss_arc = arcface(emb, Y)
                    # Local loss on (X, X_mild) pair
                    _, d1 = model(X, return_local=True)
                    _, d2 = model(X_mild, return_local=True)
                    loss_local = local_loss_fn(d1, d2)
                    loss = loss_arc + 0.3 * loss_local
                scaler2.scale(loss).backward(); scaler2.step(opt2); scaler2.update()
            else:
                emb = model(X)
                loss_arc = arcface(emb, Y)
                _, d1 = model(X, return_local=True)
                _, d2 = model(X_mild, return_local=True)
                loss_local = local_loss_fn(d1, d2)
                loss = loss_arc + 0.3 * loss_local
                loss.backward(); opt2.step()
            tl_arc += loss_arc.item()
            tl_loc += loss_local.item()
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
            n_b = len(ft_loader)
            print(f"    Epoch {epoch+1}/{N_FINETUNE}: arc={tl_arc/n_b:.4f}, "
                  f"local={tl_loc/n_b:.4f}, acc={cor/tot*100:.1f}%, "
                  f"time={time.time()-t1:.0f}s{' [SWA]' if epoch >= SWA_START else ''}")
            sys.stdout.flush()

    if swa_n > 0:
        print(f"  Updating SWA BN ({swa_n} averages)..."); sys.stdout.flush()
        # Need a simple loader for BN update
        class SimpleBNDS(Dataset):
            def __init__(self, finger_data, n_samples=20):
                self.items = []
                for k, frames in finger_data.items():
                    for _ in range(n_samples):
                        f = frames[random.randint(0, len(frames)-1)]
                        self.items.append(torch.tensor(f, dtype=torch.float32).unsqueeze(0))
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                return self.items[idx]

        bn_ds = SimpleBNDS(finger_train, n_samples=20)
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
for seed in SEEDS:
    m = train_single_model(seed)
    models.append(m)
print(f"\nAll models ready in {time.time()-t_start:.0f}s"); sys.stdout.flush()

# ============================================================
# 嵌入提取
# ============================================================
@torch.no_grad()
def extract_features(models_list, frame):
    """Extract global embedding + local descriptors from all models.
    Returns:
      global_emb: (embed_dim,) L2-normalized, averaged across models and polarities
      local_descs: list of n_models × (n_positions, local_dim) per model
    """
    global_embs = []
    local_per_model = []

    for model in models_list:
        for polarity in [1, -1]:
            x = torch.tensor(polarity * frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            g, d = model(x, return_local=True)
            global_embs.append(F.normalize(g, dim=1))

    # Average global embeddings
    g_avg = torch.mean(torch.stack(global_embs), dim=0)
    g_avg = F.normalize(g_avg, dim=1).cpu().numpy().flatten()

    # Local descriptors: average per model across polarities, keep per-model
    for model in models_list:
        d_list = []
        for polarity in [1, -1]:
            x = torch.tensor(polarity * frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            _, d = model(x, return_local=True)
            # d: (1, local_dim, H, W) → (H*W, local_dim)
            d_flat = d[0].view(LOCAL_DIM, -1).t()  # (H*W, local_dim)
            d_list.append(d_flat)
        # Average and re-normalize
        d_avg = torch.mean(torch.stack(d_list), dim=0)
        d_avg = F.normalize(d_avg, dim=1)
        local_per_model.append(d_avg)  # keep as GPU tensor

    return g_avg, local_per_model

# ============================================================
# 评估
# ============================================================
print(f"\n{'='*70}")
print(f"Evaluating V18b")
print(f"{'='*70}"); sys.stdout.flush()

# Extract templates
print("Step 1: Extracting template features..."); sys.stdout.flush()
t_tmpl = time.time()
random.seed(42)

template_global = {}   # finger -> list of global embeddings
template_local = {}    # finger -> list of [n_models × GPU tensor]
template_local_concat = {}  # finger -> list of n_models × concat GPU tensor

for finger in fingers:
    frames = finger_train[finger]
    indices = random.sample(range(len(frames)), min(N_TEMPLATES, len(frames)))
    g_list, l_list = [], []
    for i in indices:
        g, l = extract_features(models, frames[i])
        g_list.append(g)
        l_list.append(l)
    template_global[finger] = g_list
    template_local[finger] = l_list

    # Pre-concat per model
    per_model_concat = []
    for m in range(len(SEEDS)):
        parts = [template_local[finger][t][m] for t in range(len(l_list))]
        per_model_concat.append(torch.cat(parts, dim=0))  # (N_T * n_pos, local_dim)
    template_local_concat[finger] = per_model_concat

print(f"  Template extraction done in {time.time()-t_tmpl:.0f}s"); sys.stdout.flush()

# Match
print("Step 2: Computing scores..."); sys.stdout.flush()
t_match = time.time()

all_probes = []
for fi_idx, probe_finger in enumerate(fingers):
    test_frames = finger_test[probe_finger]
    for frame in test_frames:
        g_emb, l_descs = extract_features(models, frame)
        scores = {}
        for identity in fingers:
            # Global score
            global_score = max(np.dot(g_emb, t_g) for t_g in template_global[identity])

            # Dense score: per-model matching, averaged
            model_dense_scores = []
            for m in range(len(SEEDS)):
                Q = l_descs[m]  # (n_pos, local_dim) GPU tensor
                T = template_local_concat[identity][m]  # (total_pos, local_dim) GPU
                S = torch.mm(Q, T.t())  # (n_pos, total_pos)
                max_sim, _ = S.max(dim=1)  # best match per query position
                k = max(1, int(0.7 * Q.shape[0]))
                top_k, _ = max_sim.topk(k)
                model_dense_scores.append(top_k.mean().item())
            dense_score = np.mean(model_dense_scores)

            scores[identity] = {'global': global_score, 'dense': dense_score}

        all_probes.append({'probe_finger': probe_finger, 'scores': scores})

    if (fi_idx+1) % 5 == 0 or fi_idx == 0:
        print(f"  [{fi_idx+1}/{n_classes}] elapsed={time.time()-t_match:.0f}s"); sys.stdout.flush()

print(f"  Matching done in {time.time()-t_match:.0f}s"); sys.stdout.flush()

# ============================================================
# Score normalization + evaluation
# ============================================================
def apply_tnorm(raw_score, claimed_id, probe_scores, key):
    imp = [probe_scores[k][key] for k in probe_scores if k != claimed_id]
    return (raw_score - np.mean(imp)) / (np.std(imp) + 1e-8)

print("Step 3: Evaluating variants..."); sys.stdout.flush()

ALPHA_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]

def collect_scores(all_probes):
    results = {}
    variants = (['global_raw', 'global_tnorm', 'dense_raw', 'dense_tnorm'] +
                [f'fused_a{a}_{n}' for a in [0.3, 0.5, 0.7] for n in ['raw', 'tnorm']])
    for v in variants:
        results[v] = {'gen': [], 'imp': []}

    pf_gen_g = {f: [] for f in fingers}
    pf_gen_d = {f: [] for f in fingers}

    for probe in all_probes:
        pf = probe['probe_finger']
        sc = probe['scores']
        for identity in fingers:
            g = sc[identity]['global']
            d = sc[identity]['dense']
            is_gen = (identity == pf)

            g_t = apply_tnorm(g, identity, sc, 'global')
            d_t = apply_tnorm(d, identity, sc, 'dense')

            for k, v in [('global_raw', g), ('global_tnorm', g_t),
                         ('dense_raw', d), ('dense_tnorm', d_t)]:
                if is_gen: results[k]['gen'].append(v)
                else: results[k]['imp'].append(v)

            for alpha in [0.3, 0.5, 0.7]:
                fused_raw = alpha * d + (1-alpha) * g
                results[f'fused_a{alpha}_raw']['gen' if is_gen else 'imp'].append(fused_raw)

                imp_fused = [alpha * sc[o]['dense'] + (1-alpha) * sc[o]['global']
                             for o in sc if o != identity]
                fused_t = (fused_raw - np.mean(imp_fused)) / (np.std(imp_fused) + 1e-8)
                results[f'fused_a{alpha}_tnorm']['gen' if is_gen else 'imp'].append(fused_t)

            if is_gen:
                pf_gen_g[pf].append(g)
                pf_gen_d[pf].append(d)

    for k in results:
        results[k]['gen'] = np.array(results[k]['gen'])
        results[k]['imp'] = np.array(results[k]['imp'])
    return results, pf_gen_g, pf_gen_d

results, pf_gen_g, pf_gen_d = collect_scores(all_probes)

def full_analysis(name, gen, imp, verbose=True):
    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    y_scores = np.concatenate([gen, imp])
    fpr_arr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
    eer = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2

    if verbose:
        print(f"\n{'='*60}")
        print(f"Results: {name}")
        print(f"{'='*60}")
        print(f"Genuine:  n={len(gen)}, mean={gen.mean():.4f}, std={gen.std():.4f}, min={gen.min():.4f}")
        print(f"Impostor: n={len(imp)}, mean={imp.mean():.4f}, std={imp.std():.4f}, max={imp.max():.4f}")
        print(f"EER = {eer*100:.4f}%")

        print(f"\n--- FFR -> FAR ---")
        for tf in [0.0, 0.01, 0.03, 0.05, 0.10]:
            idx = np.argmin(np.abs(fnr - tf))
            print(f"  FFR={tf*100:.0f}% -> FAR={fpr_arr[idx]*100:.4f}%")

        print(f"\n--- FAR -> FFR ---")
        for tf in [0.002, 0.01, 0.1, 1.0]:
            idx = np.argmin(np.abs(fpr_arr - tf/100))
            print(f"  FAR={tf}% -> FFR={fnr[idx]*100:.2f}%")

        idx = np.argmin(np.abs(fpr_arr - 0.00002))
        print(f"\n  *** FAR=0.002% (1/50000) -> FFR={fnr[idx]*100:.2f}% ***")
        d_prime = (gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.std()**2 + imp.std()**2))
        print(f"  d-prime = {d_prime:.2f}")
        sys.stdout.flush()
    return eer

# Run all evaluations
eer_results = {}
for key in sorted(results.keys()):
    gen, imp = results[key]['gen'], results[key]['imp']
    verbose = key in ['global_tnorm', 'dense_tnorm', 'fused_a0.5_tnorm', 'fused_a0.3_tnorm']
    eer = full_analysis(key, gen, imp, verbose=verbose)
    eer_results[key] = eer

# Summary
best_key = min(eer_results, key=eer_results.get)
best_eer = eer_results[best_key]

print(f"\n{'='*70}")
print(f"ALL VARIANT EER COMPARISON")
print(f"{'='*70}")
for key in sorted(eer_results.keys()):
    marker = " ***" if key == best_key else ""
    print(f"  {key:30s}: EER = {eer_results[key]*100:.4f}%{marker}")

# Per-finger
print(f"\n--- Per-finger Genuine (dense, trained local descriptors) ---")
pf_stats = []
for finger in fingers:
    scores = pf_gen_d[finger]
    if scores:
        arr = np.array(scores)
        src = finger_source.get(finger, "?")
        pf_stats.append((finger, src, arr.mean(), arr.min(), len(arr)))
pf_stats.sort(key=lambda x: x[2])
for i, (fn, src, gm, gmin, n) in enumerate(pf_stats):
    status = "*** WORST ***" if i < 3 else ""
    print(f"  {i+1}. {fn} [{src}]: mean={gm:.4f}, min={gmin:.4f}, n={n} {status}")

# Final summary
print(f"\n{'='*70}")
print(f"V18b FINAL SUMMARY")
print(f"{'='*70}")
print(f"\n  *** Best: {best_key} with EER = {best_eer*100:.4f}% ***")
print(f"\n  --- Comparison ---")
print(f"  V14 (no SWA):        EER = 2.6776%")
print(f"  V15 CNN:             EER = 2.40%")
print(f"  V16 ConvNeXt:        EER = 2.5144%")
print(f"  V17 global+T-norm:   EER = 1.6416%")
print(f"  V18 dense (untrained): EER = 1.6865%")
print(f"  V18b best:           EER = {best_eer*100:.4f}%")

improvement = 1.6416 - best_eer * 100
print(f"\n  vs V17: {improvement:+.4f}%")
print(f"\nTotal time: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")
print(f"{'='*70}"); sys.stdout.flush()
