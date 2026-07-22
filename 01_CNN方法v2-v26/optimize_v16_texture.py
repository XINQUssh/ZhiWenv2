"""
V16: Pure Texture-Based Fingerprint Matching (v2 - GPU Gabor)
核心改进 (相对V14):
  1. Gabor滤波器组作为GPU内conv2d层 (8方向×2频率=16ch + 1原图 = 17ch)
     -> 关键优化: Gabor在GPU上批量计算, 不再CPU逐帧算, 速度接近V14
  2. ConvNeXt-Tiny backbone (主) + ResNet-18 (对照)
  3. GeM池化 (替代全局平均池化)
  4. 多尺度特征聚合 (中间层+最终层)
  5. 3阶段训练: SimCLR → ArcFace+SWA → Hard Triplet Mining
  6. 增强augmentation: Random Erasing + 亮度/对比度微调
  7. 双极性推理
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

try:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    HAS_CONVNEXT = True
except ImportError:
    HAS_CONVNEXT = False
    print("WARNING: ConvNeXt not available, will use ResNet-18 only")

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 预处理 (same as V14 + fill_internal_holes from V10)
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
# 数据加载 (27类, 排除xzc) - 1通道, 和V14一样
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
print(f"V16: Pure Texture Fingerprint (GPU Gabor + GeM + MultiScale + Triplet)")
print(f"{'='*70}")
print(f"Loading data (1-channel, Gabor computed on GPU inside model)...")
sys.stdout.flush()

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
# 模型定义
# ============================================================

# GPU Gabor Layer - 关键优化: Gabor作为固定权重conv2d在GPU上批量计算
class GaborLayer(nn.Module):
    """Non-trainable Gabor filter bank as Conv2d on GPU.
    Input: (B, 1, H, W) -> Output: (B, 1+n_filters, H, W)
    """
    def __init__(self, n_orientations=8, lambdas=[8.0, 12.0], ksize=21, gamma=0.5, sigma_ratio=0.4):
        super().__init__()
        filters = []
        for lam in lambdas:
            sigma = lam * sigma_ratio
            for i in range(n_orientations):
                theta = i * np.pi / n_orientations
                kern = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lam, gamma, 0
                )
                kern = kern / (np.abs(kern).sum() + 1e-7)
                filters.append(kern.astype(np.float32))

        n_filters = len(filters)
        weight = torch.zeros(n_filters, 1, ksize, ksize)
        for i, f in enumerate(filters):
            weight[i, 0] = torch.from_numpy(f)

        self.register_buffer('gabor_weight', weight)
        self.padding = ksize // 2
        self.n_filters = n_filters
        print(f"  GaborLayer: {n_orientations} orientations x {len(lambdas)} frequencies = {n_filters} filters")

    def forward(self, x):
        # x: (B, 1, H, W)
        gabor_out = F.conv2d(x, self.gabor_weight, padding=self.padding)  # (B, n_filters, H, W)
        return torch.cat([x, gabor_out], dim=1)  # (B, 1+n_filters, H, W)

# GeM Pooling
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (1, 1)
        ).pow(1.0 / self.p).flatten(1)

N_GABOR_CH = 17  # 1 + 16

# ConvNeXt-Tiny encoder with GPU Gabor
class TextureEncoderConvNeXt(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gabor = GaborLayer(n_orientations=8, lambdas=[8.0, 12.0])

        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        orig_conv = base.features[0][0]  # Conv2d(3, 96, 4, 4)
        new_conv = nn.Conv2d(N_GABOR_CH, 96, kernel_size=4, stride=4)
        with torch.no_grad():
            rgb_mean = orig_conv.weight.data.mean(dim=1, keepdim=True)
            new_conv.weight.data[:, 0:1, :, :] = rgb_mean
            for i in range(1, N_GABOR_CH):
                new_conv.weight.data[:, i:i+1, :, :] = rgb_mean * 0.1
            new_conv.bias.data = orig_conv.bias.data.clone()
        base.features[0][0] = new_conv

        self.features = base.features
        self.gem_mid = GeM(p=3.0)
        self.gem_final = GeM(p=3.0)
        self.projector = nn.Sequential(
            nn.Linear(384 + 768, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.gabor(x)  # (B, 17, H, W) - GPU Gabor
        for i in range(6):
            x = self.features[i](x)
        mid_feat = self.gem_mid(x)
        for i in range(6, 8):
            x = self.features[i](x)
        final_feat = self.gem_final(x)
        combined = torch.cat([mid_feat, final_feat], dim=1)
        return self.projector(combined)

# ResNet-18 encoder with GPU Gabor
class TextureEncoderResNet(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gabor = GaborLayer(n_orientations=8, lambdas=[8.0, 12.0])

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        orig_w = base.conv1.weight.data
        self.conv1 = nn.Conv2d(N_GABOR_CH, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            rgb_mean = orig_w.mean(dim=1, keepdim=True)
            self.conv1.weight.data[:, 0:1, :, :] = rgb_mean
            for i in range(1, N_GABOR_CH):
                self.conv1.weight.data[:, i:i+1, :, :] = rgb_mean * 0.1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.gem_mid = GeM(p=3.0)
        self.gem_final = GeM(p=3.0)
        self.projector = nn.Sequential(
            nn.Linear(256 + 512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        x = self.gabor(x)  # (B, 17, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        mid_feat = self.gem_mid(x)
        x = self.layer4(x)
        final_feat = self.gem_final(x)
        combined = torch.cat([mid_feat, final_feat], dim=1)
        return self.projector(combined)

# Sub-center ArcFace
class SubCenterArcFace(nn.Module):
    def __init__(self, embed_dim, n_classes, K=3, s=64.0, m=0.5):
        super().__init__()
        self.s, self.m, self.K = s, m, K
        self.n_classes = n_classes
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
        sim = sim.masked_fill(mask, -1e4)
        return F.cross_entropy(sim, labels)

class OnlineHardTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        emb_n = F.normalize(embeddings, dim=1)
        dist = 1 - torch.mm(emb_n, emb_n.t())
        n = embeddings.size(0)
        losses = []
        for i in range(n):
            pos_mask = (labels == labels[i]) & (torch.arange(n, device=labels.device) != i)
            if pos_mask.sum() == 0:
                continue
            hardest_pos = dist[i][pos_mask].max()
            neg_mask = (labels != labels[i])
            if neg_mask.sum() == 0:
                continue
            hardest_neg = dist[i][neg_mask].min()
            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            losses.append(triplet_loss)
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        return torch.stack(losses).mean()

# ============================================================
# 数据增强 (1通道, 和V14一样 + Random Erasing + brightness/contrast)
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
    # Random Erasing (20%, 2-8% area)
    if random.random() < 0.20:
        h, w = f.shape
        area = h * w
        erase_area = random.uniform(0.02, 0.08) * area
        aspect = random.uniform(0.5, 2.0)
        eh = int(np.sqrt(erase_area * aspect))
        ew = int(np.sqrt(erase_area / aspect))
        eh = min(eh, h - 1); ew = min(ew, w - 1)
        if eh > 0 and ew > 0:
            top = random.randint(0, h - eh)
            left = random.randint(0, w - ew)
            f[top:top+eh, left:left+ew] = 0
    # Brightness/contrast jitter (20%)
    if random.random() < 0.20:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-0.05, 0.05)
        f = f * alpha + beta
    return f

# ============================================================
# 数据集 (1通道输入, Gabor在模型内计算)
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
# 训练 (3阶段)
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}"); sys.stdout.flush()

EMBED_DIM = 512
SEEDS = [42, 123, 777]
USE_AMP = True
N_TEMPLATES = 20
MODEL_SAVE_DIR = 'f:/1111/指纹/models_v16/'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def train_single_model(seed, encoder_class, config_name, batch_size=32):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    N_PRETRAIN = 60
    N_FINETUNE = 120
    N_TRIPLET = 25
    SWA_START = 100

    print(f"\n{'='*60}")
    print(f"Training {config_name} seed={seed}")
    print(f"  Phase1: SimCLR {N_PRETRAIN}ep | Phase2: ArcFace {N_FINETUNE}ep (SWA@{SWA_START}) | Phase3: Triplet {N_TRIPLET}ep")
    print(f"{'='*60}"); sys.stdout.flush()

    model = encoder_class(embed_dim=EMBED_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} (trainable: {trainable_params:,})")
    sys.stdout.flush()

    simclr_loss = SimCLRLoss(temperature=0.07)

    # ---- Phase 1: SimCLR (backbone frozen, only projector trains) ----
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
    pre_loader = DataLoader(pre_ds, batch_size=batch_size, shuffle=True, num_workers=0)

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

    # ---- Phase 2: Sub-center ArcFace (progressive unfreeze + SWA) ----
    print(f"  Phase 2: ArcFace ({N_FINETUNE} epochs, progressive unfreeze, SWA@{SWA_START})")
    sys.stdout.flush()
    for param in model.parameters():
        param.requires_grad = True
    # Keep Gabor frozen (non-trainable by design via register_buffer, but ensure)
    for param in model.gabor.parameters():
        param.requires_grad = False

    arcface = SubCenterArcFace(EMBED_DIM, n_classes, K=3, s=64, m=0.5).to(device)

    # Differential learning rates
    if isinstance(model, TextureEncoderConvNeXt):
        param_groups = [
            {'params': list(model.features[0].parameters()) + list(model.features[1].parameters()) +
                       list(model.features[2].parameters()) + list(model.features[3].parameters()),
             'lr': 5e-6},
            {'params': list(model.features[4].parameters()) + list(model.features[5].parameters()),
             'lr': 3e-5},
            {'params': list(model.features[6].parameters()) + list(model.features[7].parameters()),
             'lr': 1e-4},
            {'params': list(model.gem_mid.parameters()) + list(model.gem_final.parameters()) +
                       list(model.projector.parameters()) + list(arcface.parameters()),
             'lr': 3e-4},
        ]
    else:
        param_groups = [
            {'params': list(model.conv1.parameters()) + list(model.bn1.parameters()) +
                       list(model.layer1.parameters()) + list(model.layer2.parameters()),
             'lr': 1e-5},
            {'params': list(model.layer3.parameters()), 'lr': 5e-5},
            {'params': list(model.layer4.parameters()), 'lr': 1e-4},
            {'params': list(model.gem_mid.parameters()) + list(model.gem_final.parameters()) +
                       list(model.projector.parameters()) + list(arcface.parameters()),
             'lr': 3e-4},
        ]

    opt2 = optim.AdamW(param_groups, weight_decay=1e-3)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=N_FINETUNE)
    scaler2 = GradScaler('cuda') if USE_AMP else None

    ft_ds = ClassifyDS(finger_train, finger_labels, n_samples=100)
    ft_loader = DataLoader(ft_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
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
        bn_loader = DataLoader(bn_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        update_bn(bn_loader, swa_model, device=device)
        model.load_state_dict(swa_model.module.state_dict())
        print(f"  SWA applied."); sys.stdout.flush()

    # ---- Phase 3: Hard Triplet Mining ----
    print(f"  Phase 3: Hard Triplet Mining ({N_TRIPLET} epochs)")
    sys.stdout.flush()

    triplet_loss_fn = OnlineHardTripletLoss(margin=0.2)

    if isinstance(model, TextureEncoderConvNeXt):
        param_groups3 = [
            {'params': list(model.features[0].parameters()) + list(model.features[1].parameters()) +
                       list(model.features[2].parameters()) + list(model.features[3].parameters()),
             'lr': 1e-6},
            {'params': list(model.features[4].parameters()) + list(model.features[5].parameters()),
             'lr': 6e-6},
            {'params': list(model.features[6].parameters()) + list(model.features[7].parameters()),
             'lr': 2e-5},
            {'params': list(model.gem_mid.parameters()) + list(model.gem_final.parameters()) +
                       list(model.projector.parameters()) + list(arcface.parameters()),
             'lr': 6e-5},
        ]
    else:
        param_groups3 = [
            {'params': list(model.conv1.parameters()) + list(model.bn1.parameters()) +
                       list(model.layer1.parameters()) + list(model.layer2.parameters()),
             'lr': 2e-6},
            {'params': list(model.layer3.parameters()), 'lr': 1e-5},
            {'params': list(model.layer4.parameters()), 'lr': 2e-5},
            {'params': list(model.gem_mid.parameters()) + list(model.gem_final.parameters()) +
                       list(model.projector.parameters()) + list(arcface.parameters()),
             'lr': 6e-5},
        ]

    opt3 = optim.AdamW(param_groups3, weight_decay=1e-3)
    sch3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=N_TRIPLET)
    scaler3 = GradScaler('cuda') if USE_AMP else None

    trip_ds = ClassifyDS(finger_train, finger_labels, n_samples=80)
    trip_loader = DataLoader(trip_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    t2 = time.time()
    for epoch in range(N_TRIPLET):
        model.train(); arcface.train()
        tl_arc, tl_tri = 0, 0
        for X, Y in trip_loader:
            X, Y = X.to(device), Y.to(device)
            opt3.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    emb = model(X)
                    loss_arc = arcface(emb, Y)
                    loss_tri = triplet_loss_fn(emb, Y)
                    loss = loss_arc + 0.3 * loss_tri
                scaler3.scale(loss).backward(); scaler3.step(opt3); scaler3.update()
            else:
                emb = model(X)
                loss_arc = arcface(emb, Y)
                loss_tri = triplet_loss_fn(emb, Y)
                loss = loss_arc + 0.3 * loss_tri
                loss.backward(); opt3.step()
            tl_arc += loss_arc.item(); tl_tri += loss_tri.item()
        sch3.step()
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{N_TRIPLET}: arc={tl_arc/len(trip_loader):.4f}, "
                  f"tri={tl_tri/len(trip_loader):.4f}, time={time.time()-t2:.0f}s")
            sys.stdout.flush()

    model.eval()
    total_time = time.time() - t0
    print(f"  Model done. Total training: {total_time:.0f}s ({total_time/60:.1f}min)"); sys.stdout.flush()
    return model

# ============================================================
# 推理 (双极性)
# ============================================================
@torch.no_grad()
def get_ensemble_emb(models, frame):
    """Dual polarity: original + negated, average across models and polarities."""
    x_orig = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x_flip = torch.tensor(-frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    embs = []
    for model in models:
        embs.append(F.normalize(model(x_orig), dim=1))
        embs.append(F.normalize(model(x_flip), dim=1))
    avg = torch.mean(torch.stack(embs), dim=0)
    return F.normalize(avg, dim=1).cpu().numpy().flatten()

def compute_max_sim(query_emb, template_embs):
    return max(np.dot(query_emb, t) for t in template_embs)

# ============================================================
# 评估
# ============================================================
def full_analysis(name, gen, imp):
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    print(f"\nGenuine:  n={len(gen)}, mean={gen.mean():.4f}, std={gen.std():.4f}, min={gen.min():.4f}")
    print(f"Impostor: n={len(imp)}, mean={imp.mean():.4f}, std={imp.std():.4f}, max={imp.max():.4f}")

    gen_min = gen.min()
    imp_max = imp.max()
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

    from scipy import stats as sp_stats
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

def evaluate_config(config_name, models_list):
    print(f"\n{'='*70}")
    print(f"Evaluating: {config_name}")
    print(f"{'='*70}"); sys.stdout.flush()

    print("  Extracting template embeddings..."); sys.stdout.flush()
    t_emb = time.time()
    random.seed(42)
    templates = {}
    for finger in fingers:
        train_frames = finger_train[finger]
        chosen = random.sample(range(len(train_frames)), min(N_TEMPLATES, len(train_frames)))
        templates[finger] = [get_ensemble_emb(models_list, train_frames[i]) for i in chosen]
    print(f"  Templates done in {time.time()-t_emb:.0f}s"); sys.stdout.flush()

    print("  Matching test frames..."); sys.stdout.flush()
    t_eval = time.time()

    genuine_scores = []
    impostor_scores = []
    per_finger_genuine = {f: [] for f in fingers}
    all_scores = []

    for fi_idx, finger in enumerate(fingers):
        test_frames = finger_test[finger]
        if not test_frames:
            continue
        for frame in test_frames:
            query_emb = get_ensemble_emb(models_list, frame)
            gs = compute_max_sim(query_emb, templates[finger])
            genuine_scores.append(gs)
            per_finger_genuine[finger].append(gs)
            all_scores.append((finger, finger, gs))
            for other in fingers:
                if other == finger:
                    continue
                is_score = compute_max_sim(query_emb, templates[other])
                impostor_scores.append(is_score)
                all_scores.append((finger, other, is_score))

        if (fi_idx+1) % 5 == 0 or fi_idx == 0:
            print(f"  [{fi_idx+1}/{n_classes}] elapsed={time.time()-t_eval:.0f}s"); sys.stdout.flush()

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    print(f"  Evaluation done in {time.time()-t_eval:.0f}s"); sys.stdout.flush()

    eer = full_analysis(config_name, genuine_scores, impostor_scores)

    # Group analysis
    print(f"\n  --- Group Analysis ---"); sys.stdout.flush()
    for group_name, prefix in [("无贴屏", "wtp_"), ("不贴屏", "btp_")]:
        group_f = set(f for f in fingers if f.startswith(prefix))
        g_gen, g_imp = [], []
        for qf, tf, score in all_scores:
            if qf not in group_f or tf not in group_f:
                continue
            if qf == tf:
                g_gen.append(score)
            else:
                g_imp.append(score)
        if len(g_gen) < 5 or len(g_imp) < 5:
            print(f"  [{group_name}] insufficient data"); continue
        g_gen = np.array(g_gen); g_imp = np.array(g_imp)
        y_t = np.concatenate([np.ones(len(g_gen)), np.zeros(len(g_imp))])
        y_s = np.concatenate([g_gen, g_imp])
        fpr_g, tpr_g, _ = roc_curve(y_t, y_s)
        fnr_g = 1 - tpr_g
        ei = np.nanargmin(np.abs(fnr_g - fpr_g))
        eer_g = (fpr_g[ei] + fnr_g[ei]) / 2
        print(f"  [{group_name}] gen={len(g_gen)}, imp={len(g_imp)}, EER={eer_g*100:.4f}%")
        print(f"    gen: mean={g_gen.mean():.4f}, min={g_gen.min():.4f}")
        print(f"    imp: mean={g_imp.mean():.4f}, max={g_imp.max():.4f}")
        for tf in [0.03, 0.05]:
            idx = np.argmin(np.abs(fnr_g - tf))
            print(f"    FFR={tf*100:.0f}% -> FAR={fpr_g[idx]*100:.4f}%")
        sys.stdout.flush()

    # Per-finger analysis
    print(f"\n  --- Per-finger Genuine (worst to best) ---"); sys.stdout.flush()
    stats_list = []
    for finger in fingers:
        arr = np.array(per_finger_genuine[finger])
        if len(arr) > 0:
            stats_list.append((finger, finger_source[finger], arr.mean(), arr.min(), len(arr)))
    stats_list.sort(key=lambda x: x[2])
    for i, (fn, src, gm, gmin, n) in enumerate(stats_list):
        status = "*** WORST ***" if i < 3 else ""
        print(f"  {i+1}. {fn} [{src}]: mean={gm:.4f}, min={gmin:.4f}, n={n} {status}")
    sys.stdout.flush()

    return eer, genuine_scores, impostor_scores

# ============================================================
# MAIN
# ============================================================
t_total = time.time()
results = {}

# Config A: ConvNeXt-Tiny
if HAS_CONVNEXT:
    print(f"\n{'#'*70}")
    print(f"# Config A: ConvNeXt-Tiny + GPU Gabor + GeM + MultiScale + Triplet")
    print(f"{'#'*70}"); sys.stdout.flush()

    # Check if saved models exist (from previous run)
    saved_A = all(os.path.exists(os.path.join(MODEL_SAVE_DIR, f'v16_convnext_seed{s}.pth')) for s in SEEDS)
    if saved_A:
        print("  Loading saved ConvNeXt models..."); sys.stdout.flush()
        models_A = []
        for seed in SEEDS:
            m = TextureEncoderConvNeXt(embed_dim=EMBED_DIM).to(device)
            m.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, f'v16_convnext_seed{seed}.pth'),
                                         map_location=device, weights_only=True))
            m.eval()
            models_A.append(m)
            print(f"    Loaded seed {seed}"); sys.stdout.flush()
    else:
        models_A = []
        for seed in SEEDS:
            m = train_single_model(seed, TextureEncoderConvNeXt, "ConvNeXt-A", batch_size=24)
            save_path = os.path.join(MODEL_SAVE_DIR, f'v16_convnext_seed{seed}.pth')
            torch.save(m.state_dict(), save_path)
            print(f"  Saved: {save_path}"); sys.stdout.flush()
            models_A.append(m)

    eer_A, gen_A, imp_A = evaluate_config("Config A: ConvNeXt-Tiny", models_A)
    results['A_ConvNeXt'] = eer_A

# Config C: ResNet-18 (control)
print(f"\n{'#'*70}")
print(f"# Config C: ResNet-18 + GPU Gabor + GeM + MultiScale + Triplet")
print(f"{'#'*70}"); sys.stdout.flush()

saved_C = all(os.path.exists(os.path.join(MODEL_SAVE_DIR, f'v16_resnet18_seed{s}.pth')) for s in SEEDS)
if saved_C:
    print("  Loading saved ResNet-18 models..."); sys.stdout.flush()
    models_C = []
    for seed in SEEDS:
        m = TextureEncoderResNet(embed_dim=EMBED_DIM).to(device)
        m.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, f'v16_resnet18_seed{seed}.pth'),
                                     map_location=device, weights_only=True))
        m.eval()
        models_C.append(m)
        print(f"    Loaded seed {seed}"); sys.stdout.flush()
else:
    models_C = []
    for seed in SEEDS:
        m = train_single_model(seed, TextureEncoderResNet, "ResNet18-C", batch_size=32)
        save_path = os.path.join(MODEL_SAVE_DIR, f'v16_resnet18_seed{seed}.pth')
        torch.save(m.state_dict(), save_path)
        print(f"  Saved: {save_path}"); sys.stdout.flush()
        models_C.append(m)

eer_C, gen_C, imp_C = evaluate_config("Config C: ResNet-18", models_C)
results['C_ResNet18'] = eer_C

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"V16 FINAL SUMMARY")
print(f"{'='*70}")
print(f"\n  --- V16 Results ---")
for k, v in results.items():
    print(f"  {k}: EER = {v*100:.4f}%")

print(f"\n  --- Comparison with Previous ---")
print(f"  V14 30类 (original):       EER = 3.8989%")
print(f"  V14 27类 (no xzc):         EER = 2.6776%")
print(f"  V15 CNN component (27类):   EER = 2.40%")
if HAS_CONVNEXT and 'A_ConvNeXt' in results:
    print(f"  V16 ConvNeXt-Tiny:          EER = {results['A_ConvNeXt']*100:.4f}%")
print(f"  V16 ResNet-18:              EER = {results['C_ResNet18']*100:.4f}%")

best_config = min(results, key=results.get)
best_eer = results[best_config]
print(f"\n  *** Best: {best_config} with EER = {best_eer*100:.4f}% ***")

v14_eer = 2.6776
improvement = v14_eer - best_eer * 100
print(f"  Improvement over V14 (27类): {improvement:.4f}% absolute")

print(f"\nTotal time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")
print(f"{'='*70}"); sys.stdout.flush()
