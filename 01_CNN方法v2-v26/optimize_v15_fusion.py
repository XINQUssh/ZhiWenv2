"""
V15: SIFT+CNN Score-Level Fusion
核心思路: SIFT(局部几何匹配) + CNN(全局外观嵌入) 分数级融合
  - SIFT: V10精细管线 (RootSIFT + 180°翻转补偿 + 紧RANSAC)
  - CNN: V14架构 (ResNet-18 + Sub-center ArcFace K=3 + embed512)
  - 融合: 多种归一化 × 多种融合规则 × 权重网格搜索
  - 27类 (排除xzc)
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

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 预处理 (V10完整版 GMFS mask)
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

def preprocess_dual(img):
    """双路径预处理: CNN路径 + SIFT路径"""
    enhanced = clahe_enhance(img)
    upsampled = upsample_2x(enhanced)
    mask = generate_gmfs_mask(upsampled)

    # SIFT路径: Gaussian blur (V10)
    sift_img = cv2.GaussianBlur(upsampled, (5, 5), 0.8)

    # CNN路径: mask + normalize (V14)
    masked = upsampled.copy()
    masked[mask == 0] = 0
    valid = masked[mask > 0].astype(np.float32)
    mu = valid.mean() if len(valid) > 0 else 0
    std = (valid.std() + 1e-6) if len(valid) > 0 else 1
    cnn_frame = (masked.astype(np.float32) - mu) / std
    cnn_frame[mask == 0] = 0

    mask_ratio = np.sum(mask > 0) / mask.size
    return cnn_frame, sift_img, mask, mask_ratio

# ============================================================
# SIFT: V10精细管线 (RootSIFT + 180°翻转 + 紧RANSAC)
# ============================================================
def extract_sift_features(img_blurred, mask):
    sift = cv2.SIFT_create(
        nfeatures=300, nOctaveLayers=4,
        contrastThreshold=0.03, edgeThreshold=17.5, sigma=1.7
    )
    kp, des = sift.detectAndCompute(img_blurred, mask)
    if des is not None and len(des) > 0:
        # RootSIFT
        des = des / (np.sum(np.abs(des), axis=1, keepdims=True) + 1e-7)
        des = np.sqrt(np.abs(des))
        norms = np.linalg.norm(des, axis=1, keepdims=True)
        des = des / (norms + 1e-7)
    return kp, des

def flip_descriptors_180(des):
    if des is None or len(des) == 0:
        return None
    n = des.shape[0]
    d = des.reshape(n, 4, 4, 8)
    d_flip = np.zeros_like(d)
    for r in range(4):
        for c in range(4):
            d_flip[:, r, c, :] = np.roll(d[:, 3-r, 3-c, :], 4, axis=-1)
    return d_flip.reshape(n, 128).astype(np.float32)

def match_sift_v10(kp1, des1, kp2, des2, ratio_thresh=0.85, ransac_thresh=0.5):
    if des1 is None or des2 is None:
        return 0
    if len(des1) < 4 or len(des2) < 4:
        return 0

    des1_flip = flip_descriptors_180(des1)
    des1_f32 = des1.astype(np.float32)
    des2_f32 = des2.astype(np.float32)
    des1f_f32 = des1_flip.astype(np.float32)

    a2 = np.sum(des1_f32**2, axis=1, keepdims=True)
    b2 = np.sum(des2_f32**2, axis=1, keepdims=True)
    af2 = np.sum(des1f_f32**2, axis=1, keepdims=True)

    dist_fwd = np.sqrt(np.maximum(a2 + b2.T - 2 * des1_f32 @ des2_f32.T, 0))
    dist_flip = np.sqrt(np.maximum(af2 + b2.T - 2 * des1f_f32 @ des2_f32.T, 0))
    dist_min = np.minimum(dist_fwd, dist_flip)

    matches = []
    for i in range(len(des1)):
        if len(des2) < 2:
            continue
        sorted_idx = np.argpartition(dist_min[i], min(2, len(dist_min[i])-1))[:2]
        d0, d1 = dist_min[i, sorted_idx[0]], dist_min[i, sorted_idx[1]]
        if d0 > d1:
            sorted_idx[0], sorted_idx[1] = sorted_idx[1], sorted_idx[0]
            d0, d1 = d1, d0
        if d0 < ratio_thresh * d1:
            matches.append((i, sorted_idx[0], d0))

    if len(matches) < 4:
        return 0

    best_per_train = {}
    for qi, ti, d in matches:
        if ti not in best_per_train or d < best_per_train[ti][2]:
            best_per_train[ti] = (qi, ti, d)
    matches = list(best_per_train.values())

    if len(matches) < 4:
        return 0

    pts1 = np.float32([kp1[m[0]].pt for m in matches])
    pts2 = np.float32([kp2[m[1]].pt for m in matches])
    _, inlier_mask = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
    )
    if inlier_mask is None:
        return 0
    return int(inlier_mask.sum())

# ============================================================
# CNN模型 (V14架构)
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

# ============================================================
# 数据增强 + 数据集 (V14)
# ============================================================
def elastic_transform_light(image, alpha=10, sigma=3):
    h, w = image.shape
    dx = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma)
    dy = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(image, (x + dx).astype(np.float32), (y + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=0)

def augment_frame(frame):
    f = frame.copy()
    if random.random() < 0.5: f = -f
    f += np.random.randn(*f.shape).astype(np.float32) * 0.03
    if random.random() < 0.3: f = f[:, ::-1].copy()
    if random.random() < 0.5:
        dx, dy = random.randint(-5, 5), random.randint(-5, 5)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        f = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), borderValue=0)
    if random.random() < 0.3:
        angle = random.uniform(-3, 3)
        h, w = f.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        f = cv2.warpAffine(f, M, (w, h), borderValue=0)
    if random.random() < 0.25:
        f = elastic_transform_light(f)
    return f

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
        f1, f2 = augment_frame(frames[i1]), augment_frame(frames[i2])
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
# Phase A: 数据加载 (27类, 排除xzc)
# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
SKIP_BTP = ['xzc']
WTP_REG = 'Rgd1245'
BTP_REG = 'Rgd1237'
N_TEMPLATES = 20

print("="*70)
print("V15: SIFT+CNN Score-Level Fusion (27类, 排除xzc)")
print("="*70); sys.stdout.flush()

# CNN data
finger_train_cnn = {}
finger_test_cnn = {}
# SIFT data
finger_train_sift = {}  # key -> list of (sift_img_u8, mask)
finger_test_sift = {}
# Common
finger_labels = {}
finger_source = {}
fi = 0

print("Phase A: Loading data..."); sys.stdout.flush()
t_load = time.time()

for finger in sorted(os.listdir(base1)):
    key = f"wtp_{finger}"
    rpath = os.path.join(base1, finger, WTP_REG)
    if not os.path.exists(rpath):
        continue
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    tr_cnn, te_cnn, tr_sift, te_sift = [], [], [], []
    for p in imgs_paths[:70]:
        img = load_img(p)
        if img is not None:
            cnn_f, sift_img, mask, _ = preprocess_dual(img)
            tr_cnn.append(cnn_f)
            tr_sift.append((sift_img, mask))
    for p in imgs_paths[70:]:
        img = load_img(p)
        if img is not None:
            cnn_f, sift_img, mask, _ = preprocess_dual(img)
            te_cnn.append(cnn_f)
            te_sift.append((sift_img, mask))
    if tr_cnn and te_cnn:
        finger_train_cnn[key] = tr_cnn
        finger_test_cnn[key] = te_cnn
        finger_train_sift[key] = tr_sift
        finger_test_sift[key] = te_sift
        finger_labels[key] = fi
        finger_source[key] = "无贴屏"
        fi += 1

n_wtp = fi
print(f"  无贴屏: {n_wtp} classes"); sys.stdout.flush()

for finger in sorted(os.listdir(base2)):
    if any(s in finger.lower() for s in SKIP_BTP):
        print(f"  SKIP {finger}"); sys.stdout.flush()
        continue
    rpath = os.path.join(base2, finger, BTP_REG)
    if not os.path.exists(rpath):
        continue
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    if len(imgs_paths) < 50:
        continue
    key = f"btp_{finger}"
    tr_cnn, te_cnn, tr_sift, te_sift = [], [], [], []
    for p in imgs_paths[:70]:
        img = load_img(p)
        if img is not None:
            cnn_f, sift_img, mask, _ = preprocess_dual(img)
            tr_cnn.append(cnn_f)
            tr_sift.append((sift_img, mask))
    for p in imgs_paths[70:]:
        img = load_img(p)
        if img is not None:
            cnn_f, sift_img, mask, _ = preprocess_dual(img)
            te_cnn.append(cnn_f)
            te_sift.append((sift_img, mask))
    if tr_cnn and te_cnn:
        finger_train_cnn[key] = tr_cnn
        finger_test_cnn[key] = te_cnn
        finger_train_sift[key] = tr_sift
        finger_test_sift[key] = te_sift
        finger_labels[key] = fi
        finger_source[key] = "不贴屏"
        fi += 1

n_classes = fi
fingers = list(finger_train_cnn.keys())
print(f"  Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes-n_wtp} 不贴屏)")
print(f"Phase A done in {time.time()-t_load:.0f}s"); sys.stdout.flush()

# 固定模板索引 (SIFT和CNN共用)
random.seed(42)
template_indices = {}
for key in fingers:
    n_train = len(finger_train_cnn[key])
    template_indices[key] = random.sample(range(n_train), min(N_TEMPLATES, n_train))

# ============================================================
# Phase B: CNN训练 + 保存模型
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}"); sys.stdout.flush()

N_PRETRAIN = 80
N_FINETUNE = 120
EMBED_DIM = 512
SEEDS = [42, 123, 777]
USE_AMP = True
MODEL_SAVE_DIR = 'f:/1111/指纹/models_v15/'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

finger_train = finger_train_cnn  # alias for training

def train_single_model(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"Training model with seed={seed}")
    print(f"{'='*60}"); sys.stdout.flush()

    model = FingerprintEncoder(embed_dim=EMBED_DIM).to(device)
    simclr_loss = SimCLRLoss(temperature=0.07)

    # Phase 1: SimCLR (backbone frozen)
    print(f"  Phase 1: SimCLR ({N_PRETRAIN} epochs, backbone frozen)")
    for name, param in model.named_parameters():
        if 'projector' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,}"); sys.stdout.flush()

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

    # Phase 2: Sub-center ArcFace (progressive unfreeze)
    print(f"  Phase 2: Sub-center ArcFace ({N_FINETUNE} epochs, progressive unfreeze)")
    sys.stdout.flush()
    for param in model.parameters():
        param.requires_grad = True

    arcface = SubCenterArcFace(EMBED_DIM, n_classes, K=3, s=64, m=0.5).to(device)
    param_groups = [
        {'params': list(model.conv1.parameters()) + list(model.bn1.parameters()) +
                   list(model.layer1.parameters()) + list(model.layer2.parameters()), 'lr': 1e-5},
        {'params': list(model.layer3.parameters()), 'lr': 5e-5},
        {'params': list(model.layer4.parameters()), 'lr': 1e-4},
        {'params': list(model.projector.parameters()) + list(arcface.parameters()), 'lr': 3e-4},
    ]
    opt2 = optim.AdamW(param_groups, weight_decay=1e-3)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=N_FINETUNE)
    scaler2 = GradScaler('cuda') if USE_AMP else None

    ft_ds = ClassifyDS(finger_train, finger_labels, n_samples=100)
    ft_loader = DataLoader(ft_ds, batch_size=32, shuffle=True, num_workers=0)

    t1 = time.time()
    for epoch in range(N_FINETUNE):
        model.train(); arcface.train(); tl, cor, tot = 0, 0, 0
        for X, Y in ft_loader:
            X, Y = X.to(device), Y.to(device)
            opt2.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    emb = model(X); loss = arcface(emb, Y)
                scaler2.scale(loss).backward(); scaler2.step(opt2); scaler2.update()
            else:
                emb = model(X); loss = arcface(emb, Y)
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
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_FINETUNE}: loss={tl/len(ft_loader)/2:.4f}, "
                  f"acc={cor/tot*100:.1f}%, time={time.time()-t1:.0f}s"); sys.stdout.flush()

    model.eval()
    save_path = os.path.join(MODEL_SAVE_DIR, f'v15_encoder_seed{seed}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"  Model saved to {save_path}")
    print(f"  Model done. Total: {time.time()-t0:.0f}s"); sys.stdout.flush()
    return model

print(f"\n{'='*70}")
print("Phase B: CNN Training (3 models)")
print("="*70); sys.stdout.flush()

models = []
t_total = time.time()
for seed in SEEDS:
    m = train_single_model(seed)
    models.append(m)
print(f"\nAll CNN models trained in {time.time()-t_total:.0f}s"); sys.stdout.flush()

# ============================================================
# Phase C: 预计算SIFT分数
# ============================================================
print(f"\n{'='*70}")
print("Phase C: SIFT score pre-computation (V10 refined pipeline)")
print("="*70); sys.stdout.flush()

# 预提取SIFT特征
print("  Pre-extracting SIFT features..."); sys.stdout.flush()
t_sift = time.time()

sift_feats_template = {}  # key -> list of (kp, des) for selected templates
sift_feats_test = {}      # key -> list of (kp, des) for all test frames

for key in fingers:
    indices = template_indices[key]
    sift_feats_template[key] = [
        extract_sift_features(finger_train_sift[key][i][0], finger_train_sift[key][i][1])
        for i in indices
    ]
    sift_feats_test[key] = [
        extract_sift_features(te[0], te[1]) for te in finger_test_sift[key]
    ]

print(f"  Features extracted in {time.time()-t_sift:.0f}s"); sys.stdout.flush()

# 计算所有SIFT分数 (对齐pair)
print("  Computing all SIFT match scores..."); sys.stdout.flush()
t_match = time.time()

# 存储: 按 (probe_finger, test_idx) 顺序的 genuine 和 impostor 分数
genuine_sift = []
impostor_sift = []
pair_order = []  # (probe_finger, test_idx, target_finger, is_genuine)

for fi_idx, probe_finger in enumerate(fingers):
    n_test = len(sift_feats_test[probe_finger])
    for tidx in range(n_test):
        q_kp, q_des = sift_feats_test[probe_finger][tidx]
        for target_finger in fingers:
            best_score = 0
            for t_kp, t_des in sift_feats_template[target_finger]:
                score = match_sift_v10(q_kp, q_des, t_kp, t_des)
                best_score = max(best_score, score)
            if target_finger == probe_finger:
                genuine_sift.append(best_score)
                pair_order.append((probe_finger, tidx, target_finger, True))
            else:
                impostor_sift.append(best_score)
                pair_order.append((probe_finger, tidx, target_finger, False))
    if (fi_idx+1) % 5 == 0 or fi_idx == 0:
        print(f"    [{fi_idx+1}/{n_classes}] elapsed={time.time()-t_match:.0f}s")
        sys.stdout.flush()

genuine_sift = np.array(genuine_sift, dtype=float)
impostor_sift = np.array(impostor_sift, dtype=float)
print(f"  SIFT matching done in {time.time()-t_match:.0f}s")
print(f"  genuine={len(genuine_sift)}, impostor={len(impostor_sift)}")
sys.stdout.flush()

# ============================================================
# Phase D: 预计算CNN分数 (双极性, 集成)
# ============================================================
print(f"\n{'='*70}")
print("Phase D: CNN score pre-computation (dual polarity, 3-model ensemble)")
print("="*70); sys.stdout.flush()

@torch.no_grad()
def get_ensemble_emb(models, frame):
    x = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    embs = []
    for model in models:
        emb = model(x)
        embs.append(F.normalize(emb, dim=1))
    avg = torch.mean(torch.stack(embs), dim=0)
    return F.normalize(avg, dim=1).cpu().numpy().flatten()

def get_dual_polarity_emb(models, frame):
    return (get_ensemble_emb(models, frame), get_ensemble_emb(models, -frame))

def dual_polarity_max_sim(q_embs, t_embs):
    q_orig, q_flip = q_embs
    t_orig, t_flip = t_embs
    return max(np.dot(q_orig, t_orig), np.dot(q_orig, t_flip),
               np.dot(q_flip, t_orig), np.dot(q_flip, t_flip))

# 预计算模板嵌入
print("  Pre-computing template embeddings..."); sys.stdout.flush()
t_cnn = time.time()
cnn_templates = {}
for key in fingers:
    indices = template_indices[key]
    cnn_templates[key] = [
        get_dual_polarity_emb(models, finger_train_cnn[key][i])
        for i in indices
    ]
print(f"  Templates done in {time.time()-t_cnn:.0f}s"); sys.stdout.flush()

# 计算CNN分数 (与SIFT严格对齐)
print("  Computing all CNN match scores..."); sys.stdout.flush()
t_cnn_match = time.time()

genuine_cnn = []
impostor_cnn = []

for fi_idx, probe_finger in enumerate(fingers):
    test_frames = finger_test_cnn[probe_finger]
    for tidx, frame in enumerate(test_frames):
        q_embs = get_dual_polarity_emb(models, frame)
        for target_finger in fingers:
            best_sim = -1.0
            for t_embs in cnn_templates[target_finger]:
                sim = dual_polarity_max_sim(q_embs, t_embs)
                best_sim = max(best_sim, sim)
            if target_finger == probe_finger:
                genuine_cnn.append(best_sim)
            else:
                impostor_cnn.append(best_sim)
    if (fi_idx+1) % 5 == 0 or fi_idx == 0:
        print(f"    [{fi_idx+1}/{n_classes}] elapsed={time.time()-t_cnn_match:.0f}s")
        sys.stdout.flush()

genuine_cnn = np.array(genuine_cnn)
impostor_cnn = np.array(impostor_cnn)
print(f"  CNN matching done in {time.time()-t_cnn_match:.0f}s")
sys.stdout.flush()

# 验证对齐
assert len(genuine_sift) == len(genuine_cnn), \
    f"Genuine count mismatch: SIFT={len(genuine_sift)}, CNN={len(genuine_cnn)}"
assert len(impostor_sift) == len(impostor_cnn), \
    f"Impostor count mismatch: SIFT={len(impostor_sift)}, CNN={len(impostor_cnn)}"
print(f"\n  Score alignment verified: {len(genuine_sift)} genuine, {len(impostor_sift)} impostor")

# ============================================================
# 错误相关性分析
# ============================================================
print(f"\n{'='*70}")
print("Error Correlation Analysis")
print("="*70)

corr_gen = np.corrcoef(genuine_sift, genuine_cnn)[0, 1]
corr_imp = np.corrcoef(impostor_sift, impostor_cnn)[0, 1]
print(f"  Genuine scores correlation (SIFT vs CNN):  {corr_gen:.4f}")
print(f"  Impostor scores correlation (SIFT vs CNN): {corr_imp:.4f}")
print(f"  (Low correlation = better fusion potential)")
sys.stdout.flush()

# ============================================================
# Phase E: 融合策略搜索
# ============================================================
print(f"\n{'='*70}")
print("Phase E: Score Fusion Grid Search")
print("="*70); sys.stdout.flush()

def compute_eer(gen, imp):
    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    y_scores = np.concatenate([gen, imp])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[eer_idx] + fnr[eer_idx]) / 2

def compute_far_at_ffr(gen, imp, target_ffr=0.03):
    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    y_scores = np.concatenate([gen, imp])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fnr - target_ffr))
    return fpr[idx]

# 归一化方法
def normalize_minmax(gen, imp):
    all_s = np.concatenate([gen, imp])
    mn, mx = all_s.min(), all_s.max()
    return (gen - mn) / (mx - mn + 1e-10), (imp - mn) / (mx - mn + 1e-10)

def normalize_zscore(gen, imp):
    all_s = np.concatenate([gen, imp])
    mu, std = all_s.mean(), all_s.std() + 1e-10
    return (gen - mu) / std, (imp - mu) / std

def normalize_tanh(gen, imp):
    all_s = np.concatenate([gen, imp])
    mu, std = all_s.mean(), all_s.std() + 1e-10
    return 0.5 * (np.tanh(0.01 * (gen - mu) / std) + 1), \
           0.5 * (np.tanh(0.01 * (imp - mu) / std) + 1)

# 先报告各方法单独性能
eer_sift = compute_eer(genuine_sift, impostor_sift)
eer_cnn = compute_eer(genuine_cnn, impostor_cnn)
far3_sift = compute_far_at_ffr(genuine_sift, impostor_sift, 0.03)
far3_cnn = compute_far_at_ffr(genuine_cnn, impostor_cnn, 0.03)
far5_sift = compute_far_at_ffr(genuine_sift, impostor_sift, 0.05)
far5_cnn = compute_far_at_ffr(genuine_cnn, impostor_cnn, 0.05)

print(f"\nIndividual system performance:")
print(f"  SIFT (V10 refined): EER={eer_sift*100:.4f}%, FFR=3%->FAR={far3_sift*100:.4f}%, FFR=5%->FAR={far5_sift*100:.4f}%")
print(f"  CNN  (V14 ensemble): EER={eer_cnn*100:.4f}%, FFR=3%->FAR={far3_cnn*100:.4f}%, FFR=5%->FAR={far5_cnn*100:.4f}%")
sys.stdout.flush()

# 融合搜索
normalizations = {
    'minmax': normalize_minmax,
    'zscore': normalize_zscore,
    'tanh': normalize_tanh,
}

results = []

for norm_name, norm_fn in normalizations.items():
    gs_n, is_n = norm_fn(genuine_sift, impostor_sift)
    gc_n, ic_n = norm_fn(genuine_cnn, impostor_cnn)

    # 加权求和
    for alpha in np.arange(0.1, 0.95, 0.05):
        gf = alpha * gs_n + (1 - alpha) * gc_n
        if_ = alpha * is_n + (1 - alpha) * ic_n
        eer = compute_eer(gf, if_)
        far3 = compute_far_at_ffr(gf, if_, 0.03)
        far5 = compute_far_at_ffr(gf, if_, 0.05)
        results.append((norm_name, 'weighted_sum', alpha, eer, far3, far5))

    # Max rule
    gf = np.maximum(gs_n, gc_n)
    if_ = np.maximum(is_n, ic_n)
    eer = compute_eer(gf, if_)
    far3 = compute_far_at_ffr(gf, if_, 0.03)
    far5 = compute_far_at_ffr(gf, if_, 0.05)
    results.append((norm_name, 'max', 0, eer, far3, far5))

    # Product rule
    gf = gs_n * gc_n
    if_ = is_n * ic_n
    eer = compute_eer(gf, if_)
    far3 = compute_far_at_ffr(gf, if_, 0.03)
    far5 = compute_far_at_ffr(gf, if_, 0.05)
    results.append((norm_name, 'product', 0, eer, far3, far5))

# Logistic regression (无需归一化)
from sklearn.linear_model import LogisticRegression
X_all = np.column_stack([
    np.concatenate([genuine_sift, impostor_sift]),
    np.concatenate([genuine_cnn, impostor_cnn])
])
y_all = np.concatenate([np.ones(len(genuine_sift)), np.zeros(len(impostor_sift))])
clf = LogisticRegression(max_iter=1000)
clf.fit(X_all, y_all)
probs = clf.predict_proba(X_all)[:, 1]
ng = len(genuine_sift)
eer_lr = compute_eer(probs[:ng], probs[ng:])
far3_lr = compute_far_at_ffr(probs[:ng], probs[ng:], 0.03)
far5_lr = compute_far_at_ffr(probs[:ng], probs[ng:], 0.05)
results.append(('raw', 'logistic', 0, eer_lr, far3_lr, far5_lr))
print(f"  Logistic weights: SIFT={clf.coef_[0][0]:.4f}, CNN={clf.coef_[0][1]:.4f}, "
      f"intercept={clf.intercept_[0]:.4f}")

# 排序并显示
results.sort(key=lambda x: x[3])
print(f"\n  Top 10 fusion configurations (by EER):")
print(f"  {'Norm':<8} {'Rule':<13} {'Alpha':>6} {'EER':>8} {'FAR@3%':>10} {'FAR@5%':>10}")
print(f"  {'-'*58}")
for i, (norm, rule, alpha, eer, far3, far5) in enumerate(results[:10]):
    alpha_str = f"{alpha:.2f}" if rule == 'weighted_sum' else '-'
    print(f"  {norm:<8} {rule:<13} {alpha_str:>6} {eer*100:>7.4f}% {far3*100:>9.4f}% {far5*100:>9.4f}%")
sys.stdout.flush()

# ============================================================
# Phase F: 最佳融合完整评估
# ============================================================
best = results[0]
best_norm_name, best_rule, best_alpha = best[0], best[1], best[2]
print(f"\n{'='*70}")
print(f"Phase F: Best Fusion Full Evaluation")
print(f"  Config: norm={best_norm_name}, rule={best_rule}, alpha={best_alpha:.2f}")
print("="*70); sys.stdout.flush()

# 应用最佳融合
if best_rule == 'logistic':
    fused_gen = probs[:ng]
    fused_imp = probs[ng:]
else:
    norm_fn = normalizations[best_norm_name]
    gs_n, is_n = norm_fn(genuine_sift, impostor_sift)
    gc_n, ic_n = norm_fn(genuine_cnn, impostor_cnn)
    if best_rule == 'weighted_sum':
        fused_gen = best_alpha * gs_n + (1 - best_alpha) * gc_n
        fused_imp = best_alpha * is_n + (1 - best_alpha) * ic_n
    elif best_rule == 'max':
        fused_gen = np.maximum(gs_n, gc_n)
        fused_imp = np.maximum(is_n, ic_n)
    elif best_rule == 'product':
        fused_gen = gs_n * gc_n
        fused_imp = is_n * ic_n

# 完整分析
def full_analysis(name, gen, imp):
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    print(f"\nGenuine:  n={len(gen)}, mean={gen.mean():.4f}, std={gen.std():.4f}, min={gen.min():.4f}")
    print(f"Impostor: n={len(imp)}, mean={imp.mean():.4f}, std={imp.std():.4f}, max={imp.max():.4f}")

    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    y_scores = np.concatenate([gen, imp])
    fpr_arr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
    eer = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2

    print(f"\nEER = {eer*100:.4f}%")
    print(f"\n--- FFR -> FAR ---")
    for target_ffr in [0.0, 0.01, 0.03, 0.05, 0.10]:
        idx = np.argmin(np.abs(fnr - target_ffr))
        print(f"  FFR={target_ffr*100:.0f}% -> FAR={fpr_arr[idx]*100:.4f}%")
    print(f"\n--- FAR -> FFR ---")
    for target_far in [0.002, 0.01, 0.1, 1.0]:
        idx = np.argmin(np.abs(fpr_arr - target_far/100))
        print(f"  FAR={target_far}% -> FFR={fnr[idx]*100:.2f}%")
    tf_idx = np.argmin(np.abs(fpr_arr - 0.00002))
    print(f"\n  *** FAR=0.002% (1/50000) -> FFR={fnr[tf_idx]*100:.2f}% ***")
    d_prime = (gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.std()**2 + imp.std()**2))
    print(f"  d-prime = {d_prime:.2f}")
    sys.stdout.flush()
    return eer

eer_fusion = full_analysis(
    f"V15 Fusion ({best_norm_name}/{best_rule}/α={best_alpha:.2f})",
    fused_gen, fused_imp
)

# 也报告SIFT和CNN单独的完整结果
eer_sift_full = full_analysis("SIFT (V10 refined) 单独", genuine_sift, impostor_sift)
eer_cnn_full = full_analysis("CNN (V14 ensemble) 单独", genuine_cnn, impostor_cnn)

# ============================================================
# 分组分析 (从pair_order提取)
# ============================================================
print(f"\n{'='*70}")
print("分组分析")
print("="*70); sys.stdout.flush()

# 重建逐pair的融合分数
all_fused_gen = fused_gen
all_fused_imp = fused_imp

# 用pair_order重建分组
gen_idx = 0
imp_idx = 0
group_scores = {'wtp': {'gen': [], 'imp': []}, 'btp': {'gen': [], 'imp': []}}

for probe_finger in fingers:
    prefix = 'wtp' if probe_finger.startswith('wtp_') else 'btp'
    n_test = len(finger_test_cnn[probe_finger])
    for tidx in range(n_test):
        for target_finger in fingers:
            target_prefix = 'wtp' if target_finger.startswith('wtp_') else 'btp'
            if target_finger == probe_finger:
                # genuine
                if prefix == target_prefix:  # 同组内
                    group_scores[prefix]['gen'].append(all_fused_gen[gen_idx])
                gen_idx += 1
            else:
                # impostor
                if prefix == target_prefix:  # 同组内
                    group_scores[prefix]['imp'].append(all_fused_imp[imp_idx])
                imp_idx += 1

for gname in ['wtp', 'btp']:
    label = '无贴屏' if gname == 'wtp' else '不贴屏'
    g = np.array(group_scores[gname]['gen'])
    im = np.array(group_scores[gname]['imp'])
    if len(g) < 5 or len(im) < 5:
        print(f"  [{label}] insufficient data"); continue
    eer_g = compute_eer(g, im)
    far3_g = compute_far_at_ffr(g, im, 0.03)
    print(f"\n  [{label}] gen={len(g)}, imp={len(im)}")
    print(f"  Fusion EER = {eer_g*100:.4f}%, FFR=3%->FAR={far3_g*100:.4f}%")
sys.stdout.flush()

# ============================================================
# 最终总结
# ============================================================
print(f"\n{'='*70}")
print(f"V15 总结对比 (27类, 排除xzc)")
print("="*70)
print(f"  SIFT (V10 refined)        EER = {eer_sift_full*100:.4f}%")
print(f"  CNN  (V14 ensemble)       EER = {eer_cnn_full*100:.4f}%")
print(f"  V15 Fusion ({best_norm_name}/{best_rule})  EER = {eer_fusion*100:.4f}%")
print(f"")
print(f"  Fusion improvement over best individual:")
best_individual = min(eer_sift_full, eer_cnn_full)
improvement = best_individual - eer_fusion
print(f"    {best_individual*100:.4f}% -> {eer_fusion*100:.4f}% (Δ = {improvement*100:.4f}%)")
print(f"")
print(f"  Error correlation: genuine={corr_gen:.4f}, impostor={corr_imp:.4f}")
print(f"\nAll done. Total time: {time.time()-t_total:.0f}s")
print(f"{'='*70}"); sys.stdout.flush()
