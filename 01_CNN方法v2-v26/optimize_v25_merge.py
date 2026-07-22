"""
V25: 同手指跨场景合并训练 + 跨场景评估
背景:
  客户确认: ysjz新数据中 dy_R*/lwh_R*/zyh_R* 与老数据同名类是同一根手指(贴屏再采集).
  V24把它们当独立新类导致有害监督(同一手指被ArcFace强行分开), 两个arm都掉分:
  V24a(+raw)=1.2608%, V24b(+denoised)=1.8829%, V21基线=1.1898%.

设计:
  1. 模型/训练/超参与V21完全一致
  2. 同手指合并: 9个R指新帧前70帧并入老类训练, 后30帧留作跨场景测试探针
  3. 仅新手指(dy_L*/zyh_L*/SSH_L*/yjx_R*)作为新类, 全部帧训练 (27+12=39类)
  4. Eval A: 老协议(27类, 老模板seed42, 老测试帧) → 与V21的1.1898%直接可比
     [关键] 模板仍只从老数据train帧选取(finger_train不混新帧), RNG序列与V21一致
  5. Eval B: 跨场景协议 — 老场景模板 x 新场景探针(9指x~30帧), 全变体EER
  6. 变体: python optimize_v25_merge.py raw|denoised → models_v25a / models_v25b
  排除: ssh_R0_failure(19帧<50).
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
# 预处理
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
    return result, np.sum(mask > 0) / mask.size

# ============================================================
# 数据加载
# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
WTP_REG = 'Rgd1245'
BTP_REG = 'Rgd1237'
SKIP_BTP = ['xzc']

VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'raw'
assert VARIANT in ('raw', 'denoised'), f"bad variant: {VARIANT}"
if VARIANT == 'raw':
    NEW_BASE = 'f:/1111/指纹/ysjz_raw/ysjz'
    ARM = 'v25a'
else:
    NEW_BASE = 'f:/1111/指纹/ysjz_denoised/ysjz_denoised'
    ARM = 'v25b'

# 客户确认的同手指映射(新folder -> 老类)
MERGE_MAP = {
    'dy_R0': 'wtp_dy_R0', 'dy_R1': 'wtp_dy_R1', 'dy_R2': 'wtp_dy_R2',
    'lwh_R0': 'wtp_lwh_R0', 'lwh_R1': 'wtp_lwh_R1', 'lwh_R2': 'wtp_lwh_R2',
    'zyh_R0': 'btp_zyh_R0', 'zyh_R1': 'btp_zyh_R1', 'zyh_R2': 'btp_zyh_R2',
}

finger_train = {}
finger_test = {}
finger_labels = {}
finger_source = {}
fi = 0

print(f"{'='*70}")
print(f"V25 ({ARM}): same-finger merge + cross-session eval (ysjz {VARIANT})")
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

n_eval = fi
eval_fingers = list(finger_train.keys())   # 老27类: 评估协议与V21完全一致
print(f"  Old data: {n_eval} eval classes ({n_wtp} wtp + {n_eval-n_wtp} btp)")

# ---- 新数据(ysjz): 同手指并入老类训练, 仅新手指作为新类 ----
# [关键] finger_train 保持只含老数据帧 → 模板选取RNG序列与V21完全一致
#        训练数据集使用 train_combined (老帧 + 合并的新帧 + 新类)
t_new = time.time()
extra_train = {}     # old_key -> 新场景train帧(前70)
cross_test = {}      # old_key -> 新场景跨场景探针(70之后)
new_only = {}        # new_key -> 全部帧
for folder in sorted(os.listdir(NEW_BASE)):
    fpath = os.path.join(NEW_BASE, folder)
    if not os.path.isdir(fpath):
        continue
    imgs_paths = sorted(glob.glob(os.path.join(fpath, '*.bmp')))
    if folder in MERGE_MAP:
        old_key = MERGE_MAP[folder]
        assert old_key in finger_train, f"{old_key} missing from old data"
        tr, te = [], []
        for p in imgs_paths[:70]:
            img = load_img(p)
            if img is not None:
                frame, _ = preprocess_frame(img)
                tr.append(frame)
        for p in imgs_paths[70:]:
            img = load_img(p)
            if img is not None:
                frame, _ = preprocess_frame(img)
                te.append(frame)
        extra_train[old_key] = tr
        cross_test[old_key] = te
        print(f"  MERGE {folder} -> {old_key}: +{len(tr)} train, {len(te)} cross-probes")
    elif len(imgs_paths) < 50:
        print(f"  SKIP new_{folder} ({len(imgs_paths)} frames < 50)")
    else:
        key = f"new_{folder}"
        frames = []
        for p in imgs_paths:       # 仅新手指: 全部帧训练
            img = load_img(p)
            if img is not None:
                frame, _ = preprocess_frame(img)
                frames.append(frame)
        if frames:
            new_only[key] = frames
            finger_labels[key] = fi
            finger_source[key] = f"ysjz_{VARIANT}"
            fi += 1

n_classes = fi
fingers = eval_fingers             # Eval A 遍历的是老27类
train_combined = {k: finger_train[k] + extra_train.get(k, []) for k in finger_train}
train_combined.update(new_only)
print(f"  Merged {len(extra_train)} same-finger folders into old classes")
print(f"  New-only classes: {len(new_only)} ({VARIANT}), time={time.time()-t_new:.0f}s")
print(f"  Total: {n_classes} train classes, {n_eval} eval classes, "
      f"cross-probes: {sum(len(v) for v in cross_test.values())}")
print(f"Data loading done in {time.time()-t_load:.0f}s"); sys.stdout.flush()

# ============================================================
# 模型: Layer4全局 + 注意力局部选点
# ============================================================
class HybridDelfEncoder(nn.Module):
    """V18b architecture + DELF attention on layer3.

    Global path (unchanged from V18b):
        layer3 → layer4 → GAP → Linear(512→512)+BN → 512-dim global embedding

    Attention path (NEW):
        layer3 → attention(256→128→1, Softplus) → weighted pool → Linear(256→512)+BN
        → auxiliary ArcFace classification → trains attention to select discriminative regions

    Local path (unchanged from V18b):
        layer3 → Conv1x1(256→64)+BN+L2-norm → 64-dim local descriptors

    Inference: attention selects top-K positions for local matching
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

        # Main global projector (from layer4, same as V18b)
        self.global_projector = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # DELF-style spatial attention on layer3
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Softplus(),
        )

        # Attention-weighted projector (auxiliary path for training attention)
        self.attn_projector = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # Local descriptor head (same as V18b)
        self.local_head = nn.Sequential(
            nn.Conv2d(256, local_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(local_dim),
        )

    def forward(self, x, mode='global'):
        """
        mode='global': returns global_emb only (for basic forward)
        mode='all': returns (global_emb, local_desc, attn_map, attn_emb)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # (B, 256, 14, 13)

        if mode == 'all':
            # Attention
            attn_map = self.attention(x)  # (B, 1, 14, 13)

            # Local descriptors
            local_desc = self.local_head(x)
            local_desc = F.normalize(local_desc, dim=1)

            # Attention-weighted global (auxiliary)
            feat_norm = F.normalize(x, p=2, dim=1)
            weighted = feat_norm * attn_map
            attn_feat = weighted.sum(dim=[2, 3])  # (B, 256)
            attn_emb = self.attn_projector(attn_feat)

        # Main global (layer4 path — always computed)
        x4 = self.layer4(x)
        x4 = self.avgpool(x4).flatten(1)
        global_emb = self.global_projector(x4)

        if mode == 'all':
            return global_emb, local_desc, attn_map, attn_emb
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
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, desc1, desc2):
        B, D, H, W = desc1.shape
        N = H * W
        d1 = desc1.view(B, D, N).permute(0, 2, 1)
        d2 = desc2.view(B, D, N).permute(0, 2, 1)
        total_loss = 0
        for b in range(B):
            sim = torch.mm(d1[b], d2[b].t()) / self.temperature
            labels = torch.arange(N, device=sim.device)
            total_loss += F.cross_entropy(sim, labels)
        return total_loss / B

# ============================================================
# 数据增强
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
        f = cv2.warpAffine(f, np.float32([[1,0,dx],[0,1,dy]]), (f.shape[1], f.shape[0]), borderValue=0)
    if random.random() < 0.3:
        angle = random.uniform(-3, 3)
        h, w = f.shape
        f = cv2.warpAffine(f, cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0), (w, h), borderValue=0)
    if random.random() < 0.25: f = elastic_transform_light(f)
    if random.random() < 0.15:
        h, w = f.shape
        ea = random.uniform(0.02, 0.06) * h * w
        asp = random.uniform(0.5, 2.0)
        eh = min(int(np.sqrt(ea * asp)), h - 1)
        ew = min(int(np.sqrt(ea / asp)), w - 1)
        if eh > 0 and ew > 0:
            t, l = random.randint(0, h - eh), random.randint(0, w - ew)
            f[t:t+eh, l:l+ew] = 0
    return f

def augment_frame_mild(frame):
    f = frame.copy()
    if random.random() < 0.5: f = -f
    f += np.random.randn(*f.shape).astype(np.float32) * 0.02
    if random.random() < 0.5:
        dx, dy = random.randint(-2, 2), random.randint(-2, 2)
        f = cv2.warpAffine(f, np.float32([[1,0,dx],[0,1,dy]]), (f.shape[1], f.shape[0]), borderValue=0)
    return f

# ============================================================
# 数据集
# ============================================================
class ContrastiveDS(Dataset):
    def __init__(self, finger_data, n_samples=120):
        self.finger_data = finger_data
        self.n_samples = n_samples
        self.fingers = list(finger_data.keys())
    def __len__(self): return len(self.fingers) * self.n_samples
    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        frames = self.finger_data[finger]
        frame = frames[random.randint(0, len(frames)-1)]
        f1 = augment_frame_mild(frame)
        f2 = augment_frame_mild(frame)
        return (torch.tensor(f1, dtype=torch.float32).unsqueeze(0),
                torch.tensor(f2, dtype=torch.float32).unsqueeze(0))

class ClassifyDS(Dataset):
    def __init__(self, finger_data, finger_labels, n_samples=100):
        self.finger_data = finger_data
        self.finger_labels = finger_labels
        self.n_samples = n_samples
        self.fingers = list(finger_data.keys())
    def __len__(self): return len(self.fingers) * self.n_samples
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
MODEL_SAVE_DIR = f'f:/1111/指纹/models_{ARM}/'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

N_PRETRAIN = 60
N_FINETUNE = 120
SWA_START = 100

def train_single_model(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    save_path = os.path.join(MODEL_SAVE_DIR, f'{ARM}_seed{seed}.pth')
    if os.path.exists(save_path):
        print(f"\n  Found saved model: {save_path}")
        model = HybridDelfEncoder(embed_dim=EMBED_DIM, local_dim=LOCAL_DIM).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model.eval()
        print(f"  Loaded."); sys.stdout.flush()
        return model

    print(f"\n{'='*60}")
    print(f"Training seed={seed}")
    print(f"  Phase1: SimCLR+LocalCL {N_PRETRAIN}ep | Phase2: ArcFace(main+attn)+LocalCL {N_FINETUNE}ep (SWA@{SWA_START})")
    print(f"{'='*60}"); sys.stdout.flush()

    model = HybridDelfEncoder(embed_dim=EMBED_DIM, local_dim=LOCAL_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}"); sys.stdout.flush()

    simclr_loss_fn = SimCLRLoss(temperature=0.07)
    local_loss_fn = PositionContrastiveLoss(temperature=0.1)

    # ---- Phase 1: SimCLR + Local CL ----
    print(f"  Phase 1: SimCLR(main+attn) + Local CL ({N_PRETRAIN} epochs)")
    for name, param in model.named_parameters():
        if any(k in name for k in ['global_projector', 'local_head', 'attention', 'attn_projector']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable: {trainable:,}"); sys.stdout.flush()

    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=0.001, weight_decay=1e-3)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=N_PRETRAIN)
    scaler = GradScaler('cuda') if USE_AMP else None

    pre_ds = ContrastiveDS(train_combined, n_samples=120)
    pre_loader = DataLoader(pre_ds, batch_size=32, shuffle=True, num_workers=0)

    t0 = time.time()
    for epoch in range(N_PRETRAIN):
        model.train()
        tl_g, tl_a, tl_l = 0, 0, 0
        for f1, f2 in pre_loader:
            f1, f2 = f1.to(device), f2.to(device)
            opt1.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    g1, d1, _, a1 = model(f1, mode='all')
                    g2, d2, _, a2 = model(f2, mode='all')
                    loss_g = simclr_loss_fn(F.normalize(g1, dim=1), F.normalize(g2, dim=1))
                    loss_a = simclr_loss_fn(F.normalize(a1, dim=1), F.normalize(a2, dim=1))
                    loss_l = local_loss_fn(d1, d2)
                    loss = loss_g + 0.3 * loss_a + 0.5 * loss_l
                scaler.scale(loss).backward(); scaler.step(opt1); scaler.update()
            else:
                g1, d1, _, a1 = model(f1, mode='all')
                g2, d2, _, a2 = model(f2, mode='all')
                loss_g = simclr_loss_fn(F.normalize(g1, dim=1), F.normalize(g2, dim=1))
                loss_a = simclr_loss_fn(F.normalize(a1, dim=1), F.normalize(a2, dim=1))
                loss_l = local_loss_fn(d1, d2)
                loss = loss_g + 0.3 * loss_a + 0.5 * loss_l
                loss.backward(); opt1.step()
            tl_g += loss_g.item(); tl_a += loss_a.item(); tl_l += loss_l.item()
        sch1.step()
        if (epoch+1) % 10 == 0:
            nb = len(pre_loader)
            print(f"    Epoch {epoch+1}/{N_PRETRAIN}: global={tl_g/nb:.4f}, attn={tl_a/nb:.4f}, "
                  f"local={tl_l/nb:.4f}, time={time.time()-t0:.0f}s")
            sys.stdout.flush()

    # ---- Phase 2: ArcFace(main) + ArcFace(attn) + Local CL + SWA ----
    print(f"  Phase 2: ArcFace(main+attn) + Local CL ({N_FINETUNE} epochs, SWA@{SWA_START})")
    sys.stdout.flush()

    for param in model.parameters():
        param.requires_grad = True

    arcface_main = SubCenterArcFace(EMBED_DIM, n_classes, K=3, s=64, m=0.5,
                                     label_smoothing=0.1).to(device)
    arcface_attn = SubCenterArcFace(EMBED_DIM, n_classes, K=3, s=64, m=0.5,
                                     label_smoothing=0.1).to(device)

    param_groups = [
        {'params': list(model.conv1.parameters()) + list(model.bn1.parameters()) +
                   list(model.layer1.parameters()) + list(model.layer2.parameters()),
         'lr': 1e-5},
        {'params': list(model.layer3.parameters()) + list(model.local_head.parameters()),
         'lr': 5e-5},
        {'params': list(model.layer4.parameters()),
         'lr': 1e-4},
        {'params': list(model.attention.parameters()) + list(model.attn_projector.parameters()) +
                   list(arcface_attn.parameters()),
         'lr': 1e-4},
        {'params': list(model.global_projector.parameters()) + list(arcface_main.parameters()),
         'lr': 3e-4},
    ]

    opt2 = optim.AdamW(param_groups, weight_decay=1e-3)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=N_FINETUNE)
    scaler2 = GradScaler('cuda') if USE_AMP else None

    ft_ds = ClassifyDS(train_combined, finger_labels, n_samples=100)
    ft_loader = DataLoader(ft_ds, batch_size=32, shuffle=True, num_workers=0)

    from torch.optim.swa_utils import AveragedModel
    swa_model = AveragedModel(model)
    swa_n = 0

    t1 = time.time()
    for epoch in range(N_FINETUNE):
        model.train(); arcface_main.train(); arcface_attn.train()
        tl_m, tl_a, tl_l, cor, tot = 0, 0, 0, 0, 0
        for X, X_mild, Y in ft_loader:
            X, X_mild, Y = X.to(device), X_mild.to(device), Y.to(device)
            opt2.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    g1, d1, _, a1 = model(X, mode='all')
                    loss_main = arcface_main(g1, Y)
                    loss_attn = arcface_attn(a1, Y)
                    _, d2, _, _ = model(X_mild, mode='all')
                    loss_local = local_loss_fn(d1, d2)
                    loss = loss_main + 0.3 * loss_attn + 0.3 * loss_local
                scaler2.scale(loss).backward(); scaler2.step(opt2); scaler2.update()
            else:
                g1, d1, _, a1 = model(X, mode='all')
                loss_main = arcface_main(g1, Y)
                loss_attn = arcface_attn(a1, Y)
                _, d2, _, _ = model(X_mild, mode='all')
                loss_local = local_loss_fn(d1, d2)
                loss = loss_main + 0.3 * loss_attn + 0.3 * loss_local
                loss.backward(); opt2.step()
            tl_m += loss_main.item(); tl_a += loss_attn.item(); tl_l += loss_local.item()
            with torch.no_grad():
                en = F.normalize(g1.float(), dim=1)
                wn = F.normalize(arcface_main.weight.float(), dim=1)
                cos_all = torch.mm(en, wn.t()).view(-1, n_classes, 3)
                cos, _ = cos_all.max(dim=2)
                _, p = cos.max(1)
                cor += p.eq(Y).sum().item(); tot += Y.size(0)
        sch2.step()

        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_n += 1

        if (epoch+1) % 10 == 0:
            nb = len(ft_loader)
            print(f"    Epoch {epoch+1}/{N_FINETUNE}: main={tl_m/nb:.4f}, attn={tl_a/nb:.4f}, "
                  f"local={tl_l/nb:.4f}, acc={cor/tot*100:.1f}%, "
                  f"time={time.time()-t1:.0f}s{' [SWA]' if epoch >= SWA_START else ''}")
            sys.stdout.flush()

    if swa_n > 0:
        print(f"  Updating SWA BN ({swa_n} averages)..."); sys.stdout.flush()
        class SimpleBNDS(Dataset):
            def __init__(self, finger_data, n_samples=20):
                self.items = []
                for k, frames in finger_data.items():
                    for _ in range(n_samples):
                        f = frames[random.randint(0, len(frames)-1)]
                        self.items.append(torch.tensor(f, dtype=torch.float32).unsqueeze(0))
            def __len__(self): return len(self.items)
            def __getitem__(self, idx): return self.items[idx]

        swa_mod = swa_model.module
        swa_mod.train()
        for module in swa_mod.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.reset_running_stats()
                module.momentum = None
        bn_ds = SimpleBNDS(train_combined, n_samples=20)
        bn_loader = DataLoader(bn_ds, batch_size=32, shuffle=True, num_workers=0)
        with torch.no_grad():
            for X_bn in bn_loader:
                swa_mod(X_bn.to(device), mode='all')
        model.load_state_dict(swa_mod.state_dict())
        print(f"  SWA applied."); sys.stdout.flush()

    model.eval()
    total_time = time.time() - t0
    print(f"  Model done. Total: {total_time:.0f}s ({total_time/60:.1f}min)")
    torch.save(model.state_dict(), save_path)
    print(f"  Saved: {save_path}"); sys.stdout.flush()
    return model

models_list = []
for seed in SEEDS:
    m = train_single_model(seed)
    models_list.append(m)
print(f"\nAll models ready in {time.time()-t_start:.0f}s"); sys.stdout.flush()

# ============================================================
# 嵌入提取
# ============================================================
@torch.no_grad()
def extract_features(models, frame):
    global_embs = []
    local_per_model = []
    attn_per_model = []

    for model in models:
        g_list, d_list, a_list = [], [], []
        for polarity in [1, -1]:
            x = torch.tensor(polarity * frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            g, d, a_map, _ = model(x, mode='all')
            g_list.append(F.normalize(g, dim=1))
            d_list.append(d[0].view(LOCAL_DIM, -1).t())
            a_list.append(a_map[0, 0].view(-1))

        global_embs.extend(g_list)
        d_avg = F.normalize(torch.mean(torch.stack(d_list), dim=0), dim=1)
        local_per_model.append(d_avg)
        a_avg = torch.mean(torch.stack(a_list), dim=0)
        attn_per_model.append(a_avg)

    g_avg = F.normalize(torch.mean(torch.stack(global_embs), dim=0), dim=1).cpu().numpy().flatten()
    return g_avg, local_per_model, attn_per_model

# ============================================================
# 评估
# ============================================================
print(f"\n{'='*70}")
print(f"EVAL A ({ARM}): old protocol identical to V21 (old {n_eval} classes)")
print(f"{'='*70}"); sys.stdout.flush()

print("Step 1: Extracting template features..."); sys.stdout.flush()
t_tmpl = time.time()
random.seed(42)

template_global = {}
template_local = {}
template_attn = {}
template_local_concat = {}

for finger in fingers:
    frames = finger_train[finger]
    indices = random.sample(range(len(frames)), min(N_TEMPLATES, len(frames)))
    g_list, l_list, a_list = [], [], []
    for i in indices:
        g, l, a = extract_features(models_list, frames[i])
        g_list.append(g)
        l_list.append(l)
        a_list.append(a)
    template_global[finger] = g_list
    template_local[finger] = l_list
    template_attn[finger] = a_list

    per_model_concat = []
    for m in range(len(SEEDS)):
        parts = [template_local[finger][t][m] for t in range(len(l_list))]
        per_model_concat.append(torch.cat(parts, dim=0))
    template_local_concat[finger] = per_model_concat

print(f"  Template extraction done in {time.time()-t_tmpl:.0f}s"); sys.stdout.flush()

print("Step 2: Computing scores..."); sys.stdout.flush()
t_match = time.time()

all_probes = []
for fi_idx, probe_finger in enumerate(fingers):
    test_frames = finger_test[probe_finger]
    for frame in test_frames:
        g_emb, l_descs, a_descs = extract_features(models_list, frame)
        scores = {}
        for identity in fingers:
            global_score = max(np.dot(g_emb, t_g) for t_g in template_global[identity])

            model_dense_all, model_dense_topk, model_dense_weighted = [], [], []
            for m in range(len(SEEDS)):
                Q = l_descs[m]
                T = template_local_concat[identity][m]
                Q_attn = a_descs[m]

                S = torch.mm(Q, T.t())
                max_sim, _ = S.max(dim=1)

                # V18b baseline: all positions, top-70%
                k_all = max(1, int(0.7 * Q.shape[0]))
                top_v, _ = max_sim.topk(k_all)
                model_dense_all.append(top_v.mean().item())

                # Attention top-K
                K_SEL = 90
                topk_idx = Q_attn.topk(min(K_SEL, Q.shape[0]))[1]
                model_dense_topk.append(max_sim[topk_idx].mean().item())

                # Attention-weighted
                w = Q_attn / (Q_attn.sum() + 1e-8)
                model_dense_weighted.append((max_sim * w).sum().item())

            scores[identity] = {
                'global': global_score,
                'dense_all': np.mean(model_dense_all),
                'dense_topk': np.mean(model_dense_topk),
                'dense_weighted': np.mean(model_dense_weighted),
            }
        all_probes.append({'probe_finger': probe_finger, 'scores': scores})

    if (fi_idx+1) % 5 == 0 or fi_idx == 0:
        print(f"  [{fi_idx+1}/{n_eval}] elapsed={time.time()-t_match:.0f}s"); sys.stdout.flush()

print(f"  Matching done in {time.time()-t_match:.0f}s"); sys.stdout.flush()

# ============================================================
# Score normalization + evaluation
# ============================================================
def apply_tnorm(raw_score, claimed_id, probe_scores, key):
    imp = [probe_scores[k][key] for k in probe_scores if k != claimed_id]
    return (raw_score - np.mean(imp)) / (np.std(imp) + 1e-8)

print("Step 3: Evaluating variants..."); sys.stdout.flush()

DENSE_KEYS = ['dense_all', 'dense_topk', 'dense_weighted']

def collect_scores(all_probes):
    variants = ['global_raw', 'global_tnorm']
    for dk in DENSE_KEYS:
        variants += [f'{dk}_raw', f'{dk}_tnorm']
    for dk in DENSE_KEYS:
        for alpha in [0.3, 0.5, 0.7]:
            variants += [f'fused_{dk}_a{alpha}_raw', f'fused_{dk}_a{alpha}_tnorm']

    results = {v: {'gen': [], 'imp': []} for v in variants}
    pf_gen_g = {f: [] for f in fingers}

    for probe in all_probes:
        pf = probe['probe_finger']
        sc = probe['scores']
        for identity in fingers:
            g = sc[identity]['global']
            is_gen = (identity == pf)
            g_t = apply_tnorm(g, identity, sc, 'global')

            tag = 'gen' if is_gen else 'imp'
            results['global_raw'][tag].append(g)
            results['global_tnorm'][tag].append(g_t)

            for dk in DENSE_KEYS:
                d = sc[identity][dk]
                d_t = apply_tnorm(d, identity, sc, dk)
                results[f'{dk}_raw'][tag].append(d)
                results[f'{dk}_tnorm'][tag].append(d_t)

                for alpha in [0.3, 0.5, 0.7]:
                    fused_raw = alpha * d + (1 - alpha) * g
                    results[f'fused_{dk}_a{alpha}_raw'][tag].append(fused_raw)
                    imp_fused = [alpha * sc[o][dk] + (1 - alpha) * sc[o]['global']
                                 for o in sc if o != identity]
                    fused_t = (fused_raw - np.mean(imp_fused)) / (np.std(imp_fused) + 1e-8)
                    results[f'fused_{dk}_a{alpha}_tnorm'][tag].append(fused_t)

            if is_gen:
                pf_gen_g[pf].append(g)

    for k in results:
        results[k]['gen'] = np.array(results[k]['gen'])
        results[k]['imp'] = np.array(results[k]['imp'])
    return results, pf_gen_g

results, pf_gen_g = collect_scores(all_probes)

def full_analysis(name, gen, imp, verbose=True):
    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    y_scores = np.concatenate([gen, imp])
    fpr_arr, tpr, _ = roc_curve(y_true, y_scores)
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

VERBOSE_KEYS = ['global_tnorm', 'fused_dense_all_a0.5_tnorm',
                'fused_dense_topk_a0.5_tnorm', 'fused_dense_weighted_a0.5_tnorm']

eer_results = {}
for key in sorted(results.keys()):
    gen, imp = results[key]['gen'], results[key]['imp']
    if len(gen) == 0: continue
    eer = full_analysis(key, gen, imp, verbose=(key in VERBOSE_KEYS))
    eer_results[key] = eer

best_key = min(eer_results, key=eer_results.get)
best_eer = eer_results[best_key]

print(f"\n{'='*70}")
print(f"ALL VARIANT EER COMPARISON")
print(f"{'='*70}")

categories = {
    'Global': [k for k in sorted(eer_results) if k.startswith('global_')],
    'Dense (all, V18b-style)': [k for k in sorted(eer_results) if k.startswith('dense_all')],
    'Dense (top-K attn)': [k for k in sorted(eer_results) if k.startswith('dense_topk')],
    'Dense (attn-weighted)': [k for k in sorted(eer_results) if k.startswith('dense_weighted')],
    'Fused (all, V18b-style)': [k for k in sorted(eer_results) if 'fused_dense_all' in k],
    'Fused (top-K attn)': [k for k in sorted(eer_results) if 'fused_dense_topk' in k],
    'Fused (attn-weighted)': [k for k in sorted(eer_results) if 'fused_dense_weighted' in k],
}
for cat_name, keys in categories.items():
    if not keys: continue
    print(f"\n  --- {cat_name} ---")
    for key in keys:
        marker = " ***" if key == best_key else ""
        print(f"    {key:45s}: EER = {eer_results[key]*100:.4f}%{marker}")
sys.stdout.flush()

# Per-finger
print(f"\n--- Per-finger Genuine (global, layer4 embedding) ---")
pf_stats = []
for finger in fingers:
    scores = pf_gen_g[finger]
    if scores:
        arr = np.array(scores)
        src = finger_source.get(finger, "?")
        pf_stats.append((finger, src, arr.mean(), arr.min(), len(arr)))
pf_stats.sort(key=lambda x: x[2])
for i, (fn, src, gm, gmin, n) in enumerate(pf_stats):
    status = "*** WORST ***" if i < 3 else ""
    print(f"  {i+1}. {fn} [{src}]: mean={gm:.4f}, min={gmin:.4f}, n={n} {status}")

# Best variant details
print(f"\n{'='*70}")
print(f"BEST VARIANT DETAILED ANALYSIS (EVAL A)")
print(f"{'='*70}")
full_analysis(best_key, results[best_key]['gen'], results[best_key]['imp'], verbose=True)

# ============================================================
# EVAL B: 跨场景评估 (老场景模板 x 新场景探针)
# ============================================================
print(f"\n{'='*70}")
print(f"EVAL B: CROSS-SESSION (old-session templates x new-session probes, {VARIANT})")
print(f"  V21 frozen baseline: raw EER=56.91%, denoised EER=51.26% (global_raw)")
print(f"{'='*70}"); sys.stdout.flush()
t_cross = time.time()
cross_probes = []
for ci, (cross_key, te_frames) in enumerate(cross_test.items()):
    for frame in te_frames:
        g_emb, l_descs, a_descs = extract_features(models_list, frame)
        scores = {}
        for identity in fingers:
            global_score = max(np.dot(g_emb, t_g) for t_g in template_global[identity])

            model_dense_all, model_dense_topk, model_dense_weighted = [], [], []
            for m in range(len(SEEDS)):
                Q = l_descs[m]
                T = template_local_concat[identity][m]
                Q_attn = a_descs[m]

                S = torch.mm(Q, T.t())
                max_sim, _ = S.max(dim=1)

                k_all = max(1, int(0.7 * Q.shape[0]))
                top_v, _ = max_sim.topk(k_all)
                model_dense_all.append(top_v.mean().item())

                K_SEL = 90
                topk_idx = Q_attn.topk(min(K_SEL, Q.shape[0]))[1]
                model_dense_topk.append(max_sim[topk_idx].mean().item())

                w = Q_attn / (Q_attn.sum() + 1e-8)
                model_dense_weighted.append((max_sim * w).sum().item())

            scores[identity] = {
                'global': global_score,
                'dense_all': np.mean(model_dense_all),
                'dense_topk': np.mean(model_dense_topk),
                'dense_weighted': np.mean(model_dense_weighted),
            }
        cross_probes.append({'probe_finger': cross_key, 'scores': scores})
    print(f"  [{ci+1}/{len(cross_test)}] {cross_key} done, elapsed={time.time()-t_cross:.0f}s"); sys.stdout.flush()

cross_results, cross_pf = collect_scores(cross_probes)
cross_eers = {}
for key in sorted(cross_results.keys()):
    gen, imp = cross_results[key]['gen'], cross_results[key]['imp']
    if len(gen) == 0:
        continue
    cross_eers[key] = full_analysis(key, gen, imp, verbose=False)
cross_best = min(cross_eers, key=cross_eers.get)

print(f"\n  --- Cross-session EER (all variants) ---")
for key in sorted(cross_eers):
    marker = " ***" if key == cross_best else ""
    print(f"    {key:45s}: EER = {cross_eers[key]*100:.4f}%{marker}")

print(f"\n  --- Cross-session per-finger genuine (global) ---")
for cross_key in cross_test:
    arr = np.array(cross_pf[cross_key])
    if len(arr):
        print(f"    {cross_key:14s}: mean={arr.mean():.4f}, min={arr.min():.4f}, n={len(arr)}")

print(f"\n{'='*70}")
print(f"CROSS-SESSION BEST DETAIL: {cross_best}")
print(f"{'='*70}")
full_analysis(cross_best, cross_results[cross_best]['gen'], cross_results[cross_best]['imp'], verbose=True)

# Final summary
print(f"\n{'='*70}")
print(f"V25 ({ARM}, same-finger merge, +ysjz {VARIANT}) FINAL SUMMARY")
print(f"{'='*70}")
print(f"\n  === EVAL A (old protocol, comparable to V21) ===")
print(f"  *** Best: {best_key} with EER = {best_eer*100:.4f}% ***")
print(f"  *** global_raw (V21's best variant): EER = {eer_results['global_raw']*100:.4f}% ***")
print(f"  V21 hybrid (no new data): EER = 1.1898%")
print(f"  V24 {('a' if VARIANT=='raw' else 'b')} (new data as separate classes): "
      f"EER = {'1.2608%' if VARIANT=='raw' else '1.8829%'}")
improvement = 1.1898 - best_eer * 100
print(f"  IMPROVEMENT vs V21: {improvement:+.4f}%  (positive = better)")
print(f"\n  === EVAL B (cross-session: old templates x new probes) ===")
print(f"  *** Best: {cross_best} with EER = {cross_eers[cross_best]*100:.4f}% ***")
print(f"  *** global_raw: EER = {cross_eers['global_raw']*100:.4f}% ***")
print(f"  V21 frozen baseline:     EER = {'56.91%' if VARIANT=='raw' else '51.26%'} (global_raw)")
print(f"\nTotal time: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")
print(f"{'='*70}"); sys.stdout.flush()
