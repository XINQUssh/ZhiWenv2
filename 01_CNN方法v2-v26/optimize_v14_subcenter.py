"""
V14: 基于V12验证架构的精准优化
核心策略: 保留V12已验证有效的一切, 只做精准改进
  1. ResNet-18预训练骨干 + 简单1层投影头 (V12架构)
  2. SimCLR预训练 + ArcFace微调 (V12训练流程)
  3. Sub-center ArcFace (K=3): 处理类内多模态分布
  4. 嵌入维度512 (vs V12的256)
  5. 更长训练: SimCLR 80ep + ArcFace 120ep
  6. 轻度弹性形变增强
  7. 推理: 双极性 + 帧质量过滤
  8. 3模型集成
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
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    if n_labels > 1:
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = ((labels == max_label) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=2)
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
    # 计算掩膜面积比 (用于质量评估)
    mask_ratio = np.sum(mask > 0) / mask.size
    return result, mask_ratio

# ============================================================
# 数据加载
# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
WTP_REG = 'Rgd1245'
BTP_REG = 'Rgd1237'

finger_train = {}
finger_test = {}
finger_labels = {}
finger_source = {}
finger_train_quality = {}
finger_test_quality = {}
fi = 0

print(f"Loading data (无贴屏:{WTP_REG}, 不贴屏:{BTP_REG})..."); sys.stdout.flush()
print("Preprocessing: CLAHE + 2x upsample + GMFS mask"); sys.stdout.flush()

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
print(f"Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes-n_wtp} 不贴屏)")

# 打印质量统计
all_q = []
for k in fingers:
    all_q.extend(finger_train_quality[k])
    all_q.extend(finger_test_quality[k])
all_q = np.array(all_q)
print(f"  Mask area ratio: mean={all_q.mean():.3f}, min={all_q.min():.3f}, "
      f"max={all_q.max():.3f}, <0.1={np.sum(all_q<0.1)}")
print(f"Data loading done in {time.time()-t_load:.0f}s"); sys.stdout.flush()

# ============================================================
# 模型: ResNet-18 + 简单投影头 (V12证明有效的架构)
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
        # 简单1层投影 (V12架构, 已验证优于多层)
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
# Sub-center ArcFace
# ============================================================
class SubCenterArcFace(nn.Module):
    """
    Sub-center ArcFace: 每个类有K个子中心
    处理类内多模态 (不同按压力度/角度产生的不同分布)
    匹配时取最近的子中心, 减少类内方差
    """
    def __init__(self, embed_dim, n_classes, K=3, s=64.0, m=0.5):
        super().__init__()
        self.s, self.m, self.K = s, m, K
        self.n_classes = n_classes
        # K个子中心 per class
        self.weight = nn.Parameter(torch.FloatTensor(n_classes * K, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels):
        emb_n = F.normalize(emb, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        # [B, n_classes * K]
        cos_all = torch.mm(emb_n, w_n.t()).clamp(-1+1e-7, 1-1e-7)
        # reshape → [B, n_classes, K], 每个类取max子中心
        cos_all = cos_all.view(-1, self.n_classes, self.K)
        cos, _ = cos_all.max(dim=2)  # [B, n_classes]

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
# 数据增强 (轻度弹性形变, 不过激)
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
    # 轻度弹性形变 (25%概率)
    if random.random() < 0.25:
        f = elastic_transform_light(f)
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
# 训练 (V12验证流程: SimCLR冻结 → ArcFace渐进解冻)
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}"); sys.stdout.flush()

N_PRETRAIN = 80     # SimCLR (比V12多20个epoch)
N_FINETUNE = 120    # ArcFace (比V12多40个epoch)
EMBED_DIM = 512     # 比V12的256更大
SEEDS = [42, 123, 777]
USE_AMP = True

def train_single_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"Training model with seed={seed}")
    print(f"{'='*60}"); sys.stdout.flush()

    model = FingerprintEncoder(embed_dim=EMBED_DIM).to(device)
    simclr_loss = SimCLRLoss(temperature=0.07)

    # ---- Phase 1: SimCLR (骨干冻结) ----
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
        model.train()
        tl = 0
        for f1, f2 in pre_loader:
            f1, f2 = f1.to(device), f2.to(device)
            opt1.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    z1 = F.normalize(model(f1), dim=1)
                    z2 = F.normalize(model(f2), dim=1)
                    loss = simclr_loss(z1, z2)
                scaler.scale(loss).backward()
                scaler.step(opt1)
                scaler.update()
            else:
                z1 = F.normalize(model(f1), dim=1)
                z2 = F.normalize(model(f2), dim=1)
                loss = simclr_loss(z1, z2)
                loss.backward()
                opt1.step()
            tl += loss.item()
        sch1.step()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_PRETRAIN}: loss={tl/len(pre_loader):.4f}, "
                  f"time={time.time()-t0:.0f}s"); sys.stdout.flush()

    # ---- Phase 2: Sub-center ArcFace (渐进解冻) ----
    print(f"  Phase 2: Sub-center ArcFace ({N_FINETUNE} epochs, progressive unfreeze)")
    sys.stdout.flush()

    for param in model.parameters():
        param.requires_grad = True

    arcface = SubCenterArcFace(EMBED_DIM, n_classes, K=3, s=64, m=0.5).to(device)

    # 差分学习率 (V12验证有效)
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
                scaler2.scale(loss).backward()
                scaler2.step(opt2)
                scaler2.update()
            else:
                emb = model(X)
                loss = arcface(emb, Y)
                loss.backward()
                opt2.step()
            tl += loss.item()
            with torch.no_grad():
                # 分类精度 (用子中心的max)
                en = F.normalize(emb.float(), dim=1)
                wn = F.normalize(arcface.weight.float(), dim=1)
                cos_all = torch.mm(en, wn.t()).view(-1, n_classes, 3)
                cos, _ = cos_all.max(dim=2)
                _, p = cos.max(1)
                cor += p.eq(Y).sum().item(); tot += Y.size(0)
            tl += loss.item()
        sch2.step()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_FINETUNE}: loss={tl/len(ft_loader)/2:.4f}, "
                  f"acc={cor/tot*100:.1f}%, time={time.time()-t1:.0f}s"); sys.stdout.flush()

    model.eval()
    print(f"  Model done. Total: {time.time()-t0:.0f}s"); sys.stdout.flush()
    return model

models = []
t_total = time.time()
for seed in SEEDS:
    m = train_single_model(seed)
    models.append(m)

print(f"\nAll models trained in {time.time()-t_total:.0f}s"); sys.stdout.flush()

# ============================================================
# 嵌入提取 (双极性)
# ============================================================
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
    emb_orig = get_ensemble_emb(models, frame)
    emb_flip = get_ensemble_emb(models, -frame)
    return emb_orig, emb_flip

def dual_polarity_max_sim(query_embs, template_embs):
    q_orig, q_flip = query_embs
    t_orig, t_flip = template_embs
    return max(np.dot(q_orig, t_orig), np.dot(q_orig, t_flip),
               np.dot(q_flip, t_orig), np.dot(q_flip, t_flip))

# ============================================================
# 评估
# ============================================================
print(f"\n{'='*60}")
print(f"评估: V14 (ResNet-18 + Sub-center ArcFace K=3 + embed512)")
print(f"{'='*60}"); sys.stdout.flush()

N_TEMPLATES = 20

# 模板嵌入
print("Extracting template embeddings..."); sys.stdout.flush()
t_emb = time.time()
templates = {}
for finger in fingers:
    train_frames = finger_train[finger]
    chosen = random.sample(train_frames, min(N_TEMPLATES, len(train_frames)))
    templates[finger] = [get_dual_polarity_emb(models, f) for f in chosen]
print(f"  Templates done in {time.time()-t_emb:.0f}s"); sys.stdout.flush()

# 匹配
print("Matching test frames..."); sys.stdout.flush()
t_eval = time.time()

genuine_scores = []
impostor_scores = []
genuine_quality = []
impostor_quality = []

for fi_idx, finger in enumerate(fingers):
    test_frames = finger_test[finger]
    test_q = finger_test_quality[finger]
    if not test_frames:
        continue

    for idx, frame in enumerate(test_frames):
        query_embs = get_dual_polarity_emb(models, frame)
        qr = test_q[idx]

        # Genuine
        own_sims = [dual_polarity_max_sim(query_embs, t) for t in templates[finger]]
        genuine_scores.append(max(own_sims))
        genuine_quality.append(qr)

        # Impostor
        for other in fingers:
            if other == finger:
                continue
            other_sims = [dual_polarity_max_sim(query_embs, t) for t in templates[other]]
            impostor_scores.append(max(other_sims))
            impostor_quality.append(qr)

    if (fi_idx+1) % 5 == 0 or fi_idx == 0:
        print(f"  [{fi_idx+1}/{n_classes}] elapsed={time.time()-t_eval:.0f}s"); sys.stdout.flush()

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)
genuine_quality = np.array(genuine_quality)
impostor_quality = np.array(impostor_quality)

print(f"\nEvaluation done in {time.time()-t_eval:.0f}s"); sys.stdout.flush()

# ============================================================
# 结果分析
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
    sys.stdout.flush()
    return eer

eer_main = full_analysis("V14 全部帧", genuine_scores, impostor_scores)

# ============================================================
# 质量过滤: 去掉掩膜面积小的帧
# ============================================================
print(f"\n{'='*60}")
print(f"质量过滤实验")
print(f"{'='*60}"); sys.stdout.flush()

for min_quality in [0.05, 0.10, 0.15, 0.20]:
    gq_mask = genuine_quality >= min_quality
    iq_mask = impostor_quality >= min_quality
    gen_filt = genuine_scores[gq_mask]
    imp_filt = impostor_scores[iq_mask]
    if len(gen_filt) < 10 or len(imp_filt) < 10:
        print(f"  min_quality={min_quality}: too few samples")
        continue
    y_t = np.concatenate([np.ones(len(gen_filt)), np.zeros(len(imp_filt))])
    y_s = np.concatenate([gen_filt, imp_filt])
    fpr_f, tpr_f, th_f = roc_curve(y_t, y_s)
    fnr_f = 1 - tpr_f
    ei = np.nanargmin(np.abs(fnr_f - fpr_f))
    eer_f = (fpr_f[ei] + fnr_f[ei]) / 2
    idx3 = np.argmin(np.abs(fnr_f - 0.03))
    far3 = fpr_f[idx3]
    idx5 = np.argmin(np.abs(fnr_f - 0.05))
    far5 = fpr_f[idx5]
    print(f"  min_quality={min_quality:.2f}: kept={gq_mask.sum()}/{len(genuine_scores)} gen, "
          f"EER={eer_f*100:.4f}%, FFR=3%->FAR={far3*100:.4f}%, FFR=5%->FAR={far5*100:.4f}%")
    sys.stdout.flush()

# ============================================================
# 分组分析
# ============================================================
print(f"\n{'='*60}")
print(f"分组分析 (双极性)")
print(f"{'='*60}"); sys.stdout.flush()

def group_analysis(group_name, group_fingers):
    g_gen, g_imp = [], []
    for finger in group_fingers:
        test_frames = finger_test.get(finger, [])
        for frame in test_frames:
            q_embs = get_dual_polarity_emb(models, frame)
            own_sims = [dual_polarity_max_sim(q_embs, t) for t in templates[finger]]
            g_gen.append(max(own_sims))
            for other in group_fingers:
                if other == finger:
                    continue
                other_sims = [dual_polarity_max_sim(q_embs, t) for t in templates[other]]
                g_imp.append(max(other_sims))

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

    gmin, imax = g_gen.min(), g_imp.max()
    sep = "PERFECT" if gmin > imax else f"overlap={np.sum(g_imp > gmin)}/{len(g_imp)}"
    print(f"\n  [{group_name}] gen={len(g_gen)}, imp={len(g_imp)}")
    print(f"  EER = {eer_g*100:.4f}%  {sep}")
    print(f"  gen: mean={g_gen.mean():.4f}, std={g_gen.std():.4f}, min={gmin:.4f}")
    print(f"  imp: mean={g_imp.mean():.4f}, std={g_imp.std():.4f}, max={imax:.4f}")
    for tf in [0.0, 0.03, 0.05]:
        idx = np.argmin(np.abs(fnr_g - tf))
        print(f"  FFR={tf*100:.0f}% -> FAR={fpr_g[idx]*100:.4f}%")
    tf_idx = np.argmin(np.abs(fpr_g - 0.00002))
    print(f"  FAR=0.002% -> FFR={fnr_g[tf_idx]*100:.2f}%")
    sys.stdout.flush()

wtp = [f for f in fingers if f.startswith("wtp_")]
btp = [f for f in fingers if f.startswith("btp_")]
group_analysis("无贴屏", wtp)
group_analysis("不贴屏", btp)

# ============================================================
# 参数化外推
# ============================================================
print(f"\n{'='*60}")
print(f"参数化分布外推")
print(f"{'='*60}"); sys.stdout.flush()

from scipy import stats

gen_mu, gen_std = genuine_scores.mean(), genuine_scores.std()
imp_mu, imp_std = impostor_scores.mean(), impostor_scores.std()
d_prime = (gen_mu - imp_mu) / np.sqrt(0.5 * (gen_std**2 + imp_std**2))

print(f"Genuine:  mu={gen_mu:.4f}, std={gen_std:.4f}")
print(f"Impostor: mu={imp_mu:.4f}, std={imp_std:.4f}")
print(f"d-prime = {d_prime:.2f}")

for target_ffr in [0.01, 0.03, 0.05]:
    thresh = gen_mu - stats.norm.ppf(1 - target_ffr) * gen_std
    far_theory = stats.norm.cdf(thresh, loc=imp_mu, scale=imp_std)
    print(f"  FFR={target_ffr*100:.0f}% -> 理论FAR={far_theory*100:.6f}%")

for target_far in [0.00002, 0.0001, 0.001]:
    thresh = stats.norm.ppf(target_far, loc=imp_mu, scale=imp_std)
    ffr_theory = 1 - stats.norm.cdf(thresh, loc=gen_mu, scale=gen_std)
    print(f"  FAR={target_far*100:.4f}% -> 理论FFR={ffr_theory*100:.2f}%")

sys.stdout.flush()

# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print(f"V14 总结对比")
print(f"{'='*60}")
print(f"  V14 (Sub-center ArcFace K=3, embed512) EER = {eer_main*100:.4f}%")
print(f"  V12 (ArcFace, embed256)                EER = 4.0039%")
print(f"  V11 (4层CNN, embed256)                 EER = 4.9968%")
print(f"  V10 SIFT                               EER = 1.5200% (target)")
print(f"\nAll done. Total time: {time.time()-t_total:.0f}s")
print(f"{'='*60}"); sys.stdout.flush()
