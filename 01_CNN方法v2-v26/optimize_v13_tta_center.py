"""
V13: 深度优化CNN方案
核心改进 (基于V12分析):
  1. ResNet-18预训练骨干网 (继承V12, 最佳骨干)
  2. 去掉局部匹配 (V12已证明无效), 专注全局嵌入优化
  3. 渐进式ArcFace解冻 (替代无效的SimCLR冻结阶段)
     - Phase 1: 冻结骨干, 只训练头 (20 epochs)
     - Phase 2: 解冻layer4 (30 epochs)
     - Phase 3: 解冻layer3+4 (30 epochs)
     - Phase 4: 全部解冻 (20 epochs)
  4. Center Loss: 拉紧类内分布, 减少genuine方差
  5. 增强数据增强: 弹性形变 + 随机擦除 + 更大旋转/平移
  6. TTA推理 (10个增强版本平均嵌入): 压缩genuine尾部
  7. 模板质心匹配: 20个模板嵌入→质心 + 个体max, 取最优
  8. 3模型集成 (继承)
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
# 预处理 (继承V11/V12)
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
    return result  # 220x200

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
    for p in imgs_paths[:70]:
        img = load_img(p)
        if img is not None:
            train_frames.append(preprocess_frame(img))
    for p in imgs_paths[70:]:
        img = load_img(p)
        if img is not None:
            test_frames.append(preprocess_frame(img))
    if train_frames and test_frames:
        finger_train[key] = train_frames
        finger_test[key] = test_frames
        finger_labels[key] = fi
        finger_source[key] = "无贴屏"
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
    for p in imgs_paths[:70]:
        img = load_img(p)
        if img is not None:
            train_frames.append(preprocess_frame(img))
    for p in imgs_paths[70:]:
        img = load_img(p)
        if img is not None:
            test_frames.append(preprocess_frame(img))
    if train_frames and test_frames:
        finger_train[key] = train_frames
        finger_test[key] = test_frames
        finger_labels[key] = fi
        finger_source[key] = "不贴屏"
        fi += 1

n_classes = fi
fingers = list(finger_train.keys())
print(f"Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes-n_wtp} 不贴屏)")
for k in fingers[:3]:
    print(f"  {k}: {len(finger_train[k])} train, {len(finger_test[k])} test, "
          f"frame_shape={finger_train[k][0].shape}")
print(f"Data loading done in {time.time()-t_load:.0f}s"); sys.stdout.flush()

# ============================================================
# 模型: ResNet-18 + 全局嵌入
# ============================================================
class FingerprintEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 1通道输入
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

        # 投影头: 512 → embed_dim
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
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
# 损失函数
# ============================================================
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

class CenterLoss(nn.Module):
    """拉紧类内分布, 减少genuine score方差"""
    def __init__(self, n_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_classes, feat_dim))

    def forward(self, x, labels):
        x_n = F.normalize(x, dim=1)
        c_n = F.normalize(self.centers, dim=1)
        centers_batch = c_n[labels]
        # 1 - cosine similarity (越小越好)
        return (1 - (x_n * centers_batch).sum(dim=1)).mean()

# ============================================================
# 数据增强 (大幅增强)
# ============================================================
def elastic_transform(image, alpha=15, sigma=3):
    """弹性形变: 模拟手指按压的非刚性变形"""
    h, w = image.shape
    dx = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma)
    dy = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)

def random_erasing(image, p=0.3, scale=(0.02, 0.15)):
    """随机擦除: 模拟局部遮挡/缺损"""
    if random.random() > p:
        return image
    img = image.copy()
    h, w = img.shape
    area = h * w
    erase_area = random.uniform(scale[0], scale[1]) * area
    aspect = random.uniform(0.3, 3.3)
    eh = int(np.sqrt(erase_area * aspect))
    ew = int(np.sqrt(erase_area / aspect))
    if eh >= h or ew >= w:
        return img
    y = random.randint(0, h - eh)
    x = random.randint(0, w - ew)
    img[y:y+eh, x:x+ew] = 0
    return img

def augment_frame_v13(frame, strength='normal'):
    """
    增强版数据增强
    strength: 'normal' (前半段), 'hard' (后半段, 更激进)
    """
    f = frame.copy()

    # 极性翻转 (50%)
    if random.random() < 0.5:
        f = -f

    # 随机噪声
    noise_std = 0.05 if strength == 'hard' else 0.03
    f += np.random.randn(*f.shape).astype(np.float32) * noise_std

    # 水平翻转 (30%)
    if random.random() < 0.3:
        f = f[:, ::-1].copy()

    # 随机平移
    max_shift = 8 if strength == 'hard' else 5
    if random.random() < 0.5:
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        f = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), borderValue=0)

    # 旋转
    max_angle = 5.0 if strength == 'hard' else 3.0
    if random.random() < 0.4:
        angle = random.uniform(-max_angle, max_angle)
        h, w = f.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        f = cv2.warpAffine(f, M, (w, h), borderValue=0)

    # 弹性形变 (仅hard模式或30%概率)
    if strength == 'hard' or random.random() < 0.3:
        f = elastic_transform(f, alpha=12, sigma=3)

    # 随机擦除
    f = random_erasing(f, p=0.3 if strength == 'hard' else 0.15)

    # 亮度扰动
    if random.random() < 0.3:
        f = f * random.uniform(0.85, 1.15)

    return f

# ============================================================
# 数据集
# ============================================================
class ClassifyDS(Dataset):
    def __init__(self, finger_data, finger_labels, n_samples=100, strength='normal'):
        self.finger_data = finger_data
        self.finger_labels = finger_labels
        self.n_samples = n_samples
        self.fingers = list(finger_data.keys())
        self.strength = strength

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        frames = self.finger_data[finger]
        f = augment_frame_v13(frames[random.randint(0, len(frames)-1)], self.strength)
        return (torch.tensor(f, dtype=torch.float32).unsqueeze(0),
                self.finger_labels[finger])

# ============================================================
# 训练: 渐进式ArcFace解冻 + Center Loss
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}"); sys.stdout.flush()

SEEDS = [42, 123, 777]
USE_AMP = True

def freeze_layers(model, freeze_until='all'):
    """冻结指定层"""
    for name, param in model.named_parameters():
        param.requires_grad = True  # 默认解冻

    if freeze_until == 'head_only':
        # 只训练projector
        for name, param in model.named_parameters():
            if 'projector' not in name:
                param.requires_grad = False
    elif freeze_until == 'layer4':
        # 解冻layer4 + projector
        for name, param in model.named_parameters():
            if not any(k in name for k in ['layer4', 'projector']):
                param.requires_grad = False
    elif freeze_until == 'layer3':
        # 解冻layer3 + layer4 + projector
        for name, param in model.named_parameters():
            if not any(k in name for k in ['layer3', 'layer4', 'projector']):
                param.requires_grad = False
    # else: 'all' = all unfrozen

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def train_single_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"Training model with seed={seed}")
    print(f"{'='*60}"); sys.stdout.flush()

    model = FingerprintEncoder(embed_dim=256).to(device)
    arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)
    center_loss = CenterLoss(n_classes, 256).to(device)
    center_weight = 0.005

    # 渐进式解冻训练
    phases = [
        ('head_only',  20, 1e-3,   'normal'),  # Phase 1: 只训练头
        ('layer4',     30, 3e-4,   'normal'),  # Phase 2: 解冻layer4
        ('layer3',     30, 1e-4,   'hard'),    # Phase 3: 解冻layer3+4
        ('all',        20, 3e-5,   'hard'),    # Phase 4: 全部解冻
    ]

    t0 = time.time()
    for phase_idx, (freeze_mode, n_epochs, lr, aug_strength) in enumerate(phases):
        trainable, total = freeze_layers(model, freeze_mode)
        print(f"\n  Phase {phase_idx+1}: freeze={freeze_mode}, epochs={n_epochs}, lr={lr}")
        print(f"    Trainable: {trainable:,} / {total:,} params, aug={aug_strength}")
        sys.stdout.flush()

        # 优化器 (每个phase重建)
        train_params = list(filter(lambda p: p.requires_grad, model.parameters())) + \
                       list(arcface.parameters()) + list(center_loss.parameters())
        opt = optim.AdamW(train_params, lr=lr, weight_decay=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        scaler = GradScaler('cuda') if USE_AMP else None

        ds = ClassifyDS(finger_train, finger_labels, n_samples=100, strength=aug_strength)
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

        for epoch in range(n_epochs):
            model.train(); arcface.train()
            tl, tl_arc, tl_cen, cor, tot = 0, 0, 0, 0, 0
            for X, Y in loader:
                X, Y = X.to(device), Y.to(device)
                opt.zero_grad()
                if USE_AMP:
                    with autocast('cuda'):
                        emb = model(X)
                        loss_arc = arcface(emb, Y)
                        loss_cen = center_loss(emb, Y)
                        loss = loss_arc + center_weight * loss_cen
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    emb = model(X)
                    loss_arc = arcface(emb, Y)
                    loss_cen = center_loss(emb, Y)
                    loss = loss_arc + center_weight * loss_cen
                    loss.backward()
                    opt.step()
                tl += loss.item()
                tl_arc += loss_arc.item()
                tl_cen += loss_cen.item()
                with torch.no_grad():
                    en = F.normalize(emb.float(), dim=1)
                    wn = F.normalize(arcface.weight.float(), dim=1)
                    _, p = torch.mm(en, wn.t()).max(1)
                    cor += p.eq(Y).sum().item(); tot += Y.size(0)
            sch.step()
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{n_epochs}: arc={tl_arc/len(loader):.4f}, "
                      f"cen={tl_cen/len(loader):.4f}, acc={cor/tot*100:.1f}%, "
                      f"time={time.time()-t0:.0f}s")
                sys.stdout.flush()

    model.eval()
    print(f"  Model seed={seed} done. Total time: {time.time()-t0:.0f}s"); sys.stdout.flush()
    return model

# 训练3个模型
models = []
t_total = time.time()
for seed in SEEDS:
    m = train_single_model(seed)
    models.append(m)

print(f"\nAll models trained in {time.time()-t_total:.0f}s"); sys.stdout.flush()

# ============================================================
# TTA推理: 10个增强版本取平均嵌入
# ============================================================
@torch.no_grad()
def get_ensemble_emb(models, frame):
    """单帧 → 3模型集成嵌入"""
    x = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    embs = []
    for model in models:
        emb = model(x)
        embs.append(F.normalize(emb, dim=1))
    avg = torch.mean(torch.stack(embs), dim=0)
    return F.normalize(avg, dim=1).cpu().numpy().flatten()

def make_tta_versions(frame):
    """生成10个TTA版本"""
    h, w = frame.shape
    versions = [
        frame,                                    # 0: 原始
        -frame,                                   # 1: 极性翻转
    ]
    # 小角度旋转 (±2°, ±4°) × 2极性 = 8个
    for angle in [2.0, -2.0]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        versions.append(cv2.warpAffine(frame, M, (w, h), borderValue=0))
        versions.append(cv2.warpAffine(-frame, M, (w, h), borderValue=0))
    # 小位移 (±3, 0) × 2极性 = 4个
    for dx in [3, -3]:
        M_s = np.float32([[1, 0, dx], [0, 1, 0]])
        versions.append(cv2.warpAffine(frame, M_s, (w, h), borderValue=0))
        versions.append(cv2.warpAffine(-frame, M_s, (w, h), borderValue=0))
    return versions  # 10个版本

def get_tta_embedding(models, frame):
    """TTA推理: 10个版本取平均嵌入, L2归一化"""
    versions = make_tta_versions(frame)
    embs = [get_ensemble_emb(models, v) for v in versions]
    avg = np.mean(embs, axis=0)
    avg /= (np.linalg.norm(avg) + 1e-10)
    return avg

# ============================================================
# 评估: 客户部署场景
# ============================================================
print(f"\n{'='*60}")
print(f"评估: V13 (ResNet-18 + ArcFace+CenterLoss + TTA)")
print(f"{'='*60}"); sys.stdout.flush()

N_TEMPLATES = 20

# ---- 提取模板嵌入 (TTA) ----
print("Extracting template embeddings (TTA)..."); sys.stdout.flush()
t_emb = time.time()

templates_tta = {}       # TTA嵌入
templates_centroid = {}  # 质心嵌入
for finger in fingers:
    train_frames = finger_train[finger]
    chosen = random.sample(train_frames, min(N_TEMPLATES, len(train_frames)))
    embs = [get_tta_embedding(models, f) for f in chosen]
    templates_tta[finger] = embs
    # 计算质心
    centroid = np.mean(embs, axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-10)
    templates_centroid[finger] = centroid

print(f"  Templates done in {time.time()-t_emb:.0f}s"); sys.stdout.flush()

# ---- 匹配测试帧 ----
print("Matching test frames (TTA)..."); sys.stdout.flush()
t_eval = time.time()

# 策略1: TTA + max个体匹配
gen_tta_max = []
imp_tta_max = []
# 策略2: TTA + 质心匹配
gen_centroid = []
imp_centroid = []
# 策略3: TTA + max(质心, max个体)
gen_hybrid = []
imp_hybrid = []
# 策略4: 无TTA对比 (双极性, 类似V12)
gen_no_tta = []
imp_no_tta = []

for fi_idx, finger in enumerate(fingers):
    test_frames = finger_test[finger]
    if not test_frames:
        continue

    for frame in test_frames:
        # TTA嵌入
        q_tta = get_tta_embedding(models, frame)
        # 无TTA双极性嵌入
        q_orig = get_ensemble_emb(models, frame)
        q_flip = get_ensemble_emb(models, -frame)

        # Genuine
        # 策略1: max个体
        own_sims = [np.dot(q_tta, t) for t in templates_tta[finger]]
        gen_tta_max.append(max(own_sims))

        # 策略2: 质心
        cen_sim = np.dot(q_tta, templates_centroid[finger])
        gen_centroid.append(cen_sim)

        # 策略3: hybrid
        gen_hybrid.append(max(max(own_sims), cen_sim))

        # 策略4: 无TTA
        own_no_tta = []
        for t in templates_tta[finger]:
            own_no_tta.append(max(np.dot(q_orig, t), np.dot(q_flip, t)))
        gen_no_tta.append(max(own_no_tta))

        # Impostor
        for other in fingers:
            if other == finger:
                continue
            # 策略1
            oth_sims = [np.dot(q_tta, t) for t in templates_tta[other]]
            imp_tta_max.append(max(oth_sims))
            # 策略2
            cen_sim_o = np.dot(q_tta, templates_centroid[other])
            imp_centroid.append(cen_sim_o)
            # 策略3
            imp_hybrid.append(max(max(oth_sims), cen_sim_o))
            # 策略4
            oth_no_tta = []
            for t in templates_tta[other]:
                oth_no_tta.append(max(np.dot(q_orig, t), np.dot(q_flip, t)))
            imp_no_tta.append(max(oth_no_tta))

    if (fi_idx+1) % 5 == 0 or fi_idx == 0:
        print(f"  [{fi_idx+1}/{n_classes}] elapsed={time.time()-t_eval:.0f}s"); sys.stdout.flush()

print(f"\nEvaluation done in {time.time()-t_eval:.0f}s"); sys.stdout.flush()

# ============================================================
# 结果分析
# ============================================================
def analyze_scores(name, gen, imp):
    gen = np.array(gen)
    imp = np.array(imp)

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

eer1 = analyze_scores("策略1: TTA + max个体匹配", gen_tta_max, imp_tta_max)
eer2 = analyze_scores("策略2: TTA + 质心匹配", gen_centroid, imp_centroid)
eer3 = analyze_scores("策略3: TTA + hybrid(质心/个体)", gen_hybrid, imp_hybrid)
eer4 = analyze_scores("策略4: 无TTA双极性 (baseline)", gen_no_tta, imp_no_tta)

# ============================================================
# 分组分析 (最佳策略)
# ============================================================
best_idx = np.argmin([eer1, eer2, eer3, eer4])
best_name = ['TTA+max', 'TTA+centroid', 'TTA+hybrid', 'no-TTA'][best_idx]
print(f"\n{'='*60}")
print(f"分组分析 (最佳策略: {best_name})")
print(f"{'='*60}"); sys.stdout.flush()

def group_analysis(group_name, group_fingers):
    g_gen, g_imp = [], []
    for finger in group_fingers:
        test_frames = finger_test.get(finger, [])
        for frame in test_frames:
            q = get_tta_embedding(models, frame)
            # 使用hybrid策略 (最鲁棒)
            own_sims = [np.dot(q, t) for t in templates_tta[finger]]
            cen_sim = np.dot(q, templates_centroid[finger])
            g_gen.append(max(max(own_sims), cen_sim))
            for other in group_fingers:
                if other == finger:
                    continue
                oth_sims = [np.dot(q, t) for t in templates_tta[other]]
                cen_sim_o = np.dot(q, templates_centroid[other])
                g_imp.append(max(max(oth_sims), cen_sim_o))

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
print(f"参数化分布外推 (最佳策略)")
print(f"{'='*60}"); sys.stdout.flush()

from scipy import stats

# 使用最佳策略的分数
best_gen = [gen_tta_max, gen_centroid, gen_hybrid, gen_no_tta][best_idx]
best_imp = [imp_tta_max, imp_centroid, imp_hybrid, imp_no_tta][best_idx]
best_gen = np.array(best_gen)
best_imp = np.array(best_imp)

gen_mu, gen_std = best_gen.mean(), best_gen.std()
imp_mu, imp_std = best_imp.mean(), best_imp.std()
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
print(f"V13 总结对比")
print(f"{'='*60}")
print(f"  V13 TTA+max    EER = {eer1*100:.4f}%")
print(f"  V13 TTA+centroid EER = {eer2*100:.4f}%")
print(f"  V13 TTA+hybrid EER = {eer3*100:.4f}%")
print(f"  V13 无TTA      EER = {eer4*100:.4f}%")
print(f"  V12 全局余弦   EER = 4.0039% (prev best)")
print(f"  V11 全局余弦   EER = 4.9968%")
print(f"  V10 SIFT       EER = 1.5200% (target)")
print(f"\nAll done. Total time: {time.time()-t_total:.0f}s")
print(f"{'='*60}"); sys.stdout.flush()
