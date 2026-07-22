"""
V11: 增强CNN方案 - 借鉴SIFT管线的所有预处理技术
改进点 (相比v9):
  1. 单寄存器: 无贴屏Rgd1245, 不贴屏Rgd1237 (匹配客户数据选择)
  2. CLAHE对比度增强 (clip=4.0)
  3. 2x上采样 (110x100 → 220x200)
  4. GMFS指纹掩膜 (零化背景区域)
  5. 推理时双极性: 原始+翻转各算嵌入, 取max相似度
  6. SimCLR预训练 + ArcFace微调 + 3模型集成
评估: 客户部署场景 (1帧 vs 20模板, max匹配)
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

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 预处理: CLAHE + 2x上采样 + GMFS掩膜
# ============================================================
def clahe_enhance(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def upsample_2x(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

def generate_gmfs_mask(img, sigma=13.0/3, percentile=95, threshold_ratio=0.2):
    """GMFS指纹分割掩膜"""
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
    # 保留最大连通域
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    if n_labels > 1:
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = ((labels == max_label) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=2)
    return mask

def preprocess_frame(img):
    """完整预处理: CLAHE → 2x上采样 → GMFS掩膜 → 归一化"""
    enhanced = clahe_enhance(img)
    upsampled = upsample_2x(enhanced)
    mask = generate_gmfs_mask(upsampled)
    # 应用掩膜: 背景区域置零
    masked = upsampled.copy()
    masked[mask == 0] = 0
    # 归一化 (只在有效区域上算统计量)
    valid = masked[mask > 0].astype(np.float32)
    if len(valid) > 0:
        mu, std = valid.mean(), valid.std() + 1e-6
    else:
        mu, std = 0, 1
    result = (masked.astype(np.float32) - mu) / std
    result[mask == 0] = 0  # 背景保持零
    return result  # 220x200

# ============================================================
# 数据加载: 单寄存器
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
    train_frames = []
    test_frames = []
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
    train_frames = []
    test_frames = []
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
# 模型: CNN嵌入器 (输入220x200, 比v9更深)
# ============================================================
class EnhancedFrameEncoder(nn.Module):
    """增强CNN: 适配2x上采样后的220x200输入"""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                 # 110x100
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                 # 55x50
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                 # 27x25
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),    # 4x4
        )
        self.projector = nn.Sequential(
            nn.Linear(256 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.projector(x)

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
# 数据增强
# ============================================================
def augment_frame(frame):
    """训练时数据增强: 极性翻转 + 噪声 + 水平翻转"""
    f = frame.copy()
    if random.random() < 0.5:
        f = -f  # 极性翻转
    f += np.random.randn(*f.shape).astype(np.float32) * 0.03
    if random.random() < 0.3:
        f = f[:, ::-1].copy()  # 水平翻转
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
# 训练
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}"); sys.stdout.flush()

N_PRETRAIN = 80
N_FINETUNE = 80
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

    model = EnhancedFrameEncoder(embed_dim=256).to(device)
    simclr_loss = SimCLRLoss(temperature=0.07)
    scaler = GradScaler('cuda') if USE_AMP else None

    # Phase 1: SimCLR
    print(f"  Phase 1: SimCLR Pre-training ({N_PRETRAIN} epochs)..."); sys.stdout.flush()
    opt1 = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=N_PRETRAIN)

    pre_ds = ContrastiveDS(finger_train, n_samples=120)
    pre_loader = DataLoader(pre_ds, batch_size=32, shuffle=True, num_workers=0)
    print(f"    DataLoader: {len(pre_ds)} samples, {len(pre_loader)} batches"); sys.stdout.flush()

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
            print(f"    Epoch {epoch+1}/{N_PRETRAIN}: loss={tl/len(pre_loader):.4f}, time={time.time()-t0:.0f}s")
            sys.stdout.flush()

    # Phase 2: ArcFace
    print(f"  Phase 2: ArcFace Fine-tuning ({N_FINETUNE} epochs)..."); sys.stdout.flush()
    arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)
    opt2 = optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=0.0003, weight_decay=1e-3
    )
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=N_FINETUNE)
    scaler2 = GradScaler('cuda') if USE_AMP else None

    ft_ds = ClassifyDS(finger_train, finger_labels, n_samples=100)
    ft_loader = DataLoader(ft_ds, batch_size=32, shuffle=True, num_workers=0)
    print(f"    DataLoader: {len(ft_ds)} samples, {len(ft_loader)} batches"); sys.stdout.flush()

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
                en = F.normalize(emb.float(), dim=1)
                wn = F.normalize(arcface.weight.float(), dim=1)
                _, p = torch.mm(en, wn.t()).max(1)
                cor += p.eq(Y).sum().item(); tot += Y.size(0)
        sch2.step()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_FINETUNE}: loss={tl/len(ft_loader):.4f}, acc={cor/tot*100:.1f}%, time={time.time()-t1:.0f}s")
            sys.stdout.flush()

    model.eval()
    return model

# 训练3个模型
models = []
t_total = time.time()
for seed in SEEDS:
    m = train_single_model(seed)
    models.append(m)
    print(f"  Model seed={seed} done. Elapsed: {time.time()-t_total:.0f}s"); sys.stdout.flush()

# ============================================================
# 嵌入提取 (双极性推理)
# ============================================================
def get_ensemble_emb(models, frame):
    """单帧 → 3模型集成嵌入"""
    x = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    embs = []
    for model in models:
        with torch.no_grad():
            emb = model(x)
        embs.append(F.normalize(emb, dim=1))
    avg = torch.mean(torch.stack(embs), dim=0)
    return F.normalize(avg, dim=1).cpu().numpy().flatten()

def get_dual_polarity_emb(models, frame):
    """双极性推理: 原始帧和翻转帧各算嵌入, 返回两个"""
    emb_orig = get_ensemble_emb(models, frame)
    emb_flip = get_ensemble_emb(models, -frame)  # 极性翻转
    return emb_orig, emb_flip

def dual_polarity_max_sim(query_embs, template_embs):
    """
    双极性最大相似度:
    query有2个嵌入(orig, flip), template有2个嵌入(orig, flip)
    取4种组合的max余弦相似度
    """
    q_orig, q_flip = query_embs
    t_orig, t_flip = template_embs
    sims = [
        np.dot(q_orig, t_orig),
        np.dot(q_orig, t_flip),
        np.dot(q_flip, t_orig),
        np.dot(q_flip, t_flip),
    ]
    return max(sims)

# ============================================================
# 评估: 客户部署场景 + 双极性推理
# ============================================================
print(f"\n{'='*60}")
print(f"评估: 客户部署场景 (1帧 vs 20模板, 双极性推理)")
print(f"{'='*60}"); sys.stdout.flush()

N_TEMPLATES = 20

# 提取模板嵌入 (双极性)
print("Extracting template embeddings..."); sys.stdout.flush()
t_emb = time.time()
templates = {}
for finger in fingers:
    train_frames = finger_train[finger]
    chosen = random.sample(train_frames, min(N_TEMPLATES, len(train_frames)))
    templates[finger] = [get_dual_polarity_emb(models, f) for f in chosen]

print(f"  Templates done in {time.time()-t_emb:.0f}s"); sys.stdout.flush()

# 提取测试嵌入并匹配
print("Matching test frames..."); sys.stdout.flush()
t_eval = time.time()

genuine_scores = []
impostor_scores = []

for fi, finger in enumerate(fingers):
    test_frames = finger_test[finger]
    if not test_frames:
        continue

    for frame in test_frames:
        query_embs = get_dual_polarity_emb(models, frame)

        # Genuine: 和自己的模板比, 取max
        own_sims = [dual_polarity_max_sim(query_embs, t) for t in templates[finger]]
        genuine_scores.append(max(own_sims))

        # Impostor: 和每个其他人的模板比, 取max
        for other in fingers:
            if other == finger:
                continue
            other_sims = [dual_polarity_max_sim(query_embs, t) for t in templates[other]]
            impostor_scores.append(max(other_sims))

    if (fi+1) % 5 == 0 or fi == 0:
        print(f"  [{fi+1}/{n_classes}] elapsed={time.time()-t_eval:.0f}s"); sys.stdout.flush()

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print(f"\nEvaluation done in {time.time()-t_eval:.0f}s")
sys.stdout.flush()

# ============================================================
# ROC分析
# ============================================================
print(f"\n{'='*60}")
print(f"Results: V11 Enhanced CNN (单寄存器 + CLAHE + 2x + 掩膜 + 双极性)")
print(f"{'='*60}"); sys.stdout.flush()

print(f"\nGenuine:  n={len(genuine_scores)}, mean={genuine_scores.mean():.4f}, std={genuine_scores.std():.4f}, min={genuine_scores.min():.4f}")
print(f"Impostor: n={len(impostor_scores)}, mean={impostor_scores.mean():.4f}, std={impostor_scores.std():.4f}, max={impostor_scores.max():.4f}")

y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
y_scores = np.concatenate([genuine_scores, impostor_scores])
fpr_arr, tpr, thresholds = roc_curve(y_true, y_scores)
fnr = 1 - tpr

eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
eer = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2
eer_thresh = thresholds[eer_idx]

gen_min = genuine_scores.min()
imp_max = impostor_scores.max()
if gen_min > imp_max:
    sep_str = "*** PERFECT SEPARATION ***"
else:
    overlap_count = np.sum(impostor_scores > gen_min)
    sep_str = f"overlap: {overlap_count}/{len(impostor_scores)}"

print(f"\nEER = {eer*100:.4f}% (threshold={eer_thresh:.4f})")
print(f"gen_min={gen_min:.4f}, imp_max={imp_max:.4f}")
print(f"{sep_str}")

print(f"\n--- FFR → FAR ---")
for target_ffr in [0.0, 0.01, 0.03, 0.05, 0.10]:
    idx = np.argmin(np.abs(fnr - target_ffr))
    print(f"  FFR={target_ffr*100:.0f}% -> FAR={fpr_arr[idx]*100:.4f}% (threshold={thresholds[idx]:.4f})")

print(f"\n--- FAR → FFR ---")
for target_far in [0.002, 0.01, 0.1, 1.0]:
    target_frac = target_far / 100
    idx = np.argmin(np.abs(fpr_arr - target_frac))
    print(f"  FAR={target_far}% -> FFR={fnr[idx]*100:.2f}%")

target_far_frac = 0.00002
idx = np.argmin(np.abs(fpr_arr - target_far_frac))
print(f"\n  *** FAR=0.002% (1/50000) -> FFR={fnr[idx]*100:.2f}% ***")

# d-prime
d_prime = (genuine_scores.mean() - impostor_scores.mean()) / \
          np.sqrt(0.5 * (genuine_scores.std()**2 + impostor_scores.std()**2))
print(f"  d-prime = {d_prime:.2f}")
sys.stdout.flush()

# ============================================================
# 消融: 不用双极性 (只用原始极性)
# ============================================================
print(f"\n{'='*60}")
print(f"消融实验: 不用双极性推理 (只用原始帧嵌入)")
print(f"{'='*60}"); sys.stdout.flush()

genuine_single = []
impostor_single = []

for finger in fingers:
    test_frames = finger_test[finger]
    for frame in test_frames:
        q_emb = get_ensemble_emb(models, frame)
        # Genuine
        own_sims = [max(np.dot(q_emb, t[0]), np.dot(q_emb, t[1])) for t in templates[finger]]
        genuine_single.append(max(own_sims))
        # Impostor
        for other in fingers:
            if other == finger:
                continue
            other_sims = [max(np.dot(q_emb, t[0]), np.dot(q_emb, t[1])) for t in templates[other]]
            impostor_single.append(max(other_sims))

genuine_single = np.array(genuine_single)
impostor_single = np.array(impostor_single)

y_true_s = np.concatenate([np.ones(len(genuine_single)), np.zeros(len(impostor_single))])
y_scores_s = np.concatenate([genuine_single, impostor_single])
fpr_s, tpr_s, thresh_s = roc_curve(y_true_s, y_scores_s)
fnr_s = 1 - tpr_s
eer_idx_s = np.nanargmin(np.abs(fnr_s - fpr_s))
eer_s = (fpr_s[eer_idx_s] + fnr_s[eer_idx_s]) / 2

print(f"Single-polarity EER = {eer_s*100:.4f}%")
print(f"Genuine: mean={genuine_single.mean():.4f}, min={genuine_single.min():.4f}")
print(f"Impostor: mean={impostor_single.mean():.4f}, max={impostor_single.max():.4f}")
for tf in [0.0, 0.03, 0.05]:
    idx = np.argmin(np.abs(fnr_s - tf))
    print(f"  FFR={tf*100:.0f}% -> FAR={fpr_s[idx]*100:.4f}%")
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
    print(f"  gen: mean={g_gen.mean():.4f}, min={gmin:.4f}")
    print(f"  imp: mean={g_imp.mean():.4f}, max={imax:.4f}")
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

print(f"\n{'='*60}")
print(f"All done. Total time: {time.time()-t_total:.0f}s")
print(f"{'='*60}"); sys.stdout.flush()
