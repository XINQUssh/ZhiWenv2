"""
V9: 单帧匹配方案 - 匹配客户实际部署场景
部署场景: 输入1张指纹图 → 和模板库20张比较 → 任一匹配即通过
架构: 单帧CNN → 256维嵌入, SimCLR预训练 + ArcFace微调
训练时利用所有寄存器的帧(学习寄存器不变性), 部署时只需单帧
"""
import os, glob, random, time, copy, sys
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
# 数据加载: 所有寄存器帧混合 (部署时无寄存器区分)
# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'

wtp_regs = ['Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245', 'Rgd1247']
btp_regs = ['Rgd1237', 'Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245']

finger_train = {}   # key -> list of frames (all registers mixed)
finger_test = {}
finger_labels = {}
finger_source = {}
fi = 0

print("Loading 无贴屏 (all registers pooled)..."); sys.stdout.flush()
for finger in sorted(os.listdir(base1)):
    key = f"wtp_{finger}"
    train_frames = []
    test_frames = []
    for reg in wtp_regs:
        rpath = os.path.join(base1, finger, reg)
        if not os.path.exists(rpath):
            continue
        imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
        for p in imgs_paths[:70]:
            img = load_img(p).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            train_frames.append(img)
        for p in imgs_paths[70:]:
            img = load_img(p).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            test_frames.append(img)
    if train_frames:
        finger_train[key] = train_frames
        finger_test[key] = test_frames
        finger_labels[key] = fi
        finger_source[key] = "无贴屏"
        fi += 1

n_wtp = fi
print(f"  无贴屏: {n_wtp} classes"); sys.stdout.flush()

print("Loading 不贴屏 (all registers pooled)..."); sys.stdout.flush()
for finger in sorted(os.listdir(base2)):
    primary = os.path.join(base2, finger, 'Rgd1237')
    if not os.path.exists(primary):
        continue
    if len(glob.glob(os.path.join(primary, '*.bmp'))) < 50:
        continue
    key = f"btp_{finger}"
    train_frames = []
    test_frames = []
    for reg in btp_regs:
        rpath = os.path.join(base2, finger, reg)
        if not os.path.exists(rpath):
            continue
        imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
        for p in imgs_paths[:70]:
            img = load_img(p).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            train_frames.append(img)
        for p in imgs_paths[70:]:
            img = load_img(p).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            test_frames.append(img)
    if train_frames:
        finger_train[key] = train_frames
        finger_test[key] = test_frames
        finger_labels[key] = fi
        finger_source[key] = "不贴屏"
        fi += 1

n_classes = fi
fingers = list(finger_train.keys())
print(f"Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes-n_wtp} 不贴屏)")
for k in fingers[:3]:
    print(f"  {k}: {len(finger_train[k])} train, {len(finger_test[k])} test")
sys.stdout.flush()

# ============================================================
# 模型: 单帧CNN嵌入器
# ============================================================
class SingleFrameEncoder(nn.Module):
    """单帧 → 256维嵌入, 部署时直接用"""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projector = nn.Sequential(
            nn.Linear(128 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        # x: (B, 1, H, W)
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
# 数据集: 单帧级别
# ============================================================
class ContrastiveDS(Dataset):
    """SimCLR: 同一手指的两个随机帧作为正样本对"""
    def __init__(self, finger_data, n_samples_per_finger=100):
        self.finger_data = finger_data
        self.n_samples = n_samples_per_finger
        self.fingers = list(finger_data.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        frames = self.finger_data[finger]
        # 随机选两帧 (可来自不同寄存器)
        i1, i2 = random.sample(range(len(frames)), 2)
        f1 = frames[i1].copy()
        f2 = frames[i2].copy()
        # 数据增强: 随机极性翻转 + 噪声
        if random.random() < 0.5: f1 = -f1
        if random.random() < 0.5: f2 = -f2
        f1 += np.random.randn(*f1.shape).astype(np.float32) * 0.03
        f2 += np.random.randn(*f2.shape).astype(np.float32) * 0.03
        # 随机水平翻转
        if random.random() < 0.3: f1 = f1[:, ::-1].copy()
        if random.random() < 0.3: f2 = f2[:, ::-1].copy()
        return (torch.tensor(f1, dtype=torch.float32).unsqueeze(0),
                torch.tensor(f2, dtype=torch.float32).unsqueeze(0))

class ClassifyDS(Dataset):
    """ArcFace: 单帧分类"""
    def __init__(self, finger_data, finger_labels, n_samples_per_finger=80):
        self.finger_data = finger_data
        self.finger_labels = finger_labels
        self.n_samples = n_samples_per_finger
        self.fingers = list(finger_data.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        frames = self.finger_data[finger]
        i = random.randint(0, len(frames) - 1)
        f = frames[i].copy()
        # 数据增强
        if random.random() < 0.5: f = -f
        f += np.random.randn(*f.shape).astype(np.float32) * 0.02
        if random.random() < 0.3: f = f[:, ::-1].copy()
        return (torch.tensor(f, dtype=torch.float32).unsqueeze(0),
                self.finger_labels[finger])

# ============================================================
# 训练
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}"); sys.stdout.flush()

N_PRETRAIN = 60
N_FINETUNE = 60
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

    model = SingleFrameEncoder(embed_dim=256).to(device)
    simclr_loss = SimCLRLoss(temperature=0.07)
    scaler = GradScaler('cuda') if USE_AMP else None

    # Phase 1: SimCLR 预训练
    print(f"  Phase 1: SimCLR Pre-training ({N_PRETRAIN} epochs)..."); sys.stdout.flush()
    opt1 = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=N_PRETRAIN)

    pre_ds = ContrastiveDS(finger_train, n_samples_per_finger=100)
    pre_loader = DataLoader(pre_ds, batch_size=64, shuffle=True, num_workers=0)
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

    # Phase 2: ArcFace 微调
    print(f"  Phase 2: ArcFace Fine-tuning ({N_FINETUNE} epochs)..."); sys.stdout.flush()
    arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)
    opt2 = optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=0.0003, weight_decay=1e-3
    )
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=N_FINETUNE)
    scaler2 = GradScaler('cuda') if USE_AMP else None

    ft_ds = ClassifyDS(finger_train, finger_labels, n_samples_per_finger=80)
    ft_loader = DataLoader(ft_ds, batch_size=64, shuffle=True, num_workers=0)
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

# ============================================================
# 训练3个模型
# ============================================================
models = []
t_total = time.time()
for seed in SEEDS:
    m = train_single_model(seed)
    models.append(m)
    print(f"  Model seed={seed} done. Elapsed: {time.time()-t_total:.0f}s"); sys.stdout.flush()

# ============================================================
# 单帧嵌入提取
# ============================================================
def get_single_emb(model, frame):
    """单帧 → 256维嵌入"""
    x = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)
    return F.normalize(emb, dim=1).cpu().numpy().flatten()

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

# ============================================================
# 评估: 完全匹配客户部署场景
# 注册: 每指20帧 → 20个嵌入模板
# 验证: 1帧 → 1个嵌入 → max(cos_sim with 20模板) → 判决
# ============================================================
print("\n" + "="*60)
print("评估: 客户部署场景 (1帧 vs 20模板, max匹配)")
print("="*60); sys.stdout.flush()

def evaluate_deployment(models, n_templates=20, use_ensemble=True, label=""):
    """
    完全模拟客户部署场景:
    - 注册: 从训练集随机采n_templates帧作为模板
    - 验证: 测试集每一帧单独验证
    - 分数: max(cos_sim(验证帧, 模板_i)) for i in 1..n_templates
    - 判决: 分数 > 阈值 → 匹配
    """
    emb_func = lambda f: get_ensemble_emb(models, f) if use_ensemble else get_single_emb(models[0], f)

    # 注册: 每指采n_templates帧嵌入作为模板
    templates = {}
    for finger in fingers:
        train_frames = finger_train[finger]
        if len(train_frames) < n_templates:
            chosen = train_frames
        else:
            chosen = random.sample(train_frames, n_templates)
        templates[finger] = [emb_func(f) for f in chosen]

    # 验证: 每个测试帧单独验证
    genuine_scores = []
    impostor_scores = []

    for finger in fingers:
        test_frames = finger_test[finger]
        if not test_frames:
            continue
        for frame in test_frames:
            query_emb = emb_func(frame)

            # Genuine: 和自己的模板比
            own_sims = [np.dot(query_emb, t) for t in templates[finger]]
            genuine_scores.append(max(own_sims))

            # Impostor: 和每个其他人的模板比
            for other in fingers:
                if other == finger:
                    continue
                other_sims = [np.dot(query_emb, t) for t in templates[other]]
                impostor_scores.append(max(other_sims))

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    print(f"\n  [{label}] Genuine scores: {len(genuine_scores)}, Impostor scores: {len(impostor_scores)}")
    print(f"  Genuine:  mean={genuine_scores.mean():.4f}, std={genuine_scores.std():.4f}, min={genuine_scores.min():.4f}")
    print(f"  Impostor: mean={impostor_scores.mean():.4f}, std={impostor_scores.std():.4f}, max={impostor_scores.max():.4f}")
    sys.stdout.flush()

    # ROC分析
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])

    fpr_arr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # EER
    eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
    eer = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2
    eer_thresh = thresholds[eer_idx]

    # 分离度
    gen_min = genuine_scores.min()
    imp_max = impostor_scores.max()
    if gen_min > imp_max:
        sep_str = "*** PERFECT SEPARATION ***"
    else:
        overlap_count = np.sum(impostor_scores > gen_min)
        sep_str = f"overlap: {overlap_count}/{len(impostor_scores)} impostor scores > gen_min"

    print(f"\n  EER = {eer*100:.4f}% (threshold={eer_thresh:.4f})")
    print(f"  gen_min={gen_min:.4f}, imp_max={imp_max:.4f}")
    print(f"  {sep_str}")

    # FFR → FAR
    for target_ffr in [0.0, 0.01, 0.03, 0.05, 0.10]:
        idx = np.argmin(np.abs(fnr - target_ffr))
        print(f"  FFR={target_ffr*100:.0f}% -> FAR={fpr_arr[idx]*100:.4f}% (threshold={thresholds[idx]:.4f})")

    # FAR → FFR
    for target_far in [0.002, 0.01, 0.1, 1.0]:
        target_far_frac = target_far / 100
        idx = np.argmin(np.abs(fpr_arr - target_far_frac))
        print(f"  FAR={target_far}% -> FFR={fnr[idx]*100:.2f}%")

    # 特别关注: FAR=0.002% (1/50000)
    target_far_frac = 0.00002
    idx = np.argmin(np.abs(fpr_arr - target_far_frac))
    print(f"\n  *** 关键指标: FAR=0.002% (1/50000) -> FFR={fnr[idx]*100:.2f}% ***")
    sys.stdout.flush()

    # d-prime
    d_prime = (genuine_scores.mean() - impostor_scores.mean()) / \
              np.sqrt(0.5 * (genuine_scores.std()**2 + impostor_scores.std()**2))
    print(f"  d-prime = {d_prime:.2f}")
    sys.stdout.flush()

    return genuine_scores, impostor_scores

# --- 单模型测试 ---
for mi, model in enumerate(models):
    print(f"\n{'='*60}")
    print(f"Single Model (seed={SEEDS[mi]})")
    print(f"{'='*60}"); sys.stdout.flush()
    evaluate_deployment([model], n_templates=20, use_ensemble=False,
                       label=f"Model-{mi} (seed={SEEDS[mi]})")

# --- 集成测试 ---
print(f"\n{'='*60}")
print(f"Ensemble ({len(models)} models)")
print(f"{'='*60}"); sys.stdout.flush()

# 20模板
gen_20, imp_20 = evaluate_deployment(models, n_templates=20, use_ensemble=True,
                                      label="Ensemble-20tpl")

# 30模板
gen_30, imp_30 = evaluate_deployment(models, n_templates=30, use_ensemble=True,
                                      label="Ensemble-30tpl")

# --- 分组分析 ---
print(f"\n{'='*60}")
print(f"分组分析 (Ensemble, 20模板)")
print(f"{'='*60}"); sys.stdout.flush()

def evaluate_group(models, group_fingers, n_templates=20, label=""):
    emb_func = lambda f: get_ensemble_emb(models, f)

    templates = {}
    for finger in group_fingers:
        train_frames = finger_train[finger]
        chosen = random.sample(train_frames, min(n_templates, len(train_frames)))
        templates[finger] = [emb_func(f) for f in chosen]

    genuine_scores = []
    impostor_scores = []

    for finger in group_fingers:
        test_frames = finger_test[finger]
        if not test_frames:
            continue
        for frame in test_frames:
            query_emb = emb_func(frame)
            own_sims = [np.dot(query_emb, t) for t in templates[finger]]
            genuine_scores.append(max(own_sims))
            for other in group_fingers:
                if other == finger:
                    continue
                other_sims = [np.dot(query_emb, t) for t in templates[other]]
                impostor_scores.append(max(other_sims))

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    print(f"\n  [{label}] Genuine: {len(genuine_scores)}, Impostor: {len(impostor_scores)}")
    print(f"  Genuine:  mean={genuine_scores.mean():.4f}, min={genuine_scores.min():.4f}")
    print(f"  Impostor: mean={impostor_scores.mean():.4f}, max={impostor_scores.max():.4f}")

    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    fpr_arr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
    eer = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2

    gen_min = genuine_scores.min()
    imp_max = impostor_scores.max()
    if gen_min > imp_max:
        print(f"  EER = {eer*100:.4f}%  *** PERFECT SEPARATION ***")
    else:
        overlap = np.sum(impostor_scores > gen_min)
        print(f"  EER = {eer*100:.4f}%  overlap={overlap}/{len(impostor_scores)}")

    for target_ffr in [0.0, 0.03, 0.05]:
        idx = np.argmin(np.abs(fnr - target_ffr))
        print(f"  FFR={target_ffr*100:.0f}% -> FAR={fpr_arr[idx]*100:.4f}%")

    target_far_frac = 0.00002
    idx = np.argmin(np.abs(fpr_arr - target_far_frac))
    print(f"  FAR=0.002% -> FFR={fnr[idx]*100:.2f}%")
    sys.stdout.flush()

wtp_fingers = [f for f in fingers if f.startswith("wtp_")]
btp_fingers = [f for f in fingers if f.startswith("btp_")]
for group_name, group_fingers in [("无贴屏", wtp_fingers), ("不贴屏", btp_fingers)]:
    if group_fingers:
        evaluate_group(models, group_fingers, n_templates=20, label=group_name)

# --- 参数化分布分析 ---
print(f"\n{'='*60}")
print(f"参数化分布分析")
print(f"{'='*60}"); sys.stdout.flush()

from scipy import stats

gen_mu, gen_std = gen_20.mean(), gen_20.std()
imp_mu, imp_std = imp_20.mean(), imp_20.std()
d_prime = (gen_mu - imp_mu) / np.sqrt(0.5 * (gen_std**2 + imp_std**2))

print(f"Genuine:  mu={gen_mu:.4f}, std={gen_std:.4f}")
print(f"Impostor: mu={imp_mu:.4f}, std={imp_std:.4f}")
print(f"d-prime = {d_prime:.2f}")

# Gaussian外推
for target_ffr in [0.01, 0.03, 0.05]:
    thresh = gen_mu - stats.norm.ppf(1 - target_ffr) * gen_std
    far_theory = stats.norm.cdf(thresh, loc=imp_mu, scale=imp_std)
    print(f"  FFR={target_ffr*100:.0f}% -> 理论FAR={far_theory*100:.6f}%")

for target_far in [0.00002, 0.0001, 0.001]:
    thresh = stats.norm.ppf(target_far, loc=imp_mu, scale=imp_std)
    ffr_theory = 1 - stats.norm.cdf(thresh, loc=gen_mu, scale=gen_std)
    print(f"  FAR={target_far*100:.4f}% -> 理论FFR={ffr_theory*100:.2f}%")

# 正态性检验
_, gen_p = stats.shapiro(genuine_scores[:min(500, len(genuine_scores))])
_, imp_p = stats.shapiro(impostor_scores[:min(500, len(impostor_scores))])
print(f"\n正态性检验 (Shapiro-Wilk):")
print(f"  Genuine p-value = {gen_p:.6f}")
print(f"  Impostor p-value = {imp_p:.6f}")
sys.stdout.flush()

print(f"\n{'='*60}")
print(f"All done. Total time: {time.time()-t_total:.0f}s")
print(f"{'='*60}"); sys.stdout.flush()
