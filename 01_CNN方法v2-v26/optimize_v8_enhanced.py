"""
V8增强方案: 基于v7架构 + 3项优化
1. 5模型集成 (更多多样性)
2. S-norm分数归一化 (自适应阈值)
3. 参数化分布外推 (理论估算极端FAR)
4. 更多推理采样 (K=50)
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
from scipy import stats

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 数据加载
# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
wtp_regs = ['Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245', 'Rgd1247']
btp_regs = ['Rgd1237', 'Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245']

finger_reg_train, finger_reg_test = {}, {}
finger_labels, finger_source, finger_regs = {}, {}, {}
fi = 0

print("Loading 无贴屏 (5 registers)..."); sys.stdout.flush()
for finger in sorted(os.listdir(base1)):
    key = f"wtp_{finger}"
    finger_reg_train[key], finger_reg_test[key] = {}, {}
    for ri, reg in enumerate(wtp_regs):
        rpath = os.path.join(base1, finger, reg)
        loaded = []
        for p in sorted(glob.glob(os.path.join(rpath, '*.bmp'))):
            img = load_img(p).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            loaded.append(img)
        finger_reg_train[key][ri] = loaded[:70]
        finger_reg_test[key][ri] = loaded[70:]
    finger_labels[key] = fi
    finger_source[key] = "无贴屏"
    finger_regs[key] = 5
    fi += 1
n_wtp = fi
print(f"  无贴屏: {n_wtp} classes")

print("Loading 不贴屏 (5 registers)...")
for finger in sorted(os.listdir(base2)):
    primary = os.path.join(base2, finger, 'Rgd1237')
    if not os.path.exists(primary) or len(glob.glob(os.path.join(primary, '*.bmp'))) < 50:
        continue
    key = f"btp_{finger}"
    finger_reg_train[key], finger_reg_test[key] = {}, {}
    for ri, reg in enumerate(btp_regs):
        rpath = os.path.join(base2, finger, reg)
        if not os.path.exists(rpath):
            finger_reg_train[key][ri], finger_reg_test[key][ri] = [], []
            continue
        loaded = []
        for p in sorted(glob.glob(os.path.join(rpath, '*.bmp'))):
            img = load_img(p).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            loaded.append(img)
        finger_reg_train[key][ri] = loaded[:70]
        finger_reg_test[key][ri] = loaded[70:]
    finger_labels[key] = fi
    finger_source[key] = "不贴屏"
    finger_regs[key] = 5
    fi += 1

n_classes = fi
fingers = list(finger_reg_train.keys())
print(f"Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes-n_wtp} 不贴屏)")
sys.stdout.flush()

# ============================================================
# 模型定义 (与v7相同)
# ============================================================
class FrameEncoder(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.fc = nn.Linear(128 * 9, feat_dim)
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

class MultiRegSetTransformer(nn.Module):
    def __init__(self, feat_dim=128, embed_dim=256, n_regs=5, n_heads=4, n_layers=2):
        super().__init__()
        self.frame_encoder = FrameEncoder(feat_dim)
        self.reg_embedding = nn.Embedding(n_regs, feat_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
        self.projector = nn.Sequential(nn.Linear(feat_dim, embed_dim), nn.BatchNorm1d(embed_dim))
    def forward(self, frames, reg_ids):
        B, N, H, W = frames.shape
        feats = self.frame_encoder(frames.view(B*N, 1, H, W)).view(B, N, -1)
        feats = feats + self.reg_embedding(reg_ids)
        x = torch.cat([self.cls_token.expand(B, -1, -1), feats], dim=1)
        return self.projector(self.transformer(x)[:, 0, :])

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
        sim = sim.masked_fill(mask, -1e4)
        return F.cross_entropy(sim, labels)

# ============================================================
# 采帧 + 数据集 + collate (与v7相同)
# ============================================================
def sample_multireg_raw(reg_data, n_regs=5, frames_per_reg=3):
    all_frames, all_reg_ids = [], []
    for ri in range(n_regs):
        imgs = reg_data.get(ri, [])
        if len(imgs) < 3: continue
        n_sample = min(frames_per_reg, len(imgs))
        for idx in random.sample(range(len(imgs)), n_sample):
            all_frames.append(imgs[idx]); all_reg_ids.append(ri)
    if not all_frames: return None, None
    return np.stack(all_frames, axis=0), np.array(all_reg_ids)

class MultiRegContrastiveDS(Dataset):
    def __init__(self, reg_train, regs_count, fpr=3, n_samples=80):
        self.reg_train, self.regs_count = reg_train, regs_count
        self.fpr, self.n_samples = fpr, n_samples
        self.fingers = list(reg_train.keys())
    def __len__(self): return len(self.fingers) * self.n_samples
    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        n_r = self.regs_count[finger]
        f1, r1 = sample_multireg_raw(self.reg_train[finger], n_r, self.fpr)
        f2, r2 = sample_multireg_raw(self.reg_train[finger], n_r, self.fpr)
        for i in range(len(f1)):
            if random.random() < 0.3: f1[i] = -f1[i]
        for i in range(len(f2)):
            if random.random() < 0.3: f2[i] = -f2[i]
        f1 += np.random.randn(*f1.shape).astype(np.float32) * 0.02
        f2 += np.random.randn(*f2.shape).astype(np.float32) * 0.02
        return (torch.tensor(f1, dtype=torch.float32), torch.tensor(r1, dtype=torch.long),
                torch.tensor(f2, dtype=torch.float32), torch.tensor(r2, dtype=torch.long))

class MultiRegTrainDS(Dataset):
    def __init__(self, reg_train, regs_count, labels, fpr=3, n_samples=60):
        self.reg_train, self.regs_count, self.labels = reg_train, regs_count, labels
        self.fpr, self.n_samples = fpr, n_samples
        self.fingers = list(reg_train.keys())
    def __len__(self): return len(self.fingers) * self.n_samples
    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        n_r = self.regs_count[finger]
        frames, reg_ids = sample_multireg_raw(self.reg_train[finger], n_r, self.fpr)
        for i in range(len(frames)):
            if random.random() < 0.3: frames[i] = -frames[i]
        frames += np.random.randn(*frames.shape).astype(np.float32) * 0.02
        return (torch.tensor(frames, dtype=torch.float32),
                torch.tensor(reg_ids, dtype=torch.long), self.labels[finger])

def collate_contrastive(batch):
    f1s, r1s, f2s, r2s = zip(*batch)
    max_n = max(max(x.size(0) for x in f1s), max(x.size(0) for x in f2s))
    B = len(f1s)
    H, W = f1s[0].shape[1], f1s[0].shape[2]
    F1, R1 = torch.zeros(B, max_n, H, W), torch.zeros(B, max_n, dtype=torch.long)
    F2, R2 = torch.zeros(B, max_n, H, W), torch.zeros(B, max_n, dtype=torch.long)
    for i in range(B):
        n1, n2 = f1s[i].size(0), f2s[i].size(0)
        F1[i,:n1] = f1s[i]; R1[i,:n1] = r1s[i]
        F2[i,:n2] = f2s[i]; R2[i,:n2] = r2s[i]
    return F1, R1, F2, R2

def collate_train(batch):
    fs, rs, ys = zip(*batch)
    max_n = max(x.size(0) for x in fs)
    B, H, W = len(fs), fs[0].shape[1], fs[0].shape[2]
    F_out, R_out = torch.zeros(B, max_n, H, W), torch.zeros(B, max_n, dtype=torch.long)
    for i in range(B):
        n = fs[i].size(0); F_out[i,:n] = fs[i]; R_out[i,:n] = rs[i]
    return F_out, R_out, torch.tensor(ys, dtype=torch.long)

# ============================================================
# 训练
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

FPR = 3
N_PRETRAIN = 50
N_FINETUNE = 50
SEEDS = [42, 123, 777, 2024, 3141]  # 5模型集成
USE_AMP = True

def train_single_model(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}\nTraining model with seed={seed}\n{'='*60}"); sys.stdout.flush()

    model = MultiRegSetTransformer(feat_dim=128, embed_dim=256, n_regs=5, n_heads=4, n_layers=2).to(device)
    simclr_loss = SimCLRLoss(temperature=0.1)
    scaler = GradScaler('cuda')

    # Phase 1: SimCLR
    print(f"  Phase 1: SimCLR Pre-training..."); sys.stdout.flush()
    opt1 = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=N_PRETRAIN)
    pre_loader = DataLoader(MultiRegContrastiveDS(finger_reg_train, finger_regs, fpr=FPR, n_samples=80),
                            batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_contrastive)
    t0 = time.time()
    for epoch in range(N_PRETRAIN):
        model.train(); tl = 0
        for F1, R1, F2, R2 in pre_loader:
            F1, R1, F2, R2 = F1.to(device), R1.to(device), F2.to(device), R2.to(device)
            opt1.zero_grad()
            with autocast('cuda'):
                z1 = F.normalize(model(F1, R1), dim=1)
                z2 = F.normalize(model(F2, R2), dim=1)
                loss = simclr_loss(z1, z2)
            scaler.scale(loss).backward(); scaler.step(opt1); scaler.update()
            tl += loss.item()
        sch1.step()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_PRETRAIN}: loss={tl/len(pre_loader):.4f}, time={time.time()-t0:.0f}s"); sys.stdout.flush()

    # Phase 2: ArcFace
    print(f"  Phase 2: ArcFace Fine-tuning..."); sys.stdout.flush()
    arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)
    opt2 = optim.AdamW(list(model.parameters()) + list(arcface.parameters()), lr=0.0002, weight_decay=1e-3)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=N_FINETUNE)
    scaler2 = GradScaler('cuda')
    ft_loader = DataLoader(MultiRegTrainDS(finger_reg_train, finger_regs, finger_labels, fpr=FPR, n_samples=60),
                           batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_train)
    t1 = time.time()
    for epoch in range(N_FINETUNE):
        model.train(); arcface.train(); tl, cor, tot = 0, 0, 0
        for X, R, Y in ft_loader:
            X, R, Y = X.to(device), R.to(device), Y.to(device)
            opt2.zero_grad()
            with autocast('cuda'):
                emb = model(X, R); loss = arcface(emb, Y)
            scaler2.scale(loss).backward(); scaler2.step(opt2); scaler2.update()
            tl += loss.item()
            with torch.no_grad():
                en = F.normalize(emb.float(), dim=1); wn = F.normalize(arcface.weight.float(), dim=1)
                cor += torch.mm(en, wn.t()).max(1)[1].eq(Y).sum().item(); tot += Y.size(0)
        sch2.step()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_FINETUNE}: loss={tl/len(ft_loader):.4f}, acc={cor/tot*100:.1f}%, time={time.time()-t1:.0f}s"); sys.stdout.flush()

    model.eval()
    return model

# 训练5个模型
models = []
t_total = time.time()
for seed in SEEDS:
    m = train_single_model(seed)
    models.append(m)
    print(f"  Model seed={seed} done. Elapsed: {time.time()-t_total:.0f}s"); sys.stdout.flush()

print(f"\nAll {len(models)} models trained in {time.time()-t_total:.0f}s")
sys.stdout.flush()

# ============================================================
# 提取嵌入
# ============================================================
def get_ensemble_emb(models, reg_data, n_regs, fpr=3):
    frames, reg_ids = sample_multireg_raw(reg_data, n_regs, fpr)
    if frames is None: return None
    f_t = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    r_t = torch.tensor(reg_ids, dtype=torch.long).unsqueeze(0).to(device)
    embs = []
    for model in models:
        with torch.no_grad():
            embs.append(F.normalize(model(f_t, r_t), dim=1))
    return F.normalize(torch.mean(torch.stack(embs), dim=0), dim=1).cpu().numpy().flatten()

# ============================================================
# 方法1: 标准评估 (与v7相同, 但K=50)
# ============================================================
print("\n" + "="*60)
print("方法1: 标准Cosine评估 (5-model ensemble, K=50)")
print("="*60); sys.stdout.flush()

K = 50
en_embs, ve_embs = {}, {}
for finger in fingers:
    n_r = finger_regs[finger]
    en_embs[finger] = [e for e in [get_ensemble_emb(models, finger_reg_train[finger], n_r, FPR) for _ in range(K)] if e is not None]
    ve_embs[finger] = [e for e in [get_ensemble_emb(models, finger_reg_test[finger], n_r, FPR) for _ in range(K)] if e is not None]
    if len(en_embs[finger]) % 10 == 0:
        print(f"  Extracted: {finger}"); sys.stdout.flush()

def topk_avg(e, v, k=5):
    sims = sorted([np.dot(a, b) for a in e for b in v], reverse=True)
    return np.mean(sims[:k])

def full_evaluate(enroll_embs, verify_embs, label="", finger_list=None):
    if finger_list is None: finger_list = fingers
    for k_val in [1, 3, 5, 10]:
        genuine, impostor = [], []
        for f in finger_list:
            if enroll_embs.get(f) and verify_embs.get(f):
                genuine.append(topk_avg(enroll_embs[f], verify_embs[f], k_val))
        for i in range(len(finger_list)):
            for j in range(i+1, len(finger_list)):
                fi, fj = finger_list[i], finger_list[j]
                if enroll_embs.get(fi) and verify_embs.get(fj):
                    impostor.append(topk_avg(enroll_embs[fi], verify_embs[fj], k_val))
        if not genuine or not impostor: continue

        im, ist = np.mean(impostor), np.std(impostor)
        gz = [(g - im)/(ist+1e-8) for g in genuine]
        iz = [(x - im)/(ist+1e-8) for x in impostor]
        fpr_arr, tpr, _ = roc_curve([1]*len(gz)+[0]*len(iz), gz+iz)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
        eer = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2
        gmin, imax = np.min(gz), np.max(iz)
        sep = "*** PERFECT ***" if gmin > imax else f"overlap={sum(1 for x in iz if x > gmin)}/{len(iz)}"
        print(f"\n  {label} top-{k_val}: EER={eer*100:.2f}%, gen_min={gmin:.3f}, imp_max={imax:.3f}, {sep}")
        for tf in [0.0, 0.03, 0.05, 0.10]:
            idx = np.argmin(np.abs(fnr - tf))
            print(f"    FFR={tf*100:.0f}% -> FAR={fpr_arr[idx]*100:.4f}%")
        idx_far = np.argmin(np.abs(fpr_arr - 0.00002))
        print(f"    FAR=0.002% -> FFR={fnr[idx_far]*100:.2f}%")
        sys.stdout.flush()

full_evaluate(en_embs, ve_embs, "Ensemble-5")

# ============================================================
# 方法2: S-norm分数归一化
# ============================================================
print("\n" + "="*60)
print("方法2: S-norm 分数归一化")
print("="*60); sys.stdout.flush()

def compute_snorm_score(enroll_set, verify_set, all_enroll, all_verify, target_finger, k_val=5):
    """
    S-norm: 对称归一化
    score_snorm = 0.5 * (score_znorm + score_tnorm)
    Z-norm: 用verify embedding对所有非目标enrolled的分数做归一化
    T-norm: 用enroll embedding对所有非目标verify的分数做归一化
    """
    raw_score = topk_avg(enroll_set, verify_set, k_val)

    # Z-norm: verify vs all other enrolled (cohort)
    cohort_scores_z = []
    for f in fingers:
        if f == target_finger or not all_enroll.get(f):
            continue
        cohort_scores_z.append(topk_avg(all_enroll[f], verify_set, k_val))
    if cohort_scores_z:
        z_mean, z_std = np.mean(cohort_scores_z), np.std(cohort_scores_z) + 1e-8
        z_score = (raw_score - z_mean) / z_std
    else:
        z_score = raw_score

    # T-norm: enroll vs all other verify (cohort)
    cohort_scores_t = []
    for f in fingers:
        if f == target_finger or not all_verify.get(f):
            continue
        cohort_scores_t.append(topk_avg(enroll_set, all_verify[f], k_val))
    if cohort_scores_t:
        t_mean, t_std = np.mean(cohort_scores_t), np.std(cohort_scores_t) + 1e-8
        t_score = (raw_score - t_mean) / t_std
    else:
        t_score = raw_score

    return 0.5 * (z_score + t_score)

for k_val in [3, 5]:
    genuine_snorm, impostor_snorm = [], []

    for f in fingers:
        if en_embs.get(f) and ve_embs.get(f):
            s = compute_snorm_score(en_embs[f], ve_embs[f], en_embs, ve_embs, f, k_val)
            genuine_snorm.append(s)

    for i in range(len(fingers)):
        for j in range(i+1, len(fingers)):
            fi, fj = fingers[i], fingers[j]
            if en_embs.get(fi) and ve_embs.get(fj):
                s = compute_snorm_score(en_embs[fi], ve_embs[fj], en_embs, ve_embs, fi, k_val)
                impostor_snorm.append(s)

    if not genuine_snorm or not impostor_snorm:
        print(f"  S-norm top-{k_val}: insufficient data"); continue

    # 直接用S-norm分数做ROC (已经归一化过了)
    all_scores = genuine_snorm + impostor_snorm
    all_labels = [1]*len(genuine_snorm) + [0]*len(impostor_snorm)
    fpr_arr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
    eer = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2
    gmin, imax = np.min(genuine_snorm), np.max(impostor_snorm)
    sep = "*** PERFECT ***" if gmin > imax else f"overlap={sum(1 for x in impostor_snorm if x > gmin)}/{len(impostor_snorm)}"
    print(f"\n  S-norm top-{k_val}: EER={eer*100:.2f}%, gen_min={gmin:.3f}, imp_max={imax:.3f}, {sep}")
    for tf in [0.0, 0.03, 0.05, 0.10]:
        idx = np.argmin(np.abs(fnr - tf))
        print(f"    FFR={tf*100:.0f}% -> FAR={fpr_arr[idx]*100:.4f}%")
    idx_far = np.argmin(np.abs(fpr_arr - 0.00002))
    print(f"    FAR=0.002% -> FFR={fnr[idx_far]*100:.2f}%")
    sys.stdout.flush()

# ============================================================
# 方法3: 参数化分布外推 (理论FAR估算)
# ============================================================
print("\n" + "="*60)
print("方法3: 参数化分布外推 (理论FAR估算)")
print("="*60); sys.stdout.flush()

def d_prime_to_eer(dp):
    """d-prime到理论EER的近似转换"""
    return stats.norm.cdf(-dp / 2)

for k_val in [5]:
    genuine_raw, impostor_raw = [], []
    for f in fingers:
        if en_embs.get(f) and ve_embs.get(f):
            genuine_raw.append(topk_avg(en_embs[f], ve_embs[f], k_val))
    for i in range(len(fingers)):
        for j in range(i+1, len(fingers)):
            fi, fj = fingers[i], fingers[j]
            if en_embs.get(fi) and ve_embs.get(fj):
                impostor_raw.append(topk_avg(en_embs[fi], ve_embs[fj], k_val))

    genuine_arr = np.array(genuine_raw)
    impostor_arr = np.array(impostor_raw)

    # 拟合高斯分布
    gen_mu, gen_std = np.mean(genuine_arr), np.std(genuine_arr)
    imp_mu, imp_std = np.mean(impostor_arr), np.std(impostor_arr)

    print(f"\n  原始分数分布 (top-{k_val}):")
    print(f"    Genuine:  mu={gen_mu:.4f}, std={gen_std:.4f}, min={np.min(genuine_arr):.4f}, max={np.max(genuine_arr):.4f}")
    print(f"    Impostor: mu={imp_mu:.4f}, std={imp_std:.4f}, min={np.min(impostor_arr):.4f}, max={np.max(impostor_arr):.4f}")

    # KS检验: 分数分布是否近似高斯?
    _, p_gen = stats.normaltest(genuine_arr) if len(genuine_arr) >= 8 else (0, 1)
    _, p_imp = stats.normaltest(impostor_arr) if len(impostor_arr) >= 8 else (0, 1)
    print(f"    高斯检验 p-value: Genuine={p_gen:.4f}, Impostor={p_imp:.4f}")
    print(f"    (p>0.05: 接近高斯)")

    # d-prime: 分布分离度
    d_prime = (gen_mu - imp_mu) / np.sqrt(0.5 * (gen_std**2 + imp_std**2))
    print(f"    d-prime = {d_prime:.2f} (>4.0 为优秀)")

    # 理论外推: 在不同FFR下的理论FAR
    print(f"\n  理论外推 (假设高斯分布):")
    for target_ffr in [0.0, 0.01, 0.03, 0.05, 0.10]:
        if target_ffr == 0.0:
            threshold = np.min(genuine_arr) - 0.001  # 比最小genuine分数略低
        else:
            threshold = stats.norm.ppf(target_ffr, gen_mu, gen_std)  # FFR = P(genuine < threshold)
        # 理论FAR = P(impostor > threshold)
        theoretical_far = 1 - stats.norm.cdf(threshold, imp_mu, imp_std)
        # 实际FAR (从数据)
        actual_far = np.mean(impostor_arr >= threshold) if len(impostor_arr) > 0 else 0
        print(f"    FFR={target_ffr*100:.0f}%: threshold={threshold:.4f}, 理论FAR={theoretical_far:.6f} ({theoretical_far*100:.4f}%), 实际FAR={actual_far*100:.4f}%")

    # 反过来: 在FAR=0.002%时的理论FFR
    target_far = 0.00002  # 1/50000
    threshold_strict = stats.norm.ppf(1 - target_far, imp_mu, imp_std)  # FAR = P(impostor > threshold)
    theoretical_ffr = stats.norm.cdf(threshold_strict, gen_mu, gen_std)  # FFR = P(genuine < threshold)
    print(f"\n  目标FAR=0.002% (1/50000):")
    print(f"    需要阈值: {threshold_strict:.4f}")
    print(f"    理论FFR: {theoretical_ffr*100:.2f}%")
    print(f"    理论EER: {d_prime_to_eer(d_prime)*100:.4f}% (由d-prime估算)")
    sys.stdout.flush()

    # 实际分数的分离gap
    gap = np.min(genuine_arr) - np.max(impostor_arr)
    print(f"\n  分数间隔: gen_min - imp_max = {np.min(genuine_arr):.4f} - {np.max(impostor_arr):.4f} = {gap:.4f}")
    if gap > 0:
        print(f"  ==> 完全分离! 理论FAR可无限趋近0 (在数据范围内)")
    else:
        n_overlap = np.sum(impostor_arr >= np.min(genuine_arr))
        print(f"  ==> 重叠区域: {n_overlap}/{len(impostor_arr)} impostor分数 > gen_min")

# ============================================================
# 分组分析
# ============================================================
print("\n" + "="*60)
print("分组分析 (5-model Ensemble)")
print("="*60); sys.stdout.flush()

wtp_fingers = [f for f in fingers if f.startswith("wtp_")]
btp_fingers = [f for f in fingers if f.startswith("btp_")]

for group_name, group_fingers in [("无贴屏", wtp_fingers), ("不贴屏", btp_fingers)]:
    if not group_fingers: continue
    print(f"\n  --- {group_name} ({len(group_fingers)}类) ---"); sys.stdout.flush()
    full_evaluate(en_embs, ve_embs, group_name, group_fingers)

# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print(f"V8 Enhanced 完成. 总时间: {time.time()-t_total:.0f}s")
print(f"  - 5模型集成 (seeds: {SEEDS})")
print(f"  - K={K} 密集采样")
print(f"  - S-norm 分数归一化")
print(f"  - 参数化分布外推")
print(f"{'='*60}"); sys.stdout.flush()
