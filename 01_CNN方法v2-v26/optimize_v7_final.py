"""
最终优化方案: 多寄存器Set Transformer + SimCLR + 模型集成
路径2: 所有寄存器原始帧一起输入Set Transformer, 带寄存器位置编码
路径1: 训练3个不同seed的模型, 嵌入级融合
无数据泄露: 预训练/微调/测试严格分离
优化: mixed precision + 减少采样加速训练
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
# 数据加载: 每指所有寄存器的原始帧
# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'

wtp_regs = ['Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245', 'Rgd1247']
btp_regs = ['Rgd1237', 'Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245']

finger_reg_train = {}
finger_reg_test = {}
finger_labels = {}
finger_source = {}
finger_regs = {}
fi = 0

print("Loading 无贴屏 (5 registers)..."); sys.stdout.flush()
for finger in sorted(os.listdir(base1)):
    key = f"wtp_{finger}"
    finger_reg_train[key] = {}
    finger_reg_test[key] = {}
    for ri, reg in enumerate(wtp_regs):
        rpath = os.path.join(base1, finger, reg)
        imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
        loaded = []
        for p in imgs_paths:
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
    if not os.path.exists(primary):
        continue
    if len(glob.glob(os.path.join(primary, '*.bmp'))) < 50:
        continue
    key = f"btp_{finger}"
    finger_reg_train[key] = {}
    finger_reg_test[key] = {}
    for ri, reg in enumerate(btp_regs):
        rpath = os.path.join(base2, finger, reg)
        if not os.path.exists(rpath):
            finger_reg_train[key][ri] = []
            finger_reg_test[key][ri] = []
            continue
        imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
        loaded = []
        for p in imgs_paths:
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
# 模型: 多寄存器Set Transformer
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
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, frames, reg_ids):
        B, N, H, W = frames.shape
        x = frames.view(B * N, 1, H, W)
        feats = self.frame_encoder(x)
        feats = feats.view(B, N, -1)
        reg_emb = self.reg_embedding(reg_ids)
        feats = feats + reg_emb
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, feats], dim=1)
        x = self.transformer(x)
        agg = x[:, 0, :]
        return self.projector(agg)

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
# 采帧函数
# ============================================================
def sample_multireg_raw(reg_data, n_regs=5, frames_per_reg=3):
    all_frames = []
    all_reg_ids = []
    for ri in range(n_regs):
        imgs = reg_data.get(ri, [])
        if len(imgs) < 3:
            continue
        n_sample = min(frames_per_reg, len(imgs))
        chosen = random.sample(range(len(imgs)), n_sample)
        for idx in chosen:
            all_frames.append(imgs[idx])
            all_reg_ids.append(ri)
    if not all_frames:
        return None, None
    return np.stack(all_frames, axis=0), np.array(all_reg_ids)

# ============================================================
# 数据集
# ============================================================
class MultiRegContrastiveDS(Dataset):
    def __init__(self, reg_train, regs_count, fpr=3, n_samples=80):
        self.reg_train = reg_train
        self.regs_count = regs_count
        self.fpr = fpr
        self.n_samples = n_samples
        self.fingers = list(reg_train.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

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
        self.reg_train = reg_train
        self.regs_count = regs_count
        self.labels = labels
        self.fpr = fpr
        self.n_samples = n_samples
        self.fingers = list(reg_train.keys())

    def __len__(self):
        return len(self.fingers) * self.n_samples

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        n_r = self.regs_count[finger]
        frames, reg_ids = sample_multireg_raw(self.reg_train[finger], n_r, self.fpr)
        for i in range(len(frames)):
            if random.random() < 0.3: frames[i] = -frames[i]
        frames += np.random.randn(*frames.shape).astype(np.float32) * 0.02
        return (torch.tensor(frames, dtype=torch.float32),
                torch.tensor(reg_ids, dtype=torch.long),
                self.labels[finger])

def collate_contrastive(batch):
    f1s, r1s, f2s, r2s = zip(*batch)
    max_n1 = max(x.size(0) for x in f1s)
    max_n2 = max(x.size(0) for x in f2s)
    max_n = max(max_n1, max_n2)
    B = len(f1s)
    H, W = f1s[0].shape[1], f1s[0].shape[2]
    F1 = torch.zeros(B, max_n, H, W)
    R1 = torch.zeros(B, max_n, dtype=torch.long)
    F2 = torch.zeros(B, max_n, H, W)
    R2 = torch.zeros(B, max_n, dtype=torch.long)
    for i in range(B):
        n1, n2 = f1s[i].size(0), f2s[i].size(0)
        F1[i, :n1] = f1s[i]; R1[i, :n1] = r1s[i]
        F2[i, :n2] = f2s[i]; R2[i, :n2] = r2s[i]
    return F1, R1, F2, R2

def collate_train(batch):
    fs, rs, ys = zip(*batch)
    max_n = max(x.size(0) for x in fs)
    B = len(fs)
    H, W = fs[0].shape[1], fs[0].shape[2]
    F_out = torch.zeros(B, max_n, H, W)
    R_out = torch.zeros(B, max_n, dtype=torch.long)
    Y_out = torch.tensor(ys, dtype=torch.long)
    for i in range(B):
        n = fs[i].size(0)
        F_out[i, :n] = fs[i]; R_out[i, :n] = rs[i]
    return F_out, R_out, Y_out

# ============================================================
# 训练参数
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

FPR = 3           # 每寄存器采3帧, 共15帧
N_PRETRAIN = 50   # SimCLR预训练轮数
N_FINETUNE = 50   # ArcFace微调轮数
SEEDS = [42, 123, 777]
USE_AMP = True    # mixed precision

def train_single_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"Training model with seed={seed}")
    print(f"{'='*60}"); sys.stdout.flush()

    model = MultiRegSetTransformer(feat_dim=128, embed_dim=256, n_regs=5, n_heads=4, n_layers=2).to(device)
    simclr_loss = SimCLRLoss(temperature=0.1)
    scaler = GradScaler('cuda') if USE_AMP else None

    # Phase 1: SimCLR
    print(f"  Phase 1: SimCLR Pre-training..."); sys.stdout.flush()
    opt1 = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=N_PRETRAIN)

    pre_ds = MultiRegContrastiveDS(finger_reg_train, finger_regs, fpr=FPR, n_samples=80)
    pre_loader = DataLoader(pre_ds, batch_size=16, shuffle=True, num_workers=0,
                            collate_fn=collate_contrastive)
    print(f"    DataLoader: {len(pre_ds)} samples, {len(pre_loader)} batches"); sys.stdout.flush()

    t0 = time.time()
    for epoch in range(N_PRETRAIN):
        model.train()
        tl = 0
        for F1, R1, F2, R2 in pre_loader:
            F1, R1, F2, R2 = F1.to(device), R1.to(device), F2.to(device), R2.to(device)
            opt1.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    z1 = F.normalize(model(F1, R1), dim=1)
                    z2 = F.normalize(model(F2, R2), dim=1)
                    loss = simclr_loss(z1, z2)
                scaler.scale(loss).backward()
                scaler.step(opt1)
                scaler.update()
            else:
                z1 = F.normalize(model(F1, R1), dim=1)
                z2 = F.normalize(model(F2, R2), dim=1)
                loss = simclr_loss(z1, z2)
                loss.backward()
                opt1.step()
            tl += loss.item()
        sch1.step()
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{N_PRETRAIN}: loss={tl/len(pre_loader):.4f}, time={time.time()-t0:.0f}s")
            sys.stdout.flush()

    # Phase 2: ArcFace
    print(f"  Phase 2: ArcFace Fine-tuning..."); sys.stdout.flush()
    arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)
    opt2 = optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=0.0002, weight_decay=1e-3
    )
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=N_FINETUNE)
    scaler2 = GradScaler('cuda') if USE_AMP else None

    ft_ds = MultiRegTrainDS(finger_reg_train, finger_regs, finger_labels, fpr=FPR, n_samples=60)
    ft_loader = DataLoader(ft_ds, batch_size=16, shuffle=True, num_workers=0,
                           collate_fn=collate_train)
    print(f"    DataLoader: {len(ft_ds)} samples, {len(ft_loader)} batches"); sys.stdout.flush()

    t1 = time.time()
    for epoch in range(N_FINETUNE):
        model.train(); arcface.train()
        tl, cor, tot = 0, 0, 0
        for X, R, Y in ft_loader:
            X, R, Y = X.to(device), R.to(device), Y.to(device)
            opt2.zero_grad()
            if USE_AMP:
                with autocast('cuda'):
                    emb = model(X, R)
                    loss = arcface(emb, Y)
                scaler2.scale(loss).backward()
                scaler2.step(opt2)
                scaler2.update()
            else:
                emb = model(X, R)
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
        if (epoch+1) % 5 == 0:
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
    print(f"  Model seed={seed} done. Total elapsed: {time.time()-t_total:.0f}s"); sys.stdout.flush()

# ============================================================
# 测试
# ============================================================
print("\n" + "="*60)
print("Testing (30 classes, multi-register Set Transformer)")
print("="*60); sys.stdout.flush()

def get_emb(model, reg_data, n_regs, fpr=3):
    frames, reg_ids = sample_multireg_raw(reg_data, n_regs, fpr)
    if frames is None:
        return None
    f_t = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    r_t = torch.tensor(reg_ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(f_t, r_t)
    return F.normalize(emb, dim=1).cpu().numpy().flatten()

def get_ensemble_emb(models, reg_data, n_regs, fpr=3):
    frames, reg_ids = sample_multireg_raw(reg_data, n_regs, fpr)
    if frames is None:
        return None
    f_t = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    r_t = torch.tensor(reg_ids, dtype=torch.long).unsqueeze(0).to(device)
    embs = []
    for model in models:
        with torch.no_grad():
            emb = model(f_t, r_t)
        embs.append(F.normalize(emb, dim=1))
    avg_emb = torch.mean(torch.stack(embs), dim=0)
    return F.normalize(avg_emb, dim=1).cpu().numpy().flatten()

def topk_avg(e, v, k=5):
    sims = sorted([np.dot(a, b) for a in e for b in v], reverse=True)
    return np.mean(sims[:k])

def evaluate(enroll_embs, verify_embs, label="", finger_list=None):
    if finger_list is None:
        finger_list = fingers
    for k_val in [3, 5]:
        genuine, impostor = [], []
        for f in finger_list:
            if f in enroll_embs and f in verify_embs and enroll_embs[f] and verify_embs[f]:
                genuine.append(topk_avg(enroll_embs[f], verify_embs[f], k_val))
        for i in range(len(finger_list)):
            for j in range(i+1, len(finger_list)):
                fi, fj = finger_list[i], finger_list[j]
                if fi in enroll_embs and fj in verify_embs and enroll_embs[fi] and verify_embs[fj]:
                    impostor.append(topk_avg(enroll_embs[fi], verify_embs[fj], k_val))

        if not genuine or not impostor:
            print(f"  {label} top-{k_val}: insufficient data"); sys.stdout.flush()
            continue

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

K = 20

# --- 单模型测试 ---
for mi, model in enumerate(models):
    print(f"\n--- Single Model (seed={SEEDS[mi]}) ---"); sys.stdout.flush()
    en_e, ve_e = {}, {}
    for finger in fingers:
        n_r = finger_regs[finger]
        en_e[finger] = [get_emb(model, finger_reg_train[finger], n_r, FPR) for _ in range(K)]
        ve_e[finger] = [get_emb(model, finger_reg_test[finger], n_r, FPR) for _ in range(K)]
        en_e[finger] = [e for e in en_e[finger] if e is not None]
        ve_e[finger] = [e for e in ve_e[finger] if e is not None]
    evaluate(en_e, ve_e, f"Model-{mi}")

# --- 集成测试 ---
print(f"\n--- Ensemble ({len(models)} models) ---"); sys.stdout.flush()
en_ens, ve_ens = {}, {}
for finger in fingers:
    n_r = finger_regs[finger]
    en_ens[finger] = [get_ensemble_emb(models, finger_reg_train[finger], n_r, FPR) for _ in range(K)]
    ve_ens[finger] = [get_ensemble_emb(models, finger_reg_test[finger], n_r, FPR) for _ in range(K)]
    en_ens[finger] = [e for e in en_ens[finger] if e is not None]
    ve_ens[finger] = [e for e in ve_ens[finger] if e is not None]

evaluate(en_ens, ve_ens, "Ensemble")

# --- 集成 + 更多采样 (K=30) ---
print(f"\n--- Ensemble + Dense Sampling (K=30) ---"); sys.stdout.flush()
K2 = 30
en_ens2, ve_ens2 = {}, {}
for finger in fingers:
    n_r = finger_regs[finger]
    en_ens2[finger] = [get_ensemble_emb(models, finger_reg_train[finger], n_r, FPR) for _ in range(K2)]
    ve_ens2[finger] = [get_ensemble_emb(models, finger_reg_test[finger], n_r, FPR) for _ in range(K2)]
    en_ens2[finger] = [e for e in en_ens2[finger] if e is not None]
    ve_ens2[finger] = [e for e in ve_ens2[finger] if e is not None]

evaluate(en_ens2, ve_ens2, "Ensemble-Dense")

# --- 分组分析 ---
print("\n" + "="*60)
print("分组分析 (Ensemble)")
print("="*60); sys.stdout.flush()

wtp_fingers = [f for f in fingers if f.startswith("wtp_")]
btp_fingers = [f for f in fingers if f.startswith("btp_")]

for group_name, group_fingers in [("无贴屏", wtp_fingers), ("不贴屏", btp_fingers)]:
    if not group_fingers:
        continue
    print(f"\n  --- {group_name} ({len(group_fingers)}类) ---"); sys.stdout.flush()
    evaluate(en_ens, ve_ens, group_name, group_fingers)

print(f"\n{'='*60}")
print(f"All done. Total time: {time.time()-t_total:.0f}s")
print(f"{'='*60}"); sys.stdout.flush()
