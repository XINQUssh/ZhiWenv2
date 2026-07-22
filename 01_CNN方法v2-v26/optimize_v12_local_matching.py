"""
V12: 密集局部特征匹配 CNN
核心创新:
  1. 预训练ResNet-18骨干网 (ImageNet权重) - 解决30类数据稀缺问题
  2. 保留空间特征图, 不做全局池化 - 保留空间对应关系
  3. 局部特征匹配 (互近邻 + 空间一致性) - 模仿SIFT的几何匹配
  4. 离散匹配分数 (匹配数量) - 类似SIFT inlier count
  5. 双极性推理 + CLAHE + 2x上采样 + GMFS掩膜 (继承v11)
  6. 多尺度特征 (layer3 + layer4) + 3模型集成
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
from torchvision.models import resnet18, ResNet18_Weights

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 预处理: CLAHE + 2x上采样 + GMFS掩膜 (继承v11)
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
# 数据加载: 单寄存器 (继承v11)
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
# 模型: 预训练ResNet-18 + 密集局部特征提取器
# ============================================================
class DenseLocalFeatureExtractor(nn.Module):
    """
    预训练ResNet-18骨干网, 去掉全局池化和FC层
    输出多尺度空间特征图用于局部匹配
    同时保留全局分支用于训练
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # 加载预训练ResNet-18
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 修改第一层: 1通道输入 (取RGB平均权重)
        orig_w = base.conv1.weight.data  # [64, 3, 7, 7]
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = orig_w.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1  # 64ch, stride=1
        self.layer2 = base.layer2  # 128ch, stride=2
        self.layer3 = base.layer3  # 256ch, stride=2
        self.layer4 = base.layer4  # 512ch, stride=2

        # 局部特征投影头: 降维到128维 (对layer3和layer4分别投影)
        self.local_proj3 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.local_proj4 = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False),
            nn.BatchNorm2d(128),
        )

        # 全局分支: 用于ArcFace训练
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward_features(self, x):
        """提取多尺度空间特征图"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)   # [B, 256, H3, W3]
        f4 = self.layer4(f3)  # [B, 512, H4, W4]

        # 局部特征: 投影 + L2归一化
        local3 = F.normalize(self.local_proj3(f3), dim=1)  # [B, 128, H3, W3]
        local4 = F.normalize(self.local_proj4(f4), dim=1)  # [B, 128, H4, W4]

        return local3, local4, f4

    def forward(self, x):
        """训练时: 返回全局嵌入 (用于ArcFace)"""
        _, _, f4 = self.forward_features(x)
        g = self.global_pool(f4).flatten(1)
        return self.global_proj(g)

    def extract_local(self, x):
        """推理时: 返回密集局部特征图"""
        local3, local4, _ = self.forward_features(x)
        return local3, local4

# ============================================================
# 密集局部特征匹配引擎
# ============================================================
def dense_local_match_score(feat_q, feat_t, sim_thresh=0.3):
    """
    密集局部匹配:
    1. 计算相似度矩阵
    2. 双向最近邻
    3. 互为最近邻过滤
    4. 相似度阈值过滤
    5. 空间一致性检查 (简化RANSAC)
    返回: 匹配数量 (离散整数)

    feat_q: [C, H, W] 查询特征图
    feat_t: [C, H, W] 模板特征图
    """
    C, H, W = feat_q.shape
    N = H * W

    # 展平: [N, C]
    fq = feat_q.reshape(C, -1).T  # [N, C]
    ft = feat_t.reshape(C, -1).T  # [N, C]

    # 相似度矩阵 [N_q, N_t]
    sim = fq @ ft.T

    # 双向最近邻
    nn_qt = sim.argmax(axis=1)   # query→template 最佳匹配
    nn_tq = sim.argmax(axis=0)   # template→query 最佳匹配

    # 互为最近邻
    mutual = np.zeros(N, dtype=bool)
    for i in range(N):
        if nn_tq[nn_qt[i]] == i:
            mutual[i] = True

    # 相似度阈值
    sim_vals = sim[np.arange(N), nn_qt]
    valid = mutual & (sim_vals > sim_thresh)

    if valid.sum() < 3:
        return valid.sum()

    # 空间一致性检查 (简化版)
    # 计算匹配对的位移一致性
    q_positions = np.array([(i // W, i % W) for i in range(N)])[valid]  # [M, 2]
    t_positions = np.array([(nn_qt[i] // W, nn_qt[i] % W) for i in range(N) if valid[i]])  # [M, 2]

    displacements = t_positions - q_positions  # [M, 2]

    # 用中位数位移做一致性检查 (比RANSAC简单但有效)
    median_disp = np.median(displacements, axis=0)
    disp_errors = np.linalg.norm(displacements - median_disp, axis=1)

    # 容忍2个像素的偏差 (在特征图空间)
    inliers = disp_errors < 2.5

    return int(inliers.sum())


def multi_scale_match(local3_q, local4_q, local3_t, local4_t, sim_thresh=0.3):
    """多尺度密集匹配: layer3 + layer4 分别匹配, 分数相加"""
    score3 = dense_local_match_score(local3_q, local3_t, sim_thresh)
    score4 = dense_local_match_score(local4_q, local4_t, sim_thresh)
    return score3 + score4

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
# 数据增强 (增强版: 加入弹性形变和随机裁切)
# ============================================================
def augment_frame(frame):
    f = frame.copy()
    # 极性翻转
    if random.random() < 0.5:
        f = -f
    # 随机噪声
    f += np.random.randn(*f.shape).astype(np.float32) * 0.03
    # 水平翻转
    if random.random() < 0.3:
        f = f[:, ::-1].copy()
    # 随机平移 (模拟按压位置偏移, ±5像素)
    if random.random() < 0.5:
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        f = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), borderValue=0)
    # 轻微旋转 (±3度)
    if random.random() < 0.3:
        angle = random.uniform(-3, 3)
        h, w = f.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        f = cv2.warpAffine(f, M, (w, h), borderValue=0)
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
# 训练: 渐进式解冻
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}"); sys.stdout.flush()

N_PRETRAIN = 60    # SimCLR (用全局嵌入)
N_FINETUNE = 80    # ArcFace (用全局嵌入)
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

    model = DenseLocalFeatureExtractor(embed_dim=256).to(device)
    simclr_loss = SimCLRLoss(temperature=0.07)

    # ---- Phase 1: SimCLR预训练 (冻结骨干网, 只训练投影头) ----
    print(f"  Phase 1: SimCLR Pre-training ({N_PRETRAIN} epochs, backbone frozen)...")
    sys.stdout.flush()

    # 冻结骨干网
    for name, param in model.named_parameters():
        if 'local_proj' in name or 'global_proj' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,} params")

    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=0.001, weight_decay=1e-3)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=N_PRETRAIN)
    scaler = GradScaler('cuda') if USE_AMP else None

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

    # ---- Phase 2: ArcFace微调 (逐层解冻) ----
    print(f"  Phase 2: ArcFace Fine-tuning ({N_FINETUNE} epochs, progressive unfreeze)...")
    sys.stdout.flush()

    # 解冻layer3和layer4, 低学习率
    for param in model.parameters():
        param.requires_grad = True

    arcface = ArcFace(256, n_classes, s=64, m=0.5).to(device)

    # 差分学习率
    param_groups = [
        {'params': list(model.conv1.parameters()) + list(model.bn1.parameters()) +
                   list(model.layer1.parameters()) + list(model.layer2.parameters()),
         'lr': 1e-5},  # 早期层: 很低学习率
        {'params': list(model.layer3.parameters()),
         'lr': 5e-5},  # layer3: 中等
        {'params': list(model.layer4.parameters()),
         'lr': 1e-4},  # layer4: 较高
        {'params': list(model.local_proj3.parameters()) + list(model.local_proj4.parameters()) +
                   list(model.global_proj.parameters()) + list(arcface.parameters()),
         'lr': 3e-4},  # 投影头: 最高
    ]

    opt2 = optim.AdamW(param_groups, weight_decay=1e-3)
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
# 局部特征提取
# ============================================================
@torch.no_grad()
def extract_local_features(models, frame):
    """单帧 → 多模型集成的多尺度局部特征"""
    x = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    all_local3 = []
    all_local4 = []
    for model in models:
        l3, l4 = model.extract_local(x)
        all_local3.append(l3)
        all_local4.append(l4)

    # 集成: 平均后重新L2归一化
    avg_l3 = torch.mean(torch.stack(all_local3), dim=0)
    avg_l3 = F.normalize(avg_l3, dim=1)

    avg_l4 = torch.mean(torch.stack(all_local4), dim=0)
    avg_l4 = F.normalize(avg_l4, dim=1)

    return avg_l3[0].cpu().numpy(), avg_l4[0].cpu().numpy()  # [C, H, W] each


def extract_dual_polarity_local(models, frame):
    """双极性: 原始帧和翻转帧各提取局部特征"""
    orig_l3, orig_l4 = extract_local_features(models, frame)
    flip_l3, flip_l4 = extract_local_features(models, -frame)
    return (orig_l3, orig_l4), (flip_l3, flip_l4)


def dual_polarity_local_match(query_feats, template_feats, sim_thresh=0.3):
    """
    双极性局部匹配: 4种极性组合取max
    query_feats = ((q_orig_l3, q_orig_l4), (q_flip_l3, q_flip_l4))
    template_feats = ((t_orig_l3, t_orig_l4), (t_flip_l3, t_flip_l4))
    """
    (qo3, qo4), (qf3, qf4) = query_feats
    (to3, to4), (tf3, tf4) = template_feats

    scores = [
        multi_scale_match(qo3, qo4, to3, to4, sim_thresh),  # orig-orig
        multi_scale_match(qo3, qo4, tf3, tf4, sim_thresh),  # orig-flip
        multi_scale_match(qf3, qf4, to3, to4, sim_thresh),  # flip-orig
        multi_scale_match(qf3, qf4, tf3, tf4, sim_thresh),  # flip-flip
    ]
    return max(scores)

# ============================================================
# 同时测试两种匹配策略: 全局余弦 vs 局部匹配
# ============================================================
print(f"\n{'='*60}")
print(f"评估: V12 密集局部特征匹配")
print(f"{'='*60}"); sys.stdout.flush()

N_TEMPLATES = 20

# 提取全局嵌入 (用于对比) + 局部特征 (用于新方法)
@torch.no_grad()
def get_global_emb(models, frame):
    x = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    embs = []
    for model in models:
        emb = model(x)
        embs.append(F.normalize(emb, dim=1))
    avg = torch.mean(torch.stack(embs), dim=0)
    return F.normalize(avg, dim=1).cpu().numpy().flatten()

def get_dual_global(models, frame):
    return get_global_emb(models, frame), get_global_emb(models, -frame)

# ---- 提取模板特征 ----
print("Extracting template features (local + global, dual-polarity)..."); sys.stdout.flush()
t_emb = time.time()

templates_local = {}
templates_global = {}
for finger in fingers:
    train_frames = finger_train[finger]
    chosen = random.sample(train_frames, min(N_TEMPLATES, len(train_frames)))
    templates_local[finger] = [extract_dual_polarity_local(models, f) for f in chosen]
    templates_global[finger] = [get_dual_global(models, f) for f in chosen]

print(f"  Templates done in {time.time()-t_emb:.0f}s"); sys.stdout.flush()

# 获取特征图尺寸
sample_l3, sample_l4 = extract_local_features(models, finger_train[fingers[0]][0])
print(f"  Feature map sizes: layer3={sample_l3.shape}, layer4={sample_l4.shape}")
sys.stdout.flush()

# ---- 匹配测试帧 ----
print("Matching test frames..."); sys.stdout.flush()
t_eval = time.time()

# 局部匹配分数
gen_local = []
imp_local = []
# 全局余弦分数 (对比用)
gen_global = []
imp_global = []

for fi_idx, finger in enumerate(fingers):
    test_frames = finger_test[finger]
    if not test_frames:
        continue

    for frame in test_frames:
        # 提取查询特征
        q_local = extract_dual_polarity_local(models, frame)
        q_global = get_dual_global(models, frame)

        # Genuine: 和自己的模板比
        own_local_scores = [dual_polarity_local_match(q_local, t) for t in templates_local[finger]]
        gen_local.append(max(own_local_scores))

        own_global_scores = []
        for t_g in templates_global[finger]:
            sims = [np.dot(q_global[i], t_g[j]) for i in range(2) for j in range(2)]
            own_global_scores.append(max(sims))
        gen_global.append(max(own_global_scores))

        # Impostor: 和每个其他人的模板比
        for other in fingers:
            if other == finger:
                continue
            other_local_scores = [dual_polarity_local_match(q_local, t) for t in templates_local[other]]
            imp_local.append(max(other_local_scores))

            other_global_scores = []
            for t_g in templates_global[other]:
                sims = [np.dot(q_global[i], t_g[j]) for i in range(2) for j in range(2)]
                other_global_scores.append(max(sims))
            imp_global.append(max(other_global_scores))

    if (fi_idx+1) % 5 == 0 or fi_idx == 0:
        print(f"  [{fi_idx+1}/{n_classes}] elapsed={time.time()-t_eval:.0f}s"); sys.stdout.flush()

gen_local = np.array(gen_local)
imp_local = np.array(imp_local)
gen_global = np.array(gen_global)
imp_global = np.array(imp_global)

print(f"\nEvaluation done in {time.time()-t_eval:.0f}s")
sys.stdout.flush()

# ============================================================
# 结果分析
# ============================================================
def analyze_scores(name, gen, imp):
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

    return eer, fpr_arr, fnr, thresholds

eer_local, _, _, _ = analyze_scores("V12 密集局部匹配 (ResNet-18 + Multi-scale + 互近邻)", gen_local, imp_local)
eer_global, _, _, _ = analyze_scores("V12 全局余弦对比 (ResNet-18 + ArcFace)", gen_global, imp_global)

# ============================================================
# 融合分数: 局部 + 全局
# ============================================================
print(f"\n{'='*60}")
print(f"融合策略测试: 局部匹配 + 全局余弦")
print(f"{'='*60}"); sys.stdout.flush()

# 归一化两种分数到 [0, 1]
def min_max_normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

gen_local_norm = min_max_normalize(gen_local)
imp_local_norm = min_max_normalize(imp_local)
gen_global_norm = min_max_normalize(gen_global)
imp_global_norm = min_max_normalize(imp_global)

for alpha in [0.3, 0.5, 0.7]:
    gen_fused = alpha * gen_local_norm + (1-alpha) * gen_global_norm
    imp_fused = alpha * imp_local_norm + (1-alpha) * imp_global_norm

    y_true = np.concatenate([np.ones(len(gen_fused)), np.zeros(len(imp_fused))])
    y_scores = np.concatenate([gen_fused, imp_fused])
    fpr_arr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
    eer_f = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2

    idx3 = np.argmin(np.abs(fnr - 0.03))
    far_at_3 = fpr_arr[idx3] * 100

    print(f"  alpha={alpha:.1f} (local={alpha:.0%}, global={1-alpha:.0%}): EER={eer_f*100:.4f}%, FFR=3%->FAR={far_at_3:.4f}%")
sys.stdout.flush()

# ============================================================
# 分组分析 (局部匹配)
# ============================================================
print(f"\n{'='*60}")
print(f"分组分析 (局部匹配)")
print(f"{'='*60}"); sys.stdout.flush()

def group_analysis_local(group_name, group_fingers):
    g_gen, g_imp = [], []
    for finger in group_fingers:
        test_frames = finger_test.get(finger, [])
        for frame in test_frames:
            q_local = extract_dual_polarity_local(models, frame)
            own_scores = [dual_polarity_local_match(q_local, t) for t in templates_local[finger]]
            g_gen.append(max(own_scores))
            for other in group_fingers:
                if other == finger:
                    continue
                other_scores = [dual_polarity_local_match(q_local, t) for t in templates_local[other]]
                g_imp.append(max(other_scores))

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
group_analysis_local("无贴屏", wtp)
group_analysis_local("不贴屏", btp)

# ============================================================
# 不同sim_thresh的灵敏度分析
# ============================================================
print(f"\n{'='*60}")
print(f"sim_thresh 灵敏度分析")
print(f"{'='*60}"); sys.stdout.flush()

# 用缓存的特征重新计算不同阈值下的分数
# 这里用一个子集快速测试
print("Testing different sim_thresh values on subset...")
subset_fingers = fingers[:10]
for st in [0.1, 0.2, 0.3, 0.4, 0.5]:
    sg, si = [], []
    for finger in subset_fingers:
        test_frames = finger_test[finger][:5]  # 只取5帧加速
        for frame in test_frames:
            q_local = extract_dual_polarity_local(models, frame)
            own = [dual_polarity_local_match(q_local, t, sim_thresh=st) for t in templates_local[finger]]
            sg.append(max(own))
            for other in subset_fingers:
                if other == finger:
                    continue
                oth = [dual_polarity_local_match(q_local, t, sim_thresh=st) for t in templates_local[other]]
                si.append(max(oth))
    sg, si = np.array(sg), np.array(si)
    y_t = np.concatenate([np.ones(len(sg)), np.zeros(len(si))])
    y_s = np.concatenate([sg, si])
    fpr_t, tpr_t, _ = roc_curve(y_t, y_s)
    fnr_t = 1 - tpr_t
    ei = np.nanargmin(np.abs(fnr_t - fpr_t))
    eer_t = (fpr_t[ei] + fnr_t[ei]) / 2
    print(f"  sim_thresh={st:.1f}: EER={eer_t*100:.2f}%, gen_mean={sg.mean():.2f}, imp_max={si.max():.2f}")
    sys.stdout.flush()

# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print(f"V12 总结对比")
print(f"{'='*60}")
print(f"  V12 局部匹配 EER = {eer_local*100:.4f}%")
print(f"  V12 全局余弦 EER = {eer_global*100:.4f}%")
print(f"  V11 全局余弦 EER = 4.9968% (baseline)")
print(f"  V10 SIFT     EER = 1.5200% (target)")
print(f"\nAll done. Total time: {time.time()-t_total:.0f}s")
print(f"{'='*60}"); sys.stdout.flush()
