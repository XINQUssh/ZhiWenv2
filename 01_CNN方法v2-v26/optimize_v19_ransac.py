"""
V19: RANSAC Geometric Verification (DLG-inspired)
核心改进 (相对V18b):
  V18b的局部匹配只看"哪个位置描述符最像"，不考虑空间一致性
  参考DLG代码的匹配流水线：KD-tree NN → RANSAC → LPM

  V19改进:
  1. 逐模板匹配 (不再拼接20个模板的所有位置)
  2. RANSAC仿射验证: 在14×13网格坐标上验证匹配点的几何一致性
  3. LPM邻域一致性: 检查匹配点的空间邻域结构是否保持
  4. 多种分数组合策略

  使用V18b已训练的模型，无需重新训练
"""
import os, glob, random, time, sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from torchvision.models import resnet18, ResNet18_Weights
from skimage.measure import ransac
from skimage.transform import AffineTransform, SimilarityTransform
from sklearn.neighbors import KDTree

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 预处理 (与V14/V17/V18b一致)
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
# 数据加载
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
fi = 0

print(f"{'='*70}")
print(f"V19: RANSAC Geometric Verification (DLG-inspired)")
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

n_classes = fi
fingers = list(finger_train.keys())
print(f"  Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes-n_wtp} 不贴屏)")
print(f"Data loading done in {time.time()-t_load:.0f}s"); sys.stdout.flush()

# ============================================================
# 模型定义 (与V18b一致，用于加载权重)
# ============================================================
class DenseDescriptorEncoder(nn.Module):
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
        self.global_projector = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.local_head = nn.Sequential(
            nn.Conv2d(256, local_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(local_dim),
        )

    def forward(self, x, return_local=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        local_desc = None
        if return_local:
            local_desc = self.local_head(x)
            local_desc = F.normalize(local_desc, dim=1)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        global_emb = self.global_projector(x)
        if return_local:
            return global_emb, local_desc
        return global_emb

# ============================================================
# 加载V18b模型
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}"); sys.stdout.flush()

EMBED_DIM = 512
LOCAL_DIM = 64
SEEDS = [42, 123, 777]
N_TEMPLATES = 20
MODEL_DIR = 'f:/1111/指纹/models_v18b/'

print("Loading V18b models..."); sys.stdout.flush()
model_list = []
for seed in SEEDS:
    model = DenseDescriptorEncoder(embed_dim=EMBED_DIM, local_dim=LOCAL_DIM).to(device)
    path = os.path.join(MODEL_DIR, f'v18b_seed{seed}.pth')
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    model_list.append(model)
    print(f"  Loaded {path}")
print(f"All models loaded."); sys.stdout.flush()

# ============================================================
# 网格坐标 (layer3 feature map positions)
# ============================================================
H_FEAT, W_FEAT = 14, 13
N_POS = H_FEAT * W_FEAT  # 182

GRID_POSITIONS = np.array(
    [(i, j) for i in range(H_FEAT) for j in range(W_FEAT)],
    dtype=np.float64
)  # (182, 2) — (row, col) in feature map

# ============================================================
# 特征提取
# ============================================================
@torch.no_grad()
def extract_features(frame):
    """Extract global embedding + local descriptor map.
    Local descriptors are averaged across all models and polarities.
    Returns:
        global_emb: (EMBED_DIM,) numpy, L2-normalized
        local_map: (N_POS, LOCAL_DIM) numpy, L2-normalized per position
    """
    global_embs = []
    local_maps = []

    for model in model_list:
        for polarity in [1, -1]:
            x = torch.tensor(polarity * frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            g, d = model(x, return_local=True)
            global_embs.append(F.normalize(g, dim=1).cpu())
            # d: (1, LOCAL_DIM, H, W) → (N_POS, LOCAL_DIM)
            d_flat = d[0].view(LOCAL_DIM, -1).t().cpu()  # (N_POS, LOCAL_DIM)
            local_maps.append(d_flat)

    g_avg = torch.mean(torch.stack(global_embs), dim=0)
    g_avg = F.normalize(g_avg, dim=1).numpy().flatten()

    l_avg = torch.mean(torch.stack(local_maps), dim=0)
    l_avg = F.normalize(l_avg, dim=1).numpy()  # (N_POS, LOCAL_DIM)

    return g_avg, l_avg

# ============================================================
# LPM (Locality Preserving Matching) — from DLG reference code
# ============================================================
def lpm_filter(src_pts, dst_pts, K=6, tau=0.2, lamda=0.8):
    """Filter matches using LPM (Locality Preserving Matching).
    src_pts, dst_pts: (N, 2) matched point coordinates.
    Returns: filtered (src, dst) point arrays.
    """
    if len(src_pts) < K + 1:
        return src_pts, dst_pts

    X, Y = src_pts.astype(np.float64), dst_pts.astype(np.float64)
    treeX = KDTree(X)
    treeY = KDTree(Y)
    _, indX = treeX.query(X, k=K)
    _, indY = treeY.query(Y, k=K)

    indX = indX[:, 1:]
    indY = indY[:, 1:]

    sindX = np.sort(indX, axis=1)
    sindY = np.sort(indY, axis=1)
    temp = (sindX == sindY)
    c1 = (K - 1) - temp.sum(axis=1)

    vec = X - Y
    d2 = np.square(vec).sum(axis=1)
    vx, vy = vec[:, 0], vec[:, 1]

    vxr = np.repeat(vx[:, np.newaxis], K - 1, axis=1)
    vyr = np.repeat(vy[:, np.newaxis], K - 1, axis=1)
    d2r = np.repeat(d2[:, np.newaxis], K - 1, axis=1)

    d2i_x = d2[sindX]
    d2i_y = d2[sindY]
    vxi_x = vx[sindX]
    vyi_x = vy[sindX]

    denom = np.sqrt(d2i_x * d2r)
    denom[denom < 1e-12] = 1e-12
    cos_sita = (vxi_x * vxr + vyi_x * vyr) / denom
    ratio = np.minimum(d2i_x, d2r) / (np.maximum(d2i_x, d2r) + 1e-12)

    c2i = (cos_sita * ratio) < tau
    c2 = (c2i * temp).sum(axis=1)

    C = (c1 + c2) / (K - 1)
    mask = C <= lamda

    return X[mask], Y[mask]

# ============================================================
# RANSAC 几何验证
# ============================================================
def ransac_match_score(query_descs, template_descs,
                       sim_threshold=0.5,
                       ransac_residual=2.0,
                       ransac_trials=500,
                       use_lpm=True,
                       transform_type='affine'):
    """Compute RANSAC-verified match score between query and one template.

    Args:
        query_descs: (N_POS, LOCAL_DIM) normalized descriptors
        template_descs: (N_POS, LOCAL_DIM) normalized descriptors
        sim_threshold: min cosine similarity for putative match
        ransac_residual: RANSAC residual threshold in grid units
        ransac_trials: max RANSAC iterations
        use_lpm: apply LPM post-filtering
        transform_type: 'affine' or 'similarity'

    Returns:
        n_inliers: number of RANSAC inliers
        mean_inlier_sim: mean similarity of inliers
        n_putative: number of putative matches before RANSAC
    """
    # Step 1: Compute similarity matrix (N_POS × N_POS)
    sim_matrix = query_descs @ template_descs.T  # (182, 182)

    # Step 2: Mutual nearest neighbors
    # For each query position, find best template position
    q_best_idx = sim_matrix.argmax(axis=1)  # (182,) — best template pos for each query pos
    q_best_sim = sim_matrix[np.arange(N_POS), q_best_idx]  # (182,)

    # For each template position, find best query position
    t_best_idx = sim_matrix.argmax(axis=0)  # (182,)

    # Mutual match: query i matches template j, AND template j matches query i
    mutual_mask = np.zeros(N_POS, dtype=bool)
    for i in range(N_POS):
        j = q_best_idx[i]
        if t_best_idx[j] == i:
            mutual_mask[i] = True

    # Filter by similarity threshold
    valid_mask = mutual_mask & (q_best_sim > sim_threshold)
    valid_indices = np.where(valid_mask)[0]

    n_putative = len(valid_indices)
    if n_putative < 3:
        return 0, 0.0, n_putative

    # Step 3: Get source (query) and destination (template) point coordinates
    src_pts = GRID_POSITIONS[valid_indices]  # (n_putative, 2)
    dst_pts = GRID_POSITIONS[q_best_idx[valid_indices]]  # (n_putative, 2)
    match_sims = q_best_sim[valid_indices]

    # Step 4: RANSAC geometric verification
    TransformClass = AffineTransform if transform_type == 'affine' else SimilarityTransform
    min_s = 3 if transform_type == 'affine' else 2

    try:
        model_robust, inliers = ransac(
            (src_pts, dst_pts),
            TransformClass,
            min_samples=min_s,
            residual_threshold=ransac_residual,
            max_trials=ransac_trials,
        )
        if inliers is None:
            return 0, 0.0, n_putative
    except Exception:
        return 0, 0.0, n_putative

    n_inliers = int(inliers.sum())
    if n_inliers == 0:
        return 0, 0.0, n_putative

    inlier_sims = match_sims[inliers]

    # Step 5: LPM post-filtering (optional)
    if use_lpm and n_inliers >= 4:
        inlier_src = src_pts[inliers]
        inlier_dst = dst_pts[inliers]
        filtered_src, filtered_dst = lpm_filter(inlier_src, inlier_dst, K=4, tau=0.1, lamda=0.9)
        n_after_lpm = len(filtered_src)
        if n_after_lpm > 0:
            # Recompute similarities for LPM-filtered points
            filt_sims = []
            for s, d in zip(filtered_src, filtered_dst):
                si = int(s[0]) * W_FEAT + int(s[1])
                di = int(d[0]) * W_FEAT + int(d[1])
                if si < N_POS and di < N_POS:
                    filt_sims.append(sim_matrix[si, di])
            if filt_sims:
                return n_after_lpm, np.mean(filt_sims), n_putative
        return n_inliers, float(np.mean(inlier_sims)), n_putative

    return n_inliers, float(np.mean(inlier_sims)), n_putative

def dense_pertemplate_score(query_descs, template_descs, top_ratio=0.7):
    """V18b-style dense score but per-template (not concatenated)."""
    sim = query_descs @ template_descs.T  # (182, 182)
    max_sim = sim.max(axis=1)  # (182,) best match per query pos
    k = max(1, int(top_ratio * N_POS))
    top_k = np.sort(max_sim)[-k:]
    return float(np.mean(top_k))

# ============================================================
# 评估
# ============================================================
print(f"\n{'='*70}")
print(f"Evaluating V19")
print(f"{'='*70}"); sys.stdout.flush()

# Step 1: Extract all features
print("Step 1: Extracting features for all frames..."); sys.stdout.flush()
t_feat = time.time()

# Template features: per finger, per template frame → (global_emb, local_map)
template_globals = {}   # finger → list of (EMBED_DIM,) numpy
template_locals = {}    # finger → list of (N_POS, LOCAL_DIM) numpy

random.seed(42)
for finger in fingers:
    frames = finger_train[finger]
    indices = random.sample(range(len(frames)), min(N_TEMPLATES, len(frames)))
    g_list, l_list = [], []
    for i in indices:
        g, l = extract_features(frames[i])
        g_list.append(g)
        l_list.append(l)
    template_globals[finger] = g_list
    template_locals[finger] = l_list

# Also precompute concatenated template locals for V18b-style comparison
template_local_concat = {}
for finger in fingers:
    all_l = np.concatenate(template_locals[finger], axis=0)  # (N_T * N_POS, LOCAL_DIM)
    template_local_concat[finger] = all_l

print(f"  Template extraction done in {time.time()-t_feat:.0f}s"); sys.stdout.flush()

# Step 2: Extract probe features and compute scores
print("Step 2: Computing scores with RANSAC..."); sys.stdout.flush()
t_match = time.time()

all_probes = []
for fi_idx, probe_finger in enumerate(fingers):
    test_frames = finger_test[probe_finger]
    for frame in test_frames:
        g_emb, l_map = extract_features(frame)
        scores = {}

        for identity in fingers:
            # Global score (max over templates)
            global_score = max(np.dot(g_emb, tg) for tg in template_globals[identity])

            # Dense concat score (V18b-style, for comparison)
            sim_concat = l_map @ template_local_concat[identity].T  # (182, N_T*182)
            max_sim_concat = sim_concat.max(axis=1)
            k = max(1, int(0.7 * N_POS))
            top_k_vals = np.sort(max_sim_concat)[-k:]
            dense_concat = float(np.mean(top_k_vals))

            # Per-template matching
            best_dense_pt = -1.0  # best per-template dense score
            best_ransac_inlier = 0
            best_ransac_sim = 0.0
            best_ransac_lpm = 0
            best_ransac_sim_lpm = 0.0

            for t_idx in range(len(template_locals[identity])):
                t_descs = template_locals[identity][t_idx]  # (182, 64)

                # Per-template dense score (V18b-style)
                pt_dense = dense_pertemplate_score(l_map, t_descs)
                if pt_dense > best_dense_pt:
                    best_dense_pt = pt_dense

                # RANSAC score (no LPM)
                n_inl, sim_inl, n_put = ransac_match_score(
                    l_map, t_descs,
                    sim_threshold=0.5, ransac_residual=2.0,
                    ransac_trials=500, use_lpm=False,
                    transform_type='affine'
                )
                if n_inl > best_ransac_inlier:
                    best_ransac_inlier = n_inl
                    best_ransac_sim = sim_inl

                # RANSAC + LPM score
                n_lpm, sim_lpm, _ = ransac_match_score(
                    l_map, t_descs,
                    sim_threshold=0.5, ransac_residual=2.0,
                    ransac_trials=500, use_lpm=True,
                    transform_type='affine'
                )
                if n_lpm > best_ransac_lpm:
                    best_ransac_lpm = n_lpm
                    best_ransac_sim_lpm = sim_lpm

            scores[identity] = {
                'global': global_score,
                'dense_concat': dense_concat,
                'dense_pt': best_dense_pt,
                'ransac_inlier': best_ransac_inlier,
                'ransac_sim': best_ransac_sim,
                'ransac_lpm': best_ransac_lpm,
                'ransac_sim_lpm': best_ransac_sim_lpm,
            }

        all_probes.append({'probe_finger': probe_finger, 'scores': scores})

    if (fi_idx + 1) % 3 == 0 or fi_idx == 0:
        elapsed = time.time() - t_match
        eta = elapsed / (fi_idx + 1) * n_classes - elapsed
        print(f"  [{fi_idx+1}/{n_classes}] elapsed={elapsed:.0f}s, eta={eta:.0f}s")
        sys.stdout.flush()

print(f"  Matching done in {time.time()-t_match:.0f}s"); sys.stdout.flush()

# ============================================================
# Score normalization + evaluation
# ============================================================
def apply_tnorm(raw_score, claimed_id, probe_scores, key):
    imp = [probe_scores[k][key] for k in probe_scores if k != claimed_id]
    mu = np.mean(imp)
    std = np.std(imp) + 1e-8
    return (raw_score - mu) / std

print("Step 3: Evaluating variants..."); sys.stdout.flush()

# Collect all scores
score_keys = ['global', 'dense_concat', 'dense_pt',
              'ransac_inlier', 'ransac_sim', 'ransac_lpm', 'ransac_sim_lpm']

def build_variants(all_probes):
    """Build all evaluation variants."""
    # Raw and T-norm for each score type
    variants = {}
    for sk in score_keys:
        variants[f'{sk}_raw'] = {'gen': [], 'imp': []}
        variants[f'{sk}_tnorm'] = {'gen': [], 'imp': []}

    # Fused variants: global + ransac_inlier with T-norm
    for alpha in [0.3, 0.5, 0.7]:
        for ransac_key in ['ransac_inlier', 'ransac_lpm', 'ransac_sim']:
            variants[f'fused_g{1-alpha:.1f}_{ransac_key}_a{alpha}_raw'] = {'gen': [], 'imp': []}
            variants[f'fused_g{1-alpha:.1f}_{ransac_key}_a{alpha}_tnorm'] = {'gen': [], 'imp': []}

    # Fused: global + dense_concat (V18b-style) for comparison
    for alpha in [0.3, 0.5]:
        variants[f'v18b_fused_a{alpha}_tnorm'] = {'gen': [], 'imp': []}

    # 3-way fusion: global + dense_concat + ransac_inlier
    for a_d in [0.2, 0.3]:
        for a_r in [0.2, 0.3]:
            a_g = 1 - a_d - a_r
            if a_g < 0.1:
                continue
            variants[f'triple_g{a_g:.1f}_d{a_d}_r{a_r}_tnorm'] = {'gen': [], 'imp': []}

    return variants

variants = build_variants(all_probes)

pf_gen = {sk: {f: [] for f in fingers} for sk in score_keys}

for probe in all_probes:
    pf = probe['probe_finger']
    sc = probe['scores']

    for identity in fingers:
        is_gen = (identity == pf)

        # Raw and T-norm for base scores
        for sk in score_keys:
            raw_val = sc[identity][sk]
            tnorm_val = apply_tnorm(raw_val, identity, sc, sk)

            variants[f'{sk}_raw']['gen' if is_gen else 'imp'].append(raw_val)
            variants[f'{sk}_tnorm']['gen' if is_gen else 'imp'].append(tnorm_val)

            if is_gen:
                pf_gen[sk][pf].append(raw_val)

        # Fused: global + ransac variants
        g = sc[identity]['global']
        for alpha in [0.3, 0.5, 0.7]:
            for rk in ['ransac_inlier', 'ransac_lpm', 'ransac_sim']:
                r = sc[identity][rk]
                fused_raw = (1 - alpha) * g + alpha * r
                vk_raw = f'fused_g{1-alpha:.1f}_{rk}_a{alpha}_raw'
                variants[vk_raw]['gen' if is_gen else 'imp'].append(fused_raw)

                # T-norm on fused score
                imp_fused = [(1 - alpha) * sc[o]['global'] + alpha * sc[o][rk]
                             for o in sc if o != identity]
                fused_t = (fused_raw - np.mean(imp_fused)) / (np.std(imp_fused) + 1e-8)
                vk_tnorm = f'fused_g{1-alpha:.1f}_{rk}_a{alpha}_tnorm'
                variants[vk_tnorm]['gen' if is_gen else 'imp'].append(fused_t)

        # V18b-style fused for comparison
        d = sc[identity]['dense_concat']
        for alpha in [0.3, 0.5]:
            fused_raw_v18b = alpha * d + (1 - alpha) * g
            imp_fv = [alpha * sc[o]['dense_concat'] + (1 - alpha) * sc[o]['global']
                      for o in sc if o != identity]
            fused_t_v18b = (fused_raw_v18b - np.mean(imp_fv)) / (np.std(imp_fv) + 1e-8)
            variants[f'v18b_fused_a{alpha}_tnorm']['gen' if is_gen else 'imp'].append(fused_t_v18b)

        # Triple fusion: global + dense_concat + ransac_inlier
        ri = sc[identity]['ransac_inlier']
        for a_d in [0.2, 0.3]:
            for a_r in [0.2, 0.3]:
                a_g = 1 - a_d - a_r
                if a_g < 0.1:
                    continue
                trip_raw = a_g * g + a_d * d + a_r * ri
                imp_trip = [a_g * sc[o]['global'] + a_d * sc[o]['dense_concat'] + a_r * sc[o]['ransac_inlier']
                            for o in sc if o != identity]
                trip_t = (trip_raw - np.mean(imp_trip)) / (np.std(imp_trip) + 1e-8)
                variants[f'triple_g{a_g:.1f}_d{a_d}_r{a_r}_tnorm']['gen' if is_gen else 'imp'].append(trip_t)

# Convert to arrays
for k in variants:
    variants[k]['gen'] = np.array(variants[k]['gen'])
    variants[k]['imp'] = np.array(variants[k]['imp'])

# ============================================================
# Analysis
# ============================================================
def full_analysis(name, gen, imp, verbose=True):
    if len(gen) == 0 or len(imp) == 0:
        return 1.0
    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    y_scores = np.concatenate([gen, imp])

    if np.std(y_scores) < 1e-10:
        return 1.0

    fpr_arr, tpr, thresholds = roc_curve(y_true, y_scores)
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

# Evaluate all variants
eer_results = {}
verbose_keys = {
    'global_tnorm', 'dense_concat_tnorm', 'ransac_inlier_tnorm',
    'ransac_lpm_tnorm', 'ransac_sim_tnorm',
    'v18b_fused_a0.5_tnorm',
}
for key in sorted(variants.keys()):
    gen, imp = variants[key]['gen'], variants[key]['imp']
    if len(gen) == 0:
        continue
    eer = full_analysis(key, gen, imp, verbose=(key in verbose_keys))
    eer_results[key] = eer

# ============================================================
# Summary
# ============================================================
best_key = min(eer_results, key=eer_results.get)
best_eer = eer_results[best_key]

print(f"\n{'='*70}")
print(f"ALL VARIANT EER COMPARISON")
print(f"{'='*70}")

# Group by category
categories = {
    'Base scores (raw)': [k for k in sorted(eer_results) if k.endswith('_raw') and 'fused' not in k and 'triple' not in k],
    'Base scores (T-norm)': [k for k in sorted(eer_results) if k.endswith('_tnorm') and 'fused' not in k and 'triple' not in k and 'v18b' not in k],
    'V18b-style fusion': [k for k in sorted(eer_results) if 'v18b_' in k],
    'Global+RANSAC fusion': [k for k in sorted(eer_results) if k.startswith('fused_') and 'tnorm' in k],
    'Triple fusion': [k for k in sorted(eer_results) if k.startswith('triple_')],
}

for cat_name, keys in categories.items():
    if not keys:
        continue
    print(f"\n  --- {cat_name} ---")
    for key in keys:
        marker = " ***" if key == best_key else ""
        print(f"  {key:50s}: EER = {eer_results[key]*100:.4f}%{marker}")

# Show best of each category
print(f"\n{'='*70}")
print(f"CATEGORY BESTS")
print(f"{'='*70}")
for cat_name, keys in categories.items():
    if not keys:
        continue
    best_cat_key = min(keys, key=lambda k: eer_results.get(k, 1))
    print(f"  {cat_name:30s}: {best_cat_key} = {eer_results[best_cat_key]*100:.4f}%")

# Detailed analysis of best variant
print(f"\n{'='*70}")
print(f"BEST VARIANT DETAILED ANALYSIS")
print(f"{'='*70}")
full_analysis(best_key, variants[best_key]['gen'], variants[best_key]['imp'], verbose=True)

# Per-finger analysis for ransac_inlier
print(f"\n--- Per-finger Genuine (ransac_inlier, raw) ---")
pf_stats = []
for finger in fingers:
    scores = pf_gen['ransac_inlier'][finger]
    if scores:
        arr = np.array(scores)
        src = finger_source.get(finger, "?")
        pf_stats.append((finger, src, arr.mean(), arr.min(), len(arr)))
pf_stats.sort(key=lambda x: x[2])
for i, (fn, src, gm, gmin, n) in enumerate(pf_stats):
    status = "*** WORST ***" if i < 3 else ""
    print(f"  {i+1}. {fn} [{src}]: mean_inliers={gm:.1f}, min={gmin:.0f}, n={n} {status}")

# RANSAC statistics
print(f"\n--- RANSAC Statistics ---")
all_ransac_gen = pf_gen['ransac_inlier']
all_ransac_imp = []
for probe in all_probes:
    pf = probe['probe_finger']
    for identity in fingers:
        if identity != pf:
            all_ransac_imp.append(probe['scores'][identity]['ransac_inlier'])
gen_vals = [v for f in fingers for v in all_ransac_gen[f]]
imp_vals = all_ransac_imp
print(f"  Genuine RANSAC inliers:  mean={np.mean(gen_vals):.1f}, std={np.std(gen_vals):.1f}, "
      f"min={np.min(gen_vals):.0f}, max={np.max(gen_vals):.0f}")
print(f"  Impostor RANSAC inliers: mean={np.mean(imp_vals):.1f}, std={np.std(imp_vals):.1f}, "
      f"min={np.min(imp_vals):.0f}, max={np.max(imp_vals):.0f}")

# Final comparison
print(f"\n{'='*70}")
print(f"V19 FINAL SUMMARY")
print(f"{'='*70}")
print(f"\n  *** Best: {best_key} with EER = {best_eer*100:.4f}% ***")
print(f"\n  --- Version Comparison ---")
print(f"  V14 (no SWA):            EER = 2.6776%")
print(f"  V15 CNN:                 EER = 2.40%")
print(f"  V16 ConvNeXt:            EER = 2.5144%")
print(f"  V17 global+T-norm:       EER = 1.6416%")
print(f"  V18 dense (untrained):   EER = 1.6865%")
print(f"  V18b trained local:      EER = 1.3341%")
print(f"  V19 best:                EER = {best_eer*100:.4f}%")

if best_eer * 100 < 1.3341:
    print(f"\n  IMPROVEMENT vs V18b: {1.3341 - best_eer*100:+.4f}%")
else:
    print(f"\n  vs V18b: {1.3341 - best_eer*100:+.4f}%")

print(f"\nTotal time: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")
print(f"{'='*70}"); sys.stdout.flush()
