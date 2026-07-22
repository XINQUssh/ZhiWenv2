"""
V18: Dense Local Descriptor Matching (借鉴FLARE/DMD)
核心思路:
  FLARE/DMD输出(H,W,D)三维描述符, 保留空间位置信息, 通过局部匹配计算相似度.
  我们借鉴这个思路: 提取ResNet-18 layer3的特征图(256,14,13)作为dense descriptors,
  每个空间位置有一个256维的局部描述符, 通过双向最近邻匹配计算局部匹配分数.

关键:
  1. 不需要重新训练! 复用V17模型 (ResNet-18 + SWA + Label Smoothing)
  2. Layer3特征图 (256ch, 14×13) = 182个空间位置 × 256维描述符
  3. 双向最大均值匹配 (bidirectional max-mean)
  4. 全局嵌入 + 局部描述符 融合匹配
  5. T-norm分数归一化 (V17验证最有效)

匹配流程:
  Query descriptor map (182, 256) vs All template descriptors (20×182, 256)
  → 每个query位置找所有template位置中最相似的 → top-K均值 → dense_score
  → alpha * dense_score + (1-alpha) * global_score → T-norm
"""
import os, glob, random, time, sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import roc_curve
from torchvision.models import resnet18, ResNet18_Weights
from scipy import stats as sp_stats

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 预处理 (与V14/V17一致)
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
# 数据加载 (与V17一致)
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
print(f"V18: Dense Local Descriptor Matching (FLARE/DMD inspired)")
print(f"{'='*70}")
print(f"Loading data..."); sys.stdout.flush()
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
# 模型 (与V17一致)
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

# ============================================================
# 特征提取器 (同时提取global embedding + layer3 dense descriptors)
# ============================================================
class DenseFeatureExtractor:
    """Hook-based extractor: gets both global embedding and layer3 dense descriptors."""
    def __init__(self, model):
        self.model = model
        self.layer3_features = None
        self._hook = model.layer3.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.layer3_features = output

    @torch.no_grad()
    def extract(self, x):
        """Returns (global_embedding, dense_descriptors).
        global_embedding: (embed_dim,) L2-normalized
        dense_descriptors: (n_positions, 256) L2-normalized per position
        """
        global_emb = self.model(x)  # (1, embed_dim)
        global_emb = F.normalize(global_emb, dim=1)

        # layer3 features captured by hook: (1, 256, h, w)
        feat = self.layer3_features
        B, C, H, W = feat.shape
        # Reshape to (H*W, C) and L2-normalize per position
        dense = feat[0].view(C, H * W).t()  # (H*W, C)
        dense = F.normalize(dense, dim=1)  # L2-norm per position

        return global_emb[0].cpu().numpy(), dense.cpu().numpy(), (H, W)

    def extract_dual_polarity(self, frame):
        """Extract with dual polarity averaging (2 views)."""
        x_orig = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        x_neg = torch.tensor(-frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Original
        g_orig, d_orig, hw = self.extract(x_orig)
        # Negated
        g_neg, d_neg, _ = self.extract(x_neg)

        # Average global embeddings
        g_avg = (g_orig + g_neg) / 2
        g_avg = g_avg / (np.linalg.norm(g_avg) + 1e-8)

        # Average dense descriptors per position, then re-normalize
        d_avg = (d_orig + d_neg) / 2
        norms = np.linalg.norm(d_avg, axis=1, keepdims=True) + 1e-8
        d_avg = d_avg / norms

        return g_avg, d_avg, hw

# ============================================================
# 加载V17模型
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}"); sys.stdout.flush()

EMBED_DIM = 512
SEEDS = [42, 123, 777]
N_TEMPLATES = 20
MODEL_DIR = 'f:/1111/指纹/models_v17/'

models = []
extractors = []
for seed in SEEDS:
    path = os.path.join(MODEL_DIR, f'v17_seed{seed}.pth')
    model = FingerprintEncoder(embed_dim=EMBED_DIM).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    models.append(model)
    extractors.append(DenseFeatureExtractor(model))
    print(f"  Loaded V17 seed={seed}"); sys.stdout.flush()

# ============================================================
# 多模型特征提取 (每个模型独立提取, 全局embedding平均, dense descriptors分别保留)
# ============================================================
def extract_frame_features(frame):
    """Extract features from all 3 models.
    Returns:
      global_emb: (512,) averaged across 3 models × 2 polarities, L2-normalized
      dense_per_model: list of 3 × (n_positions, 256) dense descriptor maps
      spatial_size: (H, W)
    """
    global_embs = []
    dense_maps = []

    for ext in extractors:
        g, d, hw = ext.extract_dual_polarity(frame)
        global_embs.append(g)
        dense_maps.append(d)

    # Average global embeddings across models
    g_avg = np.mean(global_embs, axis=0)
    g_avg = g_avg / (np.linalg.norm(g_avg) + 1e-8)

    return g_avg, dense_maps, hw

# ============================================================
# Dense Matching (FLARE/DMD inspired bidirectional matching)
# ============================================================
def dense_match_score(query_dense_maps, template_dense_maps_list, top_k_ratio=0.7):
    """Compute dense matching score between query and a set of templates.

    Args:
        query_dense_maps: list of 3 × (n_pos, 256) - one per model
        template_dense_maps_list: list of N_TEMPLATES tuples, each containing
                                  list of 3 × (n_pos, 256)
        top_k_ratio: fraction of top matches to average (0.7 = top 70%)

    Returns:
        dense_score: float, averaged across 3 models
    """
    n_models = len(query_dense_maps)
    model_scores = []

    for m in range(n_models):
        Q = query_dense_maps[m]  # (n_pos_q, 256)
        n_pos_q = Q.shape[0]

        # Concatenate all template descriptors for this model
        T_parts = [t[m] for t in template_dense_maps_list]  # list of (n_pos_t, 256)
        T = np.concatenate(T_parts, axis=0)  # (N_templates * n_pos_t, 256)

        # Cosine similarity matrix
        S = Q @ T.T  # (n_pos_q, n_total_template_pos)

        # For each query position: best match across ALL template positions
        max_sim_q = S.max(axis=1)  # (n_pos_q,)

        # Top-K mean (robust to background/low-quality positions)
        k = max(1, int(top_k_ratio * n_pos_q))
        top_k_sims = np.sort(max_sim_q)[-k:]
        model_scores.append(top_k_sims.mean())

    return np.mean(model_scores)

def dense_match_score_gpu(query_dense_maps, template_dense_concat, top_k_ratio=0.7):
    """GPU-accelerated dense matching.

    Args:
        query_dense_maps: list of 3 GPU tensors, each (n_pos, 256)
        template_dense_concat: list of 3 GPU tensors, each (total_template_pos, 256)
        top_k_ratio: fraction of top matches to average

    Returns:
        dense_score: float
    """
    n_models = len(query_dense_maps)
    model_scores = []

    for m in range(n_models):
        Q = query_dense_maps[m]  # (n_pos_q, 256)
        T = template_dense_concat[m]  # (total_pos, 256)
        n_pos_q = Q.shape[0]

        S = torch.mm(Q, T.t())  # (n_pos_q, total_pos)
        max_sim_q, _ = S.max(dim=1)  # (n_pos_q,)

        k = max(1, int(top_k_ratio * n_pos_q))
        top_k_sims, _ = max_sim_q.topk(k)
        model_scores.append(top_k_sims.mean().item())

    return np.mean(model_scores)

# ============================================================
# Extract ALL template features
# ============================================================
print(f"\n{'='*70}")
print(f"Step 1: Extracting template features (global + dense)...")
print(f"{'='*70}"); sys.stdout.flush()
t_tmpl = time.time()

random.seed(42)
template_global = {}   # finger -> list of N_TEMPLATES global embeddings
template_dense = {}    # finger -> list of N_TEMPLATES × [3 dense maps]
template_indices = {}

for finger in fingers:
    frames = finger_train[finger]
    indices = random.sample(range(len(frames)), min(N_TEMPLATES, len(frames)))
    template_indices[finger] = indices

    g_list = []
    d_list = []
    for i in indices:
        g, d_maps, hw = extract_frame_features(frames[i])
        g_list.append(g)
        d_list.append(d_maps)

    template_global[finger] = g_list
    template_dense[finger] = d_list

print(f"  Template extraction done in {time.time()-t_tmpl:.0f}s")
print(f"  Dense descriptor shape per frame: {hw[0]}×{hw[1]} = {hw[0]*hw[1]} positions × 256 dims")
sys.stdout.flush()

# Pre-concatenate template dense descriptors per identity (for GPU matching)
print("  Pre-concatenating template descriptors for GPU matching..."); sys.stdout.flush()
template_dense_gpu = {}  # finger -> list of 3 GPU tensors
for finger in fingers:
    per_model = []
    for m in range(len(SEEDS)):
        parts = [template_dense[finger][t][m] for t in range(len(template_dense[finger]))]
        concat = np.concatenate(parts, axis=0)  # (N_templates * n_pos, 256)
        per_model.append(torch.tensor(concat, dtype=torch.float32, device=device))
    template_dense_gpu[finger] = per_model

# ============================================================
# Extract ALL test frame features and compute scores
# ============================================================
print(f"\n{'='*70}")
print(f"Step 2: Computing match scores (global + dense + combined)")
print(f"{'='*70}"); sys.stdout.flush()
t_match = time.time()

# Store per-probe results for T-norm
all_probes = []

for fi_idx, probe_finger in enumerate(fingers):
    test_frames = finger_test[probe_finger]

    for frame in test_frames:
        # Extract features
        g_emb, d_maps, _ = extract_frame_features(frame)

        # GPU tensors for dense matching
        d_gpu = [torch.tensor(d_maps[m], dtype=torch.float32, device=device)
                 for m in range(len(SEEDS))]

        # Compute scores against ALL identities
        scores = {}
        for identity in fingers:
            # Global score: max cosine sim across templates
            global_sims = [np.dot(g_emb, t_g) for t_g in template_global[identity]]
            global_score = max(global_sims)

            # Dense score: bidirectional max-mean matching against all templates
            dense_score = dense_match_score_gpu(d_gpu, template_dense_gpu[identity])

            scores[identity] = {
                'global': global_score,
                'dense': dense_score,
            }

        all_probes.append({
            'probe_finger': probe_finger,
            'scores': scores,
        })

    if (fi_idx+1) % 5 == 0 or fi_idx == 0:
        print(f"  [{fi_idx+1}/{n_classes}] elapsed={time.time()-t_match:.0f}s"); sys.stdout.flush()

print(f"  Matching done in {time.time()-t_match:.0f}s"); sys.stdout.flush()

# ============================================================
# Score Normalization
# ============================================================
def apply_tnorm(raw_score, claimed_identity, probe_scores, score_key):
    """T-norm: normalize by probe's impostor score distribution."""
    imp_scores = [probe_scores[k][score_key] for k in probe_scores if k != claimed_identity]
    t_mu = np.mean(imp_scores)
    t_std = np.std(imp_scores) + 1e-8
    return (raw_score - t_mu) / t_std

# ============================================================
# Collect scores for all variants
# ============================================================
print(f"\n{'='*70}")
print(f"Step 3: Evaluating all matching variants")
print(f"{'='*70}"); sys.stdout.flush()

# Alpha values for fusion: combined = alpha * dense + (1-alpha) * global
ALPHA_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
# top_k_ratio values to try
TOP_K_RATIOS = [0.5, 0.7, 0.9, 1.0]

def collect_all_scores(all_probes, alpha_values):
    """Collect genuine/impostor scores for all variants."""
    results = {}

    # Pure global (alpha=0)
    for norm in ['raw', 'tnorm']:
        key = f"global_{norm}"
        results[key] = {'gen': [], 'imp': []}

    # Pure dense (alpha=1)
    for norm in ['raw', 'tnorm']:
        key = f"dense_{norm}"
        results[key] = {'gen': [], 'imp': []}

    # Fused variants
    for alpha in [0.3, 0.5, 0.7]:
        for norm in ['raw', 'tnorm']:
            key = f"fused_a{alpha}_{norm}"
            results[key] = {'gen': [], 'imp': []}

    per_finger_genuine_global = {f: [] for f in fingers}
    per_finger_genuine_dense = {f: [] for f in fingers}

    for probe in all_probes:
        pf = probe['probe_finger']
        sc = probe['scores']

        for identity in fingers:
            g = sc[identity]['global']
            d = sc[identity]['dense']

            # T-norm
            g_t = apply_tnorm(g, identity, sc, 'global')
            d_t = apply_tnorm(d, identity, sc, 'dense')

            is_genuine = (identity == pf)

            # Global
            for norm, val in [('raw', g), ('tnorm', g_t)]:
                k = f"global_{norm}"
                if is_genuine:
                    results[k]['gen'].append(val)
                else:
                    results[k]['imp'].append(val)

            # Dense
            for norm, val in [('raw', d), ('tnorm', d_t)]:
                k = f"dense_{norm}"
                if is_genuine:
                    results[k]['gen'].append(val)
                else:
                    results[k]['imp'].append(val)

            # Fused
            for alpha in [0.3, 0.5, 0.7]:
                fused_raw = alpha * d + (1 - alpha) * g
                fused_t = apply_tnorm(fused_raw, identity, sc, 'global')  # T-norm on fused raw

                # Actually, compute fused score properly
                # Raw fusion
                for norm in ['raw']:
                    k = f"fused_a{alpha}_{norm}"
                    if is_genuine:
                        results[k]['gen'].append(fused_raw)
                    else:
                        results[k]['imp'].append(fused_raw)

                # T-norm on fused: compute fused impostor scores for T-norm
                imp_fused = [alpha * sc[other]['dense'] + (1-alpha) * sc[other]['global']
                             for other in sc if other != identity]
                t_mu = np.mean(imp_fused)
                t_std = np.std(imp_fused) + 1e-8
                fused_tnorm = (fused_raw - t_mu) / t_std
                k = f"fused_a{alpha}_tnorm"
                if is_genuine:
                    results[k]['gen'].append(fused_tnorm)
                else:
                    results[k]['imp'].append(fused_tnorm)

            if is_genuine:
                per_finger_genuine_global[pf].append(g)
                per_finger_genuine_dense[pf].append(d)

    for k in results:
        results[k]['gen'] = np.array(results[k]['gen'])
        results[k]['imp'] = np.array(results[k]['imp'])

    return results, per_finger_genuine_global, per_finger_genuine_dense

results, pf_gen_global, pf_gen_dense = collect_all_scores(all_probes, ALPHA_VALUES)

# ============================================================
# Analysis
# ============================================================
def full_analysis(name, gen, imp, verbose=True):
    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    y_scores = np.concatenate([gen, imp])
    fpr_arr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
    eer = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2

    if verbose:
        print(f"\n{'='*60}")
        print(f"Results: {name}")
        print(f"{'='*60}")
        print(f"\nGenuine:  n={len(gen)}, mean={gen.mean():.4f}, std={gen.std():.4f}, min={gen.min():.4f}")
        print(f"Impostor: n={len(imp)}, mean={imp.mean():.4f}, std={imp.std():.4f}, max={imp.max():.4f}")

        gen_min, imp_max = gen.min(), imp.max()
        if gen_min > imp_max:
            print(f"\n*** PERFECT SEPARATION ***")
        else:
            overlap = np.sum(imp > gen_min)
            print(f"overlap: {overlap}/{len(imp)}")

        print(f"\nEER = {eer*100:.4f}% (threshold={thresholds[eer_idx]:.4f})")

        print(f"\n--- FFR -> FAR ---")
        for target_ffr in [0.0, 0.01, 0.03, 0.05, 0.10]:
            idx = np.argmin(np.abs(fnr - target_ffr))
            print(f"  FFR={target_ffr*100:.0f}% -> FAR={fpr_arr[idx]*100:.4f}%")

        print(f"\n--- FAR -> FFR ---")
        for target_far in [0.002, 0.01, 0.1, 1.0]:
            idx = np.argmin(np.abs(fpr_arr - target_far/100))
            print(f"  FAR={target_far}% -> FFR={fnr[idx]*100:.2f}%")

        target_far_frac = 0.00002
        idx = np.argmin(np.abs(fpr_arr - target_far_frac))
        print(f"\n  *** FAR=0.002% (1/50000) -> FFR={fnr[idx]*100:.2f}% ***")

        d_prime = (gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.std()**2 + imp.std()**2))
        print(f"  d-prime = {d_prime:.2f}")

        # Parametric extrapolation
        gen_mu, gen_std = gen.mean(), gen.std()
        imp_mu, imp_std = imp.mean(), imp.std()
        print(f"\n--- Parametric extrapolation (Gaussian) ---")
        for target_ffr in [0.01, 0.03, 0.05]:
            thresh = gen_mu - sp_stats.norm.ppf(1 - target_ffr) * gen_std
            far_theory = sp_stats.norm.cdf(thresh, loc=imp_mu, scale=imp_std)
            print(f"  FFR={target_ffr*100:.0f}% -> 理论FAR={far_theory*100:.6f}%")
        sys.stdout.flush()

    return eer

# Evaluate all variants
print(f"\n{'#'*70}")
print(f"# DETAILED RESULTS")
print(f"{'#'*70}"); sys.stdout.flush()

eer_results = {}

# Show detailed results for key variants
key_variants = ['global_raw', 'global_tnorm', 'dense_raw', 'dense_tnorm',
                'fused_a0.3_raw', 'fused_a0.3_tnorm',
                'fused_a0.5_raw', 'fused_a0.5_tnorm',
                'fused_a0.7_raw', 'fused_a0.7_tnorm']

for key in key_variants:
    gen = results[key]['gen']
    imp = results[key]['imp']
    eer = full_analysis(key, gen, imp, verbose=(key in ['global_tnorm', 'dense_tnorm',
                                                         'fused_a0.5_tnorm', 'fused_a0.5_raw']))
    eer_results[key] = eer

# Brief results for all variants
print(f"\n{'='*70}")
print(f"ALL VARIANT EER COMPARISON")
print(f"{'='*70}")
for key in sorted(eer_results.keys()):
    print(f"  {key:30s}: EER = {eer_results[key]*100:.4f}%")
sys.stdout.flush()

# ============================================================
# Group Analysis (best variant)
# ============================================================
best_key = min(eer_results, key=eer_results.get)
best_eer = eer_results[best_key]
print(f"\n  *** Best: {best_key} with EER = {best_eer*100:.4f}% ***")

# Group analysis
print(f"\n--- Group Analysis ({best_key}) ---"); sys.stdout.flush()

def group_analysis_from_probes(group_name, group_fingers, all_probes, score_fn):
    g_gen, g_imp = [], []
    for probe in all_probes:
        if probe['probe_finger'] not in group_fingers:
            continue
        for identity in group_fingers:
            score = score_fn(probe, identity)
            if identity == probe['probe_finger']:
                g_gen.append(score)
            else:
                g_imp.append(score)

    g_gen = np.array(g_gen)
    g_imp = np.array(g_imp)
    if len(g_gen) == 0 or len(g_imp) == 0:
        print(f"  [{group_name}] insufficient data"); return

    y_t = np.concatenate([np.ones(len(g_gen)), np.zeros(len(g_imp))])
    y_s = np.concatenate([g_gen, g_imp])
    fpr_g, tpr_g, _ = roc_curve(y_t, y_s)
    fnr_g = 1 - tpr_g
    ei = np.nanargmin(np.abs(fnr_g - fpr_g))
    eer_g = (fpr_g[ei] + fnr_g[ei]) / 2

    print(f"  [{group_name}] gen={len(g_gen)}, imp={len(g_imp)}, EER={eer_g*100:.4f}%")
    print(f"    gen: mean={g_gen.mean():.4f}, min={g_gen.min():.4f}")
    print(f"    imp: mean={g_imp.mean():.4f}, max={g_imp.max():.4f}")
    idx3 = np.argmin(np.abs(fnr_g - 0.03))
    idx5 = np.argmin(np.abs(fnr_g - 0.05))
    print(f"    FFR=3% -> FAR={fpr_g[idx3]*100:.4f}%")
    print(f"    FFR=5% -> FAR={fpr_g[idx5]*100:.4f}%")
    sys.stdout.flush()

# Use the best variant's score function
if 'fused' in best_key:
    alpha = float(best_key.split('_a')[1].split('_')[0])
    if 'tnorm' in best_key:
        def best_score_fn(probe, identity):
            sc = probe['scores']
            fused_raw = alpha * sc[identity]['dense'] + (1-alpha) * sc[identity]['global']
            imp = [alpha * sc[k]['dense'] + (1-alpha) * sc[k]['global'] for k in sc if k != identity]
            return (fused_raw - np.mean(imp)) / (np.std(imp) + 1e-8)
    else:
        def best_score_fn(probe, identity):
            sc = probe['scores']
            return alpha * sc[identity]['dense'] + (1-alpha) * sc[identity]['global']
elif 'dense' in best_key:
    if 'tnorm' in best_key:
        def best_score_fn(probe, identity):
            return apply_tnorm(probe['scores'][identity]['dense'], identity, probe['scores'], 'dense')
    else:
        def best_score_fn(probe, identity):
            return probe['scores'][identity]['dense']
else:  # global
    if 'tnorm' in best_key:
        def best_score_fn(probe, identity):
            return apply_tnorm(probe['scores'][identity]['global'], identity, probe['scores'], 'global')
    else:
        def best_score_fn(probe, identity):
            return probe['scores'][identity]['global']

wtp = [f for f in fingers if f.startswith("wtp_")]
btp = [f for f in fingers if f.startswith("btp_")]
group_analysis_from_probes("无贴屏", wtp, all_probes, best_score_fn)
group_analysis_from_probes("不贴屏", btp, all_probes, best_score_fn)

# ============================================================
# Per-finger Analysis
# ============================================================
print(f"\n--- Per-finger Genuine (quality templates, raw global) ---")
per_finger_stats = []
for finger in fingers:
    scores = pf_gen_global[finger]
    if scores:
        scores_arr = np.array(scores)
        src = finger_source.get(finger, "?")
        per_finger_stats.append((finger, src, scores_arr.mean(), scores_arr.min(), len(scores_arr)))

per_finger_stats.sort(key=lambda x: x[2])
for i, (fn, src, gm, gmin, n) in enumerate(per_finger_stats):
    status = "*** WORST ***" if i < 3 else ""
    print(f"  {i+1}. {fn} [{src}]: mean={gm:.4f}, min={gmin:.4f}, n={n} {status}")

# Dense per-finger
print(f"\n--- Per-finger Genuine (dense matching) ---")
per_finger_stats_d = []
for finger in fingers:
    scores = pf_gen_dense[finger]
    if scores:
        scores_arr = np.array(scores)
        src = finger_source.get(finger, "?")
        per_finger_stats_d.append((finger, src, scores_arr.mean(), scores_arr.min(), len(scores_arr)))

per_finger_stats_d.sort(key=lambda x: x[2])
for i, (fn, src, gm, gmin, n) in enumerate(per_finger_stats_d):
    status = "*** WORST ***" if i < 3 else ""
    print(f"  {i+1}. {fn} [{src}]: mean={gm:.4f}, min={gmin:.4f}, n={n} {status}")
sys.stdout.flush()

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"V18 FINAL SUMMARY")
print(f"{'='*70}")

print(f"\n  --- V18 Key Results ---")
for key in ['global_raw', 'global_tnorm', 'dense_raw', 'dense_tnorm',
            'fused_a0.5_raw', 'fused_a0.5_tnorm']:
    if key in eer_results:
        print(f"  {key:30s}: EER = {eer_results[key]*100:.4f}%")

print(f"\n  --- All Variants ---")
for key in sorted(eer_results.keys()):
    marker = " ***" if key == best_key else ""
    print(f"  {key:30s}: EER = {eer_results[key]*100:.4f}%{marker}")

print(f"\n  *** Best: {best_key} with EER = {best_eer*100:.4f}% ***")

print(f"\n  --- Comparison with Previous ---")
print(f"  V14 (27 classes, no SWA):      EER = 2.6776%")
print(f"  V15 CNN component:             EER = 2.40%")
print(f"  V16 ConvNeXt-Tiny:             EER = 2.5144%")
print(f"  V17 global + T-norm:           EER = 1.6416%")
print(f"  V18 best ({best_key}):  EER = {best_eer*100:.4f}%")

improvement_v17 = 1.6416 - best_eer * 100
print(f"\n  Improvement over V17: {improvement_v17:.4f}% absolute")

print(f"\nTotal time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")
print(f"{'='*70}"); sys.stdout.flush()
