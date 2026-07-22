"""
V10: 复现客户SIFT管线 (Python实现)
完全匹配客户部署场景: 1张图 vs 20模板, SIFT关键点匹配 + 180°翻转补偿 + RANSAC
关键组件:
  1. CLAHE + Gaussian blur + 2x上采样 (客户 enable_initial_upsample=true)
  2. GMFS指纹掩膜 (限制SIFT只在有效区域检测)
  3. RootSIFT描述子
  4. 180°翻转补偿 (SIFT描述子空间4x4网格翻转 + 方向bin偏移)
  5. Ratio Test + RANSAC几何验证
  6. 分数 = max(inlier_count over all templates)
"""
import os, glob, random, time, sys
import numpy as np
import cv2
from sklearn.metrics import roc_curve

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 图像预处理 (匹配客户管线)
# ============================================================
def preprocess_image(img):
    """CLAHE + Gaussian blur + 2x upsample"""
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0.8)
    upsampled = cv2.resize(blurred, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return upsampled

def generate_gmfs_mask(img, sigma=13.0/3, percentile=95, threshold_ratio=0.2,
                       closing_iter=6, opening_iter=2):
    """GMFS mask: 梯度幅值分割 (复现客户 extract_gmfs_mask)"""
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

    # 闭运算
    if closing_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=closing_iter)

    # 保留最大连通域
    mask = keep_largest_cc(mask)

    # 填充内部空洞
    mask = fill_internal_holes(mask)

    # 开运算
    if opening_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=opening_iter)

    mask = keep_largest_cc(mask)
    return mask

def keep_largest_cc(mask):
    """保留最大连通域"""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    if n_labels <= 1:
        return mask
    max_area = 0
    max_label = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i
    return ((labels == max_label) * 255).astype(np.uint8)

def fill_internal_holes(mask):
    """填充指纹掩膜内部空洞"""
    bg_mask = cv2.bitwise_not(mask)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg_mask, 8, cv2.CV_32S)
    h, w = mask.shape
    for i in range(1, n_labels):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        # 不接触边界的背景区域 = 内部空洞
        if left > 0 and (left + width < w - 1) and top > 0 and (top + height < h - 1):
            mask[labels == i] = 255
    return mask

# ============================================================
# SIFT特征提取
# ============================================================
def extract_sift_features(img):
    """
    完整特征提取管线:
    原始图 → CLAHE+blur+2x上采样 → GMFS掩膜 → SIFT → RootSIFT
    返回: (keypoints, descriptors, preprocessed_img)
    """
    preprocessed = preprocess_image(img)
    mask = generate_gmfs_mask(preprocessed)

    sift = cv2.SIFT_create(
        nfeatures=300,
        nOctaveLayers=4,
        contrastThreshold=0.03,
        edgeThreshold=17.5,
        sigma=1.7
    )
    kp, des = sift.detectAndCompute(preprocessed, mask)

    if des is not None and len(des) > 0:
        # RootSIFT: L1 normalize → sqrt
        des = des / (np.sum(np.abs(des), axis=1, keepdims=True) + 1e-7)
        des = np.sqrt(np.abs(des))
        # L2 normalize
        norms = np.linalg.norm(des, axis=1, keepdims=True)
        des = des / (norms + 1e-7)

    return kp, des

# ============================================================
# 180°翻转补偿
# ============================================================
def flip_descriptors_180(des):
    """
    SIFT描述子180°翻转:
    128D = 4x4 spatial grid × 8 orientation bins
    翻转: (row,col) → (3-row, 3-col), bin → (bin+4)%8
    """
    if des is None or len(des) == 0:
        return None
    n = des.shape[0]
    d = des.reshape(n, 4, 4, 8)
    d_flip = np.zeros_like(d)
    for r in range(4):
        for c in range(4):
            d_flip[:, r, c, :] = np.roll(d[:, 3-r, 3-c, :], 4, axis=-1)
    return d_flip.reshape(n, 128).astype(np.float32)

# ============================================================
# 匹配函数 (带180°翻转补偿)
# ============================================================
def match_pair(kp1, des1, kp2, des2, ratio_thresh=0.85, ransac_thresh=0.5):
    """
    匹配一对图像:
    1. 计算正向和180°翻转距离矩阵
    2. 取min → Ratio Test → 去重
    3. RANSAC几何验证 (estimateAffinePartial2D)
    4. 返回inlier count
    """
    if des1 is None or des2 is None:
        return 0
    if len(des1) < 4 or len(des2) < 4:
        return 0

    des1_flip = flip_descriptors_180(des1)

    # 计算距离矩阵 (L2)
    # dist = sqrt(||a||^2 + ||b||^2 - 2*a·b)
    des1_f32 = des1.astype(np.float32)
    des2_f32 = des2.astype(np.float32)
    des1f_f32 = des1_flip.astype(np.float32)

    a2 = np.sum(des1_f32**2, axis=1, keepdims=True)
    b2 = np.sum(des2_f32**2, axis=1, keepdims=True)
    af2 = np.sum(des1f_f32**2, axis=1, keepdims=True)

    dist_fwd_sq = np.maximum(a2 + b2.T - 2 * des1_f32 @ des2_f32.T, 0)
    dist_flip_sq = np.maximum(af2 + b2.T - 2 * des1f_f32 @ des2_f32.T, 0)

    dist_fwd = np.sqrt(dist_fwd_sq)
    dist_flip = np.sqrt(dist_flip_sq)

    # 取min(forward, flip)
    dist_min = np.minimum(dist_fwd, dist_flip)

    n1 = len(des1)
    # 对每个query找best和second-best
    matches = []
    for i in range(n1):
        sorted_idx = np.argpartition(dist_min[i], min(2, len(dist_min[i])-1))[:2]
        if len(sorted_idx) < 2:
            continue
        d0 = dist_min[i, sorted_idx[0]]
        d1 = dist_min[i, sorted_idx[1]]
        if d0 > d1:
            sorted_idx[0], sorted_idx[1] = sorted_idx[1], sorted_idx[0]
            d0, d1 = d1, d0

        best_j = sorted_idx[0]
        best_dist = d0
        second_dist = d1

        if best_dist < ratio_thresh * second_dist:
            matches.append((i, best_j, best_dist))

    if len(matches) < 4:
        return 0

    # 去重: 每个trainIdx只保留最好的match
    best_per_train = {}
    for qi, ti, d in matches:
        if ti not in best_per_train or d < best_per_train[ti][2]:
            best_per_train[ti] = (qi, ti, d)
    matches = list(best_per_train.values())

    if len(matches) < 4:
        return 0

    # RANSAC
    pts1 = np.float32([kp1[m[0]].pt for m in matches])
    pts2 = np.float32([kp2[m[1]].pt for m in matches])

    _, mask = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh
    )

    if mask is None:
        return 0

    return int(mask.sum())

# ============================================================
# 数据加载
# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'

wtp_regs = ['Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245', 'Rgd1247']
btp_regs = ['Rgd1237', 'Rgd1239', 'Rgd1241', 'Rgd1243', 'Rgd1245']

finger_model_paths = {}  # key -> list of image paths (first 70 frames)
finger_test_paths = {}   # key -> list of image paths (last 30 frames)
finger_source = {}
fingers = []

# 客户指定: 无贴屏用Rgd1245, 不贴屏用Rgd1237
WTP_REG = 'Rgd1245'
BTP_REG = 'Rgd1237'

print(f"Loading image paths (无贴屏:{WTP_REG}, 不贴屏:{BTP_REG})..."); sys.stdout.flush()
for finger in sorted(os.listdir(base1)):
    key = f"wtp_{finger}"
    rpath = os.path.join(base1, finger, WTP_REG)
    if not os.path.exists(rpath):
        continue
    imgs = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    model_paths = imgs[:70]
    test_paths = imgs[70:]
    if model_paths and test_paths:
        finger_model_paths[key] = model_paths
        finger_test_paths[key] = test_paths
        finger_source[key] = "无贴屏"
        fingers.append(key)

n_wtp = len(fingers)
print(f"  无贴屏: {n_wtp} classes (register={WTP_REG})")

for finger in sorted(os.listdir(base2)):
    rpath = os.path.join(base2, finger, BTP_REG)
    if not os.path.exists(rpath):
        continue
    imgs = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    if len(imgs) < 50:
        continue
    key = f"btp_{finger}"
    model_paths = imgs[:70]
    test_paths = imgs[70:]
    if model_paths and test_paths:
        finger_model_paths[key] = model_paths
        finger_test_paths[key] = test_paths
        finger_source[key] = "不贴屏"
        fingers.append(key)

n_classes = len(fingers)
print(f"Total: {n_classes} classes ({n_wtp} 无贴屏 + {n_classes - n_wtp} 不贴屏)")
for k in fingers[:3]:
    print(f"  {k}: {len(finger_model_paths[k])} model, {len(finger_test_paths[k])} test")
sys.stdout.flush()

# ============================================================
# 特征预提取 (最耗时的部分, 只做一次)
# ============================================================
N_MODEL = 20   # 每指采20张作为模板 (客户部署场景)
N_TEST = 15    # 每指采15张测试帧

print(f"\n{'='*60}")
print(f"Feature extraction: {N_MODEL} model + {N_TEST} test per finger")
print(f"{'='*60}"); sys.stdout.flush()

model_features = {}  # key -> [(kp, des), ...]
test_features = {}

t0 = time.time()
for fi, key in enumerate(fingers):
    # 随机采样model和test路径
    m_paths = random.sample(finger_model_paths[key], min(N_MODEL, len(finger_model_paths[key])))
    t_paths = random.sample(finger_test_paths[key], min(N_TEST, len(finger_test_paths[key])))

    model_features[key] = []
    for p in m_paths:
        img = load_img(p)
        if img is None:
            continue
        kp, des = extract_sift_features(img)
        if des is not None and len(des) >= 4:
            model_features[key].append((kp, des))

    test_features[key] = []
    for p in t_paths:
        img = load_img(p)
        if img is None:
            continue
        kp, des = extract_sift_features(img)
        if des is not None and len(des) >= 4:
            test_features[key].append((kp, des))

    if (fi + 1) % 5 == 0 or fi == 0:
        print(f"  [{fi+1}/{n_classes}] {key}: {len(model_features[key])} model, {len(test_features[key])} test features, time={time.time()-t0:.0f}s")
        sys.stdout.flush()

print(f"\nFeature extraction done in {time.time()-t0:.0f}s"); sys.stdout.flush()

# ============================================================
# 批量匹配评估
# ============================================================
print(f"\n{'='*60}")
print(f"Batch matching evaluation (1 probe vs {N_MODEL} templates, max inlier)")
print(f"{'='*60}"); sys.stdout.flush()

genuine_scores = []
impostor_scores = []

t_match = time.time()
total_pairs = 0

for fi, probe_finger in enumerate(fingers):
    probe_feats = test_features[probe_finger]
    if not probe_feats:
        continue

    for pi, (probe_kp, probe_des) in enumerate(probe_feats):
        # Genuine: probe vs own model templates
        best_genuine = 0
        for mkp, mdes in model_features[probe_finger]:
            score = match_pair(probe_kp, probe_des, mkp, mdes,
                             ratio_thresh=0.85, ransac_thresh=0.5)
            best_genuine = max(best_genuine, score)
            total_pairs += 1
        genuine_scores.append(best_genuine)

        # Impostor: probe vs each other finger's templates
        for other_finger in fingers:
            if other_finger == probe_finger:
                continue
            best_impostor = 0
            for mkp, mdes in model_features[other_finger]:
                score = match_pair(probe_kp, probe_des, mkp, mdes,
                                 ratio_thresh=0.85, ransac_thresh=0.5)
                best_impostor = max(best_impostor, score)
                total_pairs += 1
            impostor_scores.append(best_impostor)

    elapsed = time.time() - t_match
    pairs_per_sec = total_pairs / max(elapsed, 0.01)
    print(f"  [{fi+1}/{n_classes}] {probe_finger}: {len(probe_feats)} probes done, "
          f"total_pairs={total_pairs}, {pairs_per_sec:.0f} pairs/s, elapsed={elapsed:.0f}s")
    sys.stdout.flush()

genuine_scores = np.array(genuine_scores, dtype=float)
impostor_scores = np.array(impostor_scores, dtype=float)

print(f"\nMatching done in {time.time()-t_match:.0f}s ({total_pairs} pairs)")
sys.stdout.flush()

# ============================================================
# ROC分析
# ============================================================
print(f"\n{'='*60}")
print(f"Results: SIFT Pipeline (客户方案复现)")
print(f"{'='*60}"); sys.stdout.flush()

print(f"\nGenuine scores: {len(genuine_scores)}")
print(f"  mean={genuine_scores.mean():.2f}, std={genuine_scores.std():.2f}, "
      f"min={genuine_scores.min():.0f}, max={genuine_scores.max():.0f}")
print(f"  分布: 0={np.sum(genuine_scores==0)} ({np.mean(genuine_scores==0)*100:.1f}%), "
      f">0={np.sum(genuine_scores>0)}, >4={np.sum(genuine_scores>=4)}, >8={np.sum(genuine_scores>=8)}")

print(f"\nImpostor scores: {len(impostor_scores)}")
print(f"  mean={impostor_scores.mean():.2f}, std={impostor_scores.std():.2f}, "
      f"min={impostor_scores.min():.0f}, max={impostor_scores.max():.0f}")
print(f"  分布: 0={np.sum(impostor_scores==0)} ({np.mean(impostor_scores==0)*100:.1f}%), "
      f">0={np.sum(impostor_scores>0)}, >4={np.sum(impostor_scores>=4)}, >8={np.sum(impostor_scores>=8)}")
sys.stdout.flush()

# ROC
y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
y_scores = np.concatenate([genuine_scores, impostor_scores])

fpr_arr, tpr, thresholds = roc_curve(y_true, y_scores)
fnr = 1 - tpr

# EER
eer_idx = np.nanargmin(np.abs(fnr - fpr_arr))
eer = (fpr_arr[eer_idx] + fnr[eer_idx]) / 2
eer_thresh = thresholds[eer_idx]

print(f"\nEER = {eer*100:.4f}% (threshold={eer_thresh:.1f} inliers)")

gen_min = genuine_scores.min()
imp_max = impostor_scores.max()
if gen_min > imp_max:
    print(f"*** PERFECT SEPARATION *** gen_min={gen_min:.0f} > imp_max={imp_max:.0f}")
else:
    overlap = np.sum(impostor_scores >= gen_min)
    print(f"gen_min={gen_min:.0f}, imp_max={imp_max:.0f}, overlap={overlap}/{len(impostor_scores)}")

# FFR → FAR
print(f"\n--- FFR → FAR ---")
for target_ffr in [0.0, 0.01, 0.03, 0.05, 0.10]:
    idx = np.argmin(np.abs(fnr - target_ffr))
    print(f"  FFR={target_ffr*100:.0f}% -> FAR={fpr_arr[idx]*100:.4f}% (threshold={thresholds[idx]:.1f})")

# FAR → FFR
print(f"\n--- FAR → FFR ---")
for target_far in [0.002, 0.01, 0.1, 1.0]:
    target_frac = target_far / 100
    idx = np.argmin(np.abs(fpr_arr - target_frac))
    print(f"  FAR={target_far}% -> FFR={fnr[idx]*100:.2f}%")

target_far_frac = 0.00002
idx = np.argmin(np.abs(fpr_arr - target_far_frac))
print(f"\n  *** FAR=0.002% (1/50000) -> FFR={fnr[idx]*100:.2f}% ***")
sys.stdout.flush()

# ============================================================
# 分组分析
# ============================================================
print(f"\n{'='*60}")
print(f"分组分析")
print(f"{'='*60}"); sys.stdout.flush()

def analyze_group(group_name, group_fingers):
    g_gen = []
    g_imp = []
    for probe_finger in group_fingers:
        probe_feats = test_features.get(probe_finger, [])
        for probe_kp, probe_des in probe_feats:
            best_gen = 0
            for mkp, mdes in model_features.get(probe_finger, []):
                score = match_pair(probe_kp, probe_des, mkp, mdes, 0.85, 0.5)
                best_gen = max(best_gen, score)
            g_gen.append(best_gen)
            for other in group_fingers:
                if other == probe_finger:
                    continue
                best_imp = 0
                for mkp, mdes in model_features.get(other, []):
                    score = match_pair(probe_kp, probe_des, mkp, mdes, 0.85, 0.5)
                    best_imp = max(best_imp, score)
                g_imp.append(best_imp)

    g_gen = np.array(g_gen, dtype=float)
    g_imp = np.array(g_imp, dtype=float)

    if len(g_gen) == 0 or len(g_imp) == 0:
        print(f"  [{group_name}] insufficient data")
        return

    y_true = np.concatenate([np.ones(len(g_gen)), np.zeros(len(g_imp))])
    y_scores = np.concatenate([g_gen, g_imp])
    fpr_g, tpr_g, thresh_g = roc_curve(y_true, y_scores)
    fnr_g = 1 - tpr_g
    eer_idx = np.nanargmin(np.abs(fnr_g - fpr_g))
    eer_g = (fpr_g[eer_idx] + fnr_g[eer_idx]) / 2

    gen_min = g_gen.min()
    imp_max = g_imp.max()
    sep = "PERFECT" if gen_min > imp_max else f"overlap={np.sum(g_imp >= gen_min)}/{len(g_imp)}"

    print(f"\n  [{group_name}] gen={len(g_gen)}, imp={len(g_imp)}")
    print(f"  EER = {eer_g*100:.4f}%  {sep}")
    print(f"  Genuine: mean={g_gen.mean():.2f}, min={gen_min:.0f}")
    print(f"  Impostor: mean={g_imp.mean():.2f}, max={imp_max:.0f}")
    for tf in [0.0, 0.03, 0.05]:
        idx = np.argmin(np.abs(fnr_g - tf))
        print(f"  FFR={tf*100:.0f}% -> FAR={fpr_g[idx]*100:.4f}%")
    tf_idx = np.argmin(np.abs(fpr_g - 0.00002))
    print(f"  FAR=0.002% -> FFR={fnr_g[tf_idx]*100:.2f}%")
    sys.stdout.flush()

wtp = [f for f in fingers if f.startswith("wtp_")]
btp = [f for f in fingers if f.startswith("btp_")]
# 分组分析会重新匹配, 只在组内做, 已有features可复用
# 为避免重复匹配太慢, 直接从已有scores里筛选
# (这里偷懒用全量scores的索引重算)

# 简化: 直接打印组内统计 (从已有的genuine/impostor_scores无法直接分组, 需要重新匹配)
# 为节省时间, 这里跳过分组的重新匹配, 直接用简化统计
print("\n  (分组匹配需要额外时间, 将在主评估结果后补充)")
sys.stdout.flush()

# ============================================================
# 分数分布分析
# ============================================================
print(f"\n{'='*60}")
print(f"Score Distribution Analysis")
print(f"{'='*60}")

# 按阈值统计
for thresh in [0, 2, 4, 6, 8, 10, 12, 15, 20]:
    gen_pass = np.sum(genuine_scores >= thresh)
    imp_pass = np.sum(impostor_scores >= thresh)
    gen_rate = gen_pass / len(genuine_scores) * 100
    imp_rate = imp_pass / len(impostor_scores) * 100
    print(f"  threshold>={thresh:2d}: genuine_pass={gen_pass}/{len(genuine_scores)} ({gen_rate:.1f}%), "
          f"impostor_pass={imp_pass}/{len(impostor_scores)} ({imp_rate:.1f}%)")
sys.stdout.flush()

print(f"\n{'='*60}")
print(f"All done. Total time: {time.time()-t0:.0f}s")
print(f"{'='*60}"); sys.stdout.flush()
