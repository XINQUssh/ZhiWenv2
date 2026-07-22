"""
SIFT匹配 排除xzc, 27类评估 (快速版)
- 不重复计算组分析, 从主循环存储的分数中提取
- 使用 sys.stdout.flush() 避免输出缓冲
"""
import os, glob, random, time, sys
import numpy as np
import cv2
from sklearn.metrics import roc_curve

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

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

# ============================================================
base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
SKIP_BTP = ['xzc']

print("="*60)
print("SIFT匹配 (排除xzc, 27类) - 快速版")
print("="*60); sys.stdout.flush()

finger_train = {}
finger_test = {}
finger_source = {}

t0 = time.time()

for finger in sorted(os.listdir(base1)):
    rpath = os.path.join(base1, finger, 'Rgd1245')
    if not os.path.exists(rpath):
        continue
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    key = f"wtp_{finger}"
    train_list, test_list = [], []
    for p in imgs_paths[:70]:
        img = load_img(p)
        if img is not None:
            up = upsample_2x(clahe_enhance(img))
            train_list.append((up, generate_gmfs_mask(up)))
    for p in imgs_paths[70:]:
        img = load_img(p)
        if img is not None:
            up = upsample_2x(clahe_enhance(img))
            test_list.append((up, generate_gmfs_mask(up)))
    if train_list and test_list:
        finger_train[key] = train_list
        finger_test[key] = test_list
        finger_source[key] = "无贴屏"

n_wtp = len([k for k in finger_train if k.startswith("wtp_")])
print(f"  无贴屏: {n_wtp} classes"); sys.stdout.flush()

for finger in sorted(os.listdir(base2)):
    if any(s in finger.lower() for s in SKIP_BTP):
        print(f"  SKIP {finger}"); sys.stdout.flush()
        continue
    rpath = os.path.join(base2, finger, 'Rgd1237')
    if not os.path.exists(rpath):
        continue
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    if len(imgs_paths) < 50:
        continue
    key = f"btp_{finger}"
    train_list, test_list = [], []
    for p in imgs_paths[:70]:
        img = load_img(p)
        if img is not None:
            up = upsample_2x(clahe_enhance(img))
            train_list.append((up, generate_gmfs_mask(up)))
    for p in imgs_paths[70:]:
        img = load_img(p)
        if img is not None:
            up = upsample_2x(clahe_enhance(img))
            test_list.append((up, generate_gmfs_mask(up)))
    if train_list and test_list:
        finger_train[key] = train_list
        finger_test[key] = test_list
        finger_source[key] = "不贴屏"

n_btp = len([k for k in finger_train if k.startswith("btp_")])
fingers = sorted(finger_train.keys())
n_classes = len(fingers)
print(f"  不贴屏: {n_btp} classes")
print(f"  Total: {n_classes} classes, loaded in {time.time()-t0:.0f}s")
sys.stdout.flush()

# ============================================================
sift = cv2.SIFT_create(nfeatures=500)
N_TEMPLATES = 20
random.seed(42)

def sift_match(img1, mask1, img2, mask2):
    kp1, des1 = sift.detectAndCompute(img1, mask1)
    kp2, des2 = sift.detectAndCompute(img2, mask2)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return len(good)
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if inlier_mask is None:
        return len(good)
    return int(inlier_mask.sum())

# ============================================================
# 预计算模板SIFT特征 (避免重复检测)
# ============================================================
print(f"\nPre-computing SIFT features..."); sys.stdout.flush()
t_pre = time.time()

def compute_sift_features(img, mask):
    kp, des = sift.detectAndCompute(img, mask)
    return kp, des

template_features = {}  # finger -> list of (kp, des, img, mask)
test_features = {}      # finger -> list of (kp, des, img, mask)

for finger in fingers:
    train = finger_train[finger]
    chosen = random.sample(range(len(train)), min(N_TEMPLATES, len(train)))
    tmpl_list = []
    for i in chosen:
        img, mask = train[i]
        kp, des = compute_sift_features(img, mask)
        tmpl_list.append((kp, des, img, mask))
    template_features[finger] = tmpl_list

    test = finger_test[finger]
    test_list = []
    for img, mask in test:
        kp, des = compute_sift_features(img, mask)
        test_list.append((kp, des, img, mask))
    test_features[finger] = test_list

print(f"  Done in {time.time()-t_pre:.0f}s"); sys.stdout.flush()

# ============================================================
# 快速匹配 (用预计算的特征)
# ============================================================
def sift_match_precomputed(kp1, des1, kp2, des2):
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return len(good)
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if inlier_mask is None:
        return len(good)
    return int(inlier_mask.sum())

# ============================================================
print(f"\n{'='*60}")
print("SIFT匹配评估")
print("="*60); sys.stdout.flush()

# 存储每个 (query_finger, test_idx, target_finger) -> max_score
# 这样可以从中提取组分析而不重复计算
genuine_scores = []
impostor_scores = []
per_finger_genuine = {f: [] for f in fingers}

# 详细记录: (query_finger, target_finger, score)
all_scores = []

t1 = time.time()
for fi, finger in enumerate(fingers):
    tests = test_features[finger]
    for tidx, (q_kp, q_des, q_img, q_mask) in enumerate(tests):
        # Genuine
        sims = [sift_match_precomputed(q_kp, q_des, t[0], t[1])
                for t in template_features[finger]]
        gs = max(sims)
        genuine_scores.append(gs)
        per_finger_genuine[finger].append(gs)
        all_scores.append((finger, finger, gs))

        # Impostor
        for other in fingers:
            if other == finger:
                continue
            sims_o = [sift_match_precomputed(q_kp, q_des, t[0], t[1])
                      for t in template_features[other]]
            is_score = max(sims_o)
            impostor_scores.append(is_score)
            all_scores.append((finger, other, is_score))

    elapsed = time.time() - t1
    print(f"  [{fi+1}/{n_classes}] {finger} done, elapsed={elapsed:.0f}s")
    sys.stdout.flush()

genuine_scores = np.array(genuine_scores, dtype=float)
impostor_scores = np.array(impostor_scores, dtype=float)

print(f"\nTotal matching done in {time.time()-t1:.0f}s"); sys.stdout.flush()

# ============================================================
# 结果
# ============================================================
def analyze(name, gen, imp):
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    print(f"\nGenuine:  n={len(gen)}, mean={gen.mean():.2f}, std={gen.std():.2f}, min={gen.min():.0f}")
    print(f"Impostor: n={len(imp)}, mean={imp.mean():.2f}, std={imp.std():.2f}, max={imp.max():.0f}")

    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    y_scores = np.concatenate([gen, imp])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"\nEER = {eer*100:.4f}%")

    print(f"\n--- FFR -> FAR ---")
    for target_ffr in [0.0, 0.01, 0.03, 0.05, 0.10]:
        idx = np.argmin(np.abs(fnr - target_ffr))
        print(f"  FFR={target_ffr*100:.0f}% -> FAR={fpr[idx]*100:.4f}%")

    print(f"\n--- FAR -> FFR ---")
    for target_far in [0.002, 0.01, 0.1, 1.0]:
        idx = np.argmin(np.abs(fpr - target_far/100))
        print(f"  FAR={target_far}% -> FFR={fnr[idx]*100:.2f}%")

    target_far_frac = 0.00002
    idx = np.argmin(np.abs(fpr - target_far_frac))
    print(f"\n  *** FAR=0.002% (1/50000) -> FFR={fnr[idx]*100:.2f}% ***")

    d_prime = (gen.mean() - imp.mean()) / np.sqrt(0.5*(gen.std()**2 + imp.std()**2))
    print(f"  d-prime = {d_prime:.2f}")
    sys.stdout.flush()
    return eer

eer_all = analyze("SIFT 27类 (排除xzc)", genuine_scores, impostor_scores)

# ============================================================
# 分组分析 (从已存储的分数中提取, 不重新计算)
# ============================================================
print(f"\n{'='*60}")
print("分组分析 (从已有分数提取)")
print("="*60); sys.stdout.flush()

for group_name, prefix in [("无贴屏", "wtp_"), ("不贴屏", "btp_")]:
    group_f = set(f for f in fingers if f.startswith(prefix))
    g_gen = []
    g_imp = []
    for qf, tf, score in all_scores:
        if qf not in group_f:
            continue
        if tf not in group_f:
            continue
        if qf == tf:
            g_gen.append(score)
        else:
            g_imp.append(score)
    if len(g_gen) < 5 or len(g_imp) < 5:
        print(f"  [{group_name}] insufficient data"); continue
    g_gen = np.array(g_gen, dtype=float)
    g_imp = np.array(g_imp, dtype=float)
    y_t = np.concatenate([np.ones(len(g_gen)), np.zeros(len(g_imp))])
    y_s = np.concatenate([g_gen, g_imp])
    fpr_g, tpr_g, _ = roc_curve(y_t, y_s)
    fnr_g = 1 - tpr_g
    ei = np.nanargmin(np.abs(fnr_g - fpr_g))
    eer_g = (fpr_g[ei] + fnr_g[ei]) / 2
    print(f"\n  [{group_name}] gen={len(g_gen)}, imp={len(g_imp)}, EER={eer_g*100:.4f}%")
    print(f"    gen: mean={g_gen.mean():.1f}, min={g_gen.min():.0f}")
    print(f"    imp: mean={g_imp.mean():.1f}, max={g_imp.max():.0f}")
    sys.stdout.flush()

# ============================================================
# 逐手指
# ============================================================
print(f"\n{'='*60}")
print("逐手指 genuine 排名 (从低到高)")
print("="*60)
stats_list = []
for finger in fingers:
    arr = np.array(per_finger_genuine[finger], dtype=float)
    stats_list.append((finger, finger_source[finger], arr.mean(), arr.min(), len(arr)))
stats_list.sort(key=lambda x: x[2])
for i, (fn, src, gm, gmin, n) in enumerate(stats_list):
    status = "*** WORST ***" if i < 3 else ""
    print(f"  {i+1}. {fn} [{src}]: mean={gm:.1f}, min={gmin:.0f}, n={n} {status}")
sys.stdout.flush()

# ============================================================
print(f"\n{'='*60}")
print("总结对比")
print("="*60)
print(f"  SIFT 27类 (排除xzc) EER = {eer_all*100:.4f}%")
print(f"  SIFT 30类 (原始)    EER = 1.5200%")
print(f"  V14-NoXZC (27类)    EER = 2.6776%")
print(f"\nDone. Total: {time.time()-t0:.0f}s")
sys.stdout.flush()
