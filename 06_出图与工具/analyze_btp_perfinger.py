"""
不贴屏 逐手指分析: 找出哪些手指匹配质量差
使用SIFT匹配(无需训练, 快速), 同时统计数据质量
"""
import os, glob, random, time
import numpy as np
import cv2

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

# ============================================================
# 预处理 (与V14一致)
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
    mask_ratio = np.sum(mask > 0) / mask.size
    return result, mask_ratio, upsampled, mask

# ============================================================
# SIFT匹配 (与V10一致)
# ============================================================
sift = cv2.SIFT_create(nfeatures=500)

def sift_match(img1_u8, img2_u8, mask1=None, mask2=None):
    """返回RANSAC内点数"""
    kp1, des1 = sift.detectAndCompute(img1_u8, mask1)
    kp2, des2 = sift.detectAndCompute(img2_u8, mask2)
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
# 加载不贴屏数据 (Rgd1237)
# ============================================================
base = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'
print("="*70)
print("不贴屏 逐手指分析 (Rgd1237)")
print("="*70)

finger_data = {}   # finger_name -> list of (preprocessed, mask_ratio, upsampled_u8, mask)
for finger in sorted(os.listdir(base)):
    rpath = os.path.join(base, finger, 'Rgd1237')
    if not os.path.exists(rpath):
        print(f"  SKIP {finger} (no Rgd1237)")
        continue
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    if len(imgs_paths) < 50:
        print(f"  SKIP {finger} (only {len(imgs_paths)} imgs)")
        continue

    frames = []
    for p in imgs_paths:
        img = load_img(p)
        preprocessed, mask_ratio, upsampled, mask = preprocess_frame(img)
        frames.append({
            'preprocessed': preprocessed,
            'mask_ratio': mask_ratio,
            'upsampled_u8': upsampled,
            'mask': mask,
            'path': p
        })
    finger_data[finger] = frames
    print(f"  {finger}: {len(frames)} frames, "
          f"mask_ratio: mean={np.mean([f['mask_ratio'] for f in frames]):.3f}, "
          f"min={np.min([f['mask_ratio'] for f in frames]):.3f}")

fingers = sorted(finger_data.keys())
n_fingers = len(fingers)
print(f"\nTotal: {n_fingers} fingers loaded")

# ============================================================
# 逐手指SIFT匹配分析
# ============================================================
print(f"\n{'='*70}")
print("逐手指 SIFT 匹配分析 (前70帧做模板, 70帧后做测试)")
print("="*70)

N_TEMPLATES = 20  # 模板数
random.seed(42)

# 各手指统计
finger_results = {}

for fi, finger in enumerate(fingers):
    frames = finger_data[finger]
    train_frames = frames[:70]
    test_frames = frames[70:]

    if len(test_frames) < 5:
        print(f"  {finger}: test frames too few ({len(test_frames)})")
        continue

    # 随机选20个模板
    template_indices = random.sample(range(len(train_frames)), min(N_TEMPLATES, len(train_frames)))
    template_frames = [train_frames[i] for i in template_indices]

    # Genuine匹配: 每个测试帧 vs 模板, 取max
    genuine_scores = []
    genuine_details = []  # 记录每帧的匹配详情

    for tidx, test_frame in enumerate(test_frames):
        sims = []
        for tmpl in template_frames:
            score = sift_match(
                test_frame['upsampled_u8'], tmpl['upsampled_u8'],
                test_frame['mask'], tmpl['mask']
            )
            sims.append(score)
        max_score = max(sims)
        genuine_scores.append(max_score)
        genuine_details.append({
            'idx': tidx + 70,
            'score': max_score,
            'mask_ratio': test_frame['mask_ratio'],
            'path': test_frame['path']
        })

    genuine_scores = np.array(genuine_scores, dtype=float)

    # Impostor匹配: 随机选5帧 vs 其他手指的模板
    impostor_scores = []
    test_sample = random.sample(range(len(test_frames)), min(5, len(test_frames)))

    for tidx in test_sample:
        test_frame = test_frames[tidx]
        for other in fingers:
            if other == finger:
                continue
            other_train = finger_data[other][:70]
            other_tmpl_idx = random.sample(range(len(other_train)), min(N_TEMPLATES, len(other_train)))
            other_templates = [other_train[i] for i in other_tmpl_idx]
            sims = [sift_match(test_frame['upsampled_u8'], t['upsampled_u8'],
                              test_frame['mask'], t['mask']) for t in other_templates]
            impostor_scores.append(max(sims))

    impostor_scores = np.array(impostor_scores, dtype=float)

    # 统计
    gen_mean = genuine_scores.mean()
    gen_min = genuine_scores.min()
    gen_std = genuine_scores.std()
    imp_mean = impostor_scores.mean() if len(impostor_scores) > 0 else 0
    imp_max = impostor_scores.max() if len(impostor_scores) > 0 else 0

    # 失败帧 (genuine score <= impostor max, 即可能被误拒)
    n_fail = np.sum(genuine_scores <= imp_max) if len(impostor_scores) > 0 else 0
    fail_rate = n_fail / len(genuine_scores) * 100

    finger_results[finger] = {
        'gen_mean': gen_mean,
        'gen_min': gen_min,
        'gen_std': gen_std,
        'imp_mean': imp_mean,
        'imp_max': imp_max,
        'n_test': len(test_frames),
        'n_fail': n_fail,
        'fail_rate': fail_rate,
        'genuine_scores': genuine_scores,
        'impostor_scores': impostor_scores,
        'genuine_details': genuine_details,
        'avg_mask_ratio': np.mean([f['mask_ratio'] for f in frames]),
    }

    status = "*** BAD ***" if fail_rate > 30 else ("WARN" if fail_rate > 15 else "OK")
    print(f"\n  [{fi+1}/{n_fingers}] {finger}  [{status}]")
    print(f"    test_frames={len(test_frames)}, mask_ratio={finger_results[finger]['avg_mask_ratio']:.3f}")
    print(f"    genuine:  mean={gen_mean:.1f}, min={gen_min:.0f}, std={gen_std:.1f}")
    print(f"    impostor: mean={imp_mean:.1f}, max={imp_max:.0f}")
    print(f"    fail_frames: {n_fail}/{len(genuine_scores)} ({fail_rate:.1f}%)")

    # 显示最差的几帧
    sorted_details = sorted(genuine_details, key=lambda x: x['score'])
    worst = sorted_details[:3]
    for w in worst:
        fname = os.path.basename(w['path'])
        print(f"      worst: frame#{w['idx']}, score={w['score']:.0f}, mask={w['mask_ratio']:.3f}, file={fname}")

# ============================================================
# 汇总排名
# ============================================================
print(f"\n{'='*70}")
print("不贴屏 手指质量排名 (按失败率从高到低)")
print("="*70)

ranked = sorted(finger_results.items(), key=lambda x: -x[1]['fail_rate'])
for i, (finger, r) in enumerate(ranked):
    status = "*** BAD ***" if r['fail_rate'] > 30 else ("WARN" if r['fail_rate'] > 15 else "OK")
    print(f"  {i+1}. {finger}: fail={r['n_fail']}/{r['n_test']-70 if r['n_test']>70 else r['n_test']}"
          f" ({r['fail_rate']:.1f}%), gen_mean={r['gen_mean']:.1f}, "
          f"mask={r['avg_mask_ratio']:.3f}  [{status}]")

# 推荐排除的手指
bad_fingers = [f for f, r in ranked if r['fail_rate'] > 30]
warn_fingers = [f for f, r in ranked if 15 < r['fail_rate'] <= 30]

print(f"\n{'='*70}")
print("建议")
print("="*70)
if bad_fingers:
    print(f"  建议排除 (失败率>30%): {', '.join(bad_fingers)}")
if warn_fingers:
    print(f"  需关注 (失败率15-30%): {', '.join(warn_fingers)}")
if not bad_fingers and not warn_fingers:
    print(f"  所有手指质量均可接受 (失败率<15%)")

# 如果排除差手指, 估算不贴屏整体EER改善
if bad_fingers:
    print(f"\n排除 {len(bad_fingers)} 个差手指后的整体评估:")
    good_fingers = [f for f in fingers if f not in bad_fingers]
    all_gen, all_imp = [], []
    for finger in good_fingers:
        r = finger_results[finger]
        all_gen.extend(r['genuine_scores'].tolist())
        # 重新计算impostor (只在好手指之间)

    # 简化: 直接看好手指的genuine统计
    all_gen_arr = np.concatenate([finger_results[f]['genuine_scores'] for f in good_fingers])
    print(f"  good fingers genuine: mean={all_gen_arr.mean():.1f}, min={all_gen_arr.min():.0f}, std={all_gen_arr.std():.1f}")
    print(f"  原始全部genuine: mean={np.concatenate([finger_results[f]['genuine_scores'] for f in fingers]).mean():.1f}")

print(f"\nDone.")
