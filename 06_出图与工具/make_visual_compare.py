# -*- coding: utf-8 -*-
"""生成 老数据 / 新raw / 新denoised 对比图"""
import cv2
import glob
import numpy as np

def load(p):
    with open(p, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

rows = []
specs = [
    ('dy_R0',  'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏/dy_R0/Rgd1245'),
    ('lwh_R1', 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏/lwh_R1/Rgd1245'),
    ('zyh_R0', 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏/zyh_R0/Rgd1237'),
]
for name, olddir in specs:
    old = load(sorted(glob.glob(olddir + '/*.bmp'))[0])
    raw = load(sorted(glob.glob(f'f:/1111/指纹/ysjz_raw/ysjz/{name}/*.bmp'))[0])
    den = load(sorted(glob.glob(f'f:/1111/指纹/ysjz_denoised/ysjz_denoised/{name}/*.bmp'))[0])
    def norm(im):
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.resize(im, (200, 220), interpolation=cv2.INTER_NEAREST)
    row = np.hstack([norm(old), np.full((220, 8), 255, np.uint8),
                     norm(raw), np.full((220, 8), 255, np.uint8), norm(den)])
    rows.append(row)
    rows.append(np.full((8, row.shape[1]), 255, np.uint8))
    print(name, 'old', old.shape, old.mean(), '| raw', raw.shape, raw.mean(), raw.std(),
          '| den', den.shape, den.mean(), den.std(),
          '| raw-den absdiff', np.abs(raw.astype(float)-den.astype(float)).mean())

canvas = np.vstack(rows[:-1])
cv2.imwrite('f:/wk_compare.png', canvas)
print('saved f:/wk_compare.png  (cols: old | new_raw | new_denoised; rows: dy_R0, lwh_R1, zyh_R0)')
