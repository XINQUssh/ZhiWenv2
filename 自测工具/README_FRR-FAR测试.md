# 贴屏指纹数据 FRR/FAR 自测工具

用当前交付的纹理法模型（`best.pth`），在新版处理的贴屏数据上测 FRR/FAR，方便你迭代降噪版本时自己评估。

## 一、环境
- Python 3，装 `torch`、`opencv-python`、`numpy`
- 有 NVIDIA 显卡会用 GPU，没有也能用 CPU（慢一些）
- `best.pth` 放在 `fp_test.py` 同目录

## 二、数据格式
每个手指一个子文件夹，帧文件 `.bmp` 或 `.png`（文件名任意）。支持直接传 `.zip` 或已解压的文件夹：
```
你的数据.zip
  └─ 你的数据/
       ├─ dy_L0/  pair_1.bmp  pair_2.bmp ...
       ├─ dy_L1/  ...
       └─ ...
```

## 三、运行
```
# Windows: 先设一次(避免 OMP 冲突)
set KMP_DUPLICATE_LIB_OK=TRUE

# 【默认协议】每5张取1张作模板(100张→20模板), 其余80张作探针
python fp_test.py 你的数据.zip

# 已做 CLAHE 增强的数据——加 --noclahe 避免重复增强(降噪/原图不要加)
python fp_test.py clahe数据.zip --noclahe

# 改每N张取1(例如每4张取1→25模板)
python fp_test.py 你的数据.zip --stride 4

# (可选)改为把连续N帧合并成一个模板
python fp_test.py 你的数据.zip --merge 5
```

## 四、输出示例
```
数据: ...  手指数=21  NOCLAHE=False  每模板5帧
genuine 174对, impostor 3654对 (可分辨最小FAR≈0.027%)
EER = 21.22%
FAR<=1%   -> FRR = 43.70%
FAR<=0.5% -> FRR = 46.00%
FAR<=0.1% -> FRR = 64.90%
FAR<=0.05%-> FRR = 73.60%
FAR=0     -> FRR = 86.20%
```

## 五、协议与说明
- **默认协议（`--stride 5`）**：每指每 5 张取 1 张作模板（100 张 → 20 模板），其余 ~80 张全部作探针；探针对每根手指取"最高匹配分"，与本指=genuine、与他指=impostor。
- **`--stride N`**：每 N 张取 1 作模板。
- **`--merge N`**（可选，另一种思路）：把连续 N 帧合并成一个模板（合并关键点、mask 取并集），模板更全但语义不同。
- **`--noclahe`**：流程内部本身带一次 CLAHE 增强；若你的数据已经做过 CLAHE，请加此项，避免叠两次（会略微变差）。降噪版/原图**不要**加。
- **能测多低的 FAR**：取决于手指数 × 帧数。impostor 对数越多、能分辨的 FAR 越低。要可靠测到 1/5万（0.002%），大约需要 ≥5 万对 impostor（即更多手指、更多探针）。当前 20 来根手指只能测到 ~0.03% 量级。
- 指标越低越好；EER 是 FAR=FRR 时的错误率，用于快速横比不同数据版本。

## 六、用途建议
迭代降噪时，把每一版数据用同样命令跑一遍，横比 EER / FAR=0.1% 的 FRR，选最好的那版再做后续。
