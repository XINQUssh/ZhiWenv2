# JIPNet 预训练试水结论

日期: 2026-06-18
方法: [JIPNet (TIFS 2025)](https://github.com/XiongjunGuan/JIPNet) — CNN-Transformer 成对网络, 联合"身份验证+位姿对齐", 专为按压姿态不同的局部指纹设计。
做法: 克隆官方仓库, 下载官方预训练权重(best.pth 154MB, 光学指纹NIST/FVC训练), 把我们超声数据适配成 160×160 指纹格式(脊线深/白底/mask质心居中), **零微调**推理。

## 结果(单场景, 27类)

| 方法 | 单场景 EER |
|------|-----------|
| V21+V18b 融合+T-norm (我们现有最优) | **1.10%** |
| SIFT 几何匹配 | 3.5% |
| **预训练 JIPNet (零微调)** | **15.5%** |

- genuine 概率均值 0.92, impostor 0.61(max 0.97) — **impostor 也得高分**, 典型域失配特征。
- 尺度扫描确认 15.5% 稳健(target=140最优, 110→40%)。
- JIPNet 自带位姿估计(输出cos/sin/tx/ty), 设计上处理未对齐输入, 所以 15.5% 不是对齐没做好 → 是**光学→超声的域差**。

## 结论

1. **预训练 JIPNet 直接迁移到我们超声数据不具竞争力**(15.5% vs 现有1.10%)。这印证了仓库自己的警告: "公开权重只适用论文场景, 要好效果需在本地数据 retrain/fine-tune"。架构确有部分迁移能力(genuine>impostor有信号), 但域差太大。
2. **对客户的单场景目标, JIPNet 微调大概率低ROI**: 单场景已被现有方法解到 1.10%(接近饱和), 而 JIPNet 的价值在"局部/姿态困难"的样本; 在已经简单的单场景上, 即便微调也难超 1.10%。
3. **JIPNet 真正能发挥的是跨场景/强姿态差异**, 但那需要: (a)在我们数据上微调(需构建训练对生成管线, 官方make_data依赖VeriFinger刚性预对齐——许可不可用, 需自建近似对齐); (b)足够的配对数据。是数天工程 + 不确定payoff。

## 建议

- **单场景交付仍用 V21+V18b 融合 (1.10%, 稳定)**。
- JIPNet 微调: 仅当目标转向"跨场景/局部姿态困难"且愿投入数天工程+更多数据时才值得。单场景目标下不建议。
- 附带已下载的对照模型权重(AFRNet/DeepPrint/DesNet/PFVNet/RidgeNet)在 `JIPNet/dl_ckpts/`, 同样面临域差, 可按需试。

## 复现
- 环境: `E:\ANACONDA_NEW\python.exe` + timm/einops/gdown(已装), 需 `KMP_DUPLICATE_LIB_OK=TRUE`。
- 权重: `JIPNet/ckpts/JIPNet/best.pth`。评估脚本: `JIPNet/jipnet_eval.py`(sanity/full模式), `jip_scale.py`(尺度扫描)。
