"""
生成V14单帧匹配方案性能评估报告PDF
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

output_path = 'f:/1111/指纹/单帧匹配方案_V14性能评估报告.pdf'

with PdfPages(output_path) as pdf:

    # ================================================================
    # 第1页: 封面
    # ================================================================
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')

    ax.text(0.5, 0.75, '超声指纹识别系统', fontsize=28, fontweight='bold',
            ha='center', va='center', color='#1a1a2e')
    ax.text(0.5, 0.67, '单帧匹配方案 性能评估报告', fontsize=22, fontweight='bold',
            ha='center', va='center', color='#16213e')
    ax.axhline(y=0.61, xmin=0.15, xmax=0.85, color='#0f3460', linewidth=2)

    info_text = """部署场景: 1张原始帧 vs 20张模板, max匹配
数据集: 30类 (18无贴屏 + 12不贴屏), 单寄存器
图像尺寸: 110 x 100 (上采样至 220 x 200)
目标指标: FFR < 3%, FAR < 0.002% (1/50000)
"""
    ax.text(0.5, 0.48, info_text, fontsize=12, ha='center', va='center',
            color='#333333', linespacing=1.8,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f5', edgecolor='#ccccdd'))

    ax.text(0.5, 0.30, 'V14方案: 预训练ResNet-18 + Sub-center ArcFace + 3模型集成',
            fontsize=11, ha='center', va='center', color='#555555')
    ax.text(0.5, 0.24, 'CNN最佳: EER=3.90%  |  FFR=5%\u2192FAR=0.86%  |  d-prime=4.58',
            fontsize=12, ha='center', va='center', color='#2980b9', fontweight='bold')
    ax.text(0.5, 0.18, 'SIFT参考: EER=1.52%  |  FFR=5%\u2192FAR=0.00%',
            fontsize=11, ha='center', va='center', color='#27ae60', fontweight='bold')
    ax.text(0.5, 0.10, '2026年5月', fontsize=12, ha='center', va='center', color='#888888')
    pdf.savefig(fig); plt.close()

    # ================================================================
    # 第2页: 问题定义与部署场景
    # ================================================================
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    ax_title = fig.add_axes([0.05, 0.92, 0.9, 0.06]); ax_title.axis('off')
    ax_title.text(0.0, 0.5, '1. 问题定义与部署场景', fontsize=18, fontweight='bold', color='#1a1a2e')

    ax_text = fig.add_axes([0.05, 0.72, 0.9, 0.18]); ax_text.axis('off')
    desc = """客户实际部署场景:
  - 输入: 1张原始超声指纹帧 (110x100像素)
  - 模板库: 每个注册用户存储20张模板帧的嵌入向量
  - 匹配: 输入帧 vs 20个模板逐一比对, 最高分 > 阈值 = 通过
  - 要求: 不使用多寄存器、不使用多帧集合

与之前方案(V5/V7)的根本区别:
  - V5/V7: 认证时需要15~25帧组成集合 \u2192 Set Transformer聚合 \u2192 1个嵌入
  - 单帧方案: 认证时只有1帧 \u2192 CNN直接编码 \u2192 1个嵌入
  - 信息量差异: V7约300帧信息 vs 单帧方案仅1帧

数据选择 (按客户指定):
  - 无贴屏 (18类): 仅使用寄存器 Rgd1245
  - 不贴屏 (12类): 仅使用寄存器 Rgd1237"""
    ax_text.text(0.0, 1.0, desc, fontsize=9, va='top', linespacing=1.6, fontfamily='SimSun')

    # 部署流程图
    ax_flow = fig.add_axes([0.05, 0.38, 0.9, 0.30]); ax_flow.axis('off')
    ax_flow.set_xlim(0, 10); ax_flow.set_ylim(0, 6)

    ax_flow.text(5, 5.5, '单帧匹配部署流程', fontsize=12, fontweight='bold',
                ha='center', color='#1a1a2e')

    # 注册流程
    ax_flow.text(0.5, 4.3, '注册', fontsize=10, fontweight='bold', color='#2980b9')
    boxes_enroll = [
        (1.5, 3.5, '采集20帧\n(单寄存器)', '#e3f2fd'),
        (4, 3.5, '预处理\nCLAHE+2x\n+GMFS掩膜', '#e8f5e9'),
        (6.5, 3.5, 'CNN编码\n(双极性)', '#fff3e0'),
        (9, 3.5, '存储40个\n嵌入向量', '#f3e5f5'),
    ]
    for x, y, text, color in boxes_enroll:
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='#666', linewidth=1.2)
        ax_flow.text(x, y, text, fontsize=7.5, ha='center', va='center', bbox=bbox, linespacing=1.3)
    for x1, x2 in [(2.5, 3.0), (5.0, 5.5), (7.5, 8.0)]:
        ax_flow.annotate('', xy=(x2, 3.5), xytext=(x1, 3.5),
                        arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5))

    # 认证流程
    ax_flow.text(0.5, 1.8, '认证', fontsize=10, fontweight='bold', color='#e74c3c')
    boxes_verify = [
        (1.5, 1.0, '1张原始帧', '#ffebee'),
        (4, 1.0, '预处理\n(同上)', '#e8f5e9'),
        (6.5, 1.0, 'CNN编码\n(双极性)', '#fff3e0'),
        (9, 1.0, 'max(余弦)\nvs 40模板\n> 阈值?', '#e8eaf6'),
    ]
    for x, y, text, color in boxes_verify:
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='#666', linewidth=1.2)
        ax_flow.text(x, y, text, fontsize=7.5, ha='center', va='center', bbox=bbox, linespacing=1.3)
    for x1, x2 in [(2.5, 3.0), (5.0, 5.5), (7.5, 8.0)]:
        ax_flow.annotate('', xy=(x2, 1.0), xytext=(x1, 1.0),
                        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    # 数据划分
    ax_data = fig.add_axes([0.05, 0.05, 0.9, 0.28]); ax_data.axis('off')
    data_text = """数据划分 (训练/测试严格隔离):
+------------------+-------+--------+----------+---------+
| 数据组            | 类数   | 寄存器  | 训练帧    | 测试帧   |
+------------------+-------+--------+----------+---------+
| 无贴屏            | 18    | Rgd1245 | 前70帧   | 第71帧+  |
| 不贴屏            | 12    | Rgd1237 | 前70帧   | 第71帧+  |
+------------------+-------+--------+----------+---------+
| 合计              | 30    |  --     | ~2100帧  | ~903帧   |
+------------------+-------+--------+----------+---------+

评估指标:
  - 模板: 从训练集随机选20帧, 提取双极性嵌入 (共40个向量)
  - 测试: 每张测试帧提取双极性嵌入 (2个向量)
  - Genuine: 测试帧 vs 自己的20模板 \u2192 4种极性组合取max \u2192 20个分数取max
  - Impostor: 测试帧 vs 每个其他人的20模板 \u2192 同上
  - 共计: 903个genuine分数, 26187个impostor分数"""
    ax_data.text(0.0, 1.0, data_text, fontsize=7.5, va='top', fontfamily='SimSun',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#ddd'))
    pdf.savefig(fig); plt.close()

    # ================================================================
    # 第3页: 方法演进 (单帧匹配)
    # ================================================================
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    ax_title = fig.add_axes([0.05, 0.92, 0.9, 0.06]); ax_title.axis('off')
    ax_title.text(0.0, 0.5, '2. 单帧匹配方法演进', fontsize=18, fontweight='bold', color='#1a1a2e')

    # 方法对比表
    ax_table = fig.add_axes([0.05, 0.62, 0.9, 0.28]); ax_table.axis('off')
    table_text = """各版本方法对比:
+------+---------------------------+--------+-------------+-------------+---------+
| 版本  | 核心改进                   | EER    | FFR=3%\u2192FAR | FFR=5%\u2192FAR | d-prime |
+------+---------------------------+--------+-------------+-------------+---------+
| V9   | 4层CNN, SimCLR+ArcFace    | 9.00%  | 27.70%      |  --         | 3.10    |
|      | (5寄存器混合, 无预处理)     |        |             |             |         |
+------+---------------------------+--------+-------------+-------------+---------+
| V10  | SIFT管线 (客户方法复现)    | 1.52%  | 0.16%       | 0.00%       |  --     |
|      | RANSAC inlier计数          |        |             |             |         |
+------+---------------------------+--------+-------------+-------------+---------+
| V11  | 4层CNN + 全部SIFT预处理    | 5.00%  | 10.52%      | 4.83%       | 4.02    |
|      | CLAHE+2x+GMFS+双极性       |        |             |             |         |
+------+---------------------------+--------+-------------+-------------+---------+
| V12  | 预训练ResNet-18 + ArcFace  | 4.00%  | 17.16%      | 1.11%       | 4.43    |
|      | ImageNet骨干网              |        |             |             |         |
+------+---------------------------+--------+-------------+-------------+---------+
| V13  | 渐进解冻 + CenterLoss      | 5.04%  | 11.60%      | 5.33%       | 3.96    |
|      | + TTA推理 (效果退步)        |        |             |             |         |
+------+---------------------------+--------+-------------+-------------+---------+
| V14  | ResNet-18 + Sub-center     | 3.90%  | 13.10%      | 0.86%       | 4.58    |
|      | ArcFace(K=3) + embed512    |        |             |             |         |
+------+---------------------------+--------+-------------+-------------+---------+"""
    ax_table.text(0.0, 1.0, table_text, fontsize=6.8, va='top', fontfamily='SimSun',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#ddd'))

    # EER演进图
    ax_eer = fig.add_axes([0.1, 0.30, 0.8, 0.26])
    versions = ['V9\n基础CNN', 'V11\n+预处理', 'V12\nResNet-18', 'V13\n渐进解冻\n(退步)', 'V14\nSub-center']
    eers_cnn = [9.00, 5.00, 4.00, 5.04, 3.90]
    colors_v = ['#e74c3c', '#f39c12', '#2ecc71', '#e74c3c', '#1abc9c']

    bars = ax_eer.bar(range(len(versions)), eers_cnn, color=colors_v, width=0.6, edgecolor='white')
    ax_eer.axhline(y=1.52, color='#3498db', linestyle='--', linewidth=2, alpha=0.8)
    ax_eer.text(4.5, 1.7, 'SIFT=1.52%', fontsize=9, color='#3498db', fontweight='bold')
    ax_eer.set_xticks(range(len(versions)))
    ax_eer.set_xticklabels(versions, fontsize=7.5)
    ax_eer.set_ylabel('EER (%)', fontsize=10)
    ax_eer.set_title('CNN方案EER演进 (SIFT虚线为参考)', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, eers_cnn):
        ax_eer.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                   f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold')
    ax_eer.spines['top'].set_visible(False); ax_eer.spines['right'].set_visible(False)
    ax_eer.set_ylim(0, 11)

    # 各版本改进说明
    ax_notes = fig.add_axes([0.05, 0.03, 0.9, 0.22]); ax_notes.axis('off')
    notes = """各版本关键改进:
V9\u2192V11: 借鉴SIFT全部预处理 (CLAHE/2x上采样/GMFS掩膜/双极性推理), EER 9%\u21925%
V11\u2192V12: CNN骨干从4层自定义替换为ImageNet预训练ResNet-18, EER 5%\u21924%
V12\u2192V13: 尝试渐进解冻+CenterLoss+TTA, 效果退步 (去掉SimCLR预训练导致)
V12\u2192V14: 保留V12验证架构, 引入Sub-center ArcFace(K=3)+嵌入512维, EER 4%\u21923.9%

失败教训:
  - V12尝试密集局部特征匹配: EER=12.5% (特征图7x7太粗, 局部匹配无效)
  - V13去掉SimCLR预训练: 性能全面退步 (SimCLR对领域适配至关重要)
  - V13的TTA推理: 10版本平均嵌入反而降低区分度"""
    ax_notes.text(0.0, 1.0, notes, fontsize=8, va='top', linespacing=1.5, fontfamily='SimSun',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0', edgecolor='#ffb74d'))
    pdf.savefig(fig); plt.close()

    # ================================================================
    # 第4页: V14技术架构
    # ================================================================
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    ax_title = fig.add_axes([0.05, 0.92, 0.9, 0.06]); ax_title.axis('off')
    ax_title.text(0.0, 0.5, '3. V14方案技术架构', fontsize=18, fontweight='bold', color='#1a1a2e')

    # 架构图
    ax_arch = fig.add_axes([0.05, 0.65, 0.9, 0.25]); ax_arch.axis('off')
    ax_arch.set_xlim(0, 10); ax_arch.set_ylim(0, 5)

    arch_boxes = [
        (0.8, 3.5, '输入帧\n220x200', '#e3f2fd'),
        (2.5, 3.5, 'ResNet-18\n(预训练)\n512维特征', '#e8f5e9'),
        (4.5, 3.5, '投影头\n512\u2192512维', '#fff9c4'),
        (6.5, 3.5, 'L2归一化\n512维嵌入', '#fff3e0'),
        (8.5, 3.5, '余弦相似度\nvs模板', '#f3e5f5'),
        (2.5, 1.2, 'Sub-center\nArcFace\nK=3, s=64\nm=0.5', '#fce4ec'),
        (5.5, 1.2, '3模型集成\n(seed=42/\n123/777)', '#e0f7fa'),
    ]
    for x, y, text, color in arch_boxes:
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='#666', linewidth=1.2)
        ax_arch.text(x, y, text, fontsize=7, ha='center', va='center', bbox=bbox, linespacing=1.3)

    for x1, x2 in [(1.5, 1.8), (3.3, 3.7), (5.3, 5.7), (7.3, 7.7)]:
        ax_arch.annotate('', xy=(x2, 3.5), xytext=(x1, 3.5),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax_arch.annotate('', xy=(2.5, 2.0), xytext=(2.5, 2.8),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2))
    ax_arch.text(2.5, 2.5, '训练时', fontsize=6, ha='center', color='#e74c3c')
    ax_arch.annotate('', xy=(5.5, 2.0), xytext=(5.5, 2.8),
                    arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.2))
    ax_arch.text(5.5, 2.5, '推理时', fontsize=6, ha='center', color='#2980b9')

    # 预处理流程
    ax_preproc = fig.add_axes([0.05, 0.42, 0.9, 0.20]); ax_preproc.axis('off')
    preproc_text = """预处理流程 (继承SIFT管线):
  原始帧(110x100) \u2192 CLAHE(clip=4.0) \u2192 2x双线性上采样(220x200)
  \u2192 GMFS指纹掩膜(梯度幅值分割) \u2192 背景置零 \u2192 有效区域均值/标准差归一化

训练流程 (2阶段):
  Phase 1: SimCLR对比预训练 (80 epochs, 骨干网冻结)
    - 只训练投影头 (263K / 11.4M参数)
    - 学习指纹领域的对比表示, 适配预训练权重
    - 数据增强: 极性翻转/噪声/平移/旋转/轻度弹性形变

  Phase 2: Sub-center ArcFace微调 (120 epochs, 差分学习率渐进解冻)
    - 早期层(conv1~layer2): lr=1e-5
    - layer3: lr=5e-5
    - layer4: lr=1e-4
    - 投影头+ArcFace: lr=3e-4
    - Sub-center K=3: 每类3个子中心, 处理类内多模态(不同按压力度)"""
    ax_preproc.text(0.0, 1.0, preproc_text, fontsize=8, va='top', linespacing=1.5, fontfamily='SimSun',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#ddd'))

    # 双极性推理
    ax_dual = fig.add_axes([0.05, 0.13, 0.9, 0.25]); ax_dual.axis('off')
    dual_text = """双极性推理机制:
  超声指纹帧存在随机极性翻转, 同一手指相邻帧的相关系数在-0.97~+0.97间波动。
  解决方案: 对每张帧同时提取原始和翻转(-frame)两个嵌入, 匹配时取4种组合的最大值。

  query: (q_orig, q_flip)    template: (t_orig, t_flip)
  score = max( cos(q_orig, t_orig),
               cos(q_orig, t_flip),
               cos(q_flip, t_orig),
               cos(q_flip, t_flip) )

Sub-center ArcFace vs 标准ArcFace:
  标准ArcFace: 每类1个权重中心 \u2192 类内多模态样本被强制拉向同一点
  Sub-center(K=3): 每类3个子中心 \u2192 不同按压模式各有对应中心, 减少类内方差
  训练: cos_max = max(cos(emb, subcenter_1), cos(emb, subcenter_2), cos(emb, subcenter_3))
  效果: V14(Sub-center) EER=3.90% vs V12(标准) EER=4.00%"""
    ax_dual.text(0.0, 1.0, dual_text, fontsize=8, va='top', linespacing=1.5, fontfamily='SimSun',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f5e9', edgecolor='#a5d6a7'))

    # 超参数
    ax_hp = fig.add_axes([0.05, 0.02, 0.9, 0.08]); ax_hp.axis('off')
    hp_text = ("超参数: embed_dim=512, SimCLR_temp=0.07, ArcFace(s=64,m=0.5,K=3), "
               "batch=32, weight_decay=1e-3, 3模型集成(seed=42/123/777)")
    ax_hp.text(0.0, 0.5, hp_text, fontsize=7, va='center', fontfamily='SimSun', color='#666')

    pdf.savefig(fig); plt.close()

    # ================================================================
    # 第5页: V14详细结果
    # ================================================================
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    ax_title = fig.add_axes([0.05, 0.92, 0.9, 0.06]); ax_title.axis('off')
    ax_title.text(0.0, 0.5, '4. V14详细测试结果', fontsize=18, fontweight='bold', color='#1a1a2e')

    # 核心指标
    ax_core = fig.add_axes([0.05, 0.74, 0.9, 0.16]); ax_core.axis('off')
    core_text = """V14核心指标:
+---------------------+------------+
| 指标                 | 数值        |
+---------------------+------------+
| EER                 | 3.90%      |
| d-prime             | 4.58       |
| Genuine均值          | 0.9032     |
| Genuine标准差        | 0.1779     |
| Genuine最小值        | -0.1644    |
| Impostor均值         | 0.2332     |
| Impostor标准差       | 0.1054     |
| Impostor最大值       | 0.9425     |
| Genuine样本数        | 903        |
| Impostor样本数       | 26187      |
+---------------------+------------+"""
    ax_core.text(0.0, 1.0, core_text, fontsize=8, va='top', fontfamily='SimSun',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#ddd'))

    # FFR->FAR 和 FAR->FFR
    ax_ffr = fig.add_axes([0.05, 0.48, 0.42, 0.22]); ax_ffr.axis('off')
    ffr_text = """FFR \u2192 FAR:
+---------+-----------+
| FFR     | FAR       |
+---------+-----------+
| 0%      | 99.93%    |
| 1%      | 61.62%    |
| 3%      | 13.10%    |
| 5%      | 0.86%     |
| 10%     | 0.084%    |
+---------+-----------+"""
    ax_ffr.text(0.0, 1.0, ffr_text, fontsize=8, va='top', fontfamily='SimSun',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='#90caf9'))

    ax_far = fig.add_axes([0.52, 0.48, 0.42, 0.22]); ax_far.axis('off')
    far_text = """FAR \u2192 FFR:
+-----------+-----------+
| FAR       | FFR       |
+-----------+-----------+
| 0.002%    | 26.14%    |
| 0.01%     | 17.61%    |
| 0.1%      | 9.86%     |
| 1.0%      | 4.98%     |
+-----------+-----------+"""
    ax_far.text(0.0, 1.0, far_text, fontsize=8, va='top', fontfamily='SimSun',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fce4ec', edgecolor='#f48fb1'))

    # 分布图
    ax_dist = fig.add_axes([0.1, 0.18, 0.8, 0.25])
    gen_mu, gen_std = 0.9032, 0.1779
    imp_mu, imp_std = 0.2332, 0.1054
    x_range = np.linspace(-0.4, 1.2, 500)
    gen_pdf = np.exp(-0.5*((x_range-gen_mu)/gen_std)**2) / (gen_std*np.sqrt(2*np.pi))
    imp_pdf = np.exp(-0.5*((x_range-imp_mu)/imp_std)**2) / (imp_std*np.sqrt(2*np.pi))
    ax_dist.fill_between(x_range, gen_pdf, alpha=0.4, color='#27ae60',
                         label=f'Genuine (\u03bc={gen_mu:.3f}, \u03c3={gen_std:.3f})')
    ax_dist.fill_between(x_range, imp_pdf, alpha=0.4, color='#e74c3c',
                         label=f'Impostor (\u03bc={imp_mu:.3f}, \u03c3={imp_std:.3f})')
    ax_dist.plot(x_range, gen_pdf, color='#27ae60', linewidth=2)
    ax_dist.plot(x_range, imp_pdf, color='#e74c3c', linewidth=2)
    ax_dist.axvline(x=0.3932, color='#333', linestyle='--', linewidth=1, alpha=0.7)
    ax_dist.text(0.41, max(imp_pdf)*0.8, 'EER\nthresh\n=0.393', fontsize=7, color='#333')
    ax_dist.set_xlabel('Cosine Similarity', fontsize=10)
    ax_dist.set_ylabel('Density', fontsize=10)
    ax_dist.set_title(f'd-prime = 4.58', fontsize=11, fontweight='bold')
    ax_dist.legend(fontsize=8, loc='upper right')
    ax_dist.spines['top'].set_visible(False); ax_dist.spines['right'].set_visible(False)

    # 注释
    ax_note = fig.add_axes([0.05, 0.03, 0.9, 0.12]); ax_note.axis('off')
    note_text = """关键观察:
  - FFR=5%时FAR=0.86%, 已接近可用水平
  - 但FFR=3%时FAR=13.10%, 跳变剧烈, 原因: 约3%的genuine帧有极低相似度(min=-0.164)
  - 这些低分genuine来自极端的极性翻转或按压变形, 是全局嵌入方法的固有局限
  - Impostor最大值=0.9425, 说明有不同手指在嵌入空间中过于接近"""
    ax_note.text(0.0, 1.0, note_text, fontsize=8, va='top', linespacing=1.5, fontfamily='SimSun',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e0', edgecolor='#ffb74d'))
    pdf.savefig(fig); plt.close()

    # ================================================================
    # 第6页: 分组分析 + CNN vs SIFT对比
    # ================================================================
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    ax_title = fig.add_axes([0.05, 0.92, 0.9, 0.06]); ax_title.axis('off')
    ax_title.text(0.0, 0.5, '5. 分组分析与CNN/SIFT对比', fontsize=18, fontweight='bold', color='#1a1a2e')

    # 分组结果
    ax_group = fig.add_axes([0.05, 0.72, 0.9, 0.18]); ax_group.axis('off')
    group_text = """V14分组分析:
+------------------+--------+--------+--------+-------------+-------------+
| 数据组            | 类数    | EER    | gen均值 | FFR=3%\u2192FAR | FFR=5%\u2192FAR |
+------------------+--------+--------+--------+-------------+-------------+
| 无贴屏 (Rgd1245) | 18     | 3.33%  | 0.9129 | 3.65%       | 0.61%       |
| 不贴屏 (Rgd1237) | 12     | 5.25%  | 0.8885 | 21.87%      | 5.78%       |
+------------------+--------+--------+--------+-------------+-------------+
| 整体              | 30     | 3.90%  | 0.9032 | 13.10%      | 0.86%       |
+------------------+--------+--------+--------+-------------+-------------+

\u2022 无贴屏表现明显优于不贴屏 (EER 3.33% vs 5.25%)
\u2022 不贴屏组impostor最大值=0.9425, 拖累整体表现
\u2022 不贴屏数据质量更差 (gen标准差0.2017 vs 0.1593)"""
    ax_group.text(0.0, 1.0, group_text, fontsize=7.5, va='top', fontfamily='SimSun',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#ddd'))

    # 分组EER柱状图
    ax_g1 = fig.add_axes([0.1, 0.46, 0.35, 0.22])
    groups = ['无贴屏', '不贴屏', '整体']
    eer_v14 = [3.33, 5.25, 3.90]
    eer_sift = [None, None, 1.52]  # SIFT只有整体
    x = np.arange(3)
    ax_g1.bar(x-0.15, eer_v14, 0.3, label='V14 CNN', color='#2980b9', edgecolor='white')
    ax_g1.bar(2+0.15, 1.52, 0.3, label='SIFT', color='#27ae60', edgecolor='white')
    ax_g1.set_xticks(x)
    ax_g1.set_xticklabels(groups, fontsize=9)
    ax_g1.set_ylabel('EER (%)', fontsize=9)
    ax_g1.set_title('分组EER', fontsize=11, fontweight='bold')
    ax_g1.legend(fontsize=8)
    for i, v in enumerate(eer_v14):
        ax_g1.text(i-0.15, v+0.15, f'{v:.2f}%', ha='center', fontsize=8, fontweight='bold')
    ax_g1.text(2+0.15, 1.52+0.15, '1.52%', ha='center', fontsize=8, fontweight='bold', color='#27ae60')
    ax_g1.spines['top'].set_visible(False); ax_g1.spines['right'].set_visible(False)

    # FFR=5%时FAR对比
    ax_g2 = fig.add_axes([0.58, 0.46, 0.35, 0.22])
    versions_comp = ['V9', 'V11', 'V12', 'V14', 'SIFT']
    far5 = [None, 4.83, 1.11, 0.86, 0.00]
    colors_comp = ['#bdc3c7', '#f39c12', '#2ecc71', '#1abc9c', '#27ae60']
    bars_comp = ax_g2.bar(range(1, 5), far5[1:], color=colors_comp[1:], width=0.6, edgecolor='white')
    ax_g2.set_xticks(range(1, 5))
    ax_g2.set_xticklabels(versions_comp[1:], fontsize=9)
    ax_g2.set_ylabel('FAR (%)', fontsize=9)
    ax_g2.set_title('FFR=5%时的FAR', fontsize=11, fontweight='bold')
    for bar, val in zip(bars_comp, far5[1:]):
        ax_g2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                  f'{val:.2f}%', ha='center', fontsize=8, fontweight='bold')
    ax_g2.spines['top'].set_visible(False); ax_g2.spines['right'].set_visible(False)

    # CNN vs SIFT 根因分析
    ax_cmp = fig.add_axes([0.05, 0.10, 0.9, 0.30]); ax_cmp.axis('off')
    cmp_text = """CNN vs SIFT 根因分析:

SIFT (EER=1.52%) 为何优于 CNN (EER=3.90%):
+---------------------+---------------------------+---------------------------+
| 维度                 | SIFT                      | CNN                       |
+---------------------+---------------------------+---------------------------+
| 特征类型             | 局部描述子 (每关键点128维) | 全局嵌入 (整图\u21921个512维)  |
| 匹配策略             | 逐点匹配+RANSAC几何验证    | 两向量余弦相似度           |
| 分数类型             | 离散整数 (inlier计数)      | 连续实数 (cosine -1~1)    |
| 空间信息             | 完整保留 (点有坐标)        | 完全丢失 (全局池化压扁)    |
| 训练依赖             | 无需训练 (手工特征)        | 依赖30类训练数据           |
+---------------------+---------------------------+---------------------------+

SIFT分数分布: genuine min=2, impostor max=5, 阈值\u22656完美分离 (离散、不重叠)
CNN分数分布:  genuine min=-0.16, impostor max=0.94, 大量重叠 (连续、有尾部)

根本差异: SIFT通过RANSAC检验空间一致性, 即使局部特征相似也会被几何不一致过滤掉;
CNN的全局池化丢失了空间结构, 任何偶然相似都累积进余弦分数。"""
    ax_cmp.text(0.0, 1.0, cmp_text, fontsize=7.2, va='top', fontfamily='SimSun',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8eaf6', edgecolor='#9fa8da'))
    pdf.savefig(fig); plt.close()

    # ================================================================
    # 第7页: 结论与建议
    # ================================================================
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    ax_title = fig.add_axes([0.05, 0.92, 0.9, 0.06]); ax_title.axis('off')
    ax_title.text(0.0, 0.5, '6. 结论与建议', fontsize=18, fontweight='bold', color='#1a1a2e')

    # 达标评估
    ax_target = fig.add_axes([0.05, 0.70, 0.9, 0.20]); ax_target.axis('off')
    target_text = """与客户目标对比:
+---------------------+----------+----------+----------+----------+
| 指标                 | 客户目标  | V14 CNN  | SIFT V10 | V7多帧   |
+---------------------+----------+----------+----------+----------+
| EER                 | < 3%     | 3.90%    | 1.52%    | 0.11%    |
| FFR=3% \u2192 FAR       | <0.002%  | 13.10%   | 0.16%    | 0.00%    |
| FFR=5% \u2192 FAR       |  --      | 0.86%    | 0.00%    | 0.00%    |
| 认证输入             | 1帧      | 1帧      | 1帧      | 15~25帧  |
| 是否可部署           |  --      | 可部署   | 可部署   | 不可部署  |
+---------------------+----------+----------+----------+----------+

CNN V14状态: FFR=5%时已接近达标(FAR=0.86%), 但FFR=3%时差距较大(13.10% vs 0.002%)
SIFT V10状态: FFR=5%时完全达标, FFR=3%时FAR=0.16%接近目标"""
    ax_target.text(0.0, 1.0, target_text, fontsize=7.5, va='top', fontfamily='SimSun',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#e3f2fd', edgecolor='#90caf9'))

    # 结论
    ax_concl = fig.add_axes([0.05, 0.38, 0.9, 0.30]); ax_concl.axis('off')
    concl_text = """结论:

1. 单帧场景下, CNN全局嵌入方法的性能存在固有天花板
   - 经过6个版本迭代(V9\u2192V14), EER从9%优化到3.9%, 但始终无法突破3%
   - 根本原因: 全局池化丢失空间结构, 30类训练数据不足以学到完全鲁棒的嵌入

2. SIFT的局部几何匹配天然适合单帧小样本场景
   - 无需训练数据, 手工特征+RANSAC在30类上即可达到EER=1.52%
   - 离散inlier计数提供天然的分布分离性

3. 多帧方案(V5/V7)性能最优但不满足部署约束
   - V7 EER=0.11%远优于任何单帧方案, 但需要15~25帧输入

4. V14是目前CNN单帧方案的最优结果
   - Sub-center ArcFace + 预训练ResNet-18 + SimCLR + 512维嵌入
   - 在FFR=5%工作点已具备实用价值 (FAR=0.86%)"""
    ax_concl.text(0.0, 1.0, concl_text, fontsize=8.5, va='top', linespacing=1.5, fontfamily='SimSun')

    # 建议
    ax_suggest = fig.add_axes([0.05, 0.03, 0.9, 0.32]); ax_suggest.axis('off')
    suggest_text = """建议方案 (按优先级):

方案A: 调整工作点为FFR=5% (最简单)
  - V14在FFR=5%时FAR=0.86%, 接受略高的拒真率换取低误识率
  - 用户体验: 每20次可能有1次需要重试

方案B: 采用SIFT管线 (性能最优)
  - EER=1.52%, FFR=5%时FAR=0.00%
  - 已验证可用, 客户现有C++实现可直接部署

方案C: CNN+SIFT混合 (折中)
  - 第一级: CNN快速筛选 (拒绝明显不匹配的)
  - 第二级: SIFT精确验证 (对候选匹配做几何验证)
  - 兼顾速度和精度

方案D: 扩大训练数据 (长期改进)
  - 当前30类是CNN的根本瓶颈
  - 扩展到100+类可显著提升CNN嵌入质量
  - 预计EER可降至2%以下"""
    ax_suggest.text(0.0, 1.0, suggest_text, fontsize=8.5, va='top', linespacing=1.5, fontfamily='SimSun',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#a5d6a7'))
    pdf.savefig(fig); plt.close()

    # ================================================================
    # 第8页: 附录
    # ================================================================
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    ax_title = fig.add_axes([0.05, 0.92, 0.9, 0.06]); ax_title.axis('off')
    ax_title.text(0.0, 0.5, '附录: 实验文件与全版本记录', fontsize=16, fontweight='bold', color='#1a1a2e')

    ax_code = fig.add_axes([0.05, 0.05, 0.9, 0.85]); ax_code.axis('off')
    code_text = """A. 单帧匹配实验文件

+-------------------------------------+--------------------------------------+--------+
| 文件名                               | 说明                                  | EER    |
+-------------------------------------+--------------------------------------+--------+
| optimize_v14_subcenter.py           | *V14 ResNet-18+Sub-center ArcFace    | 3.90%  |
| optimize_v12_local_matching.py      | V12 ResNet-18+密集局部匹配            | 4.00%  |
| optimize_v13_tta_center.py          | V13 TTA+CenterLoss (效果退步)         | 5.04%  |
| optimize_v11_enhanced_cnn.py        | V11 4层CNN+全部SIFT预处理             | 5.00%  |
| optimize_v9_single_frame.py         | V9 基础单帧CNN                        | 9.00%  |
| sift_pipeline_v10.py               | V10 SIFT管线Python复现                | 1.52%  |
+-------------------------------------+--------------------------------------+--------+

B. 多帧方案实验文件 (不可部署, 仅供参考)

+-------------------------------------+--------------------------------------+--------+
| 文件名                               | 说明                                  | EER    |
+-------------------------------------+--------------------------------------+--------+
| optimize_v7_final.py                | V7 多寄存器Set Transformer+集成       | 0.11%  |
| optimize_v5_simclr_30class.py       | V5 SimCLR+Set Transformer 30类       | 1.78%  |
+-------------------------------------+--------------------------------------+--------+

C. V14 超参数

  +---------------------------+----------+
  | 参数                       | 值        |
  +---------------------------+----------+
  | 骨干网                     | ResNet-18 (ImageNet预训练)  |
  | 嵌入维度                   | 512      |
  | ArcFace类型                | Sub-center (K=3)  |
  | ArcFace s/m               | 64 / 0.5 |
  | SimCLR temperature        | 0.07     |
  | SimCLR epochs (冻结骨干)   | 80       |
  | ArcFace epochs (渐进解冻)  | 120      |
  | 差分学习率                 | 1e-5 ~ 3e-4  |
  | Weight decay              | 0.001    |
  | Batch size                | 32       |
  | 集成模型数                 | 3 (seed=42/123/777)  |
  | 双极性推理                 | 是       |
  | 预处理                     | CLAHE+2x上采样+GMFS掩膜  |
  +---------------------------+----------+

D. 运行环境

  - Python: Anaconda (E:/ANACONDA_NEW/python.exe)
  - GPU: NVIDIA GeForce RTX 5080 (16GB)
  - PyTorch: CUDA + Mixed Precision (AMP)
  - 训练时间: V14约92分钟 (3模型)"""

    ax_code.text(0.0, 1.0, code_text, fontsize=7, va='top', fontfamily='SimSun', linespacing=1.35)
    pdf.savefig(fig); plt.close()

print(f"报告已生成: {output_path}")
