# 真值跳变检查与初始化修复

## 结论与证据

`0422-AM1-2026/P02`、`P04` 的参考位置存在明确的起始不连续。
旧评估在删除无效真值点后，直接连接剩余点，把该跳变计入累计 10 m，
因此在预测轨迹几乎静止时就求出了不可靠的初始方向。

下表时间相对各记录的第一个共同有效真值样本，距离均为水平距离：

| 检查项 | P02 | P04 |
| --- | ---: | ---: |
| 第 2–5 s 的真值位移 | 36.872 m | 36.859 m |
| 该段平均速度 | 12.291 m/s | 12.286 m/s |
| 旧初始化结束 | 5.000 s | 5.000 s |
| 此时 RoNIN 净位移 | 0.030 m | 0.019 m |
| 此时 TLIO 净位移 | 0.056 m | 0.043 m |

原始 `trajectory-gt.csv` 已将跳变中间两行标为
`gt_valid=false, invalid_reason=distance_exceeded`。
对应原始插值端点距离约 35.79 / 35.77 m，跨度约 2.70 s；
跳变两侧的 `nearest` 匹配位置却仍标为有效。删去无效行后，
得到上述约 36.9 m / 3 s 的跨缺口连线。
跳变之后还存在逐渐收敛的尾段：下一秒约 3.8 m，随后逐步减小。

这说明参考位置流及旧评估的连续性处理有问题；现有位置 CSV
不足以确定底层设备发生了哪一种重定位或融合状态变化。
不能把这次跳变解释成真实行走，也不能据此认定全部参考位置均已校准。

已检查全部 **13 条记录中实际参与评估的参考样本**：
共标出 19 个连续性可疑转换，包含 15 个时间缺口和 6 个速度异常，
其中 2 个同时满足两项条件。`0421-AM` 两条记录后段各有一次约
3.34 m/s 的转换，仅列为待复核项，不直接认定是真值错误。
只有 `0422-AM1/P02、P04` 的初始化区间受到本次规则修改影响。

证据文件：

- [所有连续性事件及原始无效原因](../outputs/reference-jump-audit/discontinuities.csv)
- [13 条记录的检查汇总](../outputs/reference-jump-audit/per_sequence.csv)
- [P02 开头参考位置与速度图](../outputs/reference-jump-audit/0422-AM1-2026/P02/initial_reference.png)
- [P02 原始匹配元数据摘录](../outputs/reference-jump-audit/0422-AM1-2026/P02/initial_raw_reference.csv)

## 修复后的共享协议

两条 baseline 均使用 `continuous` 协议，网络推理、权重、IMU 预处理保持不变。

1. 在约 1 Hz 的有效真值位置流上计算相邻时间差及水平位移速度。
   时间差 **>1.5 s** 或速度 **>3 m/s** 时标记连续性断点。
2. 断点使当前初始化累计距离和位置锚点失效。最后一个断点后等待
   **10 s**，再从新的有效位置锚点重新累计；期间再次断开则重新等待。
   等待判定有 1 ms 容差，用于吸收参考时间戳的微小钟差。
3. 在第一个满足条件的连续段内，累计路径首次达到 **10 m** 即结束。
   无断点的记录沿用第一个共同有效位置，不增加等待期。
4. 用该区间的预测与真值净位移方向差计算一次旋转；平移也以
   **新锚点时间**的预测和真值位置计算，不能继续沿用跳变前位置。
5. 任一净位移小于 **0.5 m** 则明确报错，不依据模型表现寻找其他区间。
   初始化后固定旋转和平移，不拟合尺度，不进行后续真值纠偏。

区间选择仅依赖当时已有的真值与时间戳。RoNIN 和 TLIO 共用同一
区间；预测位置只用于在既定区间内求旋转。新变换在初始化结束后才可用，
图中的更早轨迹为回溯变换，不属于初始化后的评分结果。

这组阈值是针对本数据集步行参考流的显式筛查规则，10 s 是固定等待期，
不是参考系统已完全收敛的证明。规则是在检查该数据集后制定的，论文应
说明这一点；没有按位置 RMSE 搜索最佳区间或最佳阈值。

**原始真值不被改写，所有初始化后的位置评分样本保留。**
60 s 相对位移配对增加“不跨越任何连续性断点”的限制。
该修复阻止跳变污染初始化，不负责恢复跳变前的正确坐标、
识别全部缓慢漂移，或消除模型自身的累计误差。

## 重算结果

两条受影响记录均改为约 **16 s** 重新锚定、约 **40 s** 完成初始化。
初始化区间的 RoNIN 净位移分别为 **8.99 / 10.17 m**，
TLIO 为 **5.99 / 8.71 m**，不再用几厘米的预测位移估计方向。
其余 11 条记录的旋转、平移和对齐坐标经逐项比较完全不变。

由于新旧初始化结束时刻不同，下面使用**两者共同的初始化后时间戳**
重算旧指标，避免把减少评分样本误当成改进：

| 记录 | baseline | 旧对齐 RMSE | 新对齐 RMSE |
| --- | --- | ---: | ---: |
| 0422-AM1/P02 | RoNIN | 84.59 m | 46.60 m |
| 0422-AM1/P04 | RoNIN | 175.65 m | 119.51 m |
| 0422-AM1/P02 | TLIO | 155.54 m | 85.25 m |
| 0422-AM1/P04 | TLIO | 150.21 m | 136.57 m |

全部 13 条记录的等权平均位置 RMSE，在共同评分时间戳上：
RoNIN **105.30 → 98.06 m**，TLIO **228.65 → 222.19 m**。
旧协议原来报告的均值为 105.16 / 228.47 m，使用的是较早开始的评分区间。
完整对比见 [新旧对齐逐记录 CSV](../outputs/reference-jump-audit/alignment_comparison.csv)。

新协议的完整数据集指标：

| 指标 | RoNIN O2 | TLIO O2 + EKF |
| --- | ---: | ---: |
| 平均初始化后位置 RMSE | 98.06 m | 222.19 m |
| 中位数位置 RMSE | 69.32 m | 168.35 m |
| 平均 60 s 相对位移 RMSE | 16.71 m | 25.46 m |

位置误差仍然较大；本次修复不能解释或解决所有 baseline 漂移。
相对位移指标也受断点配对筛查影响，不应将其全部变化归因于初始方向改善。

## 输出与复现

原有 `alignment-10m/` 和 `report/` 保留。
新结果目录：

- [RoNIN 对齐结果](../outputs/ronin-current/alignment-10m-continuous/)
- [TLIO 对齐结果](../outputs/tlio-current/alignment-10m-continuous/)
- [英文数据集论文评估报告](../outputs/tlio-current/report-continuous/tlio_baseline_evaluation_en.md)
- [双 baseline 哈希、时间戳、坐标变换及指标验证](../outputs/tlio-current/report-continuous/comparison_verification.json)

激活 `eqnio` 后，从仓库根目录执行以下命令即可复用已有推理结果。
输出必须使用尚不存在的目录；下面使用新的后缀以保护已保存实验。

```bash
python -B scripts/audit_reference.py \
  --out-root outputs/reference-jump-audit-recheck

python -B scripts/evaluate_initial_alignment.py \
  --result-root outputs/ronin-current/inference \
  --out-root outputs/ronin-current/alignment-10m-recheck --label Madgwick \
  --protocol continuous --max-gt-gap-s 1.5 --max-gt-speed-mps 3 \
  --settling-s 10 --initialization-distance-m 10 --min-displacement-m 0.5

python -B scripts/evaluate_initial_alignment.py \
  --result-root outputs/tlio-current/inference \
  --out-root outputs/tlio-current/alignment-10m-recheck --label TLIO-EKF \
  --protocol continuous --max-gt-gap-s 1.5 --max-gt-speed-mps 3 \
  --settling-s 10 --initialization-distance-m 10 --min-displacement-m 0.5

python -B scripts/summarize_tlio_baseline.py \
  --alignment-subdir alignment-10m-recheck --report-subdir report-recheck
```

若需复现旧协议，设置 `--protocol legacy` 并另选新输出目录。
若需从 IMU 开始完整运行，执行 `bash scripts/run_baseline_example.sh both`；
新样例默认使用连续性筛查。

`evaluation.csv` 增加 `gt_dt_s,gt_step_m,gt_speed_mps`、
`gt_gap_flag,gt_speed_flag,gt_discontinuity,gt_segment,gt_anchor_ready`，
可查看每个断点及等待期。`summary.json` 保存新位置锚点时间；
`run.json` 保存协议、阈值及实现哈希。

验证：全部 **19 项**适配与共享流程测试通过，包括累计路径长度、
跳变/缺口重置、等待期重置、新锚点平移、未来真值独立性及相对指标断点限制。
26 组新旧评估保留完全相同的真值坐标和时间戳；双 baseline 比较核实
13 条记录的输入哈希、参考哈希、预测时间戳和初始化区间一致。
