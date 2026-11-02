# 实验报告的数据与案例分析

这些操作只读取已保存结果，不运行训练或模型推理。数据划分固定为完整 ScanQA val，4,675 题。分析前在服务器更新代码；若训练或推理仍在运行，先等运行结束。

## 1 输入文件

每个方法提供一个 `ScanQA_val_answers.json`，相邻的同名 `.views.json` 文件会自动读取。例：

```text
某次实验/full/ScanQA_val_answers.json
某次实验/full/ScanQA_val_answers.json.views.json
```

预测格式沿用本项目：列表，每项包含 `question_id`、`scene_id`、`answer_top10`。3D-VisTA 如已转换为本项目评测格式，可直接作为额外 `--run vista=...` 输入；没有二维选图记录也可以比较 EM。不要把汇总分数伪装成逐题预测。

## 2 一次生成统计和案例拼图

在项目根目录和已有实验环境中运行。先把以下四个路径改为实际结果位置；`CDV_DATA` 沿用服务器配置。路径指向预测文件本身，不是实验目录。

```bash
export CDV_DATA=/home/pengjiale/datasets/qa
export BASE_PRED=/实际路径/cdViews/full/ScanQA_val_answers.json
export SOFT_PRED=/实际路径/Soft/full/ScanQA_val_answers.json
export QMP_PRED=/实际路径/R2/full/ScanQA_val_answers.json
export COMBO_PRED=/实际路径/R2_soft/full/ScanQA_val_answers.json

python analyze_scanqa.py \
  --annotations "$CDV_DATA/ScanQA/ScanQA_v1.0/ScanQA_v1.0_val.json" \
  --run "cdViews=$BASE_PRED" \
  --run "Soft=$SOFT_PRED" \
  --run "QMP=$QMP_PRED" \
  --run "QMP_Soft=$COMBO_PRED" \
  --pair cdViews:Soft --pair cdViews:QMP --pair QMP:QMP_Soft \
  --frames-root "$CDV_DATA/scannetv2/frames_square" \
  --cases-per-category 2 \
  --output-dir output/report_analysis_v1
```

可增加 `--run Coverage=...`、`--run Uniform9=...`、`--run Retrieval17=...`、`--run vista=...` 和相应 `--pair`。名称仅用英文字母、数字、下划线或连字符。

输出目录必须是新目录，以免覆盖此前分析。只要统计表时省略 `--frames-root`；仅导出表、不筛选案例可再加 `--cases-per-category 0`。拼图依赖 Pillow，服务器环境已有；字体默认 DejaVuSans，若找不到，用 `--font /实际路径/字体.ttf` 指定。默认方法名和 ScanQA 文本为英文；若加入中文说明，使用支持中文的字体。

## 3 输出怎么用于报告

| 文件 | 用途 |
| --- | --- |
| `summary.csv` | 每种方法的精确 EM@1、正确题数、空答案数，核对四舍五入前的提升 |
| `pairwise.csv` | 错→对 fixed、对→错 regressed、共同正确和共同错误，直接填逐题变化表 |
| `by_question_type.csv` | 按问题首词和 how many 分组的 EM 及样本量；只是词面分组，不等于空间推理能力分类 |
| `view_stats.csv` | 实际平均输入图数、未满预算题数、候选不足题数；用于解释名义 k 与实际成本 |
| `view_overlap.csv` | 全部及各得失类别的平均 Jaccard、集合相同题数和顺序相同题数 |
| `per_question.csv` | 每题参考答案、两种回答、得失类别、具体视图顺序，用于检索候选案例 |
| `cases.json`、`case_*.png` | 候选案例详情和真实视图拼图；绿框为共同视图，橙框为该方法独有视图，编号是实际输入顺序 |
| `provenance.json` | 输入路径及 SHA256，记录分析依据 |

EM 严格沿用 `evaluate_scanqa.py` 的原始答案字符串匹配，不去标点、不转小写、不额外去空格。BLEU、ROUGE、CIDEr 继续使用原 `metrics.val.json`；本脚本不以分组数据重新计算 CIDEr，避免改变参考语料的 IDF 口径。`same_view_set` 与 `same_view_order` 区分集合变化和顺序变化。无视图记录的 3D 方法不计算这些二维指标。

案例默认按问题 ID 排序，从四种得失类别各取前两题，只用于提供稳定可复查的候选，不代表已挑选出最有解释力的案例。阅读 CSV 和真实图像后，使用重复的 `--question-id 实际ID` 精确导出指定案例（会代替默认筛选，仍应用于指定的每个方法对）。保存到另一个输出目录。

报告建议保留 Soft 修正、QMP 修正、QMP+Soft 退化、共同失败各一例。每例写清问题、参考答案和预测，指出被增加或移除的图中是否可见目标或空间关系；有必要时在 Word 中复制原图并裁剪放大。不要仅凭低 Jaccard 就断言“互补性更强”。缺失的图像会显式标为 Missing image 并记入 JSON；修复路径后重新导出，不能把缺图拼图作为最终案例。

## 4 参数量与效率比较

```bash
python -m cdviews.selector_stats --output output/selector_parameters.json
```

只需要 CPU 和项目已有 PyTorch，按默认模型构造实际统计参数。若训练时改变了查询数，增加 `--num-queries 实际数值`；其余维度以实际训练配置核对。报告数值仅覆盖选择器，不包括冻结 LLaVA、BLIP、缓存和预处理成本。Soft/Coverage 的新增可训练参数为 0，但其 BLIP 特征提取仍有成本。所有模型推理时都不训练，不能把构造时 requires_grad=True 解释为推理时训练。

默认配置已在本地统计：原选择器 **5,916,674**，QMP **6,639,873**，增加 **723,199（12.2%）**。这并不表示整个系统参数增长 12.2%。输出中的 `components` 加上 `direct_parameters`（包括 QMP 的查询向量）等于总参数量。

论文第 9 页表 3 给出 Retrieval/cdViews：参数 644M/5.9M，FLOPs 593.6T/294.5T，推理时间 2.8s/1.2s。这是论文原值，其统计范围和硬件协议未在该表中完整交代。与本地参数统计分表列示；不能直接据此给 QMP 或 3D-VisTA 填耗时、计算加速比，也不能把 5.9M 当作整个 LLaVA 问答系统。

当前已有输出未保存统一的阶段计时与峰值显存。本脚本不从文件修改时间或训练 epoch 秒数推算问答耗时。已有完整日志若能确认起止时间，可以列为“该次运行墙钟时间”，注明是否包含加载、缓存生成与评测；不能混同 GPU 模型计算时间。

在不新增实验的约束下，报告先填写选择器参数量、实际输入数、已有准确性指标；本地同口径耗时和峰值显存留待补。3D-VisTA 可比较问答质量，但总参数量、推理时间需从其实际实现和同硬件测量获取；本仓库未包含该模型，不能拿点云编码器和二维选择器做同范围参数比较。

若以后补计时，应固定 GPU、精度、问题集合、生成配置与缓存状态，分别记录特征准备、选图和联合问答，区分首次运行与缓存命中。CUDA 计时前后同步，报告问题数、总时间、每题均值及中位数、峰值显存；离线缓存开销单列。当前不要求重跑模型。
