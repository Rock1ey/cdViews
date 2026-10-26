# ScanQA 验证集：随机均匀采样对照

## 已有 cdViews 结果怎么解释

用户于 2026-10-26 提供的完整验证集评测包含 4,675 个问题、0 个空答案。
下面的实测值来自用户提供的评测 JSON，未在本机重新执行 GPU 推理。
论文参考为本地 `docs/cdviews.pdf` 第 13 页 Table S1。

| 指标 | 论文 cdViews val | 当前实测 | 实测减论文 |
|---|---:|---:|---:|
| EM@1 | 30.1 | 29.0695 | -1.0305 个百分点 |
| BLEU-1 | 42.6 | 44.6597 | +2.0597 |
| ROUGE-L | 46.8 | 47.3673 | +0.5673 |
| CIDEr | 94.0 | 96.5381 | +2.5381 |

这不是所有指标一致下降，也不能单凭这个表认定实现错误或“完全复现”。
EM 要求预测与至少一个参考答案严格一致，其他指标衡量词语或片段重合，
因此可能出现 EM 降低而其他指标提高。是否由答案表达差异导致，需要逐题检查。

原 launcher 固定 temperature=0.2，没有固定随机种子；qa_inference.py 在
temperature>0 时开启采样。这是已确认的潜在变动因素，不是已证实的全部差异来源。
还需区分模型/排序文件版本、候选图片、提示、答案后处理、软件及硬件后端。
可将作者公开的 **val 预测** 用同一 evaluate_scanqa.py 评分，辅助区分评测口径和预测差异；
不重新获取测试答案，不把 val 结果标成 EvalAI 成绩。

## 本次实现与公平性

- `--view-selection cdviews`：沿用作者排名和原 viewNMS（阈值 0.5）。
- `--view-selection uniform`：从同一排名文件列出的全部候选中等概率、不放回抽样；不运行 NMS。
- 采样前按文件名排序，移除输入排名信号；采样后按随机抽取顺序输入模型，不恢复相关性顺序。
- 这是论文第 4 页公式 (3) 的随机均匀采样路线，不是时间轴等间隔采样。
  本实现明确采用不放回抽样；论文公式没有完整指定随机种子和实现细节，不能声称逐位复现。
- 两组实验候选集合相同。这里“全部候选”指给定排序文件中的候选，
  不额外加入磁盘上的其他图片，也不假定文件一定囊括了所有原始视频帧。
- 候选不足 k 张时使用全部候选，不重复补图；cdViews 还可能因 NMS 选不足 k 张。
  实际视图数保存在逐题记录中，比较时应统计。
- 固定模型、提示、预处理、答案后处理和解码参数。两个方法读取缓存排名时均不加载 selector 权重或视觉特征。
- 新入口默认 seed=42、temperature=0.2。视图抽样与答案采样使用独立的逐题随机流，
  同一道题在 smoke/full 或不同分片顺序下使用相同抽样与解码种子。
  这不保证不同 GPU、依赖版本或注意力后端的输出逐位一致。

## 更新和激活

本地改动推送到私有 origin 后，在服务器确认没有正在运行的实验、没有未保存代码修改，再更新：

```bash
git status --short
git pull --ff-only origin main
git rev-parse HEAD
```

按当前服务器选择一组，在 Bash 中运行。GPU 编号应根据 `nvidia-smi` 确认；
下面的 0 是示例，不表示实验室 GPU 0 一直空闲。

```bash
# 实验室
cd /home/pengjiale/projects/cdViews
nvidia-smi
source scripts/activate.sh lab 0
```

```bash
# 租卡服务器
cd /workspace/cdviews-work/repo
nvidia-smi
source scripts/activate.sh cloud 0
```

仍为单卡、val；不需要另建训练环境或下载新的模型。

## 第一组：uniform，9 张视图

先保留旧实验的 temperature=0.2。smoke 只运行 2 个问题用于检查，不能作为完整指标。

```bash
export CDV_EXPERIMENT="$CDV_RUNS/uniform_k9_seed42_t02_v1"

python -u run_scanqa.py \
  --splits val --view-selection uniform --input-views 9 \
  --seed 42 --temperature 0.2 \
  --run-dir "$CDV_EXPERIMENT" --stage smoke
```

上一步成功后运行完整验证集：

```bash
python -u run_scanqa.py \
  --splits val --view-selection uniform --input-views 9 \
  --seed 42 --temperature 0.2 \
  --run-dir "$CDV_EXPERIMENT" --stage full
```

另开终端看日志（需设置相同路径变量）：

```bash
tail -f "$CDV_EXPERIMENT/full/gpu${CDV_GPU}/inference.log"
```

入口不会覆盖已存在的 smoke/full 目录。失败后重试使用 v2 等新名字；
当前推理仍在一个划分结束时保存答案，不支持逐题断点恢复。

## 完整评测

使用已有能成功运行 evaluate_scanqa.py 的环境（Java、pycocoevalcap==1.2）。
切换 Python 环境后保留 CDV_REPO、CDV_DATA、CDV_EXPERIMENT 路径变量。

```bash
python "$CDV_REPO/evaluate_scanqa.py" \
  --annotations "$CDV_DATA/ScanQA/ScanQA_v1.0/ScanQA_v1.0_val.json" \
  --predictions "$CDV_EXPERIMENT/full/ScanQA_val_answers.json" \
  --output "$CDV_EXPERIMENT/full/metrics.val.json"
```

检查 questions=4675、空答案数量及四项主指标。输出包含：

- `full/ScanQA_val_answers.json`：合并后的完整预测。
- `full/ScanQA_val_answers.json.views.json`：每题实际输入的图片、候选数、方法和种子。
- `full/experiment.json`：划分、参数、GPU 编号、Git commit、标注与候选文件 SHA256。
- `full/gpu*/QA.yaml`、候选排名副本、`inference.log`：实际运行配置与日志。
- `full/qa_inference_selected_splits.py`、`view_sampling.py`、`code.diff`：入口代码与修改快照。

## 与 cdViews 做配对比较

旧 cdViews 的未固定种子结果仍保留，作为历史复现记录。
uniform 首次结果可以与旧基线做初步比较，但不能把全部差异都归因于选图。
正式方法比较建议补跑相同参数的 cdViews：

```bash
export CDV_EXPERIMENT="$CDV_RUNS/cdviews_k9_seed42_t02_v2"
python -u run_scanqa.py \
  --splits val --view-selection cdviews --input-views 9 \
  --seed 42 --temperature 0.2 \
  --run-dir "$CDV_EXPERIMENT" --stage smoke
python -u run_scanqa.py \
  --splits val --view-selection cdviews --input-views 9 \
  --seed 42 --temperature 0.2 \
  --run-dir "$CDV_EXPERIMENT" --stage full
```

然后用同一个评测命令评分。若希望减少答案随机采样，把 **两组** 都改成
`--temperature 0` 并用新的 t0 目录重跑；不要只改 uniform 的温度后与旧 cdViews
直接归因比较。若资源允许，使用 42/43/44 三个预先确定的种子报告均值与标准差，不挑最高值。

主对照先统一 k=9。论文第 8 页 Table 2 的随机基线 28.3 EM@1 使用的是最佳 k=17，
所以不能把 9 张随机采样结果当作那一行的精确复现。资源允许时再添加 uniform k=17，
单独报告输入数量与计算成本；不取代 k=9 的受控比较。

## 本地验证范围

`python -m unittest discover -s tests -v` 使用 CPU、临时数据和替代模型，
验证抽样不依赖排序、无重复、候选不足处理、逐题种子稳定、uniform 跳过 selector/NMS/位姿、
配置传递、结果合并及拒绝覆盖。它不代表真实模型的 GPU 推理已经通过；服务器 smoke 仍需运行。
