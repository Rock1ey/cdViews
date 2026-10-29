# R2：问题引导的多查询池化，双卡训练与 ScanQA val 评测

R2 修正视觉 mask，只屏蔽真正的 padding；用 4 个可学习查询加上问题表示，
对融合后的视觉 patch token 做 cross-attention，再映射回 512 维并计算余弦分数。
保留原 MCAN 图文融合骨干和 `(cosine+1)/2` BCE，每问题先平均其视图损失再平均问题。
原 `ViewSelector.py` 与作者 checkpoint 路径不变。没有加入 coverage、集合 Transformer 或新伪标签。

**训练从随机初始化开始，只训练选择器。** LLaVA 的视觉编码器、投影层和词嵌入全部冻结；
先缓存其输出，训练进程不加载 7B 问答模型。沿用已有训练集视图正负标签，不重新生成 caption。
本轮按用户要求直接实现 R2；尚不能把潜在收益区分为 mask 修正和多查询结构各自的贡献。

## 1. 更新代码与选择两张卡

以下是 Bash；示例为实验室。先确认没有运行中的旧实验且工作区干净。

```bash
cd /home/pengjiale/projects/cdViews
git status --short
git pull --ff-only origin main
source scripts/activate.sh lab 0
nvidia-smi

# 必须换成实际空闲、在当前环境可见的两张卡；不要假定一定是 0 和 1。
export CDV_GPU_A=0
export CDV_GPU_B=1
export CDV_R2_CACHE="$CDV_DATA/scannetv2/r2_frozen_features_v1"
export CDV_R2_TRAIN="$CDV_RUNS/r2_q4_seed42_v1"
export CDV_R2_RANK="$CDV_R2_TRAIN/val_ranking"
export CDV_EXPERIMENT="$CDV_RUNS/r2_q4_nms_k9_seed42_t02_v1"
```

云端替换前两项为 `cd /workspace/cdviews-work/repo`、`source scripts/activate.sh cloud 0`。
其他命令使用环境变量，无需改路径。GPU 映射只设置在对应命令上，不重复映射 `run_scanqa.py` 的卡号。
示例均为单机双卡，通信使用本机 29521 端口；若该端口被其他任务占用，统一换一个空闲端口。

## 2. 训练资产预检

```bash
python -m cdviews.prepare_r2_features \
  --data-root "$CDV_DATA" --cache-dir "$CDV_R2_CACHE" --check-only
```

需要：

- `$CDV_DATA/ScanQA/ScanQA_v1.0/ScanQA_v1.0_train.json` 与同目录 val 标注。
- `$CDV_DATA/ScanQA_view_labeling.json`，每题含 `Positive`、`Negative`（可另有 `Uncertain`）。
- `$CDV_DATA/ScanQA_val_view_ranking.json`：沿用原 val 候选集合。
- 对应 train/val 图像，以及已有 `model/llava-onevision-qwen2-7b-ov` 完整模型。
- 最终 NMS 问答还需要对应 val 位姿。

推理跑通不代表已经有训练标签或 train 图像。标签缺失时使用原作者公开的
[ScanQA view labeling](https://drive.google.com/file/d/1x832zSViIkFu-3SpKNVoManKzpjXVPT3/view)，
放到上述路径；标签在其他位置可加 `--labels /实际路径/文件.json`。
预检会在加载模型前检查图像、标签冲突、候选覆盖，并报告需缓存的数量及估算容量。
只有同时有正负标签的问题用于训练；缺少标签或缺一类的问题会列在 excluded_ids 中，不把 Uncertain 当负例。

默认特征约为 730×3584、fp16，约 5 MiB/视图，完整缓存可能需要很多磁盘空间，
实际形状由模型决定。BLIP CLS 缓存无法替代它。为避免不明来源特征混用，本入口会建立独立缓存，
不会自动信任或覆盖旧 `llava_ov_feats`；特征准备不属于训练。

## 3. 双卡预计算冻结特征（首次需要）

```bash
CUDA_VISIBLE_DEVICES="$CDV_GPU_A,$CDV_GPU_B" \
python -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29521 --nproc_per_node=2 \
  --module cdviews.prepare_r2_features \
  --data-root "$CDV_DATA" --cache-dir "$CDV_R2_CACHE" --batch-size 4
```

两张卡按场景分工，各加载一份冻结 LLaVA；每张卡必须能容纳完整模型，不能将两卡显存简单相加。
图像预处理和问题 prompt 与原 selector 一致，保存投影后 patch token（含 newline）和原词嵌入。
缓存按帧存储，避免每个训练样本加载整个场景的大文件；记录来源文件身份和配置哈希。
大权重文件的身份使用名称/大小/mtime，配置文件使用 SHA256，不把前者冒充权重内容哈希。

中断后重跑同一命令可复用原配置下已完成的文件。临时文件原子替换；只有所有进程成功后才发布
`train.json` / `val.json` 完整清单。改变模型、图像身份、标注或特征配方会拒绝混用缓存，应换目录。
若显存不足，减 `--batch-size`；若磁盘不足，先解决容量问题，不删除历史预测。

可分阶段准备：加 `--splits train` 先训练，排名前再用相同模型和缓存目录加 `--splits val`。
val 准备不读取答案内容参与任何学习；训练入口只读取 train 清单。

## 4. 双卡训练

先做独立的小规模训练检查，验证机器、缓存和通信：

```bash
CUDA_VISIBLE_DEVICES="$CDV_GPU_A,$CDV_GPU_B" \
python -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29521 --nproc_per_node=2 \
  --module cdviews.train_r2 \
  --cache-dir "$CDV_R2_CACHE" --run-dir "${CDV_R2_TRAIN}_smoke" \
  --epochs 1 --batch-size 2 --accum-steps 2 --workers 2 \
  --num-queries 4 --seed 42 --max-train-questions 64 --max-dev-questions 16
```

成功后从头运行正式训练，不从 smoke checkpoint 继续：

```bash
CUDA_VISIBLE_DEVICES="$CDV_GPU_A,$CDV_GPU_B" \
python -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29521 --nproc_per_node=2 \
  --module cdviews.train_r2 \
  --cache-dir "$CDV_R2_CACHE" --run-dir "$CDV_R2_TRAIN" \
  --epochs 10 --batch-size 2 --accum-steps 2 --workers 2 \
  --num-queries 4 --lr 5e-5 --seed 42 --dev-fraction 0.1
```

这是 **DDP 数据并行**：每进程每批 2 个问题，2 张卡，累积 2 次，有效 batch 为 8 个问题。
每问题抽取至多 5 正/5 负视图。每张卡持有一份选择器，不常驻 LLaVA。
训练用 DistributedSampler，累积中间步不做梯度同步；尾部不足一组的累积按实际问题数归一化。
样本数不整除卡数时 sampler 可能补齐少量训练样本，日志记录实际采样数。
开发集按问题无重复分片并汇总指标，不使用 DDP 前向的逐批同步，避免不等长开发分片互等。

仅从 train 按场景划分约 10% 内部开发集，保存清单；开发集评分使用每题全部已标注正负视图。
`best.pt` 按每问题 AP 的平均值选择，`last.pt` 含优化器及每个进程的 RNG，用于 epoch 边界续训。
`metrics.jsonl` 保存 train BCE、dev AP/Recall@9/BCE、耗时；这些是伪标签指标，不是 ScanQA QA 分数。

中断后，在原正式训练命令后加 `--resume`；从最后完成的 epoch 恢复。保持卡数、batch、种子、数据等不变，
可增大 `--epochs` 延长预算；不支持任意 batch 中途精确续跑。新实验使用新目录，不覆盖已有训练。

两卡有并行计算条件，但实际加速受缓存读取、磁盘和通信限制，不承诺 2 倍。
日志中的 epoch 秒数可用于比较。单卡保持相同有效 batch 的写法是：

```bash
CUDA_VISIBLE_DEVICES="$CDV_GPU_A" python -m cdviews.train_r2 \
  --cache-dir "$CDV_R2_CACHE" --run-dir "${CDV_R2_TRAIN}_single" \
  --epochs 10 --batch-size 2 --accum-steps 4 --workers 2 \
  --num-queries 4 --lr 5e-5 --seed 42 --dev-fraction 0.1
```

单/双卡 dropout 与采样补齐不同，不保证逐位一致。显存不足时双卡改 `--batch-size 1 --accum-steps 4`；
开发打分显存不足改 `--view-batch 4`。只改变 workers 可改善 I/O，但恢复配置需保持原值。

## 5. 用新 checkpoint 重新生成完整 val 排名

```bash
CUDA_VISIBLE_DEVICES="$CDV_GPU_A,$CDV_GPU_B" \
python -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29521 --nproc_per_node=2 \
  --module cdviews.rank_r2 \
  --cache-dir "$CDV_R2_CACHE" --checkpoint "$CDV_R2_TRAIN/best.pt" \
  --output-dir "$CDV_R2_RANK" --view-batch 16
```

只读取 val 问题、特征和原候选集合，按场景分卡。校验全部 4,675 问与候选覆盖，拒绝 smoke checkpoint。
新排名写到 `$CDV_R2_RANK/ScanQA_val_view_ranking.json`，同路径 `.scores.json` 保存原始余弦分数，
`.metadata.json` 保存 checkpoint 内容哈希、epoch、特征来源和排名哈希。不覆盖作者原排名。
排名输出目录必须为新目录；中途失败保留分片作诊断，重跑使用另一个目录。

## 6. 问答推理：R2 排名 + 原 viewNMS

先单卡两题试跑：

```bash
python -u run_scanqa.py \
  --gpus "$CDV_GPU_A" --splits val --view-selection cdviews --input-views 9 \
  --ranked-view-file "$CDV_R2_RANK/ScanQA_val_view_ranking.json" \
  --seed 42 --temperature 0.2 --run-dir "$CDV_EXPERIMENT" --stage smoke
```

成功后双卡按问题分片跑完整验证集：

```bash
python -u run_scanqa.py \
  --gpus "$CDV_GPU_A" "$CDV_GPU_B" --splits val --view-selection cdviews --input-views 9 \
  --ranked-view-file "$CDV_R2_RANK/ScanQA_val_view_ranking.json" \
  --seed 42 --temperature 0.2 --run-dir "$CDV_EXPERIMENT" --stage full
```

这里 `cdviews` 选择的是原来的 viewNMS 后处理；**选择器排名来自明确指定的 R2 文件**。
launcher 校验与作者候选池完全相同，复制新排名与元信息，并记录其哈希，避免误用旧排名。
问答两卡不是 DDP：各卡独立加载 LLaVA 回答不同问题，最后检查 ID、重复项和覆盖再合并。
两卡都需要能容纳 9 图问答。输出为 `$CDV_EXPERIMENT/full/ScanQA_val_answers.json`。

后续若比较 R2 + soft，另用新 `--run-dir` 并改 `--view-selection soft`，保留同一 R2 排名；
此时还需要 BLIP 权重。不要将它与本轮 R2 + 原 NMS 合并当成一个结构消融。

## 7. 评测

在安装了 Java / pycocoevalcap 的原评测环境中执行，保留路径变量：

```bash
python "$CDV_REPO/evaluate_scanqa.py" \
  --annotations "$CDV_DATA/ScanQA/ScanQA_v1.0/ScanQA_v1.0_val.json" \
  --predictions "$CDV_EXPERIMENT/full/ScanQA_val_answers.json" \
  --output "$CDV_EXPERIMENT/full/metrics.val.json"
```

检查 questions=4675；只与相同 val 口径比较。参考实测：原 cdViews 29.1/44.7/47.4/96.5，
soft 29.1/45.1/47.5/97.9；它们是用户提供值。R2 结果需实际运行，不保证提升。

## 验证边界

本地检查包括：真实模型模块的有效梯度、padding 不变性、小样本拟合、原文本 prompt 等价性、
场景划分、排名合并覆盖、训练/恢复、CPU 双进程 Gloo DDP，以及新排名接入问答入口。
Mac 无法替代服务器的 CUDA/NCCL、真实 LLaVA 特征提取和完整 ScanQA 训练，务必先执行 smoke。
