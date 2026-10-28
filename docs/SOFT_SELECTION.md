# 第一个改进版本：内容与位姿软去重

后续的集合感知互补版本使用独立的 `--view-selection coverage`，见 [COVERAGE_SELECTION.md](COVERAGE_SELECTION.md)。
本页的 `soft` 算法和运行命令继续保留。

入口为 `run_scanqa.py --view-selection soft`，仅支持 ScanQA `val`，默认单卡、9 张图。
无需训练；保留原 cdViews 的问题相关排序，用 BLIP 图像内容和相机位姿重新选图。
这是验证“硬去重可能误删有用视图”的初始实验，不代表已经提高了指标，也不宣称已实现跨图证据推理。

## 算法及边界

设原排序中的零起始名次为 `rank(i)`，已选集合为 `S`：

```text
relevance(i) = 1 / (1 + rank(i) / rank_scale)
visual(i,j) = clip(cosine(BLIP_image(i), BLIP_image(j)), 0, 1)
proximity(i,j) = exp(-original_pose_distance(i,j) / pose_scale)
redundancy(i | S) = max over j in S of visual(i,j) * (0.5 + 0.5 * proximity(i,j))
score(i | S) = relevance(i) - penalty * redundancy(i | S)
```

从原排序第一张开始，每次选择当前 `score` 最大的未选图，直到选满 `min(k, 候选数)`。
同分时沿用原排名。首轮冗余项为零；不做硬删除，也不再次执行原 viewNMS。
远处的重复内容仍有惩罚，位置接近会加强该惩罚；仅有位姿接近而视觉不相似不会被抑制。
问答模型接收整组选图和问题，一次生成一个答案。

默认 `penalty=0.5`、`rank_scale=10`、`pose_scale=0.5` 是未调优的初值。
位姿距离继续使用原实现的位置距离加旋转角距离，尺度不是单独的米或弧度。
不读取参考答案，不重新训练 selector。问题信息通过原 cdViews 排名间接进入选择；
本版本的冗余项是图像级相似度，不是问题条件化的对象或关系证据匹配。

**排名先验是一个明确的近似：** 作者的排序 JSON 只有文件名顺序，没有原始 selector 分数。
本版本不将排名转换值称为模型概率或原始分数，也不需要下载 LLaVA 特征重新评分。
`--soft-penalty 0` 退化成 cdViews 原排序 top-k（无 NMS），不是原版 cdViews + NMS。

## 依赖与运行

使用已有 cdViews 推理环境、ScanQA val 图像/位姿、作者 val 排序和 LLaVA-OV 权重，
另需 retrieval 实验所用的 `$CDV_DATA/model/blip-itm-large-coco`。
BLIP 编码/软选图先在独立进程完成，再加载 LLaVA，两者不同时占用显存。
BLIP 权重如尚未下载，见 [RETRIEVAL_BASELINE.md](RETRIEVAL_BASELINE.md)。
不需要升级现有推理环境中的 Transformers。

确认没有正在运行的实验、工作区干净后，在 Bash 中更新。以下为云端；实验室改为
`cd /home/pengjiale/projects/cdViews` 和 `source scripts/activate.sh lab 实际空闲GPU编号`。

```bash
cd /workspace/cdviews-work/repo
git status --short
git pull --ff-only origin main
source scripts/activate.sh cloud 0
nvidia-smi
python check_scanqa_downloads.py --root "$CDV_DATA" --splits val

export CDV_EXPERIMENT="$CDV_RUNS/soft_k9_p05_r10_d05_seed42_t02_v1"

# 两题试跑；成功后再执行完整验证集
python -u run_scanqa.py \
  --splits val --view-selection soft --input-views 9 \
  --soft-penalty 0.5 --soft-rank-scale 10 --soft-pose-scale 0.5 \
  --seed 42 --temperature 0.2 \
  --blip-model "$CDV_DATA/model/blip-itm-large-coco" \
  --run-dir "$CDV_EXPERIMENT" --stage smoke

python -u run_scanqa.py \
  --splits val --view-selection soft --input-views 9 \
  --soft-penalty 0.5 --soft-rank-scale 10 --soft-pose-scale 0.5 \
  --seed 42 --temperature 0.2 \
  --blip-model "$CDV_DATA/model/blip-itm-large-coco" \
  --run-dir "$CDV_EXPERIMENT" --stage full
```

BLIP 编码显存不足时加 `--retrieval-batch-size 2`，不会改变问答的视图数量。
同名阶段目录已存在时改实验名，不覆盖已有输出。每题选图是确定性的，smoke/full 不改变选图规则。

## 输出及评测

- `full/experiment.json`：Git 版本、输入哈希、方法和参数。
- `full/gpu<GPU编号>/selection.log`：BLIP 编码及软选图日志。
- `full/gpu<GPU编号>/inference.log`：问答日志，选图成功后才开始。
- `full/gpu<GPU编号>/ScanQA_val_soft_views.json`：每题选中的图像及输入顺序。
- 同路径 `.trace.json`：每步原排名、排名相关性、冗余和最终选择分数。
- 同路径 `.metadata.json`：算法、参数、BLIP 权重/版本及输入/输出哈希。
- `full/ScanQA_val_answers.json`：合并后的预测；`.views.json` 保存输入图像。

图像缓存复用 retrieval 的 `scannetv2/blip_itc_cache`，不改原缓存键；只有模型、依赖版本、
图像集合与文件状态等一致时才能命中。不同的候选图像集合可能创建新的缓存文件。
位姿距离矩阵保存在各实验内部，便于审计且避免沿用未确认的外部距离缓存。

在已有评测环境（Java、pycocoevalcap）中运行：

```bash
python "$CDV_REPO/evaluate_scanqa.py" \
  --annotations "$CDV_DATA/ScanQA/ScanQA_v1.0/ScanQA_v1.0_val.json" \
  --predictions "$CDV_EXPERIMENT/full/ScanQA_val_answers.json" \
  --output "$CDV_EXPERIMENT/full/metrics.val.json"
```

确认覆盖完整 4,675 问。先与相同候选池、9 张预算、seed=42、temperature=0.2 的原 cdViews 比较；
若原 NMS 实际选图不足 9 张，应结合 `.views.json` 记录实际张数，避免把数量差异全部归因为选图质量。
retrieval 默认 17 张的旧结果可保留作参考，但同预算消融应另跑 `--input-views 9`。
建议增加 `--soft-penalty 0` 的独立实验，区分新惩罚项相对原排序 top-k 的收益。
所有参数选择需记录；如需系统调参，可从训练集另划开发数据，避免反复针对完整 val 挑最优参数。

## 本地验证范围

CPU 测试覆盖近邻互补图保留、远处重复内容惩罚、位姿作用、零惩罚退化、预算/唯一性，
以及小型随机初始化 BLIP 的缓存复用、答案不参与选择、选图先于问答和多图单次生成接口。
这些检查验证代码行为，不代表真实模型的 ScanQA 精度；服务器需先执行 smoke，再做完整评测。
