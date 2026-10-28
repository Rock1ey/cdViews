# 第二个改进版本：集合感知的互补视图选择

入口：`run_scanqa.py --view-selection coverage`。无需训练，单卡 ScanQA `val`，9 张图。
在第一版 soft 的相关性和冗余惩罚上，增加**依赖已选集合的、问题加权的覆盖增益**。
它是集合感知互补选择的无训练近似；没有训练跨图关系网络，也没有显式识别对象或关系证据。

## 已有实验依据

用户在本次讨论中反馈以下结果，均为 9 张、seed=42、temperature=0.2：

| 方法 | EM@1 | BLEU-1 | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|
| 用户复现 cdViews | 29.1 | 44.7 | 47.4 | 96.5 |
| soft v1 | 29.1 | 45.1 | 47.5 | 97.9 |

这些是用户报告值，尚未读取原始预测和 metrics 文件核查；不是论文结果。
EM@1 在报告精度下持平，其他三项分别增加 0.4、0.1、1.4。
不能仅据此断定改进来自更强的多图推理，或两种方法答对了完全相同的问题。

## 为什么属于集合感知选择

soft v1 已通过最大相似度做集合相关的负向去重。v2 增加正向目标：
“当前集合没有覆盖、但与问题较相关的候选内容，加入这张图后能覆盖多少？”

使用原始候选池，令 `i` 为待选图、`u` 为待覆盖候选内容、`S` 为已选图集合：

```text
r(u) = 1 / (1 + cdViews_zero_based_rank(u) / rank_scale)
qsim(u) = cosine(BLIP_image(u), BLIP_question)
w(u) = normalize(r(u) * exp(text_weight * qsim(u) / coverage_temperature))
K(u,i) = clip(cosine(BLIP_image(u), BLIP_image(i)), 0, 1)^4

F(S) = sum_u w(u) * max_{j in S} K(u,j)
gain(i | S) = F(S union {i}) - F(S)
score(i | S) = soft_v1_score(i | S) + coverage_weight * gain(i | S)
```

空集合的覆盖定义为 0。保持原 cdViews top-1 作为固定首图，从第二张开始按新分数贪心选择。
每加入一张图就更新 `max K`，所以已经覆盖的内容不会重复获得同样的覆盖奖励。
首图的实际覆盖增益仍写入 trace，但它的覆盖奖励为 0，因为首图直接沿用原排序。
后续依然一起输入 LLaVA，一次生成答案。

图像核采用四次方以减弱弱相似背景的贡献，指数固定为 4，不用 val 答案拟合。
覆盖权重默认 0.5、问题余弦温度默认 0.07、文本权重默认 1.0，均为待验证初值。
原 soft 参数保持 penalty=0.5、rank_scale=10、pose_scale=0.5。
问题温度不是生成温度；问答的 `--temperature` 继续使用 0.2。

局限：全图特征无法可靠区分同类不同实例或细粒度空间关系，候选采样密度也会影响覆盖权重。
该版本衡量的是特征空间代表性，不能将它解释为真实证据已被证明充分，或保证 EM@1 提升。
不将这个常见的集合覆盖形式本身宣称为已确认的新颖贡献。

## 更新与推理

沿用 soft v1 的环境、模型、图像、位姿和原 cdViews 排序，无新增模型下载或训练。
BLIP 与 LLaVA 仍然分进程、分阶段运行；图像缓存键不变，可复用之前的缓存。
只额外计算问题文本特征及集合覆盖。问题编码沿用 retrieval 的最多 70 token。
每个场景的图像相似度核在内存中复用；覆盖阶段需要 O(N²) 内存和约 O(kN²) 每题计算，
其中 N 是候选数，k 是选图数，因此选图阶段可能比 v1 慢。

以下在云端 Bash 中执行。先确认没有正在运行的实验且工作区干净：

```bash
cd /workspace/cdviews-work/repo
git status --short
git pull --ff-only origin main
source scripts/activate.sh cloud 0
nvidia-smi
python check_scanqa_downloads.py --root "$CDV_DATA" --splits val

export CDV_EXPERIMENT="$CDV_RUNS/coverage_k9_w05_tau007_seed42_t02_v1"

python -u run_scanqa.py \
  --splits val --view-selection coverage --input-views 9 \
  --soft-penalty 0.5 --soft-rank-scale 10 --soft-pose-scale 0.5 \
  --coverage-weight 0.5 --coverage-temperature 0.07 --coverage-text-weight 1 \
  --seed 42 --temperature 0.2 \
  --blip-model "$CDV_DATA/model/blip-itm-large-coco" \
  --run-dir "$CDV_EXPERIMENT" --stage smoke

# 两题试跑成功后，完整运行 4,675 问
python -u run_scanqa.py \
  --splits val --view-selection coverage --input-views 9 \
  --soft-penalty 0.5 --soft-rank-scale 10 --soft-pose-scale 0.5 \
  --coverage-weight 0.5 --coverage-temperature 0.07 --coverage-text-weight 1 \
  --seed 42 --temperature 0.2 \
  --blip-model "$CDV_DATA/model/blip-itm-large-coco" \
  --run-dir "$CDV_EXPERIMENT" --stage full
```

实验室改用 `/home/pengjiale/projects/cdViews` 和 `source scripts/activate.sh lab 实际空闲GPU编号`。
BLIP 编码显存不足时可加 `--retrieval-batch-size 2`，不改变问答视图数。
同名实验阶段已存在时使用新名字；不覆盖 v1 或历史实验文件。

## 评测与消融

切换到已有评测环境并保留路径变量：

```bash
python "$CDV_REPO/evaluate_scanqa.py" \
  --annotations "$CDV_DATA/ScanQA/ScanQA_v1.0/ScanQA_v1.0_val.json" \
  --predictions "$CDV_EXPERIMENT/full/ScanQA_val_answers.json" \
  --output "$CDV_EXPERIMENT/full/metrics.val.json"
```

保持模型、候选池、问答提示词、解码、评测口径不变。首先对比完整 val 的 soft v1 与本版本；
无需为了“关闭覆盖”重新跑已有 v1 全集，下面的参数主要用于检查或更细的消融。

| 设置 | 含义 |
|---|---|
| `--view-selection soft` | 保留第一版算法 |
| `--view-selection coverage --coverage-weight 0` | 精确退化为 v1 的选图和分数，跳过问题编码/覆盖核计算 |
| `--view-selection coverage --coverage-text-weight 0` | 仍做集合覆盖，但权重只依赖原 cdViews 排名，用于隔离问题编码贡献 |
| 默认 coverage | 排名与问题共同加权的集合覆盖 |

每组消融都要换 `--run-dir`。不要仅凭一次小幅变化就宣布稳定提升；进一步判断时应对齐逐题结果，
观察新增答对/新增答错样本，并记录重复实验。若继续系统调参，使用训练集另划的开发数据，
完整 val 的反复选优结果需如实记录，不将其当作独立泛化评估。

## 实验记录

- `full/experiment.json` 和 `gpu<编号>/QA.yaml` 保存全部新增参数。
- `gpu<编号>/selection.log`：选图日志；之后 `inference.log` 为问答日志。
- `gpu<编号>/ScanQA_val_coverage_views.json`：每题选图及输入顺序。
- `.trace.json`：每一步相关性、冗余、实际覆盖增益、累计覆盖值、覆盖奖励和总分。
- `.coverage.json`：各题候选名称、问题余弦与归一化覆盖权重，仅覆盖项启用时生成。
- `.metadata.json`：方法版本、公式、参数、问题哈希及模型/输入/输出哈希。
- `full/ScanQA_val_answers.json` 与 `.views.json`：合并预测和输入视图。

不使用参考答案参与选择。旧 `soft` 输出文件名称及默认行为保持原样。
本地测试覆盖问题驱动的选图变化、集合边际收益递减、首图固定、零权重回归、缓存复用、
独立快照执行及答案不参与选择；尚未运行真实 GPU ScanQA 实验。
