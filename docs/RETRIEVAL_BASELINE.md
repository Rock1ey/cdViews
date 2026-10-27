# BLIP 图文检索：租卡服务器运行说明

实现：BLIP ViT-L 提取问题与候选图片特征，按投影后 CLS 特征的余弦相似度排序，
取前 17 张、不使用 NMS，再用原 LLaVA-OV 回答。只使用问题文本，不使用参考答案。
BLIP 与 LLaVA 分阶段、单卡运行，不同时驻留显存。无需训练。

## 下载哪个模型

使用 Salesforce 官方的 [blip-itm-large-coco](https://huggingface.co/Salesforce/blip-itm-large-coco)。
下载 PyTorch 权重约 1.78GB，以及模型和分词器配置；不下载 TensorFlow 权重。
不要替换成 captioning、VQA 或 BLIP-2 模型。

本地论文第 4 页公式 (4)、第 9 页 Table 3 指定 BLIP ViT-L 检索，但没有公布确切
checkpoint ID 或完整检索代码。我们选择公开的 COCO 检索权重，余弦排序路线参照
[BridgeQA 的检索实现](https://github.com/matthewdm0816/BridgeQA/blob/main/eval_scene_best_views.py)。
采用原问题、最大 70 个文本 token、checkpoint 自带图像预处理、float32 编码。
这是明确记录配置的检索基线实现，不宣称与 cdViews 未公开的检索细节逐项完全一致。

## 1. 更新代码、激活推理环境

确认没有实验运行、工作区没有未保存修改后，在 Bash 中执行：

```bash
cd /workspace/cdviews-work/repo
git status --short
git pull --ff-only origin main
source scripts/activate.sh cloud 0
```

继续使用原 cdViews 推理环境，不要为此升级 Transformers，也不需要安装 LAVIS。
`conda activate cdviews-eval` 只在最后评测时使用。

## 2. 下载权重

```bash
python scripts/download_blip.py \
  --output /workspace/cdviews-work/qa/model/blip-itm-large-coco
```

下载脚本只在自己的进程内解除离线模式，推理仍保持离线；继承已有的 HF_ENDPOINT 网络配置。
首次把请求的 revision 解析为明确的 commit 再下载，并保存 `download.json`。
如果服务器无法连接下载源，可在能联网且装有 huggingface_hub 的机器运行同一下载脚本，
将输出目录整体传到上述路径。不要仅复制 `.bin` 而遗漏配置和 tokenizer 文件。

## 3. 两题试跑

```bash
export CDV_EXPERIMENT="$CDV_RUNS/retrieval_k17_seed42_t02_v1"

python -u run_scanqa.py \
  --splits val --view-selection retrieval --input-views 17 \
  --seed 42 --temperature 0.2 \
  --blip-model "$CDV_DATA/model/blip-itm-large-coco" \
  --run-dir "$CDV_EXPERIMENT" --stage smoke
```

脚本先运行 BLIP 排序，成功退出、释放显存后再启动 LLaVA 问答。
默认 BLIP 编码 batch size 为 8；若检索阶段显存不足，可加 `--retrieval-batch-size 2`，
不会改变问答阶段的 17 张视图预算。

## 4. 完整验证集

```bash
python -u run_scanqa.py \
  --splits val --view-selection retrieval --input-views 17 \
  --seed 42 --temperature 0.2 \
  --blip-model "$CDV_DATA/model/blip-itm-large-coco" \
  --run-dir "$CDV_EXPERIMENT" --stage full
```

日志分别为：

- `$CDV_EXPERIMENT/full/gpu0/retrieval.log`：BLIP 排序阶段。
- `$CDV_EXPERIMENT/full/gpu0/inference.log`：LLaVA 问答阶段，排序完成后才创建。

输出目录已存在时改用 v2 等新名字，不覆盖历史结果。
图像特征缓存保存在 `$CDV_DATA/scannetv2/blip_itc_cache`，跨 smoke/full 和后续运行复用。
缓存以模型及配置文件哈希、实现、依赖版本、图片路径/大小/修改时间为键。
不改变作者原排序文件；每个实验中另存 BLIP 排名、对应余弦分数、权重哈希和评分配置。

## 5. 评测

```bash
conda activate cdviews-eval

python -u /workspace/cdviews-work/repo/evaluate_scanqa.py \
  --annotations /workspace/cdviews-work/qa/ScanQA/ScanQA_v1.0/ScanQA_v1.0_val.json \
  --predictions /workspace/cdviews-work/qa/runs/cloud/retrieval_k17_seed42_t02_v1/full/ScanQA_val_answers.json \
  --output /workspace/cdviews-work/qa/runs/cloud/retrieval_k17_seed42_t02_v1/full/metrics.val.json
```

检查 questions=4675。论文第 8 页 Table 2 中 retrieval 的配置为 17 张、无 NMS，
报告 EM@1=29.1；这是论文参考值，不保证本实现精确达到。
与之前 uniform/cdViews 的实测比较时保持生成温度、随机种子与其他问答设置一致。

## 本地验证范围

使用 CPU 小型随机初始化 BLIP 验证了缓存后的特征计算与 Transformers 余弦评分接口一致，
并检查了候选范围、参考答案不参与检索、缓存复用、检索先于问答和关闭 NMS。
未在本地下载大模型或执行完整 GPU 实验，服务器仍需先运行 smoke。
