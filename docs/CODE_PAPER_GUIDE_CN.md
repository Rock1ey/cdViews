**cdViews 仓库与论文对应导读**

本文依据本地仓库提交 `e1df64c`（`Update README.md`），于 2026-10-06 对照阅读。论文为 [3D Question Answering via only 2D Vision-Language Models](cdviews.pdf)。页码、公式编号和图表编号均按该 PDF。

本导读覆盖 cdViews 主流程的全部非空 Python 文件，并沿调用链阅读 LLaVA 的模型加载、视觉编码、图文拼接和答案生成实现。底层通用训练、服务和 TRL 代码按子系统说明，不把其中未被主流程调用的功能当作本文方法。代码片段为当前源码的连续节选；省略的部分由源码链接定位。

验证范围：静态源码追踪，以及对实际相机距离函数、viewNMS 函数的独立小样例验证。当前检查环境没有 PyTorch，仓库未附数据和模型权重，因此没有执行完整训练、生成或复现论文分数。注意力掩码的数值后果采用等价 NumPy 运算验证，不冒充完整模型实测。

**1. 先把论文贡献与底层框架分开。**

论文新增的自动标注、选择器训练、排序、去重和问答编排，集中在 `cdviews/`。`llava/` 提供现成视觉语言模型。`trl/` 主要服务于继承的通用训练功能。

| 文件或目录 | 做什么 | 与本文主流程的关系 |
|---|---|---|
| [README.md](../README.md#L1) | 项目说明、数据和权重链接、运行顺序、结果与评测说明 | 入口文档 |
| [cfgs/QA.yaml](../cfgs/QA.yaml#L1) | 数据路径、模型路径、提示词、视图数、输出路径 | 所有核心脚本共享配置 |
| [caption_generation.py](../cdviews/caption_generation.py#L76) | 将训练集问题和答案改写为描述句 | viewAnnotator 第一步；公式（5） |
| [view_matching.py](../cdviews/view_matching.py#L76) | 用描述句逐张检查图像，生成正、负、不确定标签 | viewAnnotator 第二步；公式（6） |
| [visual_feature_processing.py](../cdviews/visual_feature_processing.py#L76) | 提前编码场景全部 RGB 帧，按场景缓存视觉 token | 选择器的特征准备 |
| [dataset.py](../cdviews/dataset.py#L8) | 训练时抽正负视图；测试时读取场景全部缓存特征 | 构造问题与视图配对样本 |
| [ViewSelector.py](../cdviews/ViewSelector.py#L8) | 特征降维、自注意力、交叉注意力、池化及 BCE 损失 | viewSelector；图 3、公式（7）（8） |
| [train.py](../cdviews/train.py#L80) | 提取问题词嵌入，训练选择器，保存 checkpoint | 本文选择器训练入口 |
| [qa_inference.py](../cdviews/qa_inference.py#L87) | 所有视图打分、排序缓存、NMS、调用大模型回答 | 本文测试入口；公式（1）（2）（9） |
| [view_distance_calculation.py](../cdviews/view_distance_calculation.py#L11) | 读取相机位姿，计算场景内两两视图距离 | 公式（10）至（12） |
| [qa_utils.py](../cdviews/qa_utils.py#L17) | 配置合并、读取 ScanQA/SQA、拼接训练 batch | 共用工具 |
| [llava/model/builder.py](../llava/model/builder.py#L27) | 加载 tokenizer、模型、视觉预处理器 | 连接核心脚本与底层大模型 |
| [llava_arch.py](../llava/model/llava_arch.py#L191) | 图像编码、投影、将视觉 token 插入语言 token 序列 | 图 3 中 Visual Encoder、Projection 及图文拼接 |
| [language_model/llava_qwen.py](../llava/model/language_model/llava_qwen.py#L46) | 将 Qwen2 与多模态输入接口结合 | 默认 LLaVA-OV 的语言生成路径 |
| [multimodal_encoder/builder.py](../llava/model/multimodal_encoder/builder.py#L13) | 根据模型配置选择 SigLIP、CLIP 等视觉编码器 | 默认模型路径相关；其他编码器是通用支持 |
| [multimodal_projector/builder.py](../llava/model/multimodal_projector/builder.py#L32) | 将视觉通道数转换到语言模型隐藏维度 | 图 3 中 Projection |
| [multimodal_resampler/builder.py](../llava/model/multimodal_resampler/builder.py#L21) | 构建通用视觉重采样模块 | 核心 `encode_images` 中相应调用已注释；不是 viewSelector |
| [llava/train/train.py](../llava/train/train.py#L1449) 与 [train_dpo.py](../llava/train/train_dpo.py#L1484) | 通用 LLaVA 指令微调、DPO 等 | 与 `cdviews/train.py` 是不同训练入口 |
| [llava/serve/](../llava/serve) | Gradio 界面、模型 worker、控制器、命令行交互 | 非论文批量问答入口 |
| [trl/](../trl) | trainer 提供 PPO/DPO/SFT 等；models 提供训练包装；extras、environment、core 提供辅助工具 | 未被 cdViews 选择器训练循环直接使用 |
| [llava/mm_utils.py](../llava/mm_utils.py#L314)、[constants.py](../llava/constants.py#L6)、[conversation.py](../llava/conversation.py#L1) | 图像处理、特殊 token、通用对话模板 | 核心脚本另有复制的 `preprocess_qwen` |
| [predict.py](../predict.py#L75)、[cog.yaml](../cog.yaml#L1) | Cog/Replicate 单图推理封装，代码加载 LLaVA-v1.5-13B | 不是 cdViews 的 7B 多视图问答入口 |
| [pyproject.toml](../pyproject.toml#L8)、[requirements.txt](../requirements.txt#L1) | 包元数据、依赖清单 | 安装支持；不能据依赖中出现 DPO 等就认为论文使用了它 |
| [REPRODUCE_A6000_CN.md](setup/REPRODUCE_A6000_CN.md#L1) | 本地已有复现说明 | 阅读时为未跟踪文件；不是据此认定原始实现的证据 |

`cdviews/__init__.py` 为空；`images/framework.png` 是 README 的架构图。

**2. 实际运行是一条有缓存的流水线。**

`cfgs/QA.yaml` 默认 `dataset: ScanQA`、`batch_size: 8`、`input_views: 9`，模型由 `LVLM_ckpt` 指定。脚本会先解析命令行，再由 YAML 更新参数。当前核心脚本真正读取模型路径的是 `args.LVLM_ckpt`，而不是名字相近的 `args.model_path`。默认配置路径 `../cfgs/QA.yaml` 相对运行目录解释。

训练流程：

`训练集 QA → caption_generation → captions JSON → view_matching → 标签 JSON`

`场景 RGB 帧 → visual_feature_processing → 每场景一个特征 .pth`

`标签 + QA + 缓存特征 → ViewLabelDataset → train_epoch → ViewSelector checkpoint`

测试流程：

`测试问题 + 全场景缓存特征 → ranking_views → 排序 JSON`

`相机 pose → calculate_view_distance → 距离 CSV`

`排序 + 距离 → viewNMS → 选中图像文件名 → 重新读取 RGB → LLaVA.generate → 答案 JSON`

关键理解：论文图 3 将选择器画在视觉编码器和大模型之间；仓库采用缓存和独立脚本实现这个逻辑。选择器输出的是用于排序的向量，最终保存的是图像文件名顺序。问答时重新读取所选图片，并没有直接把选择器内部的 512 维向量送进 LLaVA。

当前代码直接读取 ScanNet 的 `color/*.jpg` 和 `pose/*.txt`，未实现一个从点云生成候选渲染视图的模块。不要把论文“从三维场景获得视图”的任务表述，误读成仓库包含点云渲染代码。

下面是数据格式示意，示例文件名和内容不是仓库中真实样本：

```text
captions JSON:
  question_id → ["描述句 1", "描述句 2", ...]

view_labeling JSON:
  question_id → {
    "Positive": ["0.jpg", ...],
    "Negative": ["20.jpg", ...],
    "Uncertain": ["40.jpg", ...]
  }

每场景的特征 .pth:
  image_filename → 视觉 token 张量 [P, D]

ranking JSON:
  question_id → [最高分图像文件名, 次高分图像文件名, ...]

距离 CSV:
  行、列均为图像文件名；元素为相机位置与朝向的组合距离

答案 JSON:
  [{"scene_id": "...", "question_id": "...",
    "answer_top10": ["同一个答案", ... 共 10 次]}]
```

**3. viewAnnotator 的两个步骤对应公式（5）和（6）。**

公式（5）：

$$C=\operatorname{LVLM}(Q,A,\operatorname{Prompt}_R).$$

[caption_generation.py:98](../cdviews/caption_generation.py#L98) 拼接 `Q: ... A: ...`，使用 `prompt_R`，然后调用 `model.generate`：

```python
        for i in range(len(line['answers'])):
            line['from'] = 'human'
            line['value'] = 'Q: {} \n A: {}'.format(line['question'], line['answers'][i])
            input_ids = preprocess_qwen([line, {'from': 'gpt', 'value': None}], tokenizer, has_image=False,
                                        system_message=args.prompt_R).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=None,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)
```

这里 `images=None`，说明生成描述句只需要问答文本。同一个问题如果有多个参考答案，代码为每个答案都生成一个描述句，最终保存为列表。输入仅来自训练集，见该文件第 87–90 行。

公式（6）：

$$S_i=\operatorname{LVLM}(C,V_i,\operatorname{Prompt}_M).$$

[view_matching.py:110](../cdviews/view_matching.py#L110) 为每张图片构造匹配任务：

```python
        for image_name in image_files:
            line['from'] = 'human'
            line['value'] = '<image>' + 'Caption: {}'.format(captions[0])

            input_ids = preprocess_qwen([line, {'from': 'gpt', 'value': None}], tokenizer,
                                        has_image=True, system_message=args.prompt_M + args.context_M).cuda()

            image_tensors = []
            image = Image.open(os.path.join(args.image_folder, scene_id, 'color', image_name))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor.half().cuda())
```

描述句后面的图片通过 `model.generate(..., images=image_tensors)` 输入，见第 123–133 行。`prompt_M` 是 A/B/C 分类指令；`context_M` 是附录图 S1 的逐步检查示例，均位于 [QA.yaml:24](../cfgs/QA.yaml#L24)。

标签解析：

```python
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if 'A' in outputs:
                view_labeling_info[question_id]['Positive'].append(image_name)
            elif 'B' in outputs:
                view_labeling_info[question_id]['Negative'].append(image_name)
            elif 'C' in outputs:
                view_labeling_info[question_id]['Uncertain'].append(image_name)
            else:
                print('providing answer not in the options {} / {}'.format(scene_id, question_id))
```

需要区分“设计目标”和“实际处理”：

- 生成阶段为每个参考答案生成 caption，匹配阶段实际只使用 `captions[0]`，没有对多个 caption 投票。
- 分类解析是检查输出是否包含大写字母 `A/B/C`，不是严格解析选项；模型输出较长解释时可能误分类。
- 明确的 C 类进入 `Uncertain`。不含 A/B/C 的输出仅打印提示，不加入任何列表。两者都不会被训练数据集采样，但保存方式不完全等同于论文“无效输出归入不确定”的说法。
- `viewAnnotator` 不是训练中的在线子网络。它离线生成 JSON，随后选择器读取这些标签。

**4. 选择器看到的特征，是视觉投影结果和语言模型的词嵌入。**

[visual_feature_processing.py:106](../cdviews/visual_feature_processing.py#L106) 提前保存视觉特征：

```python
            with torch.inference_mode():
                image_features = model.encode_images(image_tensors)
                image_features = torch.cat((image_features,
                                            model.model.image_newline[None].unsqueeze(0).repeat(num_img, 1, 1)), dim=1)

                for image_file_name, image_feature in zip(image_files, image_features):
                    feature_dict.update({image_file_name: image_feature.cpu()})

        torch.save(feature_dict, feature_save_file)
```

其中 `model.encode_images` 的实现位于 [llava_arch.py:191](../llava/model/llava_arch.py#L191)：

```python
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
```

对应论文图 3 中：

`RGB → Visual Encoder → Projection → 可供选择器使用的视觉 token`。

`image_newline` 是额外的图像分隔 token，不是另一张视图。按仓库默认选择器的 3584 输入维度，以及 SigLIP 分支断言的 729 个视觉 patch token，再加一个 newline，单张缓存特征可解释为 `[730,3584]`。这是假定默认 LLaVA-OV/SigLIP 配置的形状，不是任何模型都必须具有的形状。视觉编码器实现见 [siglip_encoder.py:576](../llava/model/multimodal_encoder/siglip_encoder.py#L576)；实际模型组件仍由外部 checkpoint 配置决定。

问题特征提取位于 [train.py:85](../cdviews/train.py#L85)：

```python
        text_embeds_list = []
        for qs, image_files in zip(qs_list, image_file_list):
            line = {'from': 'human',
                    'value': qs}
            input_ids = preprocess_qwen([line, {'from': 'gpt', 'value': None}], tokenizer, has_image=False).cuda()
            text_embed = model.get_model().embed_tokens(input_ids)
            text_embeds_list += [text_embed.squeeze(0), ] * len(image_files)

        padded_text_embeds = pad_sequence(text_embeds_list, batch_first=True)

        text_embeds, image_embeds = view_selector(image_embeds.float().cuda(), padded_text_embeds.float())

        loss = criterion(text_embeds, image_embeds, labels.cuda())
```

注意 `embed_tokens` 是词嵌入查表，不是把问题完整运行过 Qwen2 后提取最后一层 hidden states。输入还包含 system/user/assistant 对话格式 token。选择器里的两层文本自注意力负责进一步处理这段序列。

`text_embeds_list += ... * len(image_files)` 将同一问题复制给它的每张视图。虽然变量名叫 `bs`，进入选择器后的 batch 维实际是“问题与视图配对数”。

训练采样位于 [dataset.py:54](../cdviews/dataset.py#L54)：

```python
        if self.mode == 'train':
            cur_pair = self.view_labeling_info[str(question_id)]
            pos_images, neg_images = cur_pair['Positive'], cur_pair['Negative']
            num_img = min(len(pos_images), len(neg_images), self.num_img // 2)
            image_files = [pos_images[i] for i in torch.randperm(len(pos_images))[:num_img]] + \
                          [neg_images[i] for i in torch.randperm(len(neg_images))[:num_img]]
            labels = torch.cat([torch.ones(num_img),
                                torch.zeros(num_img)]).float()
        else:
            labels = None
            image_files = [image for image in cur_feats.keys()]    # score all the image files for inference
```

每个问题最多取 5 正、5 负；样本不足时，两类一起减到相同数量。没有正样本或没有负样本的问题会被过滤。测试时完全不读取正负标签，而是为场景缓存中的全部视图打分。

若一批有 8 个问题，且各自都有至少 5 个正负视图，则形成 80 个配对样本。形状变化如下：

| 阶段 | 视觉张量 | 文本张量 |
|---|---|---|
| 选择器输入 | `[M,P,3584]` | `[M,L,3584]` |
| Linear + GELU 后 | `[M,P,256]` | `[M,L,256]` |
| MCAN_ED 后 | `[M,P,256]` | `[M,L,256]` |
| AttFlat 后 | `[M,512]` | `[M,512]` |
| 余弦相似度 | 每对得到一个标量，共 `[M]` | 同左 |

其中 `M≤80`，`P` 为视觉 token 数，`L` 为批内填充后的文本长度。`custom_collate_fn` 负责拼接图像样本与标签，见 [qa_utils.py:95](../cdviews/qa_utils.py#L95)。

**5. ViewSelector 的网络可对应到图 3 的 Linear、Transformer、Pooling。**

[ViewSelector.py:9](../cdviews/ViewSelector.py#L9) 给出默认结构：

```python
    def __init__(self, input_channel=3584, embedding_dim=256, flat_out_size=512, num_layers=2, num_heads=8):
        super(ViewSelector, self).__init__()
        self.text_feat_linear = nn.Sequential(
            nn.Linear(input_channel, embedding_dim, bias=False),
            nn.GELU()
        )
        self.image_feat_linear = nn.Sequential(
            nn.Linear(input_channel, embedding_dim, bias=False),
            nn.GELU()
        )

        self.fusion_backbone = MCAN_ED(embedding_dim, num_heads=num_heads, num_layers=num_layers,
                                       pdrop=0.1)
        self.attflat_visual = AttFlat(embedding_dim, embedding_dim, 1, flat_out_size, 0.1)
        self.attflat_lang = AttFlat(embedding_dim, embedding_dim, 1, flat_out_size, 0.1)
```

代码中的两个线性层参数独立。默认 `embedding_dim=256`、`num_layers=2`、`num_heads=8`、`flat_out_size=512`。按各层维度累计，默认网络有 5,916,674 个参数，与论文约 5.9M 一致；该数量是结构计算，不是加载作者 checkpoint 后的实测。

| 类或函数 | 作用 | 阅读重点 |
|---|---|---|
| `ViewSelector.forward`，第 50 行 | 构造 mask，降维，调用融合骨干和池化 | 返回两个向量，并不直接返回 0/1 标签 |
| `MCAN_ED`，第 76 行 | 文本编码器 SA + 视觉引导注意力 SGA | 默认两层 SA、两层 SGA |
| `SA`，第 250 行 | 自注意力 + 残差/归一化 + 前馈网络 | 用于文本序列 |
| `SGA`，第 293 行 | 视觉自注意力 + 文本引导交叉注意力 + 前馈网络 | 图像 token 查询问题 token |
| `MHAtt`，第 184 行 | Q/K/V 线性映射、多头分组、缩放点积注意力 | `forward(v,k,q,mask)` 参数顺序需注意 |
| `AttFlat`，第 93 行 | 预测 token 权重，加权池化，再投影到 512 维 | 文本和视觉各有独立模块 |
| `FFN`，第 275 行 | 256→1024→256 的前馈网络 | 两层之间 GELU、dropout |
| `FC / MLP`，第 132 / 158 行 | 构成池化网络和 FFN 的基础层 | 非独立论文模块 |
| `LayerNorm`，第 169 行 | 手写归一化，含可学习缩放和偏移 | 使用 `std`，不是直接调用 `nn.LayerNorm` |
| `BCELoss`，第 326 行 | 余弦分数映射和标签监督 | 对应公式（7）（8）的训练实现 |

融合调用：

```python
    def forward(self, x, y, x_mask, y_mask):
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y
```

其中 `x` 是文本、`y` 是图像。SGA 的关键部分是：

```python
    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x
```

`self.mhatt2(y,y,x,y_mask)` 的 Q 来自视觉特征 `x`，K/V 来自文本特征 `y`。它实现的是“视觉查询文本”的引导注意力。文本分支先独立编码，没有对称地再用图像更新文本分支。因此不宜笼统描述为双向跨模态 Transformer。

注意力的基本计算：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(QK^\top/\sqrt{d_k}+\text{mask}\right)V.$$

该式用于解释实现，并非论文另行编号的公式。源码在 [ViewSelector.py:234](../cdviews/ViewSelector.py#L234)。

从 batch 的构造可知，每张图与问题独立配对。选择器没有在这一层直接对“同一场景不同视图之间的 token”做跨视图注意力；它对各视图评分，随后由 viewNMS 处理重复，最终由问答大模型同时接收多视图。

**6. 公式（7）（8）在代码中比论文写得更完整。**

论文公式（7）：

$$s_i=\cos(q,v_i)=\frac{q^\top v_i}{\lVert q\rVert\lVert v_i\rVert}.$$

训练时的完整实现：

```python
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, output1, output2, label):
        cosine_similarity = F.cosine_similarity(output1, output2)
        similarity_score = (cosine_similarity + 1) / 2

        loss = self.bce_loss(similarity_score, label.float())
        return loss
```

因此，实际用于 BCE 的概率型分数为：

$$p_i=\frac{s_i+1}{2},$$

$$\mathcal L=-\frac1M\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].$$

这里用 `y_i` 表示伪标签，用 `p_i` 表示映射后的预测，避免论文符号混淆。论文公式（8）按其定义将标签与预测的位置写反，代码调用 `nn.BCELoss(prediction,target)` 的顺序则正确。余弦值到 `[0,1]` 的映射，也补上了正文未明确给出的步骤。

推理时 [qa_inference.py:105](../cdviews/qa_inference.py#L105) 直接按原始余弦分数排序：

```python
            image_embeds = image_embeds.float()
            if len(image_files) < chunk_size:
                text_embeds, image_embeds = view_selector(image_embeds.to(model.device), padded_text_embeds)
                scores = F.cosine_similarity(text_embeds, image_embeds)
            else:
                text_embeds_list = split_list(padded_text_embeds)
                image_embeds_list = split_list(image_embeds)
                scores_list = []
                for text_embeds, image_embeds in zip(text_embeds_list, image_embeds_list):
                    text_embeds, image_embeds = view_selector(image_embeds.to(model.device), text_embeds)
                    scores = F.cosine_similarity(text_embeds, image_embeds)
                    scores_list.append(scores)
                scores = torch.cat(scores_list)

            paired = list(zip(scores, image_files))
            paired.sort(key=lambda x: x[0], reverse=True)

            sorted_scores, sorted_image_files = zip(*paired)
            qs_id = str(qs_id_list[0])
            output_dict[qs_id] = list(sorted_image_files)
```

这是合理的，因为 `(s+1)/2` 是严格单调变换，不会改变排序。保存的 ranking JSON 只包含文件名顺序，不保存分数。

`chunk_size=50` 用于分批处理视图，减小选择器单次计算量；它不表示只从场景中取 50 张候选图。当前调用 DataLoader 的 batch_size 固定为 1，排序输出逻辑也依赖一批一个问题，不宜直接改成更大的测试 batch。

**7. 训练循环对应“仅训练轻量选择器”。**

[train.py:122](../cdviews/train.py#L122)：

```python
    learning_rate, num_epoches = 0.00005, 10
    optimizer = torch.optim.Adam(view_selector.parameters(), lr=learning_rate, weight_decay=1e-5)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    for epoch in range(num_epoches):
        save_ckpt_path = os.path.join(args.work_dir, 'model_epoch_{}.pth'.format(epoch + 1))
        train_epoch(pair_dataloader_train, tokenizer, model, view_selector,
                        optimizer, epoch, save_path=save_ckpt_path)
```

学习率 `5e-5`、训练 10 轮，使用 Adam，weight decay 为 `1e-5`。优化器只持有 `view_selector.parameters()`。反向传播与保存：

```python
        loss = criterion(text_embeds, image_embeds, labels.cuda())
        print('epoch: {} -- iteration: {} -- loss: {}'.format(epoch, iter, loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.save({'epoch': epoch,
                'model': view_selector.state_dict(),
                'optimizer': optimizer.state_dict()}, save_path)
```

保存内容是 epoch、选择器参数和优化器状态，不包含一套微调后的 LLaVA 参数。本文件的 `eval_model` 名字容易误导：它实际执行训练，没有验证集打分或自动选取最佳 epoch 的步骤。

需要精确区分“不更新”和“不求梯度”：训练脚本没有显式对整个 LLaVA 调用 `requires_grad_(False)`，提取问题嵌入时也没有 `no_grad` 或 `detach`。因此，若 embedding 权重保留可求梯度状态，反向传播仍会为其计算、累积梯度；它不在优化器中，所以不会被该优化器更新。`disable_torch_init` 只是跳过默认初始化，见 [utils.py:163](../llava/utils.py#L163)，不能把它解释为冻结模型。

**8. viewNMS 与相机距离基本直接对应公式（9）至（12）。**

公式（10）旋转距离、公式（11）位置距离、公式（12）组合距离：

$$D_{\rm ori}(i,j)=2\arccos(|p_i^\top p_j|),\quad
D_{\rm pos}(i,j)=\lVert t_i-t_j\rVert_2,$$

$$D(i,j)=D_{\rm pos}(i,j)+D_{\rm ori}(i,j).$$

位姿读取函数用 `pose_matrix[:3,3]` 作为位置，`pose_matrix[:3,:3]` 作为旋转矩阵，再转四元数，见 [view_distance_calculation.py:11](../cdviews/view_distance_calculation.py#L11)。因此它要求输入位姿符合该约定；不能不加转换地换入另一种外参约定。

距离函数：

```python
def camera_distance(pose1, pose2, position_weight=1.0, orientation_weight=1.0):
    """
    Calculate a distance metric between two camera poses.
    Consider position and orientation (using quaternions).
    """
    pos1, quat1 = pose1
    pos2, quat2 = pose2

    # Position distance
    pos_distance = np.linalg.norm(pos1 - pos2)

    # Orientation distance (angle between quaternions)
    quat_dot_product = np.abs(np.dot(quat1, quat2))
    orientation_distance = 2 * np.arccos(np.clip(quat_dot_product, -1.0, 1.0))

    # Combine distances with weights
    return (position_weight * pos_distance +
            orientation_weight * orientation_distance)
```

默认两个权重都为 1，与论文相加一致；`clip` 是浮点数保护，避免点积略超出反余弦定义域。四元数点积取绝对值处理 `p` 与 `-p` 表示相同旋转的情况。

`calculate_image_distance_` 为所有图像生成对称距离表，复杂度随视图数约为 `O(N²)`。`calculate_view_distance` 将 `pose/123.txt` 对应到 `123.jpg`。推理时按场景缓存 CSV，重复问题可复用相机距离。

公式（9）的实现位于 [qa_inference.py:126](../cdviews/qa_inference.py#L126)：

```python
def viewNMS(list_of_images, neighbour_df, num_images, distance_threshold=0.5):
    selected_images = []
    remaining_images = list_of_images.copy()

    while len(selected_images) < num_images and remaining_images:
        current_image = remaining_images.pop(0)
        selected_images.append(current_image)

        sorted_distances = neighbour_df.loc[current_image].sort_values()
        filtered_images = sorted_distances[sorted_distances < distance_threshold]
        neighbours_to_remove = set(filtered_images.index.tolist())
        remaining_images = [img for img in remaining_images if img not in neighbours_to_remove]

    return selected_images
```

`list_of_images` 必须已经按分数降序排列。每保留一个最高分剩余视图，就删除距它太近的其他候选。由于之前选中视图的邻居都已经被删除，后续保留项会同时满足对所有已选项的距离要求。

与论文的两个细小区别：

- 论文条件写成 `D>T`；代码删除 `D<T`，所以 `D=T` 的视图仍可保留，即实现的边界是 `D≥T`。
- `num_images` 是上限。候选被抑制完时，返回结果可能少于 9，代码不会强行补满。

独立执行实际函数的小样例得到：相同位姿距离为 0；四元数符号相反距离仍为 0；仅平移 1 个坐标单位距离为 1；仅旋转 90° 距离为 `π/2`。按 A、B、C、D 排序，若 A/B 接近、C/D 接近，则可保留 A/C；目标数设为 3 时仍只返回 2 项。距离恰为 0.5 的两项会同时保留。这些结果核实了上述解释。

**9. 最终问答对应公式（1）（2），底层通过视觉 token 替换占位符。**

公式（1）选择视图子集：

$$\mathcal M'=F(\mathcal M,Q,k).$$

对应 `ranking_views` 排序，再在 [qa_inference.py:193](../cdviews/qa_inference.py#L193) 调用 NMS：

```python
            question_id = str(line['question_id'])
            image_files = image_file_list[question_id]
            image_files = viewNMS(image_files, distance_df, num_images=args.input_views)
            image_files = image_files if len(image_files) < args.input_views else image_files[:args.input_views]
            num_image = len(image_files)

            line['from'] = 'human'
            question = line['question'] if args.dataset == 'ScanQA' else line['situation'] + line['question']
            line['value'] = '<image>' * num_image + question
            input_ids = preprocess_qwen([line, {'from': 'gpt', 'value': None}], tokenizer, has_image=True).to(
                model.device)
```

SQA 在这里将 `situation` 与 `question` 拼接；`dataset.py` 返回给选择器的却只有 `question`。caption 生成阶段也只拼接 `question` 与答案。除非外部数据预先把 situation 放进 question 字段，否则当前代码的标注、选择与最终问答并不使用完全相同的文本上下文。

公式（2）：

$$A=\operatorname{LVLM}(\mathcal M',Q).$$

它在 [qa_inference.py:205](../cdviews/qa_inference.py#L205) 读取所选 RGB 帧，并调用生成：

```python
            image_tensors = []
            for image_file in image_files:
                image = Image.open(os.path.join(args.image_folder, scene_id, 'color', image_file))
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensors.append(image_tensor.half().to(model.device))
            # image_tensors = torch.cat(image_tensors, dim=0)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip().lower()
            ans = outputs.split('\n')[0]
            # ScanQA result format
            answers.append({"scene_id": scene_id,
                            "question_id": question_id,
                            "answer_top10": [ans for i in range(10)]})
```

`answer_top10` 实际是同一答案重复 10 次以适配输出结构，不代表模型产生了十个候选答案。输出被转为小写、去首尾空白，并只保留第一行。

生成调用的底层链路：

`qa_inference.eval_model → LlavaQwenForCausalLM.generate → prepare_inputs_labels_for_multimodal → encode_images → 图文 embedding 拼接 → Qwen2 自回归生成`。

[llava_qwen.py:130](../llava/model/language_model/llava_qwen.py#L130)：

```python
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
```

`preprocess_qwen` 把每个 `<image>` 转成特殊索引 `IMAGE_TOKEN_INDEX=-200`；它不是普通词表中一个能表示整张图片的 token。随后 [llava_arch.py:468](../llava/model/llava_arch.py#L468) 用对应图片的整段视觉特征替代占位符：

```python
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
```

因此 9 张图片可能对应数千个视觉 token，这也解释了论文为何关注输入视图数。模型还可能按自身配置裁剪最大序列长度，见同文件第 489–494 行。

当前默认生成参数是 `temperature=0.2`、`num_beams=1`、`max_new_tokens=1024`；默认启用采样。核心脚本中重复出现的 `preprocess_qwen` 构造了 `targets`，但只返回 `input_ids`。不能据此认为选择器训练包含语言模型 next-token loss。

**10. 论文全部编号公式可按下表定位。**

| 论文位置 | 内容 | 仓库对应 | 一致性说明 |
|---|---|---|---|
| 公式（1），第 4 页 | 选择子集 `F(M,Q,k)` | [qa_inference.py:87](../cdviews/qa_inference.py#L87)、[qa_inference.py:193](../cdviews/qa_inference.py#L193) | 排序与 NMS 组合；实际可能返回少于 k 张 |
| 公式（2），第 4 页 | 所选视图与问题生成答案 | [qa_inference.py:212](../cdviews/qa_inference.py#L212) | 对应 `model.generate` |
| 公式（3），第 4 页 | 均匀采样基线 | 当前核心目录未提供专用实验入口 | 训练集正负随机采样不是这一基线 |
| 公式（4），第 4 页 | BLIP 图文检索基线 | 上游未提供；本项目新增 `cdviews/blip_retrieval.py`，运行见 [检索说明](RETRIEVAL_BASELINE.md) | 使用独立 BLIP 权重，不是 viewSelector 的余弦相似度；具体公开 checkpoint 选择已单独记录 |
| 公式（5），第 5 页 | QA→caption | [caption_generation.py:98](../cdviews/caption_generation.py#L98) | 为每个答案生成，匹配时只取第一条 |
| 公式（6），第 6 页 | caption 与视图匹配标签 | [view_matching.py:110](../cdviews/view_matching.py#L110) | A/B/C 通过字符串包含关系解析 |
| 公式（7），第 6 页 | 问题、视图向量的余弦分数 | [ViewSelector.py:332](../cdviews/ViewSelector.py#L332)、[qa_inference.py:106](../cdviews/qa_inference.py#L106) | 训练与推理均使用；训练后接 `(s+1)/2` |
| 公式（8），第 6 页 | BCE 损失 | [ViewSelector.py:329](../cdviews/ViewSelector.py#L329) | 代码标签/预测顺序正确，论文写法疑似笔误 |
| 公式（9），第 6 页 | 按距离阈值保留视图 | [qa_inference.py:126](../cdviews/qa_inference.py#L126) | `D=T` 的边界与正文不同 |
| 公式（10），第 6 页 | 四元数旋转距离 | [view_distance_calculation.py:33](../cdviews/view_distance_calculation.py#L33) | 增加 `clip` 数值保护 |
| 公式（11），第 6 页 | 欧氏位置距离 | [view_distance_calculation.py:30](../cdviews/view_distance_calculation.py#L30) | 直接对应 |
| 公式（12），第 7 页 | 位置与朝向距离之和 | [view_distance_calculation.py:37](../cdviews/view_distance_calculation.py#L37) | 支持权重，默认均为 1 |
| 公式（13），第 14 页 | 直接使用 QA 做匹配的消融 | 当前匹配脚本未提供开关 | 发布脚本固定走 caption 路线 |

附录中的不同骨干、BridgeQA+LLaVA-OV 微调等实验，也不能仅凭 `llava/` 有通用训练功能就认定已有可直接运行的复现实验入口。

**11. 视觉 mask 是当前源码最需要单独核对的实现疑点。**

[ViewSelector.py:50](../cdviews/ViewSelector.py#L50)：

```python
    def forward(self, image_embedding, text_embedding):
        bs, N_i, _ = image_embedding.shape
        text_mask = self.make_mask(text_embedding).view(bs, 1, 1, -1)
        image_mask = torch.ones(bs, 1, 1, N_i).bool().to(image_embedding.device)

        image_feat = self.image_feat_linear(image_embedding)
        text_feat = self.text_feat_linear(text_embedding)
```

`image_mask` 对所有视觉 token 都为 True，而后续 mask 的语义是“True 的位置需要被遮住”。例如注意力函数：

```python
    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
```

视觉池化也有同样行为：

```python
    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
```

这会产生明确的数值后果：视觉自注意力 logits 和视觉池化 logits 都被统一替换为 `-1e9`。在当前选择器使用的 float32 下，对整排相同的有限值做 softmax 得到均匀权重，而不是全零权重。

例如原 logits 是 `[-0.3,1.2,4.0]`，掩码后为 `[-1e9,-1e9,-1e9]`，softmax 为 `[1/3,1/3,1/3]`。视觉池化的可学习打分会失效，成为等权汇聚；视觉自注意力中由 Q/K 得到的内容差异也被覆盖。

这不等于模型完全看不到图像：V 路径、残差连接仍保留视觉信息；SGA 中视觉查询文本的交叉注意力使用文本 mask，也仍然存在。准确判断是“发布实现的部分注意力/池化行为与通常预期不符”，不能扩大成“整套方法不工作”。

如进行后续复现，应区分保持发布代码与 checkpoint 的原始基线，和修正 mask 后重新训练的实验。本次只是解释与核对，没有改动该实现。

**12. 缓存、评测和运行约定会影响对结果的理解。**

| 实现细节 | 源码证据 | 对理解或复现的影响 |
|---|---|---|
| 已有 ranking 文件时跳过重新排序 | [qa_inference.py:163](../cdviews/qa_inference.py#L163) | 换选择器 checkpoint 后若仍读旧 ranking，最终结果并未评测新选择器 |
| 即使复用 ranking，仍先加载 selector 权重 | [qa_inference.py:148](../cdviews/qa_inference.py#L148) | 当前脚本并不是完全无需 selector checkpoint 的缓存入口 |
| 当前测试 split 写在代码中 | [qa_inference.py:155](../cdviews/qa_inference.py#L155) | ScanQA 默认跑两个 test，SQA 跑 test；没有直接的 val 命令行开关 |
| NMS 阈值默认 0.5，调用未传入新阈值 | [qa_inference.py:126](../cdviews/qa_inference.py#L126)、[qa_inference.py:195](../cdviews/qa_inference.py#L195) | 调节 `input_views` 不会同时调节 NMS 阈值 |
| 提取特征时硬编码每批 100 张 | [visual_feature_processing.py:88](../cdviews/visual_feature_processing.py#L88) | 和训练的 batch_size=8 是不同概念 |
| 不自动补满被 NMS 抑制后的视图数 | [qa_inference.py:130](../cdviews/qa_inference.py#L130) | `input_views=9` 是最多 9 张 |
| 声明了 num-chunks/chunk-idx，但主循环没有使用 | [qa_inference.py:244](../cdviews/qa_inference.py#L244) | 不能据这些参数认定已实现多进程数据分片 |
| 选择器训练无 DDP 调用、无验证选模循环 | [train.py:107](../cdviews/train.py#L107) | 默认是简单训练脚本，不是完整分布式实验框架 |
| 推理只保存答案 JSON | [qa_inference.py:232](../cdviews/qa_inference.py#L232) | 本地仓库没有附带本文完整 EM/BLEU/ROUGE/CIDEr 评测入口 |
| SQA 分数存在评测口径说明 | [README.md:168](../README.md#L168) | README 称 56.8 来自官方 LLM-based evaluator，55.6 来自其所引用的 Refined EM 严格匹配脚本 |

最后这一项补充了仅看论文无法获知的信息：对比 SQA 数字时需要同时写明评价脚本和口径，不能把 56.8 与 55.6 当成完全相同指标下的性能变化。此处转述的是本地 README 的说明，没有独立重新运行外部评测器。

**13. 建议沿同一个样本读完下面这条路径。**

先读 [QA.yaml](../cfgs/QA.yaml#L1) 理清输入输出，再读 [dataset.py](../cdviews/dataset.py#L46) 看一条问题如何变成多个图文配对。随后读 [ViewSelector.forward](../cdviews/ViewSelector.py#L50)、[SGA.forward](../cdviews/ViewSelector.py#L310)、[BCELoss.forward](../cdviews/ViewSelector.py#L331)，即可掌握选择器主体。

接着读 [train_epoch](../cdviews/train.py#L80) 看监督怎样进入网络，再读 [ranking_views](../cdviews/qa_inference.py#L87)、[viewNMS](../cdviews/qa_inference.py#L126)、[最终生成](../cdviews/qa_inference.py#L193) 看测试链路。最后回看两个 viewAnnotator 脚本，追溯标签从哪里来。

读懂这条主线之后，再深入 `llava_arch.py` 的图像 token 插入逻辑。理解本文贡献不需要先通读全部通用 LLaVA 训练器、服务代码和 TRL。
