# cdViews 复现指南：4 × RTX A6000、Miniforge、国内网络

> 历史环境安装记录。当前实验统一为单卡、ScanQA 验证集；机器路径和最新运行命令见 [服务器指南](../SERVERS.md)。不要直接照搬本文中的历史多卡或测试集命令。

本文针对当前仓库 `e1df64c5490c75d117f0ece5c334a6d818cf5d83`，即 **3D Question Answering via only 2D Vision-Language Models**。假设服务器是 Linux x86_64，A6000 为每卡 48 GB 的 RTX A6000。以下是根据源码和上游文档核对的操作方案；尚未在你的服务器安装或完成 GPU 实验，不能把它当成已验证的论文分数复现记录。

## 1. 先确定复现范围

推荐依次完成：

1. **作者排序文件 + 作者 viewSelector 权重 + LLaVA-OV 推理**：先确认环境、图像、相机位姿和评测流程可用。原脚本即使复用排序文件，仍会加载 selector 权重，但不会遍历特征 DataLoader，因此不需要预提取图像特征。
2. **作者 viewSelector 权重重新排序 + 推理**：需要额外提取视觉特征；结果目录中不能残留作者排序文件，否则代码直接跳过排序。
3. **重新训练 viewSelector + 重新排序 + 推理**：额外下载作者的 view labeling；可跳过成本很高的 caption generation 和 view matching。
4. 如要复现全部数据构建过程，再自行运行 caption generation 和 view matching。

该方法不微调 7B LLaVA，但会训练 viewSelector。当前 `train.py` 没有 DDP，`qa_inference.py` 虽声明 `--num-chunks/--chunk-idx`，却没有使用它们分片。不能直接加 `torchrun --nproc_per_node=4`，否则会重复计算、争抢显存或覆盖结果。

## 2. 网络策略和目录

| 内容 | 首选来源 | 失败时处理 |
|---|---|---|
| Miniforge 安装器、conda-forge 包 | 清华镜像 | 已安装 Miniforge 可跳过安装器 |
| Python 包、NVIDIA pip 运行时依赖 | 清华 PyPI | 检查 DNS/HTTPS 或切换单位可用 PyPI 镜像 |
| PyTorch CUDA wheels | 交大镜像 | 清华 PyPI 对应 Linux wheel，或官方源经代理 |
| LLaVA-OV、SigLIP | HF-Mirror | 降低并发、断点重试，或中转机器下载 |
| Transformers 固定 Git 提交 | GitHub 官方归档，经代理/中转 | PyPI 镜像不能代理 Git URL |
| FlashAttention | 清华源码包本地编译；或官方匹配 wheel 经代理 | pip 换源不影响其内部 GitHub wheel 下载 |
| 作者权重、排序、标签；ScanQA 标注 | Google Drive，经代理/中转 | 未核实到作者认可的国内镜像 |
| ScanNet frames | TUM 官方服务器，断点下载 | 代理/中转；未核实到等价国内镜像 |
| SQA 标注 | Zenodo 官方，断点下载 | 代理/中转；HF_ENDPOINT 不会代理 Zenodo |

来源：[清华 Conda 文档](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)、[清华 PyPI 文档](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)、[交大 PyTorch 文档](https://mirrors.sjtug.sjtu.edu.cn/docs/pytorch-wheels)、[HF-Mirror 文档](https://hf-mirror.com/)。镜像入口已核实，不代表从你的服务器到每个大文件的链路已经实测。

以下命令在服务器的 **Bash** 终端执行。先进入服务器上的代码仓库根目录，再设置路径；`/data/qa` 必须是你有写权限、空间充足的数据盘目录，可以改为其他绝对路径。

```bash
export CDV_REPO="$PWD"
export CDV_DATA=/data/qa
mkdir -p "$CDV_DATA"/{model,ScanQA,SQA,scannetv2,downloads,runs}
mkdir -p "$CDV_DATA/scannetv2"/{llava_ov_feats,view_distance}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$CDV_DATA/hf_cache"
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10
```

把这些变量保存为自己的环境脚本，每次登录后重新加载。数据齐全后再打开离线模式，不要在下载前设置 `HF_HUB_OFFLINE=1`。

如服务器已有可用 HTTP 代理，使用真实端口，例如：

```bash
# 仅当服务器的这个端口确实有代理服务时设置
export https_proxy=http://127.0.0.1:7890
export http_proxy="$https_proxy"
export no_proxy=localhost,127.0.0.1,.tuna.tsinghua.edu.cn,mirror.sjtu.edu.cn,hf-mirror.com
```

你电脑的 `127.0.0.1` 不是服务器的 `127.0.0.1`。无服务器代理时，在能访问外网的机器下载，再用 `rsync -avP 下载目录/ 用户@服务器:/data/qa/对应目录/` 传输。HF 缓存包含软链接，中转时保留整个缓存目录结构；只拷贝 snapshots 子目录容易漏文件。不要关闭 SSL 校验。

## 3. Miniforge 和基础环境

已有 Miniforge：直接执行 `source "$HOME/miniforge3/etc/profile.d/conda.sh"`（按实际安装路径调整）。没有安装时，可用已核实的清华镜像安装器：

```bash
cd "$CDV_DATA/downloads"
curl -fL --retry 5 -C - -O \
  https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
cd "$CDV_REPO"

conda create -n cdViews --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  python=3.10 pip git curl unzip -y
conda activate cdViews
python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0 \
  packaging==24.0 ninja==1.11.1.1 psutil==5.9.8
```

使用完整 channel URL 和 `--override-channels`，避免误走 Miniforge 默认国外 channel，也无需覆盖现有 `.condarc`。

先检查 GPU 和驱动：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

推荐沿用仓库的 **Python 3.10、PyTorch 2.1.2、torchvision 0.16.2、CUDA 12.1 wheel**。CUDA 12.x 的驱动兼容范围见 [NVIDIA 官方说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)；建议服务器使用支持 A6000 的维护中驱动，例如 535 或更新分支，已有可用较新驱动通常无需降级。`nvidia-smi` 顶部 CUDA 数字表示驱动能力，不代表安装了对应 `nvcc`。

## 4. 安装 Python 依赖

**不要直接执行 `pip install -r requirements.txt`。** 文件是作者环境导出，包含 `byted*` 内部包、重复包和版本冲突。也不建议第一步使用 `pip install -e '.[train]'`：它会访问 GitHub 并拉入本任务不需要的训练框架。

先安装 GPU 包。版本组合来自 [PyTorch 历史版本说明](https://docs.pytorch.org/get-started/previous-versions/)，下载入口采用交大镜像：

```bash
python -m pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
  --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121/ \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

若交大缺失旧 wheel 或不可达，在 Linux x86_64/Python 3.10 下可尝试清华 PyPI 的对应发布：

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

无论采用哪种安装方式，下面的验证必须显示 CUDA 12.1 和 CUDA 可用；不能把成功安装 CPU 包当作成功配置 GPU 环境。

```bash
python -m pip install \
  numpy==1.26.1 scipy==1.13.0 pandas==2.2.2 \
  pillow==10.3.0 PyYAML==6.0.1 easydict==1.13 \
  accelerate==0.29.3 tokenizers==0.15.2 huggingface-hub==0.23.5 \
  safetensors==0.4.3 sentencepiece==0.1.99 protobuf==3.20.3 \
  einops==0.6.1 einops-exts==0.0.4 \
  requests==2.31.0 tqdm==4.66.2 shortuuid==1.0.13 \
  ftfy==6.2.0 httpx==0.24.0 gdown==5.2.0
```

这是针对 `cdviews/*.py` 主流程的精简依赖，不是全部 LLaVA 工具的环境。DeepSpeed、bitsandbytes、Gradio、wandb、ENet、GLoVE、点云训练框架不是这条 FP16 推理和 selector 训练路线的必需项。后续如用仓库其他工具，再按该工具添加依赖。

### Transformers 必须固定源码提交

仓库 `pyproject.toml` 指定提交 `1c39974a4c4036fd641bc1191cc32799f85715a4`，其版本号为 `4.40.0.dev0`。不要直接升级最新 Transformers，也不要把 PyPI 的 `4.40.0` 当作相同源码。该提交的 tokenizers 要求是 `>=0.14,<0.19`，此处固定 `0.15.2`。[提交的依赖定义](https://raw.githubusercontent.com/huggingface/transformers/1c39974a4c4036fd641bc1191cc32799f85715a4/setup.py)

```bash
curl -fL --retry 5 -C - \
  https://codeload.github.com/huggingface/transformers/tar.gz/1c39974a4c4036fd641bc1191cc32799f85715a4 \
  -o "$CDV_DATA/downloads/transformers-1c39974.tar.gz"

python -m pip install "$CDV_DATA/downloads/transformers-1c39974.tar.gz"
cd "$CDV_REPO"
python -m pip install -e . --no-deps --no-build-isolation
```

归档必须经可用代理或中转下载。清华 PyPI 和 `HF_ENDPOINT` 都不能代理 `codeload.github.com`。当前没有核实可用且保持该提交一致的国内官方源码镜像，因此不提供未经核实的 GitHub 代理域名。

### FlashAttention 2.5.7

`llava/model/builder.py` 默认使用 `flash_attention_2`。PyPI 换源后，FlashAttention 的构建脚本仍会尝试从 GitHub Releases 获取 wheel；这一行为在 [2.5.7 的 setup.py](https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.5.7/setup.py) 中明确写死。

**有 CUDA 12.1 Toolkit 和 GCC 时，建议从清华源码包本地编译：**

```bash
# 只在此目录实际存在时这样设置，或先使用集群提供的 CUDA module
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
gcc --version

FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
  python -m pip install flash-attn==2.5.7 \
  --no-build-isolation --no-deps --no-binary=flash-attn
```

建议 GCC 11/12，确保和 CUDA 编译器兼容；内存不足将 `MAX_JOBS` 降到 2。pip 的 CUDA 运行时依赖不包含完整 nvcc 工具链。

没有 Toolkit 时有两种选择：

* 下载 [FlashAttention v2.5.7 官方 release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.7) 的匹配 wheel，经代理/中转传入服务器后 `python -m pip install /绝对路径/实际文件名.whl --no-deps`。需匹配 Python cp310、Linux x86_64、torch2.1、CUDA 12 系列及 `torch._C._GLIBCXX_USE_CXX11_ABI`。2.5.7 的 CUDA 12 wheel 名称可能标 `cu122`，这是该版本上游构建方式，不要自己拼造未经核实的文件地址。
* 为本地编译安装 Toolkit：可用管理员提供的 `/usr/local/cuda-12.1`，或者 NVIDIA 官方 conda channel（这部分无已核实的清华 nvidia 镜像，可经代理）：

```bash
conda install --override-channels \
  -c https://conda.anaconda.org/nvidia/label/cuda-12.1.1 \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  cuda-toolkit=12.1.1 -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
```

如果只想先排查环境，可在调用 `load_pretrained_model` 时显式传 `attn_implementation="sdpa"`，从而绕过 FA2。但这属于注意力后端变更，应单独记录；正式对齐作者路线优先使用 FA2。

安装后检查：

```bash
python -m pip check
python - <<'PY'
import torch, torchvision, transformers, tokenizers, flash_attn
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('torchvision:', torchvision.__version__)
print('transformers:', transformers.__version__)
print('tokenizers:', tokenizers.__version__, 'flash_attn:', flash_attn.__version__)
assert torch.cuda.is_available()
assert torch.version.cuda == '12.1'
print('visible GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## 5. 模型权重：必须准备两套 Hugging Face 模型

| 模型 | 用途 | 必需性 |
|---|---|---|
| `lmms-lab/llava-onevision-qwen2-7b-ov` | 问答、特征提取、自动标注 | 所有模型运行路线必需 |
| `google/siglip-so400m-patch14-384` | 视觉编码器的初始化加载 | 当前代码加载路径必需 |
| `ScanQA_view_selector.pth` | ScanQA 视图选择器 | 使用作者权重时必需 |
| `SQA_view_selector.pth` | SQA 视图选择器 | 使用作者权重时必需 |

无需再下载基础 `Qwen/Qwen2-7B-Instruct`；使用的是完整 LLaVA checkpoint。SigLIP 名称来自 [LLaVA 官方模型配置](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov/blob/main/config.json)。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$CDV_DATA/hf_cache"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_ENABLE_HF_TRANSFER

python -u - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(os.environ['CDV_DATA'])
snapshot_download(
    repo_id='lmms-lab/llava-onevision-qwen2-7b-ov',
    local_dir=str(root / 'model/llava-onevision-qwen2-7b-ov'),
    endpoint=os.environ['HF_ENDPOINT'], max_workers=2,
)
# 故意不指定 local_dir：原代码用 repo_id 从标准缓存加载 SigLIP。
snapshot_download(
    repo_id='google/siglip-so400m-patch14-384',
    cache_dir=str(Path(os.environ['HF_HOME']) / 'hub'),
    endpoint=os.environ['HF_ENDPOINT'], max_workers=2,
)
PY
```

遇到超时重复执行同一命令；不要删除未完成文件。Hub 0.23.5 的 `huggingface-cli download` 不支持 `--max-workers`，这里使用支持 `max_workers=2` 的 Python 接口。不要为使用新版 `hf` 命令升级整个研究环境。

**不要直接把 config.json 的 `mm_vision_tower` 改成本地路径。** 当前 `multimodal_encoder/builder.py` 先判断“路径是否存在”，会把本地 SigLIP 误分配给 CLIPVisionTower。保留原始 repo_id 并预热标准缓存即可；如一定要用本地路径，须另行修正编码器类型判断顺序。

建议为模型和临时缓存预留约 40–60 GB，这是部署余量估算，并非精确下载大小。下载完成后记录 Hub revision 和文件校验值，固定实验资产。

## 6. 数据、作者权重和中间产物

### 6.1 ScanQA 标注

需要 `ScanQA_v1.0_train.json`、`ScanQA_v1.0_val.json`、`ScanQA_v1.0_test_w_obj.json`、`ScanQA_v1.0_test_wo_obj.json`。只跑测试推理可只准备对应 test 标注；建议四个都保留。

作者提供的是 [Google Drive 文件夹](https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2)，可经代理执行：

```bash
gdown --folder 'https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2' \
  -O "$CDV_DATA/ScanQA"
```

检查是否多了一层子目录，配置须指向实际文件。gdown 是下载工具，本身不提供网络镜像；Google Drive 限流/权限问题也不会通过更换 PyPI 源解决。

### 6.2 ScanNet RGB 帧与相机位姿

ScanQA 和 SQA 共用 ScanNet 场景。对当前代码，RGB `color/*.jpg` 和对应 `pose/*.txt` 是必要输入；viewNMS 实现计算的是相机位置与朝向的组合距离。当前主流程没有读取 depth、网格、ENet 或 GLoVE，能取得完整 RGB/pose 帧包时无需再为主流程下载这些资产。

```bash
curl -fL --retry 10 --connect-timeout 30 -C - \
  http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip \
  -o "$CDV_DATA/downloads/scannet_train_images.zip"
unzip -l "$CDV_DATA/downloads/scannet_train_images.zip" | head -40
unzip "$CDV_DATA/downloads/scannet_train_images.zip" -d "$CDV_DATA/scannetv2"
```

这是 cdViews/ScanQA 引用的官方帧包，不是已经验证可以替换成其他 Hugging Face 数据集的资产。包名里的 train 不代表只覆盖 ScanQA train；应按问答 JSON 的 scene_id 检查覆盖情况。

源码要求最终布局：

```text
/data/qa/scannetv2/frames_square/
  scene0000_00/
    color/0.jpg ...
    pose/0.txt ...
  scene0000_01/...
```

实际压缩包若使用另一层目录，令 `image_folder` 指向真正包含 `scene*` 的目录，或建立软链接；不要未经检查就认为压缩包会自动生成 `frames_square`。README 没有提供明确的方形帧预处理脚本；如果原图尺寸与作者的 `frames_square` 设定不同，应先向作者核实裁剪/缩放规则再声称严格复现。推理会调用 SigLIP 的预处理，但这不能证明此前裁剪完全一致。必须保留排序/标签文件引用的原始帧名。

如果帧包不可用，回到 ScanNet 官方申请/下载途径取得原始 RGB-D 扫描并提取 RGB 和 pose；这还需要对齐作者的采样间隔与文件命名，不能随意重新采样替代原帧包。

### 6.3 SQA 标注

官方 [Zenodo 记录](https://zenodo.org/records/7792397) 提供 ScanQA 格式压缩包约 2.8 MB：

```bash
curl -fL --retry 10 -C - \
  'https://zenodo.org/records/7792397/files/ScanQA_format.zip?download=1' \
  -o "$CDV_DATA/downloads/ScanQA_format.zip"
printf '7f96e0bfec9d26f2c8fd43ccdf42d53f  %s\n' \
  "$CDV_DATA/downloads/ScanQA_format.zip" | md5sum -c -
unzip -l "$CDV_DATA/downloads/ScanQA_format.zip"
unzip "$CDV_DATA/downloads/ScanQA_format.zip" -d "$CDV_DATA/SQA"
```

`cfgs/QA.yaml` 期待 `SQA_train.json`、`SQA_val.json`、`SQA_test.json`。上游转换脚本输出 `SQA_balanced_{split}.json`，解压后按实际名称配置 YAML，或复制/软链接到期望名称。每项应有 `scene_id/question_id/question/situation/answers`；不要在 `question` 中再次拼接 situation，当前问答推理会自行拼接。

### 6.4 作者发布的权重、排序和标签

以下 ID 和用途来自本仓库 README；需要 Google Drive 可达。下载到指定文件名后，仍要确认下载内容不是 HTML 错误页或另外一层压缩包。

```bash
# 作者 viewSelector 权重：推理脚本总是加载
gdown 1J_A0SZzFjkzgtkFTsLjDH8EO6BVZbDX0 -O "$CDV_DATA/model/ScanQA_view_selector.pth"
gdown 1PHeQ5yeHYEGyN1JFTysh1HP8kvuW-KCn -O "$CDV_DATA/model/SQA_view_selector.pth"

# 快速复现路线：复用视图排序
gdown 151RXXI43CaMRhhv9GWbmnHNQ4s9t9C4w -O "$CDV_DATA/ScanQA_val_view_ranking.json"
gdown 1RQVmMNr0Z6a-CJzgKwkvHq1Vvdo256ud -O "$CDV_DATA/ScanQA_test_w_obj_view_ranking.json"
gdown 1IEedi70-rASTSwS43V5y_-taI3XKpcFp -O "$CDV_DATA/ScanQA_test_wo_obj_view_ranking.json"
gdown 1jq37eEIkQgDy5mzuejt6leScXzkIvhQj -O "$CDV_DATA/SQA_test_view_ranking.json"

# 重新训练 selector 才需要：正负视图标签
gdown 1x832zSViIkFu-3SpKNVoManKzpjXVPT3 -O "$CDV_DATA/ScanQA_view_labeling.json"
gdown 1r0BqRz5uXXxUzWBz3vTXpQflzn2UTW9a -O "$CDV_DATA/SQA_view_labeling.json"

# 仅重新做 view matching 时需要；已有上面标签就不必下载
gdown 1Yo23d61fQApWJIdfAbV67M2X7C_5SEKl -O "$CDV_DATA/ScanQA_QA_captions.json"
gdown 1lTdkHAVsAHScTeCbawbpG9wDKphcfaDL -O "$CDV_DATA/SQA_QA_captions.json"
```

作者提供的 predicted answers 是评测对照材料，不是运行模型的必需输入。可选：ScanQA val `1e6eagh4xj78yisULoHg0amZL15-NH5Av`；test_w_obj `1IZHuLuAfxeJxpkRmTmr6UkW5KL_OubL6`；test_wo_obj `1D9ahw-Va1e1HOy5n690hj0uuhcIiaD9k`；SQA test `1kQjMpr8aldDKrSQROE-GLnLRwLqFiGiw`。请存到独立 `reference_answers/`，避免覆盖自己的生成结果。

## 7. 配置路径、离线加载和单卡试跑

以下在仓库根目录执行，生成两份配置，保留原始配置：

```bash
cd "$CDV_REPO"
python - <<'PY'
import copy, os
from pathlib import Path
import yaml
root = Path(os.environ['CDV_DATA']).resolve()
base = yaml.safe_load(Path('cfgs/QA.yaml').read_text())
def relocate(x):
    if isinstance(x, dict): return {k: relocate(v) for k, v in x.items()}
    if isinstance(x, list): return [relocate(v) for v in x]
    if isinstance(x, str): return x.replace('/data/qa', str(root))
    return x
base = relocate(base)
for dataset in ['ScanQA', 'SQA']:
    cfg = copy.deepcopy(base)
    cfg['dataset'] = dataset
    cfg['work_dir'] = str(root / 'runs' / dataset / 'selector')
    cfg['answers_file'] = str(root / 'runs' / '{}_{}_answers.json')
    Path(f'cfgs/{dataset}.server.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

解压目录和 SQA 名称如不同，编辑生成的 YAML：`ScanQA_file`、`SQA_file`、`image_folder`、`LVLM_ckpt`、`pretrained_view_selector_ckpt`、`ranked_view_file` 等必须对应实际位置。代码使用 `args.LVLM_ckpt`，单独传 README 示例里的 `--model-path` 不能替代该配置。`--cfg_file` 是下划线。

数据齐全之后，离线加载验证：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import os
from pathlib import Path
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
p = str(Path(os.environ['CDV_DATA']) / 'model/llava-onevision-qwen2-7b-ov')
tokenizer, model, processor, context_len = load_pretrained_model(
    p, None, get_model_name_from_path(p))
print(type(model).__name__, model.device, context_len)
print('vision:', type(model.get_vision_tower()).__name__)
PY
```

加载失败若指向 `google/siglip...`，优先检查同一 `HF_HOME` 下是否下载了 SigLIP，以及是否把完整缓存传上了服务器。

先在配置副本中把对应 test JSON 改为前 5 条的小 JSON，输出设置为独立 `smoke` 目录，使用同一完整作者 ranking 文件即可。不要直接修改原始数据集。5 条通过之后，再跑全量：

```bash
cd "$CDV_REPO"
CUDA_VISIBLE_DEVICES=0 python cdviews/qa_inference.py --cfg_file cfgs/ScanQA.server.yaml
CUDA_VISIBLE_DEVICES=1 python cdviews/qa_inference.py --cfg_file cfgs/SQA.server.yaml
```

这两条顺序执行；需要并行则分别在两个终端启动。每个进程只看到一张 GPU，可规避自动模型分配和源码 `.cuda()` 混用引起的设备问题。7B FP16 在 48 GB A6000 上有望单卡运行，但仍应实测 9 视图样本峰值显存，不能把四卡总显存当成单进程可用显存。

当前默认 `temperature=0.2`、`num_beams=1`、`max_new_tokens=1024`、`input_views=9`、viewNMS 阈值 0.5。随机采样会造成分数波动。将 temperature 改为 0 能方便确定性排错，但不再是原脚本默认生成配置，须在结果中标注。

当前 ScanQA 推理只遍历两个 test split。跑 val 需要把 `qa_inference.py` 中 `test_mode` 的 ScanQA 分支改为 `['val']`，或者在单独副本中增加 split 选项；不存在现成 `--split val` 参数。

## 8. 四张 A6000 跑推理：四个独立分片进程

以下做法适用于**已下载作者 ranking 文件**的路线，不需要改主程序。将每个 split 的问答记录分为四份，每张卡处理一份；共用只读 ranking，但各自输出独立答案和距离缓存。

先设置 `CDV_BENCH=ScanQA` 跑两个 ScanQA test split，完成后改为 `SQA` 跑 SQA test：

```bash
cd "$CDV_REPO"
export CDV_BENCH=ScanQA
python - <<'PY'
import copy, json, os
from pathlib import Path
import yaml
bench = os.environ['CDV_BENCH']
cfg = yaml.safe_load(Path(f'cfgs/{bench}.server.yaml').read_text())
modes = ['test_w_obj', 'test_wo_obj'] if bench == 'ScanQA' else ['test']
root = Path(os.environ['CDV_DATA']).resolve() / 'runs' / bench / 'four_gpu'
data = {m: json.loads(Path(cfg[f'{bench}_file'][m]).read_text()) for m in modes}
for m in modes:
    rank = Path(cfg['ranked_view_file'].format(bench, m))
    assert rank.is_file(), f'先下载完整 ranking: {rank}'
for gpu in range(4):
    run = root / f'gpu{gpu}'
    run.mkdir(parents=True, exist_ok=True)
    c = copy.deepcopy(cfg)
    for mode in modes:
        p = run / f'{mode}.json'
        p.write_text(json.dumps(data[mode][gpu::4]))
        c[f'{bench}_file'][mode] = str(p)
    c['answers_file'] = str(run / '{}_{}_answers.json')
    c['view_distance_folder'] = str(run / 'view_distance')
    Path(c['view_distance_folder']).mkdir(exist_ok=True)
    (run / 'QA.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
print(root)
PY

CDV_RUN="$CDV_DATA/runs/$CDV_BENCH/four_gpu"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 \
    python -u cdviews/qa_inference.py --cfg_file "$CDV_RUN/gpu$gpu/QA.yaml" \
    > "$CDV_RUN/gpu$gpu/run.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
test "$failed" -eq 0
```

确认四个进程成功后合并；用原始 question_id 校验完整性，避免把失败分片当作完整实验：

```bash
python - <<'PY'
import json, os
from pathlib import Path
import yaml
bench = os.environ['CDV_BENCH']
cfg = yaml.safe_load(Path(f'cfgs/{bench}.server.yaml').read_text())
root = Path(os.environ['CDV_DATA']) / 'runs' / bench / 'four_gpu'
modes = ['test_w_obj', 'test_wo_obj'] if bench == 'ScanQA' else ['test']
for mode in modes:
    original = json.loads(Path(cfg[f'{bench}_file'][mode]).read_text())
    rows = []
    for gpu in range(4):
        rows += json.loads((root / f'gpu{gpu}/{bench}_{mode}_answers.json').read_text())
    indexed = {str(r['question_id']): r for r in rows}
    assert len(indexed) == len(rows), '重复 question_id'
    assert set(indexed) == {str(r['question_id']) for r in original}, '有遗漏/多余问题'
    merged = [indexed[str(r['question_id'])] for r in original]
    target = root / f'{bench}_{mode}_answers.json'
    target.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(target, len(merged))
PY
```

原始推理每个 split 结束才保存答案，不支持逐题断点续跑。应使用 tmux/作业调度器避免会话中断；长任务可另外实现定期保存，但需保持每个分片输出独立。

## 9. 重新提取特征和训练 viewSelector

不复用作者 ranking 时，需要预提取所有候选视图的 LLaVA 特征：

```bash
cd "$CDV_REPO"
CUDA_VISIBLE_DEVICES=0 python cdviews/visual_feature_processing.py \
  --cfg_file cfgs/ScanQA.server.yaml
```

代码 `batch_size_saving=100`，这是一次送入编码器的图片数。A6000 上 OOM 时先改为 8 或 16；它不等于训练 batch size。该程序遍历 image_folder 下所有场景；两套 QA 使用相同模型、图像和预处理时可以共用同一份 `llava_ov_feats`。程序没有自动跳过已有场景，分卡提取必须先按 scene_id 切分，不能四个进程都遍历整个目录。

特征保存的是 token 序列，不是单个向量。按当前 SigLIP 384/14、3584 通道、约 730 token/视图、FP16 粗算，每视图约 5 MiB，十万视图约 0.5 TiB，仅作容量规划。实际大小以提取一个 scene 的 `.pth` 实测为准；完整特征缓存建议按 TB 级评估空间，不能只预留几十 GB。

训练前建议在 `cdviews/train.py` 的 `load_pretrained_model(...)` 后添加：

```python
model.eval()
model.requires_grad_(False)
```

这是为了落实“只训练 viewSelector”：当前 optimizer 虽只更新 selector，但文本嵌入未显式冻结，会产生无用的大模型参数梯度。冻结和 eval 不改变已加载权重，属于应记录的运行修正。

分别给 ScanQA/SQA 设置独立 `work_dir`，准备对应 view_labeling 后训练：

```bash
CUDA_VISIBLE_DEVICES=0 python cdviews/train.py --cfg_file cfgs/ScanQA.server.yaml
CUDA_VISIBLE_DEVICES=1 python cdviews/train.py --cfg_file cfgs/SQA.server.yaml
```

原始训练设置是 batch_size 8、Adam lr 5e-5、weight_decay 1e-5、10 epochs。单卡显存不足先检查冻结，再降低 batch_size；改变 batch size 的实验需单独记录。当前训练脚本没有验证集选优，不能假定第 10 个 epoch 一定等于作者发布权重。

要测试自己训练的 checkpoint，修改配置中的 `pretrained_view_selector_ckpt` 指向实际 `.pth`，并将 `ranked_view_file` 改为**全新实验目录**，例如 `/data/qa/runs/retrained/{}_{}_view_ranking.json`，提前创建父目录。否则会读取作者旧 ranking，造成“换了 checkpoint 但实际上没评测新 selector”的假象。

如果要从头生成训练标签，才额外运行：

```bash
CUDA_VISIBLE_DEVICES=2 python cdviews/caption_generation.py --cfg_file cfgs/ScanQA.server.yaml
CUDA_VISIBLE_DEVICES=2 python cdviews/view_matching.py --cfg_file cfgs/ScanQA.server.yaml
```

SQA 换用对应配置。view matching 会遍历大量 QA × 候选图像组合，通常比直接下载作者标签昂贵得多。

另外，当前 `ViewSelector.py` 用全 True 的 image_mask，并将其传入 `masked_fill(..., -1e9)`。这是值得向作者核实的实现疑点：先保持发布代码和 checkpoint 一致建立基线，不要静默改动后还声称严格复现。

## 10. 评测、完整性和实验记录

该仓库没有附带完整的独立评测管线，`qa_inference.py` 只生成答案。输出格式是 list，每项 `scene_id`、字符串 `question_id` 和 `answer_top10`；后者实际把同一个答案复制十次，不是真正采样十个答案。

* ScanQA：使用 [ScanQA 官方项目](https://github.com/ATR-DBI/ScanQA) 的评测协议；有标注的 val 可本地评测，test 隐藏答案需要对应官方测试评测入口/协议。确认 split 顺序后再对比 README 中两列 test 分数，不能用 val 分数替代 test。
* SQA：本地 README 明确区分 `eval_sqa3d_llm.py` 对应的 **56.8** 与 LLaVA-3D **Refined EM 55.6**。应分别报告评测器名称、版本和选项，不要把 56.8 称作普通严格字符串 EM。[作者评测讨论](https://github.com/fereenwong/cdViews/issues/4)
* [SQA3D LLM 评测说明](https://github.com/SilongYong/SQA3D/blob/master/LLM/README.md) 同时包含自定义答案匹配和可选 `--use_gpt_metric`。仅凭脚本名字不能认定作者一定启用了在线 GPT judge，也不能假定必须下载另一个裁判模型。复现其精确分数还需核对作者实际使用的选项和生成答案到评测输入的转换。

传统文本指标可能额外需要 `pycocoevalcap==1.2`、`pycocotools==2.0.7`、`nltk==3.8.1`、`rouge-score==0.1.2`；METEOR/SPICE 可能还需 Java 和相应资源。建议为官方评测器另建环境，按其文档安装，避免其依赖升级破坏推理环境。没有确认评测脚本和输入格式前，不提供假定可直接运行的评测命令。

跑全量前至少检查：

1. 对应 QA JSON 的 question_id 均有 ranking，ranking 中每个帧都存在 RGB 和同名 pose，pose 为有限数值的 4×4 矩阵。
2. SQA 每项有 situation，数据解压目录和 YAML 一致。
3. 两个 Hugging Face 模型在离线模式下成功加载；selector checkpoint 是包含 `model` 键的字典且可严格加载。
4. 用 5 条问题得到合法输出，再扩大到完整 split；四卡合并时按 question_id 检查无重复、无缺失。
5. 分开保存作者参考答案、自己的答案、不同 checkpoint 的 ranking。

建议保存环境和代码信息：

```bash
python -m pip freeze > "$CDV_DATA/runs/pip-freeze.txt"
conda list --explicit > "$CDV_DATA/runs/conda-explicit.txt"
git rev-parse HEAD > "$CDV_DATA/runs/code-commit.txt"
git diff > "$CDV_DATA/runs/local-changes.patch"
nvidia-smi > "$CDV_DATA/runs/gpu-info.txt"
```

同时保存实验 YAML、模型 revision、下载文件校验值、随机种子、attention 后端、温度和评测器版本。四张 A6000 可用于独立推理分片或两个数据集并行训练；它们不会自动把当前单进程代码变成四卡训练。
