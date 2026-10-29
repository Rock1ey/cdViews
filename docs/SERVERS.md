# 两台服务器与单卡实验

代码通过私有仓库 `https://github.com/Rock1ey/cdViews` 同步；数据、权重、环境与输出保留在各自服务器，不进入 Git。作者原仓库为 `https://github.com/fereenwong/cdViews`，本地保留为 `upstream`。

## 已知路径

| 项目 | 实验室：lab | 优云智算：cloud |
|---|---|---|
| 登录账户 | `pengjiale@shen`；SSH 外网地址未记录 | `ssh -p 23 root@117.50.221.186` |
| 实例 | 实验室共享主机 | `uhost-1wwu92i4j3an`，单张 A800 |
| 已有仓库 | `/home/pengjiale/projects/cdViews` | `/workspace/cdviews-work/repo` |
| 数据根目录 | `/home/pengjiale/datasets/qa` | `/workspace/cdviews-work/qa` |
| 推理环境 | `/home/pengjiale/miniforge3/envs/cdViews` | `/workspace/cdviews-work/envs/cdViews` |
| 环境激活 | Miniforge 的 `conda activate cdViews` | `source /workspace/cdviews-work/envs/cdViews/bin/activate` |
| HF_HOME | `$CDV_DATA/hf_cache` | `$CDV_DATA/hf_cache` |
| 新实验输出 | `$CDV_DATA/runs/lab` | `$CDV_DATA/runs/cloud` |
| 默认运行 | 单卡；每次确认空闲 GPU 编号 | 单卡；一般为 GPU 0，实际以 `nvidia-smi` 为准 |

以上信息来自用户迁移记录，后续重建实例、换 IP 或改目录时更新本文件。优云智算 `/workspace` 已确认位于系统盘的 overlay 文件系统，不是独立数据盘；当时可用约 74GiB，不代表目前剩余容量。实例关机/回收期限以平台当前控制台为准。

Mac 工作目录为 `/Users/leye/Projects/cdViews`，用于开发和版本管理，不运行 CUDA 实验。GitHub 登录账户为 `Rock1ey`。

## 每次登录后的入口

以下在 Bash 中执行；先进入这台机器的实际仓库目录。

```bash
# 实验室；例：确认 GPU 2 空闲之后
cd /home/pengjiale/projects/cdViews
source scripts/activate.sh lab 2

# 租卡服务器
cd /workspace/cdviews-work/repo
source scripts/activate.sh cloud 0
```

两组命令选一组，不要连续执行。脚本从自身位置推导 `CDV_REPO`，因此也支持克隆到其他代码目录；数据和环境按表中路径设置。临时路径变化可在被 Git 忽略的 `scripts/machine.local.sh` 中覆盖 `CDV_DATA`、`CDV_ENV_PREFIX` 等变量，不要提交密码、令牌或私钥。

脚本设置 `CDV_HOST_ID`、`CDV_GPU`、`CDV_RUNS` 和离线 Hugging Face 环境变量。`CDV_GPU` 是传给 launcher 的 GPU 编号；launcher 为子进程设置 `CUDA_VISIBLE_DEVICES`，不需要再人为把同一卡重复映射。如果平台仅暴露单张卡，就使用进程可见的编号。

原迁移工具环境 `cdviews-migrate` 只用于搬运文件，不能用于推理或训练。云端自带 `py310` 也不是已验证的项目环境。运行前检查 `command -v python`，应位于上表对应的 `cdViews` 环境。

## 验证集检查与推理

```bash
nvidia-smi
python check_scanqa_downloads.py --root "$CDV_DATA" --splits val

# 每组实验选择新的名字，smoke/full 是不同子目录。
export CDV_EXPERIMENT="$CDV_RUNS/baseline_v1"
python -u run_scanqa.py --run-dir "$CDV_EXPERIMENT" --stage smoke
python -u run_scanqa.py --run-dir "$CDV_EXPERIMENT" --stage full
```

默认只跑 `val`，只使用 `CDV_GPU` 指定的一张卡。完整输出为 `$CDV_EXPERIMENT/full/ScanQA_val_answers.json`，包含 4,675 个问题。已有同名阶段目录时使用新实验名，避免覆盖；上游脚本不支持逐题断点恢复。

`run_scanqa_dual_gpu.py` 仅作为旧命令兼容入口保留，默认行为与 `run_scanqa.py` 一致。不要使用原始 `qa_inference.py` 的默认测试集入口。

当前入口支持 `--view-selection cdviews|uniform|retrieval|soft|coverage`、`--input-views`、`--seed`、`--temperature`。
默认 cdviews、9 张、seed=42、temperature=0.2；旧入口未固定随机种子，因此历史采样输出不保证重现。
随机均匀采样的完整命令和公平比较说明见 [UNIFORM_BASELINE.md](UNIFORM_BASELINE.md)。
BLIP retrieval 默认 17 张、无 NMS，需要额外检索权重，下载和运行见 [RETRIEVAL_BASELINE.md](RETRIEVAL_BASELINE.md)。
soft 默认 9 张，保留原 cdViews 排名先验、使用 BLIP 内容与位姿软去重，无需训练；见 [SOFT_SELECTION.md](SOFT_SELECTION.md)。
coverage 在 soft 上加入问题加权的集合覆盖增益，默认 9 张、无需训练；见 [COVERAGE_SELECTION.md](COVERAGE_SELECTION.md)。
R2 的冻结特征准备、双卡 DDP 训练、新排名和验证集评测见 [R2_TRAINING.md](R2_TRAINING.md)。
这是用户明确要求的双卡实验，日常其他运行仍默认单卡；`--ranked-view-file` 可指定新 selector 排名。

修改 selector 后必须重新生成排序，不能继续把作者排序作为新模型结果。当前 launcher 读取 `$CDV_DATA/ScanQA_val_view_ranking.json`，并保存实验内候选副本；uniform 使用同一候选集合但忽略排名且关闭 NMS。做新的排序方法时仍需独立管理排序来源。训练脚本原始 `cfgs/QA.yaml` 中仍有旧绝对路径，必须创建适用于当前机器的实验配置后再运行，不要直接启动。

## 本地指标

在安装了 Java 和 `pycocoevalcap==1.2` 的评测环境中执行（已有 `cdviews-eval` 可继续使用；也可将预测传回实验室 CPU 评测）：

```bash
python "$CDV_REPO/evaluate_scanqa.py" \
  --annotations "$CDV_DATA/ScanQA/ScanQA_v1.0/ScanQA_v1.0_val.json" \
  --predictions "$CDV_EXPERIMENT/full/ScanQA_val_answers.json" \
  --output "$CDV_EXPERIMENT/full/metrics.val.json"
```

切换评测环境后保留上述路径变量。`--em-only` 可进行无第三方依赖的 EM 检查，但不能替代完整指标。

## 两台服务器接入新远端

先在每台机器检查已有代码，保留本地修改，不要执行 `git reset --hard` 或 `git clean`：

```bash
git status --short
git remote -v
```

若 `origin` 仍指向作者且尚无 `upstream`：

```bash
git remote rename origin upstream
git remote add origin https://github.com/Rock1ey/cdViews.git
git fetch origin
```

私有仓库需要每台服务器已有的 GitHub 登录凭据；Mac Safari 登录不会自动传到服务器。不要把令牌写进 remote URL。若原目录没有 `.git`，新克隆到旁边目录后再核对本地改动，不要删除旧目录：

```bash
git clone https://github.com/Rock1ey/cdViews.git cdViews-git
```

**第一次同步：** 如果旧服务器有与新仓库同名的未跟踪脚本，先备份到仓库外；`git pull` 会因覆盖风险拒绝执行。已修改的跟踪文件也需先提交到备份分支或保存补丁。存在浅克隆导致的历史问题时，先从作者远端补齐历史，不要强制覆盖。

工作区干净且历史可快进后：

```bash
git switch main
git pull --ff-only origin main
git branch --set-upstream-to=origin/main main
# 新克隆目录或 editable 安装仍指向旧位置时，重新绑定：
python -m pip install -e . --no-deps --no-build-isolation --no-index
```

## 日常同步

Mac 修改、验证、提交并推送；服务器在启动新实验前拉取：

```bash
# Mac：只添加本次确实需要同步的文件
git add <本次修改的文件>
git commit -m "Describe experiment change"
git push origin main

# 实验室 / 云端：先确认没有未保存修改
git status --short
git pull --ff-only origin main
git rev-parse HEAD
```

实验运行期间不拉取代码。服务器临时改代码时先保留修改，通过分支提交、推送再同步回 Mac，不要让两边各自覆盖同名脚本。实验记录保存 Git commit、配置、GPU 型号和输出目录；不同卡结果不保证逐位一致。

## 本地归档

Mac 的 `.local-archive/migration/` 保留迁移脚本与旧迁移指南，不上传 GitHub。`tmp/`、`output/` 中已有文档渲染文件也保留但忽略。数据、模型、环境包、运行输出、Python 缓存均由 `.gitignore` 排除。

论文保留在 `docs/cdviews.pdf`；阅读笔记为 `docs/CODE_PAPER_GUIDE_CN.md`。历史环境搭建记录位于 `docs/setup/REPRODUCE_A6000_CN.md`，其中旧测试集或多卡命令仅供历史参考；当前运行规则以本文件及根目录 `AGENTS.md` 为准。
