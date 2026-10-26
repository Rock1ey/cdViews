"""Read-only checks for ScanQA inference using the authors' view rankings.

Uses Python's standard library. Tensor payload corruption still requires hashes
or an actual load; a valid safetensors header alone cannot establish provenance.
"""

import argparse
import json
import math
import os
import struct
import zipfile
from pathlib import Path


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"文件缺失、为空或软链接失效: {path}")


def check_weight_file(path):
    require_file(path)
    if path.suffix == ".safetensors":
        # The header records every tensor's byte range in the data section.
        with path.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            if not 2 <= header_length <= min(100_000_000, path.stat().st_size - 8):
                raise ValueError(f"无效 safetensors 头: {path}")
            header = json.loads(stream.read(header_length))
        offsets = [entry["data_offsets"] for name, entry in header.items()
                   if name != "__metadata__"]
        end = max((last for first, last in offsets), default=0)
        if any(first < 0 or last < first for first, last in offsets):
            raise ValueError(f"无效 tensor 偏移: {path}")
        if 8 + header_length + end != path.stat().st_size:
            raise ValueError(f"权重大小与头信息不符，可能未下载完整: {path}")
    elif not zipfile.is_zipfile(path):
        # Legacy PyTorch pickle files are possible, but not safely verifiable
        # by this standard-library-only checker.
        raise ValueError(f"不是可识别的 PyTorch ZIP 权重，需在模型环境确认格式: {path}")


def check_model(directory, tokenizer=False):
    config = read_json(directory / "config.json")
    if not isinstance(config, dict):
        raise ValueError("config.json 不是对象")
    if tokenizer:
        read_json(directory / "tokenizer_config.json")
        if (directory / "tokenizer.json").is_file():
            require_file(directory / "tokenizer.json")
        else:
            require_file(directory / "vocab.json")
            require_file(directory / "merges.txt")
    # Follow the format preference of the usual Transformers loading path.
    weight_files = None
    for filename in ["model.safetensors", "model.safetensors.index.json",
                     "pytorch_model.bin", "pytorch_model.bin.index.json"]:
        path = directory / filename
        if not path.exists():
            continue
        if filename.endswith(".index.json"):
            weight_map = read_json(path)["weight_map"]
            weight_files = [directory / name for name in sorted(set(weight_map.values()))]
        else:
            weight_files = [path]
        break
    if not weight_files:
        raise ValueError(f"未找到模型权重或分片索引: {directory}")
    for path in weight_files:
        check_weight_file(path)
    size = sum(path.stat().st_size for path in weight_files) / 1024**3
    legacy = any(path.suffix == ".bin" for path in weight_files)
    suffix = "；BIN 仅验证 ZIP 容器可识别，需后续实际加载" if legacy else "；分片长度检查通过"
    return f"{len(weight_files)} 个权重文件，{size:.2f} GiB{suffix}"


def check_annotation(path):
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("标注必须是非空列表")
    required = {"scene_id", "question_id", "question"}
    if any(not required <= row.keys() for row in rows):
        raise ValueError("标注缺少 scene_id/question_id/question")
    ids = [str(row["question_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("标注存在重复 question_id")
    return rows


def check_ranking(path, rows):
    ranking = read_json(path)
    if not isinstance(ranking, dict):
        raise ValueError(f"排序文件顶层应为对象，实际为 {type(ranking).__name__}: {path}")
    missing = []
    invalid = []
    for row in rows:
        key = str(row["question_id"])
        if key not in ranking:
            missing.append(key)
            continue
        views = ranking[key]
        if not isinstance(views, list) or not views or not all(isinstance(v, str) and v for v in views):
            invalid.append(key)
    if missing or invalid:
        samples = {key: repr(ranking[key])[:180] for key in list(ranking)[:3]}
        raise ValueError(
            f"标注 {len(rows)} 条，排序 {len(ranking)} 个键；"
            f"缺少 question_id {len(missing)} 个（示例 {missing[:3]}）；"
            f"帧列表格式异常 {len(invalid)} 个（示例 {invalid[:3]}）；"
            f"排序内容示例 {samples}。先核对标注与排序的 ID/格式，不要直接重编号。"
        )
    return ranking


def check_frames(frames, rows, ranking):
    if not frames.is_dir():
        raise ValueError(f"图像目录不存在；可能尚未解压或 --frames 指错: {frames}")
    pairs = {(row["scene_id"], frame) for row in rows
             for frame in ranking[str(row["question_id"]) ]}
    for scene, frame in sorted(pairs):
        require_file(frames / scene / "color" / frame)
        pose = frames / scene / "pose" / (Path(frame).stem + ".txt")
        require_file(pose)
    # The implementation reads every pose in the scene, including poses not
    # explicitly referenced by the ranking. Validate all of them.
    for scene in sorted({row["scene_id"] for row in rows}):
        for pose in (frames / scene / "pose").glob("*.txt"):
            values = [line.split() for line in pose.read_text().splitlines()]
            if len(values) != 4 or any(len(row) != 4 for row in values):
                raise ValueError(f"pose 不是 4×4: {pose}")
            if not all(math.isfinite(float(x)) for row in values for x in row):
                raise ValueError(f"pose 含非有限数值: {pose}")
    return f"{len(pairs)} 个场景/帧组合存在，位姿格式通过；未逐张解码图片"


def check_archive(path, full_crc):
    require_file(path)
    if Path(str(path) + ".aria2").exists():
        raise ValueError(f"仍有 .aria2 控制文件；确认下载进程是否完成: {path}")
    with zipfile.ZipFile(path) as archive:
        count = len(archive.infolist())
        if full_crc:
            damaged = archive.testzip()
            if damaged:
                raise ValueError(f"ZIP CRC 检查失败: {damaged}")
    return f"{count} 个条目；" + ("全量 CRC 通过" if full_crc else "目录可读，尚未进行全量 CRC")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ["CDV_DATA"]) if os.environ.get("CDV_DATA") else None)
    parser.add_argument("--splits", nargs="+", default=["val"],
                        choices=("train", "val", "test_w_obj", "test_wo_obj"),
                        help="Default: val; train checks annotations only, not selector training readiness")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--zip", dest="archive", type=Path)
    parser.add_argument("--full-zip-check", action="store_true",
                        help="对 ScanNet ZIP 全量解压到内存做 CRC 检查，不写出数据")
    args = parser.parse_args()
    if args.root is None:
        parser.error("Set CDV_DATA or pass --root")
    root = args.root
    annotation_dir = args.annotations or root / "ScanQA/ScanQA_v1.0"
    frames = args.frames or root / "scannetv2/frames_square"
    hf_home = args.hf_home or root / "hf_cache"
    archive = args.archive or root / "downloads/scannet_train_images.zip"
    failures = []

    def report(label, function):
        # Continue after each failure so a single run lists all missing assets.
        try:
            value = function()
            detail = value if isinstance(value, str) else "检查通过"
            print(f"[PASS] {label}: {detail}", flush=True)
            return value
        except Exception as error:
            failures.append(label)
            print(f"[FAIL] {label}: {error}", flush=True)
            return None

    print("范围：ScanQA 作者排序 + 作者 selector 的推理路线；只读、无网络、无 GPU。", flush=True)
    for split in args.splits:
        rows = report(f"标注 {split}", lambda split=split: check_annotation(
            annotation_dir / f"ScanQA_v1.0_{split}.json"))
        if rows is None:
            continue
        print(f"       问题数: {len(rows)}", flush=True)
        if split == "train":
            continue
        ranking = report(f"排序 {split}", lambda split=split, rows=rows: check_ranking(
            root / f"ScanQA_{split}_view_ranking.json", rows))
        if ranking is not None:
            report(f"图像/位姿 {split}", lambda: check_frames(frames, rows, ranking))
        else:
            print(f"[SKIP] 图像/位姿 {split}: 排序未通过，尚未验证引用帧。", flush=True)

    selector = root / "model/ScanQA_view_selector.pth"

    def selector_check():
        check_weight_file(selector)
        with zipfile.ZipFile(selector) as checkpoint:
            damaged = checkpoint.testzip()
            if damaged:
                raise ValueError(f"checkpoint ZIP CRC 失败: {damaged}")
        return "容器 CRC 通过；尚未检查 model 键和网络参数匹配"

    report("selector 权重", selector_check)
    report("LLaVA", lambda: check_model(root / "model/llava-onevision-qwen2-7b-ov", tokenizer=True))

    def siglip_check():
        cache = hf_home / "hub/models--google--siglip-so400m-patch14-384"
        revision = (cache / "refs/main").read_text().strip()
        return check_model(cache / "snapshots" / revision)

    report("SigLIP 缓存", siglip_check)
    if archive.exists() or args.full_zip_check:
        report("ScanNet 压缩包", lambda: check_archive(archive, args.full_zip_check))
    else:
        print("[INFO] 未保留默认路径的 ZIP；若图像/位姿已通过，不必为此重新下载。", flush=True)

    for location in [root / "model/llava-onevision-qwen2-7b-ov", hf_home / "hub"]:
        unfinished = list(location.rglob("*.incomplete"))
        if unfinished:
            print(f"[WARN] {location}: {len(unfinished)} 个 .incomplete 文件；可能是其他格式/旧下载残留。", flush=True)

    print("\n结论：" + (f"{len(failures)} 项失败，请按 FAIL 定位。" if failures else
          "以上结构/长度检查通过；仍需离线模型加载及少量问答试跑。"), flush=True)
    print("本检查不证明模型与作者版本一致，也不替代权重哈希校验。")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
