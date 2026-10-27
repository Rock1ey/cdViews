"""Download only BLIP retrieval's PyTorch checkpoint and processor files."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="Resolved to an immutable commit before download")
    args = parser.parse_args()
    # activate.sh enables offline inference. Lift it only in this downloader
    # process, before importing huggingface_hub; the parent shell stays offline.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import HfApi, snapshot_download

    repo = "Salesforce/blip-itm-large-coco"
    revision = HfApi().model_info(repo, revision=args.revision).sha
    files = ["pytorch_model.bin", "config.json", "preprocessor_config.json",
             "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "tokenizer.json"]
    print(f"Downloading {repo}@{revision} to {args.output}", flush=True)
    snapshot_download(repo_id=repo, revision=revision, allow_patterns=files,
                      local_dir=str(args.output), local_dir_use_symlinks=False, max_workers=2)
    (args.output / "download.json").write_text(json.dumps(
        {"repo_id": repo, "revision": revision, "files": files}, indent=2))
    print("BLIP download complete. Inference can stay offline.", flush=True)


if __name__ == "__main__":
    main()
