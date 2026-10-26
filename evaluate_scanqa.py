"""Evaluate labeled ScanQA predictions with the metrics reported by cdViews.

Protocol: https://github.com/ATR-DBI/ScanQA/blob/main/scripts/score.py
EM uses exact membership in the original reference answer list, without extra
normalization. Caption metrics use PTBTokenizer + pycocoevalcap; all are x100.
This is a standalone adaptation for cdViews JSON files, not the EvalAI server.
Needs pycocoevalcap==1.2 and Java. No NLTK data downloads or GPT judges are used.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil


def load_aligned(annotations, predictions):
    gold = json.loads(annotations.read_text())
    answers = json.loads(predictions.read_text())
    if not isinstance(gold, list) or not gold or not isinstance(answers, list):
        raise ValueError("Annotations must be a nonempty list; predictions must be a list")
    if any(not isinstance(row.get("answers"), list) or not row["answers"]
           or not all(isinstance(answer, str) for answer in row["answers"]) for row in gold):
        raise ValueError("Ground-truth answers are missing/invalid. Public test annotations cannot be scored locally; use val or official test evaluation.")
    gold_ids = [str(row["question_id"]) for row in gold]
    pred_ids = [str(row["question_id"]) for row in answers]
    if len(set(gold_ids)) != len(gold_ids) or len(set(pred_ids)) != len(pred_ids):
        raise ValueError("Duplicate question_id in annotations or predictions")
    if set(gold_ids) != set(pred_ids):
        raise ValueError(f"Question IDs do not match: missing={len(set(gold_ids)-set(pred_ids))}, extra={len(set(pred_ids)-set(gold_ids))}. Do not evaluate smoke outputs against the full validation set.")
    by_id = {str(row["question_id"]): row for row in answers}
    aligned = []
    for row in gold:
        prediction = by_id[str(row["question_id"])]
        if prediction["scene_id"] != row["scene_id"]:
            raise ValueError(f"Scene mismatch: {row['question_id']}")
        candidates = prediction.get("answer_top10")
        if not isinstance(candidates, list) or not candidates or not all(isinstance(x, str) for x in candidates):
            raise ValueError(f"Invalid answer_top10: {row['question_id']}")
        aligned.append(prediction)
    return gold, aligned


def exact_match(gold, predictions):
    hits = sum(pred["answer_top10"][0] in row["answers"]
               for row, pred in zip(gold, predictions))
    return 100.0 * hits / len(gold)


def caption_metrics(gold, predictions):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    tokenizer = PTBTokenizer()
    references = {str(row["question_id"]): [{"caption": answer} for answer in row["answers"]]
                  for row in gold}
    hypotheses = {str(row["question_id"]): [{"caption": row["answer_top10"][0]}]
                  for row in predictions}
    references = tokenizer.tokenize(references)
    hypotheses = tokenizer.tokenize(hypotheses)
    metrics = {}
    for scorer, names in [(Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
                          (Rouge(), ["ROUGE-L"]), (Cider(), ["CIDEr"])]:
        score, _ = scorer.compute_score(references, hypotheses)
        values = score if isinstance(score, (list, tuple)) else [score]
        metrics.update({name: float(value) * 100.0 for name, value in zip(names, values)})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--em-only", action="store_true", help="Exact match only; no third-party dependencies")
    args = parser.parse_args()
    if args.output.resolve() in (args.annotations.resolve(), args.predictions.resolve()):
        parser.error("Output must not overwrite annotations or predictions")
    gold, predictions = load_aligned(args.annotations, args.predictions)
    metrics = {"EM@1": exact_match(gold, predictions)}
    empty = sum(not row["answer_top10"][0].strip() for row in predictions)
    print(f"Aligned questions: {len(gold)}; empty top-1 answers: {empty}", flush=True)
    package_version = None
    if not args.em_only:
        if shutil.which("java") is None:
            raise RuntimeError("Java is required by PTBTokenizer; install OpenJDK in the evaluation environment")
        package_version = version("pycocoevalcap")
        metrics.update(caption_metrics(gold, predictions))
    report = {
        "questions": len(gold), "empty_top1_answers": empty,
        "metrics_x100": metrics,
        "protocol": "ScanQA score.py: raw exact match, PTBTokenizer, Bleu(4), Rouge, Cider; not an EvalAI result",
        "pycocoevalcap_version": package_version,
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
