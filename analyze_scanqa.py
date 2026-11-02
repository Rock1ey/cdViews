"""Offline ScanQA val comparison and real-view case sheets; never runs a model."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from evaluate_scanqa import load_aligned


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_type(question):
    # Lexical groups, not manually verified reasoning categories.
    words = question.lower().strip().split()
    if words[:2] == ["how", "many"]:
        return "how many"
    return words[0] if words else "empty"


def replay_view_nms(candidates, distances, budget, threshold):
    """Mirror qa_inference.viewNMS, without importing the GPU inference stack."""
    selected, remaining = [], candidates.copy()
    while remaining and len(selected) < budget:
        current = remaining.pop(0)
        selected.append(current)
        sorted_distances = distances.loc[current].sort_values()
        suppressed = set(sorted_distances[sorted_distances < threshold].index.tolist())
        remaining = [image for image in remaining if image not in suppressed]
    return selected


def reconstruct_cdviews(gold, predictions_path, ranking_path, distance_dir, budget, threshold):
    import pandas as pd

    ranking = json.loads(ranking_path.read_text())
    ranking_hash = sha256(ranking_path)
    experiment_path = predictions_path.parent / "experiment.json"
    verified = False
    if experiment_path.exists():
        experiment = json.loads(experiment_path.read_text())
        if experiment.get("method") != "cdviews" or experiment.get("input_views") != budget:
            raise ValueError("cdViews reconstruction method/budget disagrees with experiment.json")
        expected_hash = experiment.get("inputs", {}).get("val_candidates", {}).get("sha256")
        if expected_hash and expected_hash != ranking_hash:
            raise ValueError("Ranking SHA256 disagrees with the original experiment.json; use its original ranking")
        verified = bool(expected_hash)
    by_scene = {}
    for row in gold:
        by_scene.setdefault(row["scene_id"], []).append(row)
    views, distance_sources = {}, {}
    # Load one scene at a time; full scene distance matrices can be large.
    for scene, questions in by_scene.items():
        distance_path = distance_dir / f"{scene}.csv"
        if not distance_path.exists():
            raise FileNotFoundError(f"Missing original view-distance CSV: {distance_path}. "
                                    "Set --view-distance-dir to the distance folder used for cdViews.")
        distances = pd.read_csv(distance_path, index_col=0)
        distance_sources[scene] = {"path": str(distance_path.resolve()), "sha256": sha256(distance_path)}
        for row in questions:
            qid = str(row["question_id"])
            candidates = ranking.get(qid)
            if not isinstance(candidates, list) or not candidates or len(set(candidates)) != len(candidates):
                raise ValueError(f"Missing/empty/duplicate ranking candidates for {qid}")
            if not set(candidates) <= set(distances.index) or not set(candidates) <= set(distances.columns):
                raise ValueError(f"Distance matrix does not cover ranking candidates: {scene}/{qid}")
            views[qid] = {"question_id": row["question_id"], "scene_id": scene,
                          "candidate_count": len(candidates),
                          "images": replay_view_nms(candidates, distances, budget, threshold)}
    source = {"kind": "reconstructed_cdviews_nms", "ranking": str(ranking_path.resolve()),
              "ranking_sha256": ranking_hash, "input_views": budget, "distance_threshold": threshold,
              "ranking_hash_verified_against_experiment": verified,
              "experiment_sha256": sha256(experiment_path) if experiment_path.exists() else None,
              "distance_files": distance_sources,
              "note": "Reconstruction, not a historical trace. Exact recovery requires the original ranking, distances, budget and NMS rule."}
    return views, source


def load_run(name, path, annotations, expected_questions=4675, *, ranking_path=None,
             distance_dir=None, cdviews_budget=9, nms_threshold=0.5):
    gold, predictions = load_aligned(annotations, path)
    if len(gold) != expected_questions:
        raise ValueError(f"Expected {expected_questions} val questions, got {len(gold)}")
    views_path = Path(str(path) + ".views.json")
    views, budget = {}, None
    source = {"kind": "unavailable"}
    if views_path.exists():
        trace = json.loads(views_path.read_text())
        if trace["split"] != "val":
            raise ValueError(f"Non-val view trace: {views_path}")
        rows = trace["questions"]
        views = {str(row["question_id"]): row for row in rows}
        if len(views) != len(rows) or set(views) != {str(row["question_id"]) for row in gold}:
            raise ValueError(f"Incomplete/duplicate view trace IDs: {views_path}")
        for row in gold:
            v = views[str(row["question_id"])]
            if v["scene_id"] != row["scene_id"] or len(v["images"]) != len(set(v["images"])):
                raise ValueError(f"Invalid scene or repeated image: {row['question_id']}")
        budget = trace["input_views"]
        source = {"kind": "recorded", "path": str(views_path.resolve()), "sha256": sha256(views_path)}
    elif ranking_path is not None:
        if distance_dir is None:
            raise ValueError("--view-distance-dir is required to reconstruct missing cdViews traces")
        if cdviews_budget < 1 or not 0 <= nms_threshold < float("inf"):
            raise ValueError("cdViews budget must be positive; NMS threshold must be finite and nonnegative")
        views, source = reconstruct_cdviews(gold, path, ranking_path, distance_dir,
                                            cdviews_budget, nms_threshold)
        budget = cdviews_budget
    return {
        "name": name, "path": path, "predictions": predictions,
        "answers": [row["answer_top10"][0] for row in predictions],
        "hits": [pred["answer_top10"][0] in row["answers"]
                 for row, pred in zip(gold, predictions)],
        "views": views, "budget": budget, "view_source": source,
    }, gold


def compare(gold, before, after):
    rows = []
    for i, item in enumerate(gold):
        qid = str(item["question_id"])
        a, b = before["hits"][i], after["hits"][i]
        category = {(False, True): "fixed", (True, False): "regressed",
                    (True, True): "both_correct", (False, False): "both_wrong"}[a, b]
        av = before["views"].get(qid, {}).get("images")
        bv = after["views"].get(qid, {}).get("images")
        union = set(av) | set(bv) if av is not None and bv is not None else None
        rows.append({
            "before": before["name"], "after": after["name"], "question_id": qid,
            "scene_id": item["scene_id"], "question": item["question"],
            "question_type": question_type(item["question"]),
            "references": json.dumps(item["answers"], ensure_ascii=False),
            "answer_before": before["answers"][i], "answer_after": after["answers"][i],
            "correct_before": int(a), "correct_after": int(b), "category": category,
            "answer_changed": int(before["answers"][i] != after["answers"][i]),
            "view_jaccard": (len(set(av) & set(bv)) / len(union) if union else ""),
            "same_view_set": int(set(av) == set(bv)) if union is not None else "",
            "same_view_order": int(av == bv) if union is not None else "",
            "views_before": json.dumps(av), "views_after": json.dumps(bv),
            "view_source_before": before["view_source"]["kind"],
            "view_source_after": after["view_source"]["kind"],
        })
    counts = Counter(row["category"] for row in rows)
    summary = {"before": before["name"], "after": after["name"], "questions": len(rows),
               **{key: counts[key] for key in ("fixed", "regressed", "both_correct", "both_wrong")},
               "delta_em_pp": 100 * (counts["fixed"] - counts["regressed"]) / len(rows),
               "changed_answers": sum(row["answer_changed"] for row in rows)}
    return rows, summary


def draw_case(row, frames_root, output, font_path=None):
    """Preserve input order; green = shared, orange = present only in that method."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    font = ImageFont.truetype(str(font_path or "DejaVuSans.ttf"), 19)
    width, pad, tile_w, tile_h = 1080, 20, 340, 255
    canvas = Image.new("RGB", (width, 10000), "white")
    draw = ImageDraw.Draw(canvas)
    y = pad

    def text_lines(text):
        nonlocal y
        # Pixel-aware wrapping handles long IDs and paths as well as questions.
        line = ""
        for char in text:
            if char == "\n" or draw.textlength(line + char, font=font) > width - 2 * pad:
                draw.text((pad, y), line, font=font, fill="black")
                y += 27
                line = "" if char == "\n" else char
            else:
                line += char
        draw.text((pad, y), line, font=font, fill="black")
        y += 31

    text_lines(f"{row['question_id']} | {row['scene_id']} | {row['category']}")
    text_lines(row["question"])
    text_lines("References: " + row["references"])
    text_lines("Green: shared view. Orange: unique to this method. Numbers: input order.")
    a, b = json.loads(row["views_before"]), json.loads(row["views_after"])
    shared = set(a or []) & set(b or [])
    missing = []
    for side, images in (("before", a), ("after", b)):
        text_lines(f"{row[side]} | answer: {row['answer_' + side]} | EM: {row['correct_' + side]}")
        text_lines("View source: " + row["view_source_" + side])
        if images is None:
            text_lines("No 2D view trace available for this method.")
            continue
        for j, image_name in enumerate(images):
            x = pad + (j % 3) * (tile_w + 10)
            top = y + (j // 3) * (tile_h + 34)
            path = frames_root / row["scene_id"] / "color" / image_name
            if path.exists():
                with Image.open(path) as source:
                    tile = ImageOps.contain(source.convert("RGB"), (tile_w - 8, tile_h - 8))
                canvas.paste(tile, (x + (tile_w - tile.width) // 2, top + (tile_h - tile.height) // 2))
            else:
                missing.append(str(path))
                draw.text((x + 8, top + 15), "Missing image", font=font, fill="red")
            draw.rectangle((x, top, x + tile_w, top + tile_h),
                           outline="#298352" if image_name in shared else "#c77316", width=4)
            draw.text((x, top + tile_h + 3), f"{j + 1}. {image_name}", font=font, fill="black")
        y += ((len(images) + 2) // 3) * (tile_h + 34) + 10
    canvas.crop((0, 0, width, y + pad)).save(output)
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PREDICTIONS.json")
    parser.add_argument("--pair", action="append", metavar="BEFORE:AFTER",
                        help="Repeat for comparisons; default: first run versus all others")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--frames-root", type=Path, help="scannetv2/frames_square; omit for tables only")
    parser.add_argument("--font", type=Path, help="TrueType/OpenType font; default DejaVuSans.ttf")
    parser.add_argument("--cases-per-category", type=int, default=2)
    parser.add_argument("--question-id", action="append", help="Manual case IDs, replaces automatic candidates")
    parser.add_argument("--cdviews-ranking", action="append", default=[], metavar="NAME=RANKING.json",
                        help="Explicit fallback only for this run's missing view trace; replays cdViews NMS")
    parser.add_argument("--view-distance-dir", type=Path, help="Original scene distance CSV directory")
    parser.add_argument("--cdviews-input-views", type=int, default=9, help="Original budget for reconstructed runs")
    parser.add_argument("--cdviews-nms-threshold", type=float, default=0.5)
    args = parser.parse_args()
    fallbacks = {}
    for spec in args.cdviews_ranking:
        name, template = spec.split("=", 1)
        if name in fallbacks:
            parser.error(f"Duplicate cdViews fallback: {name}")
        fallbacks[name] = Path(template.format("ScanQA", "val"))
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in runs:
            parser.error("Use unique run names with letters, digits, underscores or hyphens")
        runs[name], gold = load_run(name, Path(path), args.annotations,
                                   ranking_path=fallbacks.get(name), distance_dir=args.view_distance_dir,
                                   cdviews_budget=args.cdviews_input_views,
                                   nms_threshold=args.cdviews_nms_threshold)
        print(f"{name}: view source = {runs[name]['view_source']['kind']}", flush=True)
    if set(fallbacks) - set(runs):
        parser.error("Each --cdviews-ranking NAME must match a --run NAME")
    names = list(runs)
    pairs = [pair.split(":") for pair in args.pair] if args.pair else [[names[0], n] for n in names[1:]]
    if not pairs or any(len(pair) != 2 or any(n not in runs for n in pair) for pair in pairs):
        parser.error("Supply at least two runs and valid pairs")
    if args.cases_per_category < 0:
        parser.error("cases-per-category must be nonnegative")
    if args.question_id and not set(args.question_id) <= {str(q["question_id"]) for q in gold}:
        parser.error("Unknown manual question ID")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, run in runs.items():
        if run["view_source"]["kind"] == "reconstructed_cdviews_nms":
            trace = {"split": "val", "method": "cdviews", "input_views": run["budget"],
                     "source": run["view_source"],
                     "questions": [run["views"][str(q["question_id"])] for q in gold]}
            (args.output_dir / f"{name}.reconstructed.views.json").write_text(json.dumps(trace, indent=2))
    summary, by_type, view_stats = [], [], []
    for name, run in runs.items():
        summary.append({"method": name, "questions": len(gold), "correct": sum(run["hits"]),
                        "EM@1": 100 * sum(run["hits"]) / len(gold),
                        "empty_answers": sum(not a.strip() for a in run["answers"])})
        types = sorted({question_type(q["question"]) for q in gold})
        for group in types:
            hits = [hit for q, hit in zip(gold, run["hits"]) if question_type(q["question"]) == group]
            by_type.append({"method": name, "lexical_group": group, "questions": len(hits),
                            "correct": sum(hits), "EM@1": 100 * sum(hits) / len(hits)})
        views = list(run["views"].values())
        view_stats.append({"method": name, "view_trace_available": bool(views),
                           "view_source": run["view_source"]["kind"],
                           "budget": run["budget"],
                           "mean_input_views": sum(len(v["images"]) for v in views) / len(views) if views else "",
                           "under_budget_questions": sum(len(v["images"]) < run["budget"] for v in views) if views else "",
                           "candidate_shortage_questions": sum(v["candidate_count"] < run["budget"] for v in views) if views else ""})
    details, pair_summaries, overlaps, cases = [], [], [], []
    for a, b in pairs:
        rows, counts = compare(gold, runs[a], runs[b])
        details.extend(rows)
        pair_summaries.append(counts)
        for category in ("all", "fixed", "regressed", "both_correct", "both_wrong"):
            subset = [r for r in rows if (category == "all" or r["category"] == category) and r["view_jaccard"] != ""]
            overlaps.append({"before": a, "after": b, "category": category, "questions_with_views": len(subset),
                             "mean_jaccard": sum(r["view_jaccard"] for r in subset) / len(subset) if subset else "",
                             "same_set_questions": sum(r["same_view_set"] for r in subset),
                             "same_order_questions": sum(r["same_view_order"] for r in subset)})
        # Deterministic candidates, not evidence of representativeness or causality.
        selected = []
        for category in ("fixed", "regressed", "both_correct", "both_wrong"):
            candidates = sorted((r for r in rows if r["category"] == category), key=lambda r: r["question_id"])
            selected.extend([r for r in candidates if r["question_id"] in args.question_id]
                            if args.question_id else candidates[:args.cases_per_category])
        for row in selected:
            entry = dict(row)
            entry["missing_images"] = []
            if args.frames_root:
                filename = f"case_{len(cases)+1:03d}_{a}_to_{b}_{row['category']}.png"
                entry["missing_images"] = draw_case(row, args.frames_root, args.output_dir / filename, args.font)
                entry["figure"] = filename
            cases.append(entry)
    for filename, rows in (("summary.csv", summary), ("by_question_type.csv", by_type),
                           ("pairwise.csv", pair_summaries), ("per_question.csv", details),
                           ("view_stats.csv", view_stats), ("view_overlap.csv", overlaps)):
        write_csv(args.output_dir / filename, rows)
    (args.output_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    provenance = {"split": "val", "questions": len(gold), "annotations_sha256": sha256(args.annotations),
                  "protocol": "raw exact membership, same as evaluate_scanqa.py; no caption metric recomputation",
                  "case_selection": "manual IDs" if args.question_id else "first IDs per transition category; inspect manually",
                  "runs": {n: {"predictions": str(r["path"].resolve()), "sha256": sha256(r["path"]),
                                 "views_sha256": r["view_source"].get("sha256"),
                                 "view_source": r["view_source"]}
                           for n, r in runs.items()}}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(pair_summaries, indent=2))
    missing = sum(len(c["missing_images"]) for c in cases)
    print(f"Saved {args.output_dir}; {len(cases)} candidate cases; {missing} missing image occurrences")


if __name__ == "__main__":
    main()
