"""Count selector parameters on CPU; no weights, data or inference required.

Counts exclude frozen feature extractors, BLIP and the LLaVA answering model.
Soft/Coverage introduce no learned parameters but still use BLIP features.
"""

import argparse
import json
from pathlib import Path

from .ViewSelector import ViewSelector
from .r2_model import QueryViewSelector


def parameter_counts(model):
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_at_construction": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "direct_parameters": {name: p.numel() for name, p in model.named_parameters(recurse=False)},
            "components": {name: sum(p.numel() for p in module.parameters())
                           for name, module in model.named_children()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, default=4)
    args = parser.parse_args()
    baseline = parameter_counts(ViewSelector())
    qmp = parameter_counts(QueryViewSelector(num_queries=args.num_queries))
    report = {"scope": "selector only; default dimensions 3584/256/512, 2 layers, 8 heads",
              "num_queries": args.num_queries, "cdViews": baseline, "QMP-Selector": qmp,
              "qmp_added_parameters": qmp["parameters"] - baseline["parameters"],
              "qmp_relative_increase_percent": 100 * (qmp["parameters"] / baseline["parameters"] - 1),
              "Soft_Coverage_added_learned_parameters": 0,
              "note": "Construction counts are not trainable parameters during inference. All models are frozen at inference. Frozen BLIP and LLaVA costs are excluded, not zero. No runtime or FLOPs are measured."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
