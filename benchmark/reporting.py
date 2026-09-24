"""CSV output helpers shared by the benchmark driver scripts."""

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def write_csv(output_dir: Path, name: str, rows: List[Dict]) -> None:
    path = output_dir / name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)", flush=True)


def prediction_metric_rows(
    shot: int,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
    count_field: str = "n_predictions",
) -> Tuple[List[Dict], Dict, List[Dict]]:
    """Per-class, aggregate, and confusion-matrix rows for predictions pooled
    over all episodes of one (shot, method) cell.

    ``class_labels[i]`` names integer label ``i``. Confusion rows are emitted
    only for non-zero cells.
    """
    labels = range(len(class_labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = [
        {
            "shot": shot, "method": method, "class": name,
            "precision": precision[i], "recall": recall[i],
            "f1": f1[i], "support": int(support[i]),
        }
        for i, name in enumerate(class_labels)
    ]

    def _f1(average: str) -> float:
        return precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )[2]

    aggregate = {
        "shot": shot, "method": method, "macro_f1": _f1("macro"), "micro_f1": _f1("micro"),
        "weighted_f1": _f1("weighted"),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        count_field: len(y_true),
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    confusion = [
        {
            "shot": shot, "method": method, "true_class": true_name,
            "pred_class": pred_name, "count": int(cm[i, j]),
        }
        for i, true_name in enumerate(class_labels)
        for j, pred_name in enumerate(class_labels)
        if cm[i, j] > 0
    ]
    return per_class, aggregate, confusion
