"""
retrain_pipeline.py
Retrains the recommendation model on the WHOLE feedback stream with NO filtering
(this is the vulnerable pipeline we are attacking), and also trains:
  - a baseline model on clean data
  - one model per corruption type (clean rows + only that corruption)
so we can compare how damaging each corruption type is.

Models are saved as plain JSON (weights + bias) in ./models
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

SEED = 7
FEATURES = ["topic_match", "engagement_rate", "freshness", "is_mobile"]
DATA_DIR = "data"
MODEL_DIR = "models"


def load(name):
    with open(os.path.join(DATA_DIR, name)) as fh:
        return json.load(fh)


def to_xy(rows):
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["liked"] for r in rows], dtype=int)
    return X, y


def train_and_save(name, rows):
    X, y = to_xy(rows)
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X, y)
    train_acc = model.score(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"{name}.json"), "w") as fh:
        json.dump(
            {
                "features": FEATURES,
                "weights": model.coef_[0].tolist(),
                "bias": float(model.intercept_[0]),
                "train_rows": len(rows),
                "train_accuracy": train_acc,
            },
            fh,
            indent=2,
        )
    print(f"trained {name:<22} rows={len(rows):<5} train_acc={train_acc:.3f}")


def main():
    clean = load("train.json")
    stream = load("feedback_stream.json")

    # baseline: clean data only
    train_and_save("baseline", clean)

    # vulnerable pipeline: retrain on the whole stream, no filtering
    train_and_save("poisoned_full_stream", stream)

    # per-corruption-type experiments: clean rows + one corruption type
    clean_in_stream = [r for r in stream if r["corruption"] == "clean"]
    for kind in ("duplicate_flood", "flipped_label", "stale_feature"):
        bad = [r for r in stream if r["corruption"] == kind]
        train_and_save(f"only_{kind}", clean_in_stream + bad)


if __name__ == "__main__":
    main()
