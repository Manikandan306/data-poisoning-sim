"""
eval.py
Scores every saved model on held_out_test.json using plain math
(dot product + sigmoid + accuracy by hand, no sklearn) and checks the
accuracy against a threshold.

Exit code is 1 if the production model (poisoned_full_stream) falls below
the threshold - this is the "quality gate" failing, which is expected when
the poisoning works.
"""
import glob
import json
import math
import os
import sys

DATA_DIR = "data"
MODEL_DIR = "models"
THRESHOLD = 0.75          # minimum acceptable held-out accuracy
PRODUCTION_MODEL = "poisoned_full_stream"


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def predict(model, row):
    z = model["bias"]
    for w, name in zip(model["weights"], model["features"]):
        z += w * row[name]
    return 1 if sigmoid(z) >= 0.5 else 0


def accuracy(model, rows):
    correct = sum(1 for r in rows if predict(model, r) == r["liked"])
    return correct / len(rows)


def main():
    with open(os.path.join(DATA_DIR, "held_out_test.json")) as fh:
        test = json.load(fh)

    pos = sum(r["liked"] for r in test)
    print(f"held-out test set: {len(test)} rows ({pos} liked / {len(test) - pos} not liked)")
    print(f"accuracy threshold: {THRESHOLD:.2f}\n")

    results = {}
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as fh:
            model = json.load(fh)
        results[name] = accuracy(model, test)

    base = results.get("baseline")
    print(f"{'model':<24}{'accuracy':>10}{'vs baseline':>14}   gate")
    print("-" * 58)
    for name, acc in results.items():
        drop = "" if base is None or name == "baseline" else f"{(acc - base) * 100:+.1f} pts"
        gate = "PASS" if acc >= THRESHOLD else "FAIL"
        print(f"{name:<24}{acc:>10.3f}{drop:>14}   {gate}")

    if results.get(PRODUCTION_MODEL, 1.0) < THRESHOLD:
        print(f"\nQuality gate FAILED for '{PRODUCTION_MODEL}': poisoning degraded the model.")
        sys.exit(1)
    print("\nQuality gate passed.")


if __name__ == "__main__":
    main()
