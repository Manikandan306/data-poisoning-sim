"""
generate_data.py
Creates the synthetic datasets for the data poisoning simulation.

Outputs (in ./data):
  train.json            clean baseline training data
  feedback_stream.json  clean data mixed with poisoned rows (~24% corrupted)
  held_out_test.json    balanced test set (50% liked, 50% not liked)

Every row has 4 features and a label:
  topic_match, engagement_rate, freshness, is_mobile  ->  liked (0/1)
Rows in the feedback stream also carry a "corruption" tag so we can study
each corruption type separately.
"""
import json
import os

import numpy as np

SEED = 7
N_TRAIN = 4000          # clean baseline rows
N_STREAM_CLEAN = 3040   # clean rows inside the feedback stream
N_STREAM_POISON = 960   # 960 / 4000 = 24% corrupted
N_TEST_PER_CLASS = 1000
OUT_DIR = "data"

# A hardcoded, repeatable bad pattern used for duplicate flooding.
# High topic match + high engagement + fresh + mobile, but labeled "not liked".
DUPLICATE_PATTERN = {
    "topic_match": 0.90,
    "engagement_rate": 0.85,
    "freshness": 0.90,
    "is_mobile": 1,
    "liked": 0,
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sample_features(rng, n):
    return {
        "topic_match": rng.random(n),
        "engagement_rate": rng.random(n),
        "freshness": rng.random(n),
        "is_mobile": rng.integers(0, 2, n),
    }


def hidden_process(rng, f):
    """Ground truth: a hidden process decides if the user likes a recommendation.
    Random noise is added, so labels are NOT a deterministic function of features."""
    logit = (
        -5.5
        + 5.0 * f["topic_match"]
        + 4.0 * f["engagement_rate"]
        + 2.5 * f["freshness"]
        + 0.8 * f["is_mobile"]
        + rng.normal(0.0, 0.7, len(f["topic_match"]))
    )
    return (rng.random(len(logit)) < sigmoid(logit)).astype(int)


def to_rows(f, y, tag=None):
    rows = []
    for i in range(len(y)):
        row = {
            "topic_match": round(float(f["topic_match"][i]), 4),
            "engagement_rate": round(float(f["engagement_rate"][i]), 4),
            "freshness": round(float(f["freshness"][i]), 4),
            "is_mobile": int(f["is_mobile"][i]),
            "liked": int(y[i]),
        }
        if tag:
            row["corruption"] = tag
        rows.append(row)
    return rows


def make_clean(rng, n):
    f = sample_features(rng, n)
    y = hidden_process(rng, f)
    return f, y


def make_duplicate_flood(n):
    row = dict(DUPLICATE_PATTERN)
    rows = []
    for _ in range(n):
        r = dict(row)
        r["corruption"] = "duplicate_flood"
        rows.append(r)
    return rows


def make_flipped_labels(rng, n):
    """Targeted label flipping: rows that users really liked are logged as
    'not liked'. (Flipping random rows both ways mostly cancels out.)"""
    f, y = make_clean(rng, n * 4)
    idx = np.where(y == 1)[0][:n]
    sub = {k: v[idx] for k, v in f.items()}
    return to_rows(sub, np.zeros(len(idx), dtype=int), tag="flipped_label")


def make_stale_features(rng, n):
    """Stale features: labels come from the TRUE features, but the logged
    engagement/freshness values are old, outdated numbers."""
    f, y = make_clean(rng, n)
    stale = dict(f)
    stale["freshness"] = rng.random(n) * 0.2                  # looks very old
    stale["engagement_rate"] = np.clip(
        f["engagement_rate"] * 0.4 + rng.normal(0, 0.05, n), 0, 1
    )                                                         # outdated, lower
    return to_rows(stale, y, tag="stale_feature")


def make_balanced_test(rng):
    f, y = make_clean(rng, 20000)
    pos = np.where(y == 1)[0][:N_TEST_PER_CLASS]
    neg = np.where(y == 0)[0][:N_TEST_PER_CLASS]
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    sub = {k: v[idx] for k, v in f.items()}
    return to_rows(sub, y[idx])


def save(name, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as fh:
        json.dump(rows, fh)
    print(f"saved {name}: {len(rows)} rows")


def main():
    rng = np.random.default_rng(SEED)

    # 1) clean baseline
    f, y = make_clean(rng, N_TRAIN)
    save("train.json", to_rows(f, y, tag="clean"))

    # 2) feedback stream = clean + poison
    f, y = make_clean(rng, N_STREAM_CLEAN)
    stream = to_rows(f, y, tag="clean")

    n_each = N_STREAM_POISON // 3
    stream += make_duplicate_flood(n_each)
    stream += make_flipped_labels(rng, n_each)
    stream += make_stale_features(rng, N_STREAM_POISON - 2 * n_each)

    order = rng.permutation(len(stream))
    stream = [stream[i] for i in order]
    save("feedback_stream.json", stream)

    poisoned = sum(1 for r in stream if r["corruption"] != "clean")
    print(f"corrupted share of stream: {poisoned / len(stream):.1%}")

    # 3) balanced held-out test set
    save("held_out_test.json", make_balanced_test(rng))


if __name__ == "__main__":
    main()
