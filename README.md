# ML Data Poisoning Simulation

A small, fully reproducible simulation that shows how **corrupted feedback data
degrades a recommendation model** when it is retrained with no filtering, and
which type of corruption does the most damage.

## The scenario

A recommender predicts whether a user will **like** a recommendation, using:

| Feature | Meaning |
|---|---|
| `topic_match` | how well the topic matches the user's interests |
| `engagement_rate` | the user's past engagement rate |
| `freshness` | how fresh the content is |
| `is_mobile` | whether the user is on mobile |

A hidden process (with random noise, so labels are not deterministic) decides
the true label. The model is retrained on a **feedback stream** in which about
**24% of rows are corrupted**, with no filtering.

## Corruption types

- **Duplicate flooding**: one hardcoded bad pattern (high match, high engagement, fresh, mobile, labeled "not liked") repeated many times.
- **Flipped labels**: rows users really liked are logged as "not liked".
- **Stale features**: labels are correct, but logged freshness/engagement values are old and outdated.

## Files

| File | What it does |
|---|---|
| `generate_data.py` | Creates `data/train.json` (clean), `data/feedback_stream.json` (clean + poisoned) and `data/held_out_test.json` (balanced 50/50) |
| `retrain_pipeline.py` | Retrains on the whole stream with no filtering. Also trains a clean baseline and one model per corruption type |
| `eval.py` | Scores each model on the held-out set using plain math (no sklearn) and checks accuracy against a threshold |

Random seed is **7**, so results are reproducible.

## Run locally

```bash
pip install -r requirements.txt
python generate_data.py
python retrain_pipeline.py
python eval.py
```

(Optional) with Docker:

```bash
docker build -t poisoning-sim .
docker run --rm poisoning-sim
```

## Example results (seed 7)

Held-out test set: 2,000 rows, balanced. Threshold: 0.75.

| Model | Accuracy | vs baseline | Gate |
|---|---|---|---|
| baseline (clean data) | 0.765 | | PASS |
| only duplicate flooding | 0.730 | -3.5 pts | FAIL |
| only flipped labels | 0.775 | +1.0 pts | PASS |
| only stale features | 0.747 | -1.8 pts | FAIL |
| **full poisoned stream** | **0.712** | **-5.3 pts** | **FAIL** |

Exact numbers may differ slightly across library versions.

**Takeaways**

- Retraining blindly on the poisoned stream drops accuracy by about 5 points and fails the quality gate.
- Duplicate flooding is the most damaging single corruption type.
- Combined corruption is worse than any single type.
- Targeted label flipping barely moved accuracy on the balanced test set. It mostly shifts the model's bias toward "not liked", and a 50/50 test set hides that shift. An imbalanced test set would show it differently.

`eval.py` exits with code 1 when the production model fails the gate. That is intentional: it simulates a CI quality gate catching a bad retrain.

## Ideas to extend

- Add a filter (deduplication, outlier detection) to the pipeline and measure how much accuracy it recovers.
- Vary the corruption share (5%, 10%, 24%, 40%) and plot accuracy against it.
