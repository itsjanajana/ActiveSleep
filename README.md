# ActiveSleep

Budgeted EEG acquisition with microstructure supervision for EEG-only sleep
staging. A differentiable acquisition policy learns **which patches of each 30 s
epoch to observe** at high fidelity under a strict sensing budget, replacing the
random masking of MASS, with optional CAP Phase-A microstructure supervision to
steer acquisition toward physiologically informative micro-events.

- **Base methods:** MASS (multi-level masking + global prompt) and MC²SleepNet
  (raw/spectrogram cross-view contrastive alignment).
- **Contribution:** a budget-constrained, differentiable Top-B acquisition policy
  + CAP Phase-A auxiliary head.

## Install

```bash
pip install -r requirements.txt
```

## Data layout

Sleep-EDF Expanded (extracted), pointed at by `sleepedf.raw_root`:

```
sleep-edf-database-expanded-1.0.0/
  sleep-cassette/   SC4ssNE0-PSG.edf + SC4ssNX-Hypnogram.edf
  sleep-telemetry/  ST7ssNJ0-PSG.edf + ST7ssNX-Hypnogram.edf
```

Subject id = first 5 filename chars (`SC400`, `ST701`); both nights of a subject
share it, which the subject-wise split uses as the leakage guard.

## Run order

```bash
# 1. preprocess  ->  data/processed/sleepedf/{*.npz, meta.json, splits.json}
python scripts/prepare_sleepedf.py --config configs/base.yaml configs/sleepedf.yaml \
    --set sleepedf.raw_root=/path/to/sleep-edf-database-expanded-1.0.0

# 2. train (three-phase curriculum)
python scripts/train.py --config configs/base.yaml configs/sleepedf.yaml

# 3. evaluate on the test split (staging + calibration)
python scripts/evaluate.py --ckpt results/checkpoints/best.pt --split test

# 4. performance-budget curves (learned vs random vs energy)
python scripts/budget_sweep.py --ckpt results/checkpoints/best.pt --split test

# 5. ablations
python scripts/ablate.py --config configs/base.yaml configs/sleepedf.yaml
```

Common cohorts/subsets:

```bash
# classic 20-subject cassette benchmark
--set sleepedf.subjects=edf20
# all 78 cassette subjects
--set sleepedf.subjects=edf78
# include telemetry too
--set sleepedf.cohorts=[cassette,telemetry]
```

## Adding CAP (no code changes)

The model always builds the CAP head, the dataset returns `-1` (ignore index) when
a recording has no `cap` array, and the trainer applies a masked multi-task loss.
So CAP is a config + one preprocessing run — nothing else is edited:

```bash
# set cap.raw_root in configs/cap.yaml, then:
python scripts/prepare_cap.py  --config configs/base.yaml configs/cap.yaml
python scripts/train.py        --config configs/base.yaml configs/cap.yaml
# cross-dataset (train Sleep-EDF, test CAP):
python scripts/evaluate.py --ckpt results/checkpoints/best.pt --data data/processed/cap
```

`configs/cap.yaml` raises `lambda_cap` and sets `cap_granularity: patch`.

> The single fragile point is parsing the CAP `.txt` annotation files
> (`activesleep/data/cap.py:_parse_annotation_file`). It auto-detects the header
> and matches Phase-A as `A[123]`; if your copy uses a different column layout,
> that one function is the only thing to adjust. `_alignment_warning` prints if
> fewer than 70% of Phase-A spans fall in NREM, flagging a clock-misalignment.

## Design decisions (all config-swappable)

| Decision | Default | Config key | Alternatives |
|---|---|---|---|
| Backbone-input normalization | per-epoch z-norm | `signal.norm` | `patch`, `none` |
| Policy "cheap glance" features | band-power, `D=7` | `signal.summary_type` | `raw` (downsampled) |
| Patch granularity | 1.0 s (`P=30`) | `signal.patch_sec` | 0.5 s, 2.0 s |
| Context window | 11 epochs | `data.context` | any odd K |
| Target budget | 6/30 patches | `budget.target_patches` | any 1..P |
| Cross-view alignment | off | `model.crossview` | on |
| Spectrogram source | full epoch (proxy) | `model.crossview_full_epoch` | selected patches |
| CAP granularity | epoch (patch in cap.yaml) | `model.cap_granularity` | `epoch`, `patch` |

The band-power summary and per-epoch z-norm are computed in
`activesleep/data/signal.py`; the summary is computed pre-normalization and
standardized per-recording so the policy sees which patches stand out without
cross-subject amplitude dominating.

## Curriculum

| Phase | Budget | Selection | Active losses |
|---|---|---|---|
| 1 (warmup) | `P` (full) | none | staging |
| 2 (acquire) | `P → target` (annealed) | learned | staging + stability |
| 3 (refine) | `target` | learned | staging + stability + contrastive + CAP |

Phase fractions: `train.phase1_frac`, `train.phase2_frac`.

## Layout

```
configs/            base + sleepedf + cap + experiment YAMLs
activesleep/
  data/             signal (shared preprocessing), sleepedf, cap, splits, dataset
  models/           acquisition, encoder, backbone (prompt), crossview, heads, activesleep
  losses.py metrics.py budget.py trainer.py utils.py
baselines/external/ placeholder for cloning MASS + MC2SleepNet
scripts/            prepare_sleepedf, prepare_cap, train, evaluate, budget_sweep, ablate
```

The metrics reported are accuracy, macro-F1, Cohen's κ, per-class F1, ECE
(calibration), and — with CAP — Phase-A AUPRC/F1.
