# External baselines

The random-masking and energy-heuristic acquisition baselines are built into the
pipeline as selection modes (`--mode random` / `--mode energy` in
`scripts/budget_sweep.py` and `scripts/evaluate.py`), so they share the exact same
backbone and evaluation as the learned policy.

To reproduce the *original* published numbers from the two base papers, clone the
authors' repositories here:

    git clone https://github.com/AnsonAiTRAY/MASS.git           mass
    git clone https://github.com/younghoonNa/MC2SleepNet.git    mc2sleepnet

These are kept out of the package on purpose — they have their own dependencies and
data layout. They are reference points for the "original method" rows of the
comparison table, not part of the ActiveSleep training path.
