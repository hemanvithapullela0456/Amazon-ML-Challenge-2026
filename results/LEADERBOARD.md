# Leaderboard progress

Public leaderboard F0.5 of each submitted `matching_results.tsv`. The files themselves live in
`output/runs/runN_matching_results.tsv` (not in git: ~95 MB each); the md5 identifies the exact file.

| run | what | CV F0.5 (US+India) | public LB | md5 |
|---|---|---|---|---|
| run1 | stage 1, key blocking top-30 + LightGBM | 0.9598 | **0.948** | (rebuilt from saved scores) |
| run2 | stage 1, key top-30 + dense top-10 (fine-tuned multilingual-e5-small) + LightGBM | 0.9829 | **0.970** | 4cb0190ffd311c34f09d775390f5f40c |
| run3 | run2 + stricter threshold (0.95) for countries unseen in training (France) | 0.9829 | **0.973** | c5044bebd010b8bd029d4a9467f6327f |

Implied France F0.5 (test mix 38% US / 47% India / 15% France, US+India taken at CV):
run1 ≈ 0.90, run2 ≈ 0.89, run3 ≈ 0.91 — France (absent from training) is the remaining gap.
