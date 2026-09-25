# Plan to 0.98+ : from "better matcher" to "matcher that transfers to an unseen country"

## Where we are (public LB)
| run | approach | LB | France (implied) |
|---|---|---|---|
| run1 | key blocking + LightGBM on string features | 0.948 | ~0.90 |
| run2 | + fine-tuned multilingual-e5-small dense retrieval (candidates + cos features) | 0.970 | ~0.89 |
| run3/4 | + stricter threshold for unseen countries | 0.973 | ~0.91 |
| run5 | + Qwen3-0.6B LoRA cross-encoder, stacked (validation 0.9868) | 0.974 | ~0.90 (Qwen hurt France) |
| run6 | Qwen for US/India, France from matcher without dense features | not uploaded | ? |

LB ≈ 0.383·US + 0.468·India + 0.150·France. US/India are at ~0.987 on validation and the remaining errors are
largely label noise (identical records labelled different entities). **Every remaining point is France.**
0.980 needs France ≈ 0.935; 0.985 needs France ≈ 0.97.

## Diagnosis (measured, not guessed)
- The generator is the same in every country: true matches per S1 have the same distribution in US and India
  (5.6% singletons, mean 3.46), and France's predicted counts already have that shape. France loses on *which*
  records are matched, not how many.
- France is full of "siblings": same street, different house number, templated names (city + generic word +
  legal form). The model, trained where siblings are rarer, over-trusts name/retriever similarity.
- Labelled stand-in for France ("India unseen": retriever + matcher trained on US only, scored on India):
  in-domain 0.983 -> unseen 0.823. Precision 0.927 / recall 0.805; missed matches cost 2x false ones.
  - dropping retriever `cos` features: +2.6  (in run6)
  - self-training on the target country: +0.9  (grey area: uses unlabelled test data; not recommended)
  - house-number relation (dropped digit vs neighbouring number): moderately predictive, a small feature gain
- Rerankers do not transfer: US->India AUC Qwen0.6B 0.785, Qwen1.7B 0.830 vs LightGBM 0.939.

## Research basis
- Domain adaptation for ER: DADER (Tu et al., SIGMOD 2022) — labelled source, unlabelled target; feature
  alignment (MMD, CORAL), adversarial (DANN gradient reversal, Ganin et al. 2016), reconstruction.
- Covariate-shift importance weighting (Shimodaira 2000): weight source pairs by p(target)/p(source) from a
  domain classifier, so training looks like the target.
- Ditto (Li et al., VLDB 2021): data augmentation for EM (span deletion/shuffle/replacement, MixDA) — robustness
  from training data alone.
- Contrastive EM: R-SupCon (Peeters & Bizer 2022), Sudowoodo (Wang et al., ICDE 2023) — hard negatives decide
  what the encoder learns; siblings are exactly the hard negatives France needs.
- Unicorn (Tu et al., SIGMOD 2023) and fine-tuned LLM matchers (Peeters, Steiner, Bizer, EDBT 2025): larger
  generic matchers transfer better (our 0.6B -> 1.7B: +0.045 AUC on the unseen country).
- Losses: BCE (now); focal loss (Lin et al. 2017) for hard pairs; listwise LambdaRank/LambdaMART (Burges 2010)
  grouped by S1 — the decision is per S1, so learning to order an S1's candidates matches the metric better.

## Method: every idea passes two gates before it costs a submission
1. In-domain validation (200k S1, 5-fold) must not drop.
2. "India unseen" F0.5 must rise clearly (>= +1.5). Proxy-to-France scale is calibrated once by run6
   (proxy said +2.6 for dropping dense features; run6's LB delta shows what fraction reaches France).

## Phases (cheapest, most in-rules first)
| # | Shift | Uses | Cost | Gate result |
|---|---|---|---|---|
| A | Feature-group ablation / per-country quantile normalisation / regularisation | train only | laptop, 35 min | running |
| B | Sibling-aware training: synthetic hard negatives from TRAIN S1s (same street, other number, name word or legal form swapped) + generator-style corruptions for positives (digit drop, letter suffix, accents, name-only copies) — Ditto-style augmentation | train only | laptop, ~2 h | |
| C | House-number relation features (exact / letter / dropped digit / transposed / near / other) | train only | laptop, ~1.5 h (features + retrain) | |
| D | Listwise objective (LightGBM lambdarank grouped by S1) and focal loss vs BCE | train only | laptop, ~1 h | |
| E | Importance weighting by a domain classifier (source pairs reweighted toward the target) | test *features* (no labels) | laptop, ~1 h | grey: decide after A-D |
| F | Cross-encoder retrained on B's augmented pairs with sibling hard negatives (+ optional DANN head) | train (+ test text for DANN) | Jarvis A100 ~4 h ≈ ₹350 | only if B works on LightGBM |
| G | Ensemble of the transfer-robust matchers for the unseen country (average of the best 2-3 variants) | — | laptop | last |

Submission rule: upload only a run whose projected LB (0.851·CV + 0.150·France estimate) is >= 0.98.
