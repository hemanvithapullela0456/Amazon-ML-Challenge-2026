# Experiment scoreboard

Same held-out S1s for every row of a table. Fill in as results arrive.

## Stage 1 (laptop, LightGBM on string features)
| Run | Blocking recall@30 | OOF F0.5 | US | India | US→India | India→US | Public LB |
|---|---|---|---|---|---|---|---|
| key blocking + ranker, 200k S1 | 0.9426 | 0.9598 (t=0.70, 1:1) | 0.9707 | 0.9433 | | | **0.948** (implies France ≈ 0.90) |
| **key-30 + dense-10**, 200k S1 (ceiling 0.9958) | 0.9958 | **0.9829** (t=0.75, 1:1) | 0.9831 | 0.9827 | | | **0.97** (run2; implies France ≈ 0.89) |

## Track A — retrieval (recall on the 10% blocking-eval S1s)
| Run | dense@10 | dense@20 | key@30 | union key30+dense10 | union key30+dense20 | India union |
|---|---|---|---|---|---|---|
| e5-small, all countries (1 epoch, 1.39M triples) | 0.9879 | 0.9946 | 0.9426 | **0.9958** (34.7 cands/S1) | 0.9981 (44.5) | key30+dense10: 0.9968 |
| e5-small, US only (France proxy) | | | | | | |

## Tracks B, C — rerankers (CE fold 0 of the uncertain pairs)
| Run | CE AUC | stage-1 AUC (same pairs) | stage-1 F0.5 | stage-2 F0.5 | pairs/s (T4) |
|---|---|---|---|---|---|
| B xlm-roberta-base | 0.9709 | 0.9399 | 0.9501 | **0.9676** (expected-F0.5, 1:1) | |
| B xlm-roberta-base US→India (lr 2e-5; 3e-5 never learned) | **0.8199** | 0.9461 | | | 150 train / 766 infer |
| C Qwen3-0.6B LoRA | **0.9803** | 0.9399 | 0.9501 | **0.9693** | 21 train / 71 infer |
| C Qwen3-0.6B LoRA US→India | **0.9129** | 0.9461 | | | 36 train / 53 infer |

| avg(XLM-R, Qwen) (score corr 0.95) | | | 0.9501 | 0.9693 (no gain over Qwen) | |

## New bundle (key+dense stage 1), full runs
| Run | CE AUC (fold 0/1/2) | stage-1 AUC same pairs | stacked CV F0.5 (all 200k S1) |
|---|---|---|---|
| XLM-R base, 2 ep (Kaggle B; a duplicate run gave 0.946/0.937/0.942) | 0.939 / 0.944 / 0.945 | 0.940 / 0.939 / 0.939 | not stacked (≈ stage 1) |
| **Qwen3-0.6B LoRA, 1 ep (Jarvis A100, ~4.2 h)** | **0.958 / 0.959 / 0.959** | 0.940 / 0.939 / 0.939 | **0.98303 → 0.98682** |

## Unseen-country transfer, new bundle (train US, eval India, fold 0)
| Model | CE AUC | stage-1 AUC same pairs |
|---|---|---|
| Qwen3-0.6B LoRA | 0.7845 | 0.9386 |
| Qwen3-1.7B LoRA | 0.8296 | 0.9386 |
| Retriever trained on US only: India recall dense@10 / key30+dense10 | 0.786 / 0.946 | (in-domain 0.992 / 0.997) |
| Matcher on India (US-only retriever + US-only matcher), best F0.5: with dense feats / without | 0.8227 / **0.8490** | |
=> rerankers and retriever similarity do not transfer; for the unseen country use the string-feature matcher without dense features.

## Approach C - non-fine-tuned LLM judge (Qwen2.5-7B-Instruct, Kaggle T4x2), 3000 hard labelled pairs (stage-1 prob 0.05-0.97)
| AUC | US | India |
|---|---|---|
| in-domain stage 1 | 0.862 | 0.872 |
| unseen-country matcher (US-trained), same India pairs | - | 0.661 |
| LLM zero-shot | 0.624 | 0.610 |
| LLM few-shot (8 examples) | 0.565 | 0.611 |
=> rejected: below the unseen-country matcher; hard pairs follow generator conventions an LLM cannot infer.

## Approach A - synthetic target-country training data (generator emulation, src/synth.py), India simulation
| training | India F0.5 | France-like (Latin-only India) F0.5 |
|---|---|---|
| US only | 0.8487 | 0.9009 |
| US + synthetic India | 0.8459 | 0.8991 |
| synthetic India only | 0.8044 | 0.8432 |
=> rejected. Note: 77% of India's unseen-country misses are native-script pairs, so only the Latin-only subset is France-like
(in-domain 0.9755 vs unseen 0.901 there; the loss sits in ordinary "name + address similar" pairs).

## LightGBM self-training on the France-like subset (US-trained, pseudo-labels on India, hi 0.97 / lo 0.03)
round 0 0.9038 -> round 1 0.9113 (+0.75); round 2 died out of memory (3 jobs in parallel). Small gain, kept as an option.

## Word-role features (src/word_roles.py): per-country insertion/substitution rates of name words from UNLABELLED records
Unsupervised lists recover known conventions (India noise-like: shri, dr, center; France: france, fils, groupe, 5arl/5as typos).
| matcher (no dense feats) | US -> India(Latin) | India -> US |
|---|---|---|
| baseline | 0.9037 | 0.9532 |
| + 12 word-role features | 0.9070 (+0.33) | 0.9584 (+0.52) |
Small but consistent in both directions (gain share 2.4%) -> used in the France matcher (run7 candidate).
Name-only copies per S1 behave like independent noise (0:85%, 1:13.9%, 2:1.1%; S2/S3 split even): no exploitable rule.

## Domain-adapted cross-encoder (src/da_encoder.py, XLM-R base, source US labels, target India unlabelled band pairs)
| run | match loss (end) | target AUC ce / lgbm | LATIN-ONLY ce / lgbm |
|---|---|---|---|
| B  MMD lam=1.0 | 0.528 (= base-rate entropy: learned nothing) | 0.536 / 0.774 | 0.498 / 0.853 |
| C  GRL lam=1.0 | 0.324 | 0.473 / 0.774 | 0.520 / 0.853 |
=> both collapse (MMD: constant representation; GRL: domain loss 1.06 > ln2, over-alignment). Retry only with lam ~0.1.

## France students v3 (teacher-student, France-like simulation, AUC Latin band / veto AUC on teacher-confident pairs)
XLM-R 0.934 / 0.967 | Qwen3-1.7B LoRA (Jarvis) 0.944 / 0.978 | mDeBERTa-v3-base 0.956 / 0.983
Fusion on true labels (src/fuse_v3.py): lgbm 0.1 + mDeBERTa 0.9408; mixing students hurts; + veto 2% 0.9447.
run8 (mDeBERTa + 2% veto) LB 0.976795 < run7 0.977405: the veto removed more true than false matches in the
simulation too (only 43.6% of vetoed were false) - a thin trade that France tipped negative. Veto dropped.

## US/India stage 2, new rerankers (Jarvis A100, fold 0 of the uncertain-pair bundle)
mDeBERTa-v3-base 2 ep: CE AUC 0.9555 (Qwen3-0.6B 0.958, stage 1 0.940). Qwen3-1.7B LoRA: pending.
