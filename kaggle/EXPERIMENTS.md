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
