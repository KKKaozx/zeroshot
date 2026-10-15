# Prediction target controlled comparison: jobs 197783 / 197845

Only prediction target and corresponding sampling interpretation differ. Same initialization, data order, training noise/timesteps, 18360 updates, original action scale and multiscale U-Net. Full native DDPM100 evaluation; three paired sampling seeds; one training seed.

## Findings

| Population | Demonstrations | x0 path position worse than static | epsilon path position worse than static | epsilon path position worse than x0 |
|---|---:|---:|---:|---:|
| train | 405 | 268 | 344 | 345 |
| validation | 10 | 6 | 7 | 9 |

All three sampling seeds show worse aggregate epsilon position and rotation errors in both partitions. Sampling seeds are not independent training replicates.

Training failure is broad, not only validation generalization failure. Some task groups improve over static, so total absence of learning is not supported.

This rejects changing prediction target alone as a successful repair under the fixed recipe. It does not establish a universal superiority of x0 or identify the root cause.

## Next discriminating experiment specification

Before another full-data training sweep, test the smallest generation contract: one fixed training action window, fixed known context, frozen encoder/adapter, train the current x0 decoder from scratch. Evaluate generation starting from pure noise using the native full 100-step sampler, across the same three seeds. Measure pose error against the one known target and static baseline; do not use forward-noised ground truth as the generation input.

Purpose: remove image/text discrimination and cross-demonstration variability while retaining action representation, current decoder, objective and sampler. If this cannot learn the single target, inspect the decoder/training/sampling contract before attributing failure to the adapter or generalization. If it does, proceed to a small set of distinct windows with actual vs explicit per-window identity conditions to test conditioning.

This is a module isolation test, not a generalization result or proof that the dataloader is correct. Official same-configuration U-Net forward/backward equivalence and DDPM update comparisons already passed; do not repeat them without changed evidence.

The minimal test has not been implemented or executed in this analysis. Do not assume an outcome or automatically extend the training budget.
