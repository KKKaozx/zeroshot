# Supervisor requirements and status

Summarized from the project correspondence; private emails are not published.
The later dataset/version clarification takes precedence over earlier suggestions.

| Requirement | Status on 2026-10-06 |
|---|---|
| Documented PyTorch codebase on GitHub; commit at least weekly | Actual source and documented diagnostic progress included in this update |
| One main training and one evaluation entry | `code/train.py`, `code/test.py` (offline evaluation) |
| Configuration rather than editing hyperparameters in code | Historical `.args` files and recorded JSON configs; some diagnostic export paths remain hard-coded |
| Frozen CLIP ViT-L/14, eight cross-attention layers, 512 attention channels, DDPM/U-Net | Implementation exists; recent evidence uses smaller/frozen adapter contexts, not complete target-architecture validation |
| Four core OXE subsets, justify inclusion/exclusion and versions | Teacher choices documented; formal multi-source action contracts/training incomplete |
| Four augmentation conditions | Formal controlled study not completed |
| LIBERO and CLIPort, 100 rollouts per task split | Not completed; offline errors are not task-success rates |
| At least three independent training seeds; mean ± SD | Not completed; three diffusion sampling seeds do not satisfy this |
| RT-1 and original CLIPort comparisons on identical protocol/splits | Not completed; small Bridge LCBC checks are not substitutes |
| Adapter depth ablation | Not completed as formal benchmark study |
| Cross-embodiment quantitative evaluation | Not completed |
| Paraphrase robustness distribution and failure analysis | Not completed |
| Reproducible tables/figures, splits, seeds, settings, source hashes | Selected existing diagnostic evidence recorded; full artifacts stored outside Git |
| Approximately 10,000-word thesis and conference-length manuscript | Outside the completed code/diagnostic work |
| Progress email every two weeks | This repository update supplies material; it does not send an email |

The project is currently at an implementation/diagnostic milestone. Teacher
requirements should not be marked complete because a function or parser exists.
