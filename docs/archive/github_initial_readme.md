# Zero Shot Robotic Manipulation with Vision Language Models

This repository contains the implementation and experiments for the dissertation project on parameter-efficient adaptation of vision-language models for zero-shot robotic manipulation.

## Project objectives

- Build a language-conditioned tabletop manipulation policy.
- Keep the pretrained vision-language backbone frozen.
- Train lightweight cross-attention adapters and a diffusion-based action decoder.
- Evaluate generalization to unseen objects, instruction paraphrases, and robot embodiments.
- Study paraphrase and counterfactual instruction augmentation.

## Repository structure

```text
code/               Training, evaluation, model, and data-loading code
configs/            Experiment configuration files
data_readme.txt      Dataset sources and expected local paths
results/             Final metrics, tables, and figures
logs/                Training and evaluation logs
thesis_tables/       Reproduction metadata for every thesis table and figure
```

Large datasets, checkpoints, logs, and generated outputs must not be committed to Git. Store their locations in configuration files or `data_readme.txt`.

## Current baseline

The first reproduction target is VLA-Adapter on LIBERO-Spatial:

- VLA-Adapter: https://github.com/OpenHelix-Team/VLA-Adapter
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO

## Getting started

1. Clone this repository on the GPU server.
2. Create the Python environment described in `environment.yml`.
3. Download datasets using the links in `data_readme.txt`.
4. Copy `configs/baseline.example.yaml` to a new experiment-specific configuration.
5. Record every reported experiment in `thesis_tables/`.

The concrete training and evaluation commands will be added after the VLA-Adapter and LIBERO dependencies are integrated.

## Reproducibility rules

- Keep all hyperparameters in version-controlled configuration files.
- Record the random seed, dataset split, checkpoint, hardware, and command for every reported result.
- Preserve final checkpoints and metric logs for core experiments.
- Report means and standard deviations over repeated runs.
- Commit progress at least once per week.
