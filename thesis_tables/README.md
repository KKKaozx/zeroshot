# Experiment records

## Current recorded diagnostics

- [Full training-set x0 fit and failed development evaluation](bridge_full_x0.md)
- [Matched epsilon/x0 small-window fitting](bridge_target_pair.md)
- Original regression development result and module conclusions: [progress](../docs/PROGRESS.md)

These are diagnostic evidence, not completed benchmark task-success tables.
Original source fingerprints and output checksums are retained separately from
the current publication commit. The numerical evidence manifest is
`reports/evidence_manifest.json`.

Create one text or Markdown file for every thesis table and figure. Record:

- dataset and split;
- model and source commit;
- configuration file;
- random seed;
- learning rate, batch size, and training steps;
- checkpoint path;
- GPU model and number of GPUs;
- evaluation command;
- mean, standard deviation, and number of runs.
