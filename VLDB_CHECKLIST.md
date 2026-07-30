# VLDB/PVLDB Compliance Checklist

This checklist is based on the public PVLDB/VLDB author expectations: ACM/PVLDB formatting, page limit, reproducibility, and artifact availability.

## Current article status

- Format: `acmart` with `sigconf, nonacm`.
- Current compiled manuscript: `paper/main.pdf`.
- Current page count observed locally: 11 pages.
- Artifact package: this companion folder contains code, notebooks, commands, paper-ready CSVs, figures, and preliminary feature-selection reports.

## Things to verify before submission

- Use the official PVLDB template required by the target call.
- Confirm the page limit for the exact target issue/track.
- Keep bibliography and appendix rules aligned with the current call.
- If anonymous submission is required, remove author names and any direct public GitHub URL until the camera-ready phase.
- For a non-anonymous public companion, keep this folder public and cite a stable release tag or DOI.
- Do not upload private API keys, local absolute paths, downloaded external model checkpoints, or very large raw outputs.

## Reproducibility coverage

- RQ1 effectiveness and baselines: `results/rq1_effectiveness_baselines/`.
- RQ2 meta-feature selection: `results/rq2_meta_feature_selection/`.
- RQ3 efficiency: `results/rq3_efficiency/`.
- RQ4 family ablation: `results/rq4_family_ablation/`.
- Preliminary correlation/feature-selection grids: `results/preliminary/`.
- Executable commands: `run_commands/`.
- Environment: `environment/`.
