# Paper draft

Compile with:

```bash
cd /Volumes/F/work/Gunqin-agent/paper
latexmk -pdf main.tex
```

The draft separates implemented claims from experimental claims. `main.tex` loads
`related_work.tex`, `experiments.tex`, and the appendix `additional_results.tex`;
bibliography entries live in `references.bib`.

The current experiment text uses `DOCS/experiment_result.md` and the verified
`evaluation_results/warning_ablation_current48_{on,off}/summary.json` and
`evaluation_results/single_stage_current48/summary.json` and
`evaluation_results/react_handbook_full_test50/summary.json` results.
The fixed-workflow training comparison uses
`evaluation_results/fixed_reference_ablation_current48/summary.json`; its
system/user prompts partly differ from the standard single-stage recipe.
Section 4 separates setup, main results, ablation studies, and expert evaluation.
The evaluation currently contains 48 scores and 1,031 work units. Pitch accuracy
compares predicted realizations with the input melody; action F1 compares with
expert notation. Report evaluable event counts alongside pitch accuracy and
aligned tone-pair counts alongside MAE.

The main table uses valid coverage, pitch accuracy, pitch MAE, string F1,
technique distribution similarity (1 minus total variation distance), ornament
F1, and final-warning phrase rate. Detailed hand-fingering and tone-type metrics
are in the appendix. String F1 must not be labeled joint string--hui F1, which
has not been computed in the current evaluation.

Dashes in the main table mark pending direct-generation and GEPA experiments.
The original-checkpoint agent has 1.07% valid-output coverage and N/A full-test
musical metrics. The single-stage student has complete coverage and measured
scores. Expert assessment and remaining training/inference configuration details
are still to be filled in. No pending row should be read as a measured zero.

The GEPA baseline uses a commercial GLM API as the agent backend. Record its
exact model version and reflection model separately; it is a comparison across
model backends, rather than a fixed-backbone ablation of prompt optimization.

