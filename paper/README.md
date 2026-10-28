# Paper draft

Compile with:

```bash
cd /Volumes/F/work/Gunqin-agent/paper
latexmk -pdf main.tex
```

The draft separates implemented claims from experimental claims. `main.tex` loads
`related_work.tex` and `experiments.tex`; bibliography entries live in `references.bib`.

The current experiment text uses `DOCS/experiment_result.md` and the verified
`evaluation_results/warning_ablation_current48_{on,off}/summary.json` results.
The evaluation currently contains 48 scores and 1,031 work units. Pitch accuracy
compares predicted realizations with the input melody; action F1 compares with
expert notation. Report evaluable event counts alongside pitch accuracy and
aligned tone-pair counts alongside MAE.

Dashes in the main table mark pending original-checkpoint, GEPA, and single-stage
experiments. Expert assessment and exact training/inference configurations remain
to be filled in. No pending row should be read as a measured zero.

The GEPA baseline uses a commercial GLM API as the agent backend. Record its
exact model version and reflection model separately; it is a comparison across
model backends, rather than a fixed-backbone ablation of prompt optimization.

