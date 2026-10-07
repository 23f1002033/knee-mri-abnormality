## Notebook 04: silver labels and holdout check

Inputs: competition train.csv, labels_shard0.csv and labels_shard1.csv from 03b.
Outputs: silver_scores.csv (one score per study and finding, 4407 studies),
silver_meta.json. Metrics logged to W&B run silver-v1 (no data uploaded, only a
SHA-256 of silver_scores.csv).

Merge
- 4407 studies, 0 missing. 4406 parsed as clean JSON, 1 by the regex fallback.
- Gold split rebuilt with the same seed; dev and holdout counts match notebook 03.

D4 decision on dev (rule fixed before the run: adopt if the mean dev AUC of
Synovitis, Contusion and Fracture does not go down)
- Synovitis 0.525 -> 0.545, Contusion 0.650 -> 0.665, Fracture 0.909 -> 0.889.
- Mean change +0.005, so D4 is adopted. The evidence for it is weak either way.
- Dev macro with D1+D3 is 0.816 here vs 0.814 in 03. The full run used a different batch
  mix and a 600-token cap, and greedy decoding is not bit-exact across batches, so a few
  dev answers changed.

Holdout, scored once (38 studies, bootstrap 95% CI)
- Macro AUC 0.884 (0.841 to 0.921).
- Per finding: ACL 0.994, MCL 0.970, Lateral OA 0.931, Medial Meniscus 0.920,
  Medial OA 0.885, PF OA 0.875, Baker's 0.871, Lateral Meniscus 0.864, Contusion 0.849,
  Synovitis 0.819, Effusion 0.818, Fracture 0.814.
- Holdout is higher than dev (0.817). With 20 and 38 studies this gap is within sampling
  noise; dev also holds the cases we studied closely because they were hard.
- For reference only: a public LLM label table reports 0.878 on all 58 gold studies.

Silver label profile
- Unmentioned (score 0.5): Synovitis 84%, Contusion 86%, Fracture 93% of studies.
  A competitor measured Synovitis as unaddressed in about 84% of reports, which matches.
- Most common positives: Medial Meniscus (about 36% at severity 3 or 4), PF OA, Medial OA.

D5: synovitis tie-breaking (dev-only decision, made after the holdout check)
- 84% of studies had Synovitis = 0.5 (unmentioned), so they were all tied.
- For those, score = 0.5 + p, where p is the probability that a report mentions synovitis
  (severity 2 or more), predicted by logistic regression from the other 11 silver scores
  and their unmentioned flags. Fitted on non-gold silver labels only.
- Rule set before running: adopt if the silver 5-fold AUC is at least 0.70 and dev
  Synovitis AUC goes up.
- Silver 5-fold AUC 0.894 (387 positives of 4349). Strongest positive weight: Effusion
  (+1.03), then Medial OA, Baker's, PF OA, Lateral OA. Unmentioned effusion is the
  strongest negative (-1.17).
- Dev Synovitis AUC 0.545 -> 0.551. Adopted by the rule, but the gold evidence is flat:
  9 dev positives cannot show a change this small. The support comes from the silver check.
- Some weights reflect report style rather than anatomy (for example "ACL unmentioned"
  is positive), so part of the signal may be site style.
- Saved as silver_scores_v2.csv. silver_scores.csv is unchanged, so v1 vs v2 can be
  compared as a training experiment in 05.

Rule from here on
- The labeler and the scoring do not change because of the holdout result. Any later
  change is measured on dev only and reported as dev-only.

Next: 05-baseline (image model trained on silver_scores.csv).
