## Notebook 03: report labeling

Inputs: competition train.csv (reports and the 58 gold studies).
Outputs: labels_shard0.csv and labels_shard1.csv from 03b (raw severity, certainty and
evidence per finding for all 4407 studies). Scoring and the holdout check are in 04.

Goal
- Turn every report into a silver label per finding. Gold labels come from radiologists
  reading the images with severity thresholds, so the labeler grades severity instead of
  only detecting mentions.

Design
- The LLM extracts, code decides. For each finding the model returns
  [severity 0-4 or null, certainty 0-2, evidence quote]. A fixed formula turns that into a
  score: severity + 0.25 * certainty, explicit normal = 0, unmentioned = 0.5.
- AUC only needs a ranking, so the model is never asked to apply a threshold.
- Evidence must be copied from the report, so every answer can be checked.

Gold split
- 58 gold studies split once (seed 42) into dev (20) and holdout (38).
- Dev is read freely and used for decisions. Holdout is scored once, in 04.

Model and prompt runs (dev macro AUC, 20 studies)
- Qwen3-8B fp16 split over 2 T4s, prompt v2: 0.814; with D1: 0.820. About 10 h for all reports.
- Qwen3-4B fp16, one copy per GPU, prompt v3 + D1: 0.752. About 3 h. Misreads in
  non-English reports and findings filed under the wrong label.
- Qwen3-8B NF4 4-bit, one copy per GPU, prompt v3 + D1: 0.811; with D3: 0.814.
  About 9 h, run as two shards on two team accounts.
- Chosen: Qwen3-8B NF4. Rule set before the run: dev macro at least 0.80.

Decisions made from dev errors
- D1: severity 0 with empty evidence is treated as unmentioned.
- D2: prompt v3. Severity 0 needs an explicit quote; a finding that is present with no
  severity words is severity 3; never use 0 for an unmentioned finding.
- D3: severity 0 whose evidence is not found verbatim in the report is treated as
  unmentioned (the model wrote "nicht erwaehnt" = "not mentioned" as evidence).
- D4 (candidate, decided on dev in 04): for Synovitis, Contusion and Fracture treat
  severity 0 as unmentioned. The model often quotes a sentence about another finding as
  the "normal" evidence for these.

Full run (03b)
- Shard 0: 2204 studies, 6.4 h. Shard 1: 2203 studies, 6.6 h. 4406 clean JSON, 1 recovered
  by the regex fallback, 0 missing. Evidence found verbatim: 88.5% in both shards.
- Severity distributions of the two shards match closely.

Engineering notes
- 8B split over two GPUs ran out of memory at batch 20 and later hit an illegal memory
  access. One model copy per GPU with 4-bit weights was stable.
- Outputs hit the 400-token cap on Cyrillic and Greek reports. Cap raised to 600, retry at
  1000. A regex fallback recovers findings from cut-off JSON.

Known issues
- Size words are sometimes misgraded in non-English reports (Bulgarian "small Baker's
  cyst" got severity 4).
- Evidence quotes often exceed the 8-word cap.
- About 12% of quotes are paraphrased, so evidence is not fully trustworthy.
- Some gold positives are never mentioned in the report. No report labeler can recover
  these; only the image model can.

Next: 04-silver-labels (merge shards, decide D4, score holdout once).
