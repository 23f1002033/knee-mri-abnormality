# Knee MRI Abnormality Detection

Working repo for the RSNA Knee Abnormality Detection competition on Kaggle (2026).
The task is to predict 12 knee findings per MRI study (ACL, MCL, menisci, three OA
compartments, effusion, synovitis, Baker's cyst, contusion, fracture). Metric is macro
ROC-AUC. This is a research and learning project, not a clinical tool.

The hard part is the labels: only 58 of 4407 training studies have radiologist labels.
The rest come with a free-text report in one of nine languages, and the reports are not
available at test time.

## Layout

    notebooks/   Kaggle notebooks, one job each, outputs cleared before commit
    src/         code shared between notebooks (mirrored on Kaggle as utility scripts)
    reports/     short write-up of what each notebook found

## Notebooks

| Notebook | What it does | Status |
| --- | --- | --- |
| 00-dicom-decoders | offline wheels for JPEG and JPEG 2000 DICOM decoding | done |
| 01-data-exploration | inventory of CSVs, series, DICOM headers, reports, leakage risks | done |
| 02-dicom-processing | series selection, slice ordering, laterality, preprocessing pilot | done |
| 02b-cache-build | full train image cache using src/knee_prep.py | done |
| 03-report-labeling | LLM labeler design and dev evaluation | done |
| 03b-label-shard0/1 | full labeling run, split over two accounts | done |
| 04-silver-labels | merge shards, final scoring, one-time holdout check | in progress |
| 05a-train | baseline image model: EfficientNet-B0, 2.5D, 5 site-grouped folds | done |
| 05b-infer | offline inference and submission, public LB 0.908 | done |
| 05c-laterality | side classifier for the laterality flip experiment | in progress |

## Tracking

- Kaggle notebook versions hold all outputs and pass artifacts between notebooks.
- Weights & Biases logs metrics and configs only. No competition data, reports or labels
  are uploaded there.
- Hugging Face is for a final selected checkpoint, if the competition rules allow it.

## Running

Everything runs on Kaggle. Notebooks read the competition data from
`/kaggle/input/competitions/rsna-knee-abnormality-detection` and pass results to each
other as Kaggle notebook outputs.

## Data

No competition data is stored here: no DICOM, no reports, no CSVs, no cached volumes,
no notebook outputs. Competition rules do not allow sharing the data outside the
competition.
