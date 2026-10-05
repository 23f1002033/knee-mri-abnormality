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
    src/         code shared between notebooks (also uploaded to Kaggle as a utility script)
    reports/     short write-up of what each notebook found

## Notebooks

| Notebook | What it does | Status |
| --- | --- | --- |
| 01-data-exploration | inventory of CSVs, series, DICOM headers, reports, leakage risks | done |
| 02-dicom-processing | series selection, slice ordering, laterality, preprocessing pilot | done |
| 02b-cache-build | full train image cache using src/knee_prep.py | in progress |
| 03-report-labeling | silver labels from reports, checked against the 58 gold studies | planned |

## Running

Everything runs on Kaggle. Notebooks read the competition data from
`/kaggle/input/competitions/rsna-knee-abnormality-detection` and pass results to each
other as Kaggle notebook outputs. `src/knee_prep.py` is kept in sync with a Kaggle utility
script of the same name, which later notebooks attach and import.

## Data

No competition data is stored here: no DICOM, no reports, no CSVs, no cached volumes,
no notebook outputs. Competition rules do not allow sharing the data outside the
competition.
