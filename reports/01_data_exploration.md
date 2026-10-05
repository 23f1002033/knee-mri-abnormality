## Notebook 01: data exploration

Inputs: competition data only.
Outputs: series_inventory.csv (one row per series: header fields, size, iop_plane, vendor),
study_inventory.csv (study id, report language, is_gold, vendor). No report text saved.

Data
- 4407 train studies, 24371 series, 569 GB, 0 read errors.
- Layout: train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm. The public test folder
  has 3 studies; the hidden test set replaces it at submission time.
- Uncompressed DICOM, int16, about 32 MB/s single-thread decode.

Labels
- 58 studies have radiologist labels (gold). 4349 have only a report. Reports are not in test.
- Gold positives out of 58: ACL 24, MCL 9, Medial Meniscus 26, Lateral Meniscus 23,
  Medial OA 15, Lateral OA 11, PF OA 21, Effusion 35, Synovitis 27, Baker's 12,
  Contusion 19, Fracture 18. The gold set is enriched, so prevalence in the other 4349
  studies will be lower.
- Gold can contradict the report text (moderate effusion labeled 0, lateral meniscus
  blunting labeled 0). Competitor write-ups say gold comes from radiologists reading the
  images with severity thresholds. Not yet checked against the official label definitions.

Reports
- 9 languages by langdetect: en 1736, es 682, tr 546, hr 406, el 321, de 262, bg 220,
  nl 153, fr 81. RSNA says 12, so langdetect is probably merging close languages.
- No French studies in gold.
- 177 studies share their report text with another study (46 distinct texts), mostly site
  templates. One Turkish normal report appears 37 times.

Series
- Every study has all 3 planes. Host plane labels match DICOM orientation in 100% of series.
- Fluid_Sensitive == Fat_Suppression everywhere.
- 3991 studies have an FS series in every plane. Missing FS: 257 sagittal, 159 coronal, 0 axial.
- 1166 of 13221 study-planes (about 9%) have 2+ FS series, so notebook 02 needs a selection rule.
- Geometry varies a lot: 11 to 320 slices, 160 to 1280 px, 0.6 to 6 mm thickness.
- No PatientID repeats across studies. InstitutionName, TR and TE are stripped.

Leakage risks
- Site/scanner: language + vendor is the site proxy. Group validation folds by it.
- PatientID is unique per study; returning patients cannot be ruled out if hashing is per study.
- Template reports come from site style, not patient leakage, but they make silver labels
  for those sites very uniform.
- Only 58 gold studies: tuning the labeler and the model on the same 58 overfits the evaluation.
  Every decision checked against gold has to be counted and kept to a minimum.

Next: 02-dicom-processing reads both inventories. 03-report-labeling uses study_inventory.csv
for language and the gold flag.
