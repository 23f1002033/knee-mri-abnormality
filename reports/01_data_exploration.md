## Notebook 01 summary

Inputs: competition data only.
Outputs: series_inventory.csv (one row per series, header fields, size, iop_plane, vendor),
study_inventory.csv (study id, report language, is_gold, vendor). No report text saved.

Findings
- 4407 train studies, 24371 series, 569 GB, 0 read errors.
- 58 studies have radiologist labels (gold). 4349 have only a report. Reports are not in test.
- Gold appears to be image-based with severity thresholds; it can contradict the report text.
- Reports in 9 languages (by langdetect). No French studies in gold.
- Every study has all 3 planes. Host plane labels match DICOM orientation in 100% of series.
- Fluid_Sensitive == Fat_Suppression everywhere.
- 3991 studies have an FS series in every plane. Missing FS: 257 sagittal, 159 coronal, 0 axial.
- Around 12% of study-planes have 2+ FS series, so notebook 02 needs a selection rule.
- Geometry varies a lot: 11 to 320 slices, 160 to 1280 px, 0.6 to 6 mm thickness.
- Uncompressed DICOM, int16, about 32 MB/s single-thread decode.

Leakage risks
- Site/scanner: language + vendor is the site proxy. Group validation folds by it.
- PatientID is unique per study; returning patients cannot be ruled out if hashing is per study.
- Template reports (one Turkish normal report appears 37 times) come from site style, not
  patient leakage, but they make silver labels for those sites very uniform.
- Only 58 gold studies: tuning the labeler and the model on the same 58 overfits the evaluation.
  Every decision checked against gold has to be counted and kept to a minimum.

Next: 02_dicom_processing (CPU) and 03_report_labeling (GPU), independent, both read this output.