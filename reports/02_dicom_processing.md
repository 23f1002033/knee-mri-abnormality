## Notebook 02 summary

Inputs: competition data, series_inventory.csv and study_inventory.csv from notebook 01.
Outputs: laterality.csv (pilot). The full cache is built in 02b-cache-build.

02b-cache-build
- Built with src/knee_prep.py at commit b1f7e63, attached on Kaggle as the knee_prep
  utility script.
- Rescans all 24371 series (94 s, 0 errors) and redoes selection and laterality with
  the shared code, so train and test go through the same path.
- Laterality there reads tags from every series in a study, not only the selected ones.
  16 studies have series that disagree, so they drop the tag and fall back to geometry:
  tag 2178, geo 1131, unknown 1098.
- 20-study test: 0.4 min, no errors, no QA flags.
- Outputs: cache/*.npz, series_scan.csv, selected_series.csv, laterality.csv,
  cache_qa.csv, cache_meta.json.
- Full run results: pending.

Series selection
- One series per study and plane: readable first, fluid-sensitive first, slice thickness
  closest to 3 mm, then most rows.
- 13221 series selected (3 per study). Non-FS fallback: 257 sagittal, 159 coronal, 0 axial.
- 401 selected series have more than 80 slices (3D acquisitions). About 310 GB to read.

Orientation and slice order
- Every selected series has the same orientation for its plane:
  axial (right=L, down=P), coronal (right=L, down=I), sagittal (right=P, down=I).
  No flips or rotations are needed for orientation.
- Slices are sorted by position along the slice normal, not by file name.
  Pilot QA: no duplicate positions, no gaps, no mixed image sizes.

Laterality
- Image right is always patient left, so medial is on opposite sides for left and right
  knees. Five labels depend on this (MCL, both menisci, medial and lateral OA).
- DICOM tag gives the side for 2194 studies, balanced L/R. Tag agrees with the side named
  in the report for 98% of 806 studies where both exist.
- Coordinate x of the coronal image center gives the side for Siemens (99%) and Philips
  (98%), not for GE or others. Used for 1115 untagged studies; 98% agreement with reports
  on the 99 that name a side.
- 1098 studies still unknown, mostly GE (868), Toshiba (181), Canon (45).
  Plan: small classifier on coronal slices (fibula position), trained on known sides.
- The cache keeps native orientation. Left knees are flipped at load time from
  laterality.csv, so the cache never needs rebuilding when laterality improves.

Preprocessing
- Per-volume 0.5 to 99.5 percentile clip, pad to square, antialiased resize to 256x256,
  linear resample to 32 slices, uint8, compressed npz with keys sag, cor, axi.
- 100-study pilot: 1.11 s per study with 4 workers (about 1.35 h for train),
  3.82 MB per study (about 16.8 GB for train), no errors.

Known issues
- Knee size in the frame varies with field of view. Cropping to the body is a possible
  later improvement.
- Depth resampling of 3D series (up to 320 slices) uses linear sampling without
  antialiasing.

Next: 02b-cache-build (full train cache), laterality classifier for unknown studies,
03-report-labeling.
