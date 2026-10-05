## Notebook 02: DICOM processing

Inputs: competition data, series_inventory.csv and study_inventory.csv from notebook 01.
Outputs: laterality.csv (pilot). This notebook is exploration and pilots only. The full cache
is built by 02b-cache-build with the shared module src/knee_prep.py.

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
- Sources, in order: DICOM tag, then scanner geometry, then unknown.
- Tags are balanced L/R and agree with the side named in the report for 98% of the 806
  studies where both exist.
- Geometry: the x coordinate of the coronal image center gives the side for Siemens (99%)
  and Philips (98%), not for GE or the others. On untagged studies whose report names a
  side, geometry agrees 98% (n=99).
- Final counts (02b): tag 2178, geo 1131, unknown 1098. Notebook 02 had 2194/1115 because
  it read tags from the selected series only; 16 studies have series that disagree on the
  side, so they now fall back to geometry.
- Unknown is mostly GE (868), Toshiba (181), Canon (45). Plan: small classifier on coronal
  slices (fibula position), trained on studies with a known side.
- The cache keeps native orientation. Left knees are flipped at load time from
  laterality.csv, so the cache never needs rebuilding when laterality improves.

Preprocessing
- Per-volume 0.5 to 99.5 percentile clip, pad to square, antialiased resize to 256x256,
  linear resample to 32 slices, uint8, compressed npz with keys sag, cor, axi.
- 100-study pilot: 1.11 s per study with 4 workers (about 1.35 h for train),
  3.82 MB per study (about 16.8 GB for train), no errors.

Cache build (02b-cache-build)
- src/knee_prep.py at commit b1f7e63, attached on Kaggle as the knee_prep utility script.
- Rescans all 24371 series (94 s, 0 errors) and redoes selection and laterality, so train
  and test go through the same code.
- 20-study test: 0.4 min, no errors, no QA flags.
- Outputs: cache/*.npz, series_scan.csv, selected_series.csv, laterality.csv,
  cache_qa.csv, cache_meta.json.
- Full run: 4407 studies in 74.6 min (1.12 s per study with 4 workers), 0 errors,
  no duplicate positions, gaps or mixed shapes. Header scan took 89 s.
- Cache size: about 16.8 GB (estimate from the 100-study pilot).
- cache_meta.json in that run still says git_commit FILL_ME. The run used the knee_prep
  utility script uploaded from commit b1f7e63.

Known issues
- Knee size in the frame varies with field of view. Cropping to the body is a possible
  later improvement.
- Depth resampling of 3D series (up to 320 slices) uses linear sampling without
  antialiasing.

Next: 03-report-labeling, laterality classifier for unknown studies.
