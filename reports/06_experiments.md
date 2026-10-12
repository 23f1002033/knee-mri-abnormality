## Experiments after the baseline

Each run changes one thing from the baseline b0-v1-noflip (EfficientNet-B0, 2.5D, px cache,
v1 labels, no flip, 8 epochs, 5 site-grouped folds). Judged on three numbers: silver
out-of-fold (4407 studies, stable but report-based), gold out-of-fold (58 studies, the
right labels but noisy) and the public leaderboard (part of the test set, also noisy).
A change is kept only when the three point the same way.

| Run | Change | Silver OOF | Gold OOF | Public LB | Decision |
| --- | --- | --- | --- | --- | --- |
| b0-v1-noflip | baseline | 0.817 | 0.853 | 0.908 | reference |
| b0-v2-noflip | v2 labels (D5 synovitis tie-breaking) | 0.814 | 0.864 | 0.905 | drop |
| b0-v1-flip | laterality flip | 0.819 | 0.857 | 0.910 | keep |
| b0-v1-mm | physical-scale cache | 0.827 | 0.867 | 0.914 | keep |
| b0-v1-mm-flip | mm cache + flip | 0.828 | 0.865 | 0.917 | keep |
| b2-v1-mm-flip | same, EfficientNet-B2 | 0.823 | 0.861 | not submitted | ensemble member |
| b0 + b2 average | 50/50 mean of the two runs | 0.839 | 0.876 | 0.919 | best so far |

Noise level
- Between runs, single findings move by up to about 0.03 to 0.08 on gold even when the change
  cannot affect them (v2 only changes Synovitis labels, yet MCL moved +0.05). With 58 gold
  studies, a macro change below about 0.01 cannot be told apart from training randomness.
- Repeated training with the same settings moves the public LB by roughly 0.005.

v2 labels (dropped)
- Gold Synovitis 0.752 -> 0.766, silver Synovitis 0.803 -> 0.776. Silver drops by design:
  v2 raises report-silent studies, which disagrees with report-based silver labels.
- Gold macro +0.011 came mostly from findings v2 does not touch, so it is noise.
- Silver and LB both went down, so v2 is not used. The checkpoints may still serve as a
  diverse member of an ensemble.

Laterality flip (kept)
- 05c laterality classifier (ResNet-18 on three posterior coronal slices): 99.3%
  out-of-fold accuracy on unseen sites, 98.5% agreement with the side named in the report
  on the 454 previously unknown studies that name one. The report check itself agrees with
  DICOM tags 97.9%, so this is close to its ceiling.
- Every left knee is mirrored to look like a right knee before training and inference.
  At inference the side comes from the DICOM tag, then scanner geometry, then the 05c model.
- Silver +0.002, gold +0.004, LB +0.002. Small and consistent.
- The network had mostly worked out the side by itself (fibula and other anatomy), so
  normalising it adds little.

Physical-scale cache (kept)
- Pixel spacing in a 10-study sample ranged from 0.14 to 0.56 mm, so in the px cache the
  same structure could differ in size by about 4x.
- mm cache: 140 mm per 256 px (about 0.55 mm per pixel), crop centred on the largest
  connected body region of the mean slice, same 32 x 256 x 256 shape, so the model code
  does not change.
- Coverage check on 120 pilot series: median fraction of the leg kept 0.999 (axial, whole
  cross-section) and 0.999 / 0.996 (coronal / sagittal, width within the kept rows).
  6 of 120 series below 0.9, all large legs (field of view 175 to 260 mm) where outer soft
  tissue is cut.
- A first crop centred on all bright pixels; the coverage check showed it is pulled by
  anything else in the field of view, so the crop uses the largest connected region.
- Built in two shards (9.65 GB + 9.62 GB) because the full cache is too close to the 20 GB
  output limit. 4407 studies, 0 errors, every series had pixel spacing.
- Silver +0.010, gold +0.014, LB +0.006. Best silver per fold 0.829, 0.857, 0.842, 0.820,
  0.821 (up in 4 of 5 folds). Gains spread over ACL, MCL, Baker's and Synovitis; PF OA
  stays flat, likely limited by in-plane resolution for thin patellar cartilage.

Gold out-of-fold per finding
| Finding | baseline | v2 | flip | mm | mm + flip | b2 | b0 + b2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACL | 0.912 | 0.898 | 0.914 | 0.915 | 0.938 | 0.934 | 0.953 |
| MCL | 0.785 | 0.836 | 0.846 | 0.800 | 0.828 | 0.771 | 0.837 |
| Medial Meniscus | 0.838 | 0.858 | 0.838 | 0.898 | 0.886 | 0.889 | 0.913 |
| Lateral Meniscus | 0.778 | 0.779 | 0.848 | 0.835 | 0.825 | 0.784 | 0.822 |
| Medial OA | 0.974 | 0.972 | 0.964 | 0.953 | 0.963 | 0.943 | 0.971 |
| Lateral OA | 0.857 | 0.861 | 0.778 | 0.901 | 0.861 | 0.787 | 0.824 |
| PF OA | 0.757 | 0.789 | 0.770 | 0.746 | 0.734 | 0.830 | 0.780 |
| Effusion | 0.955 | 0.937 | 0.960 | 0.920 | 0.959 | 0.950 | 0.963 |
| Synovitis | 0.752 | 0.766 | 0.774 | 0.774 | 0.762 | 0.754 | 0.780 |
| Baker's | 0.902 | 0.928 | 0.904 | 0.973 | 0.924 | 0.911 | 0.911 |
| Contusion | 0.850 | 0.839 | 0.862 | 0.821 | 0.842 | 0.892 | 0.876 |
| Fracture | 0.875 | 0.903 | 0.822 | 0.871 | 0.862 | 0.888 | 0.883 |
| Macro | 0.853 | 0.864 | 0.857 | 0.867 | 0.865 | 0.861 | 0.876 |

mm cache + flip (kept)
- Needed the laterality classifier retrained on mm coronal slices (05c-laterality-mm):
  out-of-fold accuracy 0.9934 on 3309 known studies, 98.5% agreement with the report side on
  454 unknown studies, 2.7% of unknown studies uncertain (p between 0.1 and 0.9). Training
  sides still come from the px 05c; both classifiers agree on about 98% of studies.
- Best silver per fold 0.838, 0.858, 0.850, 0.828, 0.823 vs 0.829, 0.857, 0.842, 0.820, 0.821
  for mm alone: up in all 5 folds. Gold flat (0.865 vs 0.867, within noise). LB +0.003.
- Silver per fold has pointed the same way as the LB in every run so far; gold has not
  (v2 and mm + flip went the other way). Fold-paired silver is the main decision signal.

EfficientNet-B2 (ensemble member only)
- Same setup with a bigger backbone, 246 min vs 181 min for B0.
- Best silver per fold 0.845, 0.860, 0.848, 0.825, 0.818: wins 2 of 5 folds against B0, mean
  difference about 0. Best epoch was 2 to 4 in every fold; after that train loss keeps
  falling while validation loss rises, so B2 overfits the silver labels sooner than B0.
  Future B2 runs can stop at 4 to 5 epochs.
- Gold per finding differs from B0 in a useful way: better on PF OA, Contusion, Fracture,
  worse on MCL, Lateral Meniscus, Lateral OA.

B0 + B2 average (05d, kept)
- Rule fixed before running: adopt if the 50/50 average beats B0 in at least 4 of 5 folds on
  silver and the mean fold difference is positive.
- Silver per fold 0.855, 0.870, 0.861, 0.834, 0.829: wins 5 of 5, mean +0.010. Weights 0.3
  and 0.7 on B2 were slightly lower than 0.5, so no weight tuning.
- Gold macro 0.876 vs 0.865, paired bootstrap 95% interval of the difference -0.004 to +0.026.
- LB 0.919 vs 0.917. The silver gain (+0.010) was larger than the LB gain (+0.002); part of
  what averaging fixes is noise in the silver labels themselves.
- 05b now takes a RUNS list, needs exactly 5 checkpoints per run (equal weight per run) and
  stops if a checkpoint from any other run is attached. Studies that cannot be predicted get
  the per-finding median of the other test predictions instead of 0.5.

Next
- Public notebooks reach about 0.94 on the LB with report labels no better than ours (public
  label tables score about 0.89 to 0.90 against gold). The gap is on the image side: more
  series per study (up to 6), higher resolution (about 0.4 mm per pixel), attention pooling
  per finding and a transformer across slices.
- Planned: a multi-series, higher-resolution cache, then a model with learned pooling, then
  an ensemble with the current B0 + B2.
