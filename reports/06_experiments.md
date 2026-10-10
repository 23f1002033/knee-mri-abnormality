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
| Finding | baseline | v2 | flip | mm |
| --- | --- | --- | --- | --- |
| ACL | 0.912 | 0.898 | 0.914 | 0.915 |
| MCL | 0.785 | 0.836 | 0.846 | 0.800 |
| Medial Meniscus | 0.838 | 0.858 | 0.838 | 0.898 |
| Lateral Meniscus | 0.778 | 0.779 | 0.848 | 0.835 |
| Medial OA | 0.974 | 0.972 | 0.964 | 0.953 |
| Lateral OA | 0.857 | 0.861 | 0.778 | 0.901 |
| PF OA | 0.757 | 0.789 | 0.770 | 0.746 |
| Effusion | 0.955 | 0.937 | 0.960 | 0.920 |
| Synovitis | 0.752 | 0.766 | 0.774 | 0.774 |
| Baker's | 0.902 | 0.928 | 0.904 | 0.973 |
| Contusion | 0.850 | 0.839 | 0.862 | 0.821 |
| Fracture | 0.875 | 0.903 | 0.822 | 0.871 |
| Macro | 0.853 | 0.864 | 0.857 | 0.867 |

Next
- Combined model: mm cache + flip + v1 labels. Needs the laterality classifier retrained on
  mm coronal slices (05c-laterality-mm) so inference sees the same geometry.
- Then, one change at a time on top of it: a bigger backbone, self-training, ensembling.
