## Notebook 05: first image model (baseline)

Notebooks: 05a-train (training), 05b-infer (inference and submission), 05c-laterality
(side classifier, for the next experiment). Shared code: src/knee_prep.py, src/knee_model.py.

Setup (experiment b0-v1-noflip)
- Input: the 02b cache, three planes per study, 32 slices of 256x256, random 224 crop
  in training and centre crop at inference.
- Model: EfficientNet-B0 (ImageNet weights) over 2.5D images (three neighbouring slices,
  15 per plane), mean and max pooling per plane, three planes joined, one linear head
  for 12 findings. A missing plane is masked.
- Targets: silver score / 4.5 from silver_scores.csv (v1). Soft-label BCE.
- No laterality flip and no horizontal flip (a flip would swap medial and lateral).
- 5 folds grouped by language x vendor (site proxy). Inside a fold the best epoch is
  picked on silver validation AUC only; gold is used only at the end.
- AdamW, one-cycle LR 3e-4, 8 epochs, batch 2 studies with 4-step gradient accumulation,
  fp16. Batch 8 ran out of memory on a T4.

Training
- About 7 to 8 minutes per epoch per fold, one fold per GPU, 167 minutes for 5 folds.
- The data loader is CPU bound: 4 CPUs at full load while each GPU sits near 50%.
- Best silver macro per fold: 0.818, 0.846, 0.840, 0.822, 0.808. Best epochs 7, 4, 3, 4, 7.
  Silver AUC levels off after epoch 3 to 4 while training loss keeps falling.

Out-of-fold results (all 4407 studies; gold = the 58 radiologist-labelled studies)
| Finding | Silver | Gold |
| --- | --- | --- |
| ACL | 0.866 | 0.912 |
| MCL | 0.749 | 0.785 |
| Medial Meniscus | 0.870 | 0.838 |
| Lateral Meniscus | 0.794 | 0.778 |
| Medial OA | 0.851 | 0.974 |
| Lateral OA | 0.780 | 0.857 |
| PF OA | 0.766 | 0.757 |
| Effusion | 0.794 | 0.955 |
| Synovitis | 0.803 | 0.752 |
| Baker's | 0.861 | 0.902 |
| Contusion | 0.837 | 0.850 |
| Fracture | 0.834 | 0.875 |
| Macro | 0.817 | 0.853 |

- Gold is above silver: the model learns findings the reports leave out.

Leaderboard
- Public LB 0.908, average of the 5 fold models.
- Expected about 0.86 to 0.87 from gold out-of-fold. The gap (+0.055) means the
  site-grouped CV is pessimistic: each fold is scored on sites it never saw, while the
  hidden test most likely comes from the same sites as train. Gold out-of-fold also rests
  on only 58 studies, and the public LB is only part of the test set.
- Use CV for comparing experiments, not for predicting the absolute LB score.

Inference
- 05b ran on CPU (accelerator left off) and still finished. Next runs use the GPU, which
  matters once models get bigger.
- Hidden-test path verified earlier with a 0.5 prior submission: offline decoders,
  series fallback, and a prior submission written first.

Next
- Laterality flip (05c classifier, then to_canonical in training).
- Physical-scale cache: fixed mm per pixel and a knee-centred crop.
- 6 epochs instead of 8 for experiments; self-training; a bigger backbone.
