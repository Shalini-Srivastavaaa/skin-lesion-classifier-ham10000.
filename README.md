HAM10000 Skin Lesion Classification — Deep Learning / CNN / ResNet-50

A deep-learning image-classification project using a CNN-based ResNet-50 architecture to classify dermoscopic skin-lesion images into seven diagnostic categories.

A reproducible seven-class skin lesion image classification project built with **PyTorch** and **ImageNet-pretrained ResNet-50** on the **HAM10000** dataset.

The project focuses not only on model accuracy, but also on **evaluation integrity, lesion-level data splitting, controlled experimentation, class imbalance, error analysis, and reproducibility**.

> **Final test performance:** 82.23% accuracy | 0.666 macro-F1

---

## Project Overview

The goal is to classify dermoscopic skin-lesion images into seven diagnostic classes.

The model uses transfer learning with ResNet-50 and is evaluated using a lesion-level train/validation/test split to reduce leakage caused by multiple images belonging to the same lesion.

The project follows:

```text
Data quality & EDA
        ↓
Lesion-level split
        ↓
Baseline
        ↓
Controlled experiments
        ↓
Validation-based model selection
        ↓
Single final test evaluation
        ↓
Error analysis & conclusions
```

---

## Dataset

**Dataset:** HAM10000 — Human Against Machine with 10,000 training images

The project uses:

* 10,015 dermoscopic images
* 7 lesion classes
* 7,470 unique lesions
* Multiple images may correspond to the same lesion

Class labels:

| ID | Label |
| -: | ----- |
|  0 | nv    |
|  1 | bkl   |
|  2 | mel   |
|  3 | bcc   |
|  4 | akiec |
|  5 | vasc  |
|  6 | df    |

The dataset is highly imbalanced, with `nv` representing the largest class.

---

## Why Lesion-Level Splitting?

A major consideration in HAM10000 is that multiple images can belong to the same lesion.

A naive image-level random split could place different images of the same lesion into both training and test sets, making evaluation overly optimistic.

To reduce this risk, I:

1. Grouped metadata by `lesion_id`.
2. Performed a stratified split at the lesion level.
3. Used a 70/15/15 train-validation-test split.
4. Saved the resulting CSV splits to Google Drive.
5. Verified that no lesion IDs overlap between train, validation, and test sets.

Final split sizes:

| Split                 |                           Images |
| --------------------- | -------------------------------: |
| Train                 |                            7,002 |
| Validation            |                            1,532 |
| Test                  | 1,481 before corruption handling |
| Final test evaluation |                            1,480 |

---

## Model

### Backbone

```text
ResNet-50
ImageNet pretrained weights
```

The original ResNet-50 classification head was replaced with a seven-class linear layer:

```python
nn.Linear(2048, 7)
```

The model was fine-tuned end-to-end.

### Training

```text
Optimizer: AdamW
Learning rate: 1e-4
Epochs: 5
Batch size: 32
```

Training and evaluation used PyTorch `DataLoader`s with multiple workers and pinned memory.

---

## Preprocessing

### Baseline training augmentation

* Random horizontal flip
* Resize to 232
* Center crop to 224
* ImageNet normalization

### Stronger augmentation used in the selected experiments

```text
RandomResizedCrop(224, scale=(0.8, 1.0))
RandomHorizontalFlip()
RandomVerticalFlip()
RandomRotation(15°)
ImageNet normalization
```

Validation and test images used the standard preprocessing associated with the pretrained ResNet-50 weights.

---

## Experiments

The experiments were designed as controlled comparisons rather than random model trials.

| Experiment       | Main change                                 | Validation Accuracy | Validation Macro-F1 | Decision     |
| ---------------- | ------------------------------------------- | ------------------: | ------------------: | ------------ |
| Baseline         | Original training setup                     |              80.35% |               0.635 | Baseline     |
| Experiment 1     | Strong augmentation + WeightedRandomSampler |              78.07% |               0.646 | Rejected     |
| **Experiment 2** | Strong augmentation + normal sampling       |          **81.85%** |           **0.674** | **Selected** |
| Experiment 3     | Weighted Cross-Entropy                      |              78.52% |               0.624 | Rejected     |

### Experiment 1

Stronger augmentation was introduced while retaining weighted oversampling.

Although some minority-class metrics improved, the overall model did not improve sufficiently.

**Decision: rejected.**

### Experiment 2

The stronger augmentation was retained, but `WeightedRandomSampler` was removed.

The loader used standard shuffled sampling instead.

This produced the best validation result:

```text
Accuracy: 81.85%
Macro-F1: 0.674
```

**Decision: selected as the final candidate.**

### Experiment 3

I tested class-weighted Cross-Entropy using inverse-square-root frequency weighting.

The experiment increased melanoma recall but reduced overall performance:

```text
Accuracy: 78.52%
Macro-F1: 0.624
```

**Decision: rejected.**

---

## Final Model

The final selected model is **Experiment 2**.

Configuration:

```text
ResNet-50
ImageNet pretrained
224 × 224 input
Strong augmentation
No WeightedRandomSampler
AdamW
Learning rate = 1e-4
5 epochs
Cross-Entropy Loss
```

The trained model checkpoint was saved so that the model can be restored without retraining.

---

## Final Test Evaluation

The test set was not used for model selection or tuning.

After selecting Experiment 2 using validation performance, the model was evaluated once on the held-out test set.

One test image was found to be corrupted/unreadable:

```text
ISIC_0026915.jpg
```

The image was excluded from the final evaluation and the exclusion was recorded.

### Final results

| Metric       |       Test |
| ------------ | ---------: |
| Accuracy     | **82.23%** |
| Macro-F1     |  **0.666** |
| Weighted-F1  |  **0.826** |
| Test samples |  **1,480** |

### Per-class performance

| Class | Precision | Recall |        F1 |
| ----- | --------: | -----: | --------: |
| nv    |     0.935 |  0.907 | **0.921** |
| bkl   |     0.674 |  0.702 |     0.688 |
| mel   |     0.588 |  0.570 |     0.578 |
| bcc   |     0.679 |  0.746 |     0.711 |
| akiec |     0.453 |  0.630 |     0.527 |
| vasc  |     0.882 |  0.789 |     0.833 |
| df    |     0.360 |  0.450 |     0.400 |

---

## Error Analysis

The final model performs strongly on the dominant `nv` class but has difficulty with several minority classes.

A particularly important confusion is between:

```text
melanoma ↔ nevus
```

For the final test set, 94 of 165 melanoma cases were correctly classified, while 37 were predicted as `nv`.

This highlights a key limitation of the image-only classifier: visually similar lesion categories remain difficult to separate.

---

## Why Macro-F1?

HAM10000 is highly imbalanced.

A model can obtain high accuracy by performing very well on the dominant `nv` class while underperforming on rare classes.

Macro-F1 gives equal importance to each of the seven classes, so it provides a more informative summary of multiclass performance.

For this reason, model selection considered both:

* accuracy
* macro-F1
* per-class metrics
* confusion matrix

---

## Reproducibility

The project saves:

* train/validation/test CSV splits
* model checkpoints
* optimizer state
* training-loss history
* validation predictions
* validation metrics
* final test predictions
* final test metrics

This makes the experiments reproducible and allows trained models to be restored without retraining from scratch.

---

## Example Checkpoint Artifacts

```text
resnet50_ham10000_baseline_v2.pth

resnet50_ham10000_experiment2_no_sampler.pth

resnet50_ham10000_experiment2_no_sampler_validation.pth

resnet50_ham10000_experiment3_weighted_ce.pth

resnet50_ham10000_experiment3_weighted_ce_validation.pth

resnet50_ham10000_experiment2_final_test_results.pth
```

---

## Limitations

This project is a machine-learning research/portfolio implementation rather than a clinical diagnostic system.

Important limitations include:

* The model is trained and evaluated on a single public dataset.
* There is no external clinical validation dataset.
* The class distribution is highly imbalanced.
* Rare classes remain difficult.
* Melanoma and nevus remain an important source of confusion.
* The model operates on images only and does not use richer clinical context.
* Probability calibration was not performed.
* The results should not be interpreted as clinical diagnostic accuracy.

---

## Technologies

```text
Python
PyTorch
Torchvision
Scikit-learn
Pandas
NumPy
Matplotlib
Seaborn
Google Colab
Google Drive
```

---

## Project Outcome

This project demonstrates a complete deep-learning workflow:

```text
Dataset analysis
      ↓
Leakage-aware splitting
      ↓
Data pipeline
      ↓
Transfer learning
      ↓
Controlled experimentation
      ↓
Error analysis
      ↓
Validation-based model selection
      ↓
Final held-out test evaluation
      ↓
Reproducible artifacts
```

The final selected model achieved:

**82.23% test accuracy**
**0.666 test macro-F1**

The main value of the project is not only the final score, but the disciplined process used to obtain and evaluate it.
