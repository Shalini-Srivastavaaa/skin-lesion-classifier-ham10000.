# Dataset

This directory contains the dataset split definitions used by the HAM10000 skin-lesion classification experiments.

## Dataset Source

The project uses the **HAM10000 (Human Against Machine with 10,000 training images)** dataset.

The original HAM10000 image files and metadata are **not included in this repository**.

Please download the dataset from its original distribution and place the image files locally under:

```text
data/images/
```

The expected image naming format is:

```text
ISIC_*.jpg
```

The original metadata file should be available locally as:

```text
data/HAM10000_metadata.csv
```


The raw image files and original metadata are intentionally excluded from GitHub because of their size and dataset licensing/distribution considerations.

## Split Files

The repository includes the three split files used in the experiments:

* `train.csv` — training split
* `val.csv` — validation split
* `test.csv` — test split

The splits were created at the **lesion level using `lesion_id`**, rather than randomly splitting individual images.

This is important because HAM10000 contains multiple images belonging to the same lesion. Splitting by lesion helps prevent images from the same lesion from appearing across training, validation, and test sets.

The final experiment used approximately:

```text
Train      7,002 images
Validation 1,532 images
Test       1,481 images
```

One corrupted/unreadable test image (`ISIC_0026915.jpg`) was excluded from the final test evaluation, resulting in **1,480 evaluated test images**.

## Reproducing the Dataset Setup

After downloading HAM10000:

1. Place the image files under `data/images/`.
2. Place `HAM10000_metadata.csv` under `data/`.
3. Keep the provided `train.csv`, `val.csv`, a
