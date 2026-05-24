# Data Directory

This directory is reserved for dataset documentation. Do not commit raw
datasets, generated point clouds, checkpoints, or submission artifacts.

Expected training data location:

```text
dataset_clean/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
```

Expected test data location:

```text
test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

The sample lists used by the baseline are stored in `datalist/`.
