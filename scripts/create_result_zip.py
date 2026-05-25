#!/usr/bin/env python
"""Generate outputs1/result.zip from a trained VM checkpoint."""

from pathlib import Path
import argparse
import os
import random
import shutil
import sys
import zipfile

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run denoising inference and create a competition-format result.zip."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs1/experiments/vm2/checkpoint_43.pkl",
        help="Checkpoint path relative to repo root, or an absolute path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs1/submission_checkpoint_43",
        help="Directory used to write denoised .npy files before zipping.",
    )
    parser.add_argument(
        "--zip-path",
        default="outputs1/result.zip",
        help="Path of the generated submission zip.",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/predict.yaml",
        help="Predict data config.",
    )
    parser.add_argument(
        "--transform-config",
        default="configs/transform/predict.yaml",
        help="Predict transform config.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/vm.yaml",
        help="Model config.",
    )
    parser.add_argument(
        "--system-config",
        default="configs/system/vm.yaml",
        help="System config.",
    )
    parser.add_argument(
        "--test-list",
        default="datalist/test.txt",
        help="Datalist used for expected submission file structure.",
    )
    parser.add_argument(
        "--test-root",
        default="test_noisy",
        help="Root directory of noisy test inputs.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run without CUDA. By default CUDA is enabled.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not delete the output directory before inference.",
    )
    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_under_repo(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def clean_previous_outputs(root, output_dir, zip_path, keep_output):
    outputs_root = (root / "outputs1").resolve()
    output_dir = output_dir.resolve()
    zip_path = zip_path.resolve()

    if not keep_output and output_dir.exists():
        if outputs_root not in output_dir.parents and output_dir != outputs_root:
            raise RuntimeError(f"refusing to remove outside outputs1: {output_dir}")
        shutil.rmtree(output_dir)
        print(f"Removed old output dir: {output_dir}")

    if zip_path.exists():
        if outputs_root not in zip_path.parents:
            raise RuntimeError(f"refusing to remove outside outputs1: {zip_path}")
        zip_path.unlink()
        print(f"Removed old zip: {zip_path}")


def load_yaml(path):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(str(path)))


def read_test_items(test_list):
    with open(test_list, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def validate_and_zip(root, output_dir, zip_path, test_root, test_items):
    zip_items = []
    for rel in test_items:
        noisy_path = test_root / rel / "noisy.npy"
        pred_path = output_dir / rel / "denoised.npy"

        if not noisy_path.exists():
            raise FileNotFoundError(f"missing noisy input: {noisy_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"missing prediction: {pred_path}")

        noisy = np.load(noisy_path)
        pred = np.load(pred_path)
        if pred.dtype != np.float32:
            raise TypeError(f"{pred_path} dtype must be float32, got {pred.dtype}")
        if pred.shape != noisy.shape:
            raise ValueError(
                f"{pred_path} shape mismatch: expected {noisy.shape}, got {pred.shape}"
            )
        if pred.ndim != 2 or pred.shape[1] != 3:
            raise ValueError(f"{pred_path} must have shape (N, 3), got {pred.shape}")

        arcname = (Path(rel) / "denoised.npy").as_posix()
        zip_items.append((pred_path, arcname))

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in sorted(zip_items, key=lambda item: item[1]):
            zf.write(path, arcname)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(zf.namelist())
    expected = sorted(arcname for _, arcname in zip_items)
    if names != expected:
        missing = sorted(set(expected) - set(names))[:5]
        extra = sorted(set(names) - set(expected))[:5]
        raise RuntimeError(f"zip contents mismatch: missing={missing}, extra={extra}")

    print(f"Saved submission zip: {zip_path}")
    print(
        f"Validated {len(zip_items)} files: float32, shape matches noisy input, "
        "zip structure OK."
    )


def main():
    args = parse_args()
    root = repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root))

    checkpoint = resolve_under_repo(root, args.checkpoint)
    output_dir = resolve_under_repo(root, args.output_dir)
    zip_path = resolve_under_repo(root, args.zip_path)
    data_config_path = resolve_under_repo(root, args.data_config)
    transform_config_path = resolve_under_repo(root, args.transform_config)
    model_config_path = resolve_under_repo(root, args.model_config)
    system_config_path = resolve_under_repo(root, args.system_config)
    test_list = resolve_under_repo(root, args.test_list)
    test_root = resolve_under_repo(root, args.test_root)

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not test_list.exists():
        raise FileNotFoundError(f"test list not found: {test_list}")

    clean_previous_outputs(root, output_dir, zip_path, args.keep_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import jittor as jt

    jt.flags.use_cuda = 0 if args.cpu else 1
    from src.data.dataset import DatasetConfig, PCDatasetModule
    from src.model.parse import get_model
    from src.system.parse import get_system, get_writer

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_config = load_yaml(data_config_path)
    transform_config = load_yaml(transform_config_path)
    model_config = load_yaml(model_config_path)
    system_config = load_yaml(system_config_path)

    predict_dataset_config = DatasetConfig.parse(
        **data_config["predict_dataset"]
    ).split_by_cls()

    model = get_model(model_config=model_config, transform_config=transform_config)
    model.load(str(checkpoint))

    dataset_module = PCDatasetModule(
        process_fn=model._process_fn,
        predict_dataset_config=predict_dataset_config,
        predict_transform=model.get_predict_transform(),
        debug=False,
    )
    writer = get_writer(
        __target__="vm",
        save_dir=str(output_dir),
        save_name="denoised",
        output_format="npy",
    )
    system = get_system(
        dataset_module=dataset_module,
        model=model,
        writer=writer,
        **system_config,
    )

    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Writing denoised files under: {output_dir}")
    system.predict()

    test_items = read_test_items(test_list)
    validate_and_zip(root, output_dir, zip_path, test_root, test_items)


if __name__ == "__main__":
    main()
