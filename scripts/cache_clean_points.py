#!/usr/bin/env python
"""Pre-sample clean point clouds from ShapeNet OBJ meshes."""

from pathlib import Path
import argparse
import sys

import numpy as np
from tqdm import tqdm
import trimesh


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_mesh(path: Path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _read_datalists(paths):
    items = []
    seen = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rel = line.strip()
                if not rel or rel in seen:
                    continue
                seen.add(rel)
                items.append(rel)
    return items


def main():
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    from src.data.utils import sample_vertex_groups

    parser = argparse.ArgumentParser(
        description="Cache clean point clouds sampled from dataset_clean OBJ meshes."
    )
    parser.add_argument("--input-dir", default="dataset_clean")
    parser.add_argument("--output-dir", default="cache_clean")
    parser.add_argument("--data-name", default="models/model_normalized.obj")
    parser.add_argument("--output-name", default="clean.npy")
    parser.add_argument(
        "--lists",
        nargs="+",
        default=["datalist/train.txt"],
        help="Datalist files containing paths such as shapenet/<synset>/<model>.",
    )
    parser.add_argument("--num-samples", type=int, default=32768)
    parser.add_argument("--num-vertex-samples", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = (repo_root / args.input_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    datalists = [(repo_root / p).resolve() for p in args.lists]
    items = _read_datalists(datalists)

    missing = 0
    failed = 0
    cached = 0

    for rel in tqdm(items, desc="Caching clean point clouds"):
        src = input_dir / rel / args.data_name
        dst = output_dir / rel / args.output_name
        if dst.exists() and not args.overwrite:
            cached += 1
            continue
        if not src.exists():
            missing += 1
            continue

        try:
            mesh = _load_mesh(src)
            points, _, _, _ = sample_vertex_groups(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.faces),
                num_samples=args.num_samples,
                num_vertex_samples=args.num_vertex_samples,
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, points.astype(np.float32))
            cached += 1
        except Exception as exc:
            failed += 1
            print(f"failed: {src} ({exc})")

    print(
        f"cache done: cached={cached}, missing={missing}, failed={failed}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
