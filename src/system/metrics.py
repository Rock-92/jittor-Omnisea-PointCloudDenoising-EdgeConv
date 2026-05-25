from typing import Dict, Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree

try:
    import point_cloud_utils as pcu

    HAS_PCU = True
except ImportError:
    pcu = None
    HAS_PCU = False


def normalize_to_unit_sphere(pc: np.ndarray):
    """Return point cloud normalized by bbox center and max radius."""
    center = (pc.max(axis=0) + pc.min(axis=0)) / 2.0
    pc_centered = pc - center
    scale = np.sqrt((pc_centered**2).sum(axis=1)).max()
    if scale < 1e-12:
        return pc_centered, center, scale
    return pc_centered / scale, center, scale


def chamfer_distance(
    pc_a: np.ndarray,
    pc_b: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Compute squared Chamfer Distance between two point clouds.

    Args:
        pc_a: Predicted/noisy point cloud, shape (N, 3).
        pc_b: Reference clean point cloud, shape (M, 3).
        normalize: If True, normalize pc_b to the unit sphere and apply the
            same transform to pc_a before nearest-neighbor search.
    """
    if normalize:
        pc_b, center, scale = normalize_to_unit_sphere(pc_b)
        if scale < 1e-12:
            return 0.0
        pc_a = (pc_a - center) / scale

    tree_b = cKDTree(pc_b)
    dist_a2b, _ = tree_b.query(pc_a, k=1)

    tree_a = cKDTree(pc_a)
    dist_b2a, _ = tree_a.query(pc_b, k=1)

    return float((dist_a2b**2).mean() + (dist_b2a**2).mean())


def point_to_surface_distance(
    pc: np.ndarray,
    mesh_v: Optional[np.ndarray],
    mesh_f: Optional[np.ndarray],
    normalize_ref_pc: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Compute mean squared point-to-surface distance.

    Args:
        pc: Query point cloud, shape (N, 3).
        mesh_v: Mesh vertices, shape (V, 3).
        mesh_f: Mesh faces, shape (F, 3).
        normalize_ref_pc: Optional reference point cloud. If provided, the
            reference normalization transform is applied to both pc and mesh_v.
    """
    if mesh_v is None or mesh_f is None:
        return None

    vertices = mesh_v.copy()
    if normalize_ref_pc is not None:
        _, center, scale = normalize_to_unit_sphere(normalize_ref_pc)
        if scale < 1e-12:
            return 0.0
        pc = (pc - center) / scale
        vertices = (vertices - center) / scale

    if HAS_PCU and pcu is not None:
        dists, _, _ = pcu.closest_points_on_mesh(
            pc.astype(np.float32),
            vertices.astype(np.float32),
            mesh_f.astype(np.int32),
        )
        return float((dists**2).mean())

    tree = cKDTree(vertices)
    dists, _ = tree.query(pc, k=1)
    return float((dists**2).mean())


def metric_to_score(val_pred: float, val_noisy: float) -> float:
    """Map a metric value to the competition's [0, 100] improvement score."""
    if val_noisy < 1e-15:
        return 100.0 if val_pred < 1e-15 else 0.0
    score = 100.0 * (1.0 - val_pred / val_noisy)
    return max(0.0, min(100.0, float(score)))


def compute_denoising_scores(
    pc_pred: np.ndarray,
    pc_noisy: np.ndarray,
    pc_clean: np.ndarray,
    mesh_v: Optional[np.ndarray] = None,
    mesh_f: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute CD/P2S metrics and competition-style scores for one sample.

    Returns a dict containing raw metric values, score values, and final_score.
    If mesh is unavailable, final_score falls back to CD score only.
    """
    pc_pred = np.asarray(pc_pred, dtype=np.float64)
    pc_noisy = np.asarray(pc_noisy, dtype=np.float64)
    pc_clean = np.asarray(pc_clean, dtype=np.float64)

    cd_pred = chamfer_distance(pc_pred, pc_clean, normalize=True)
    cd_noisy = chamfer_distance(pc_noisy, pc_clean, normalize=True)
    cd_score = metric_to_score(cd_pred, cd_noisy)

    p2s_pred = point_to_surface_distance(
        pc_pred,
        mesh_v,
        mesh_f,
        normalize_ref_pc=pc_clean,
    )
    p2s_noisy = point_to_surface_distance(
        pc_noisy,
        mesh_v,
        mesh_f,
        normalize_ref_pc=pc_clean,
    )
    p2s_score = None
    if p2s_pred is not None and p2s_noisy is not None:
        p2s_score = metric_to_score(p2s_pred, p2s_noisy)

    final_score = cd_score if p2s_score is None else 0.5 * cd_score + 0.5 * p2s_score

    return {
        "cd_pred": cd_pred,
        "cd_noisy": cd_noisy,
        "cd_score": cd_score,
        "p2s_pred": p2s_pred,
        "p2s_noisy": p2s_noisy,
        "p2s_score": p2s_score,
        "final_score": final_score,
    }


def aggregate_denoising_scores(records: Iterable[Dict[str, Optional[float]]]):
    """Average a sequence of per-sample denoising score records."""
    records = list(records)
    if not records:
        return None

    def mean_value(key: str):
        values = [r[key] for r in records if r.get(key) is not None]
        if not values:
            return None
        return float(np.mean(values))

    cd_score = mean_value("cd_score")
    p2s_score = mean_value("p2s_score")
    final_score = cd_score if p2s_score is None else 0.5 * cd_score + 0.5 * p2s_score

    return {
        "num_samples": len(records),
        "cd_pred": mean_value("cd_pred"),
        "cd_noisy": mean_value("cd_noisy"),
        "cd_score": cd_score,
        "p2s_pred": mean_value("p2s_pred"),
        "p2s_noisy": mean_value("p2s_noisy"),
        "p2s_score": p2s_score,
        "final_score": final_score,
    }
