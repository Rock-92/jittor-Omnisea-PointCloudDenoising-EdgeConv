from typing import List, Dict, Optional

import numpy as np
import os
import trimesh
from pathlib import Path

from .spec import DummySystem, DummyWriter
from .metrics import compute_denoising_scores
from ..data.asset import Asset, Exporter

class VMWriter(DummyWriter):
    
    def __init__(self, save_dir: str="tmp_predict", save_name: str="predict", output_format: str="npy"):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.output_format = output_format

    def _get_dataset_roots(self, dataset_module=None):
        if dataset_module is None:
            return []
        config = getattr(dataset_module, "predict_dataset_config", None)
        if config is None:
            return []
        configs = config.values() if isinstance(config, dict) else [config]
        roots = []
        for item in configs:
            root = getattr(item.datapath, "input_dataset_dir", None)
            if root:
                roots.append(os.path.abspath(root))
        return roots

    def _get_output_dir(self, asset_path: str, dataset_module=None):
        asset_dir = os.path.abspath(os.path.dirname(asset_path))
        for root in self._get_dataset_roots(dataset_module):
            try:
                rel_dir = os.path.relpath(asset_dir, root)
            except ValueError:
                continue
            if rel_dir == "." or not rel_dir.startswith(".."):
                return os.path.join(self.save_dir, rel_dir)
        return os.path.join(self.save_dir, os.path.dirname(asset_path))
    
    def write(self, batch, prediction: List[Dict], dataset_module=None):
        pc_noisy_batch = batch['pc_noisy']
        for i, asset in enumerate(batch['asset']):
            path = asset.path
            assert path is not None, "asset path is None"
            dirname = self._get_output_dir(path, dataset_module=dataset_module)
            os.makedirs(dirname, exist_ok=True)
            denoised = prediction[i]['pc_denoised']
            if isinstance(denoised, np.ndarray):
                denoised_np = denoised
            else:
                denoised_np = denoised.numpy()
            expected_shape = tuple(pc_noisy_batch[i].shape)
            if denoised_np.shape != expected_shape:
                raise ValueError(
                    f"denoised point cloud shape mismatch for {path}: "
                    f"expected {expected_shape}, got {denoised_np.shape}"
                )
            if self.output_format == 'npy':
                np.save(os.path.join(dirname, f"{self.save_name}.npy"), denoised_np.astype(np.float32))
            else:
                Exporter.export_obj(denoised_np, os.path.join(dirname, f"{self.save_name}.obj"))

class VMSystem(DummySystem):
    
    def __init__(
        self,
        dataset_module,
        model,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        super().__init__(
            dataset_module=dataset_module,
            model=model,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            writer=writer,
            ckpt_save_dir=ckpt_save_dir,
            ckpt_save_name=ckpt_save_name,
        )

    def _source_mesh_path_from_cache(self, path: Optional[str]):
        if path is None:
            return None
        parts = Path(path).resolve().parts
        for cache_name in ("cache_clean", "cache_clean_points"):
            if cache_name not in parts:
                continue
            cache_idx = parts.index(cache_name)
            base = Path(*parts[:cache_idx]) if cache_idx > 0 else Path(".")
            rel_parts = parts[cache_idx + 1:-1]
            if rel_parts:
                return base / "dataset_clean" / Path(*rel_parts) / "models" / "model_normalized.obj"
        return None

    def _normalized_mesh(self, asset: Asset):
        vertices = asset.vertices
        faces = asset.faces
        if vertices is None or faces is None:
            mesh_path = self._source_mesh_path_from_cache(asset.path)
            if mesh_path is None or not mesh_path.exists():
                return None, None
            mesh = trimesh.load(str(mesh_path), process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
        meta = asset.meta or {}
        center = meta.get("normalize_center", None)
        scale = meta.get("normalize_scale", None)
        if center is None or scale is None or float(scale) < 1e-12:
            return vertices, faces
        vertices = (vertices - center) / float(scale)
        return vertices.astype(np.float32, copy=False), faces.astype(np.int32, copy=False)

    @staticmethod
    def _reshape_patch_tensor(tensor):
        if len(tensor.shape) == 4:
            batch_size, num_patches, patch_size, dim = tensor.shape
            return tensor.reshape(batch_size * num_patches, patch_size, dim), batch_size, num_patches
        if len(tensor.shape) == 3:
            batch_size, patch_size, dim = tensor.shape
            return tensor.reshape(batch_size, patch_size, dim), batch_size, 1
        return None, None, None

    def validation_metric_step(self, batch):
        if "pc_noisy" not in batch or "pc_clean" not in batch:
            return None

        pc_noisy, batch_size, num_patches = self._reshape_patch_tensor(batch["pc_noisy"])
        pc_clean, _, _ = self._reshape_patch_tensor(batch["pc_clean"])
        if pc_noisy is None or pc_clean is None:
            return None

        patch_seed = batch.get("patch_seed")
        if patch_seed is not None:
            patch_seed = patch_seed.reshape(pc_noisy.shape[0], 1, pc_noisy.shape[2])
            pc_noisy_abs = pc_noisy + patch_seed
            pc_clean_abs = pc_clean + patch_seed
        else:
            pc_noisy_abs = pc_noisy
            pc_clean_abs = pc_clean

        pc_pred, _ = self.model.denoise_langevin_dynamics(pc_noisy)
        pc_pred_abs = pc_pred + patch_seed if patch_seed is not None else pc_pred

        pred_np = pc_pred_abs.detach().numpy()
        noisy_np = pc_noisy_abs.detach().numpy()
        clean_np = pc_clean_abs.detach().numpy()
        assets = list(batch.get("asset", []))

        metrics = []
        for i in range(pred_np.shape[0]):
            try:
                mesh_v, mesh_f = None, None
                asset_idx = i // num_patches
                if asset_idx < len(assets):
                    mesh_v, mesh_f = self._normalized_mesh(assets[asset_idx])
                item = compute_denoising_scores(
                    pc_pred=pred_np[i],
                    pc_noisy=noisy_np[i],
                    pc_clean=clean_np[i],
                    mesh_v=mesh_v,
                    mesh_f=mesh_f,
                )
                metrics.append(item)
            except Exception as exc:
                self._validation_score_errors.append(f"patch {i}: {exc}")
        return metrics
