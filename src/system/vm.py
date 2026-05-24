from typing import List, Dict, Optional

import numpy as np
import os

from .spec import DummySystem, DummyWriter
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
    
    # override functions in dummy system if you want to implement training/validation/prediction logic
