from collections import defaultdict
from jittor import optim
from typing import Dict, List, Optional
from tqdm import tqdm

import jittor as jt
import numpy as np
import os
import zipfile

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec
from .metrics import aggregate_denoising_scores, compute_denoising_scores

def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x

def _to_jittor(value):
    if isinstance(value, np.ndarray):
        return jt.array(value)
    if isinstance(value, jt.Var):
        return value
    if isinstance(value, dict):
        return {k: _to_jittor(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_jittor(v) for v in value)
    if isinstance(value, list):
        return [_to_jittor(v) for v in value]
    return value

def get_optimizer(optimizer_config, model):
    __target__ = optimizer_config.pop('__target__')
    MAPPING = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
    }
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    optimizer = OptimizerClass(model.parameters(), **optimizer_config)
    return optimizer

class DummyWriter():
    
    def __init__(self):
        pass
    
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule]=None):
        pass

class DummySystem():
    
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self.epochs = trainer_config.get('epochs', 1)
        self.log_score = trainer_config.get('log_score', True)
        self.score_every_n_epochs = trainer_config.get('score_every_n_epochs', 1)
        self.score_max_samples = trainer_config.get('score_max_samples', 0)
        self.create_submission_on_train_end = trainer_config.get(
            'create_submission_on_train_end', False
        )
        self.submission_use_best = trainer_config.get('submission_use_best', True)
        self.submission_source_dir = trainer_config.get('submission_source_dir', None)
        self.submission_zip_path = trainer_config.get(
            'submission_zip_path', os.path.join('outputs', 'result.zip')
        )
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None
        
        self._validation_loss = defaultdict(list)
        self._validation_scores = []
        self._validation_score_errors = []
        self._validation_score_summary = None
        self.best_loss = None
        self.best_epoch = None
        self.current_epoch = 0
    
    def forward(self, batch, validate: bool=False): # return loss sum
        loss_dict = self.model.training_step(batch)
        assert isinstance(loss_dict, dict), "loss_dict must be a dict containing loss/metrics"
        assert self.loss_config is not None, "do not have loss_confing"
        loss_sum = 0.
        if validate:
            assets: List[Asset] = [a for a in batch['asset']]
            cls = assets[0].cls # guaranteed to be the same cls in dataloader
            for name in loss_dict:
                assert name in self.loss_config, f'unspecified loss {name}'
                self._validation_loss[f"val/{cls}_{name}"].append(_get_item(loss_dict[name]))
                loss_sum += self.loss_config[name] * loss_dict[name]
            self._validation_loss[f"val/{cls}_loss_sum"].append(_get_item(loss_sum))
            # TODO: log
            # self.log('val/loss_sum', loss_sum, prog_bar=True, logger=True, sync_dist=True, batch_size=len(assets))
        else:
            for name in loss_dict:
                assert name in self.loss_config, f"unspecified loss name: `{name}`"
                if self.loss_config[name] > 0:
                    loss_sum += self.loss_config[name] * loss_dict[name]
            loss_dict['loss_sum'] = loss_sum
            # TODO: log
            # # add train prefix to loss_dict
            # prefixed_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            # d = dict(sorted(prefixed_loss_dict.items()))
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_batch_start(self):
        pass
    
    def training_step(self, batch):
        return self.forward(batch, validate=False)
    
    def on_train_batch_end(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
        self._validation_scores = []
        self._validation_score_errors = []
        self._validation_score_summary = None
    
    def on_validation_batch_start(self):
        pass
    
    def validation_step(self, batch):
        assert self.loss_config is not None, "do not have loss_confing"
        loss = self.forward(batch, validate=True)
        self.validation_score_step(batch)
        return loss
    
    def on_validation_batch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        loss_summary = self._get_validation_loss_summary()
        if loss_summary is not None:
            print(
                f"Epoch {self.current_epoch}, Validate Loss: "
                f"loss_sum={loss_summary['loss_sum']:.8f}, "
                f"samples={loss_summary['num_samples']}"
            )
            self._save_best_checkpoint(loss_summary)

        if not self._should_log_score():
            return
        summary = aggregate_denoising_scores(self._validation_scores)
        if summary is None:
            print(f"Epoch {self.current_epoch}, Validate Score: no valid samples")
            return
        self._validation_score_summary = summary

        msg = (
            f"Epoch {self.current_epoch}, Validate Score: "
            f"final={summary['final_score']:.4f}, "
            f"cd_score={summary['cd_score']:.4f}, "
            f"cd_pred={summary['cd_pred']:.8f}, "
            f"cd_noisy={summary['cd_noisy']:.8f}, "
        )
        if summary['p2s_score'] is not None:
            msg += (
                f"p2s_score={summary['p2s_score']:.4f}, "
                f"p2s_pred={summary['p2s_pred']:.8f}, "
                f"p2s_noisy={summary['p2s_noisy']:.8f}, "
            )
        msg += f"samples={summary['num_samples']}"
        print(msg)
        if self._validation_score_errors:
            print(
                f"Epoch {self.current_epoch}, Validate Score warnings: "
                f"{len(self._validation_score_errors)} samples skipped"
            )
    
    def on_before_optimizer_step(self, optimizer):
        pass

    def prepare_batch(self, batch):
        return _to_jittor(batch)

    def _get_validation_loss_summary(self):
        losses = []
        for key, values in self._validation_loss.items():
            if key.endswith("_loss_sum"):
                losses.extend(values)
        if not losses:
            return None
        return {
            "loss_sum": float(np.mean(losses)),
            "num_samples": len(losses),
        }

    def _save_best_checkpoint(self, summary):
        loss = summary.get('loss_sum', None)
        if loss is None:
            return
        if self.best_loss is not None and loss >= self.best_loss:
            return

        self.best_loss = loss
        self.best_epoch = self.current_epoch
        os.makedirs(self.ckpt_save_dir, exist_ok=True)

        best_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_best.pkl')
        self.model.save(best_path)

        meta_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_best.txt')
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"best_epoch: {self.best_epoch}\n")
            f.write(f"best_loss: {self.best_loss:.8f}\n")
            for key in ["loss_sum", "num_samples"]:
                value = summary.get(key, None)
                if value is not None:
                    f.write(f"{key}: {value}\n")

        print(
            f"Epoch {self.current_epoch}, Best checkpoint saved: "
            f"{best_path} (loss={self.best_loss:.8f})"
        )

    def _should_log_score(self):
        if not self.log_score:
            return False
        if self.score_every_n_epochs <= 0:
            return False
        return (self.current_epoch + 1) % self.score_every_n_epochs == 0

    def _get_normalized_mesh(self, asset: Asset):
        if asset.vertices is None or asset.faces is None:
            return None, None
        meta = asset.meta or {}
        center = meta.get('normalize_center', None)
        scale = meta.get('normalize_scale', None)
        if center is None or scale is None or scale < 1e-12:
            return asset.vertices, asset.faces
        vertices = (asset.vertices - center) / scale
        return vertices, asset.faces

    @jt.no_grad()
    def validation_score_step(self, batch):
        if not self._should_log_score():
            return
        if self.score_max_samples > 0 and len(self._validation_scores) >= self.score_max_samples:
            return

        assets: List[Asset] = batch.get('asset', [])
        for asset in assets:
            if self.score_max_samples > 0 and len(self._validation_scores) >= self.score_max_samples:
                return
            if asset.sampled_vertices is None or asset.sampled_vertices_noisy is None:
                self._validation_score_errors.append("missing point cloud")
                continue
            try:
                pc_noisy = asset.sampled_vertices_noisy.astype(np.float32)
                pred = self.model.predict_step({"pc_noisy": jt.array(pc_noisy[None, ...])})
                pc_pred = pred[0]["pc_denoised"]
                mesh_v, mesh_f = self._get_normalized_mesh(asset)
                score = compute_denoising_scores(
                    pc_pred=pc_pred,
                    pc_noisy=asset.sampled_vertices_noisy,
                    pc_clean=asset.sampled_vertices,
                    mesh_v=mesh_v,
                    mesh_f=mesh_f,
                )
                self._validation_scores.append(score)
            except Exception as e:
                path = asset.path or "<unknown>"
                self._validation_score_errors.append(f"{path}: {e}")
    
    def on_predict_epoch_start(self):
        pass
    
    def on_predict_batch_start(self):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model.predict_step(batch)
    
    def on_predict_batch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass

    def on_train_end(self):
        if self.create_submission_on_train_end:
            self.create_submission()

    def create_submission(self):
        if self.writer is None:
            raise ValueError(
                "create_submission_on_train_end=True requires a writer config."
            )

        if self.submission_use_best:
            best_path = os.path.join(
                self.ckpt_save_dir, f'{self.ckpt_save_name}_best.pkl'
            )
            if os.path.exists(best_path):
                self.model.load(best_path)
                print(f"Loaded best checkpoint for submission: {best_path}")
            else:
                print(
                    f"Best checkpoint not found: {best_path}. "
                    "Using current model for submission."
                )

        print("Generating submission predictions...")
        self.predict()
        self._zip_submission()

    def _zip_submission(self):
        source_dir = self.submission_source_dir
        if source_dir is None:
            source_dir = getattr(self.writer, 'save_dir', None)
        if source_dir is None:
            raise ValueError(
                "Cannot create result.zip because submission_source_dir is not set "
                "and writer has no save_dir."
            )

        source_dir = os.path.abspath(source_dir)
        zip_path = os.path.abspath(self.submission_zip_path)
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(
                f"Submission source directory not found: {source_dir}"
            )

        npy_files = []
        for root, _, files in os.walk(source_dir):
            for filename in files:
                if filename.endswith(".npy"):
                    path = os.path.join(root, filename)
                    arcname = os.path.relpath(path, source_dir)
                    npy_files.append((path, arcname))

        if not npy_files:
            raise FileNotFoundError(
                f"No .npy files found under submission source: {source_dir}"
            )

        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path, arcname in sorted(npy_files, key=lambda x: x[1]):
                zf.write(path, arcname)

        print(
            f"Submission zip saved: {zip_path} "
            f"({len(npy_files)} denoised point clouds)"
        )
    
    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.model.train()
            self.on_train_epoch_start()
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            pbar = tqdm(train_dataloader, total=len(train_dataloader)//train_dataloader.batch_size) # type: ignore
            for batch in pbar:
                batch = self.prepare_batch(batch)
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                pbar.set_description(f"Epoch {epoch}, Loss: {_get_item(loss)}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.on_train_batch_end()
            self.on_train_epoch_end()
            
            self.model.eval()
            validate_dataloader = self.dataset_module.validate_dataloader()
            if validate_dataloader is not None:
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size)
                        for batch in pbar:
                            batch = self.prepare_batch(batch)
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                            self.on_validation_batch_end()
                else:
                    pbar = tqdm(validate_dataloader, total=len(validate_dataloader)//validate_dataloader.batch_size)
                    for batch in pbar:
                        batch = self.prepare_batch(batch)
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
                self.on_validation_epoch_end()
            
            checkpoint_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_{epoch}.pkl')
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            self.model.save(checkpoint_path)
        self.on_train_end()
    
    def predict(self):
        # only iterate once
        self.model.set_predict(True)
        self.model.eval()
        self.on_predict_epoch_start()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        for dataloader_name, dataloader in predict_dataloader.items():
            pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                batch = self.prepare_batch(batch)
                self.on_predict_batch_start()
                output = self.predict_step(batch, batch_idx)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
                pbar.set_description(f"Predicting {dataloader_name}, Batch {batch_idx}")
        self.on_predict_epoch_end()
