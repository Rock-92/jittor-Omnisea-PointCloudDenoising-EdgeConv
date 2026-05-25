from collections import defaultdict
from jittor import optim
from typing import Dict, List, Optional
from tqdm import tqdm

import csv
import jittor as jt
import numpy as np
import os
import zipfile

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec
from .metrics import aggregate_denoising_scores

def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x

def _to_jittor(value):
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32, copy=False)
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
        self.training_log_path = trainer_config.get(
            'training_log_path', os.path.join('outputs', 'training_log.csv')
        )
        self.lr_decay_patience = trainer_config.get('lr_decay_patience', 10)
        self.lr_decay_factor = trainer_config.get('lr_decay_factor', 0.8)
        self.lr_decay_min = trainer_config.get('lr_decay_min', 0.0)
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None
        
        self._train_loss = defaultdict(list)
        self._validation_loss = defaultdict(list)
        self._validation_scores = []
        self._validation_score_errors = []
        self._validation_score_summary = None
        self._last_train_loss_dict = {}
        self._best_checkpoint_updated = False
        self._epochs_since_best_checkpoint = 0
        self._last_lr_decayed = False
        self.best_loss = None
        self.best_score = None
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
            self._last_train_loss_dict = loss_dict.copy()
            # TODO: log
            # # add train prefix to loss_dict
            # prefixed_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            # d = dict(sorted(prefixed_loss_dict.items()))
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        self._train_loss = defaultdict(list)
        self._last_train_loss_dict = {}
        self._best_checkpoint_updated = False
        self._last_lr_decayed = False
    
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
        self.record_validation_scores(self.validation_metric_step(batch))
        return loss

    def validation_metric_step(self, batch):
        return None

    def record_validation_scores(self, metrics):
        if not self._should_log_score() or metrics is None:
            return
        if isinstance(metrics, dict):
            metrics = [metrics]
        for metric in metrics:
            if self.score_max_samples > 0 and len(self._validation_scores) >= self.score_max_samples:
                return
            self._validation_scores.append(metric)
    
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

        if not self._should_log_score():
            return
        summary = aggregate_denoising_scores(self._validation_scores)
        if summary is None:
            print(f"Epoch {self.current_epoch}, Validate Score: no valid samples")
            return
        if loss_summary is not None:
            summary["loss_sum"] = loss_summary["loss_sum"]
            summary["loss_num_samples"] = loss_summary["num_samples"]
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
        self._save_best_checkpoint(summary)
    
    def on_before_optimizer_step(self, optimizer):
        pass

    def prepare_batch(self, batch):
        return _to_jittor(batch)

    def record_train_losses(self, loss_dict):
        for name, value in loss_dict.items():
            self._train_loss[name].append(float(_get_item(value)))

    def _get_train_loss_summary(self):
        if not self._train_loss:
            return None
        summary = {}
        for name, values in self._train_loss.items():
            if values:
                summary[name] = float(np.mean(values))
        summary["num_batches"] = len(self._train_loss.get("loss_sum", []))
        return summary

    def _get_validation_loss_summary(self):
        losses = []
        for key, values in self._validation_loss.items():
            if key.endswith("_loss_sum"):
                losses.extend(values)
        if not losses:
            return None
        summary = {
            "loss_sum": float(np.mean(losses)),
            "num_samples": len(losses),
        }
        if self.loss_config is not None:
            for loss_name in self.loss_config:
                values = []
                suffix = f"_{loss_name}"
                for key, key_values in self._validation_loss.items():
                    if key.endswith(suffix):
                        values.extend(key_values)
                if values:
                    summary[loss_name] = float(np.mean(values))
        return summary

    def _training_log_fields(self):
        loss_names = list(self.loss_config.keys()) if self.loss_config is not None else []
        return [
            "epoch",
            *[f"train_{name}" for name in loss_names],
            "train_loss_sum",
            "train_num_batches",
            *[f"val_{name}" for name in loss_names],
            "val_loss_sum",
            "val_num_samples",
            "final_score",
            "cd_score",
            "cd_pred",
            "cd_noisy",
            "p2s_score",
            "p2s_pred",
            "p2s_noisy",
            "score_num_samples",
            "validation_score_errors",
            "best_epoch",
            "best_score",
            "current_lr",
            "epochs_since_best_checkpoint",
            "lr_decayed",
            "checkpoint_path",
        ]

    def _append_training_log(self, epoch, train_summary, validation_summary, score_summary, checkpoint_path):
        row = {field: "" for field in self._training_log_fields()}
        row["epoch"] = epoch
        if train_summary is not None:
            for name, value in train_summary.items():
                if name == "num_batches":
                    row["train_num_batches"] = value
                else:
                    row[f"train_{name}"] = value
        if validation_summary is not None:
            for name, value in validation_summary.items():
                if name == "num_samples":
                    row["val_num_samples"] = value
                else:
                    row[f"val_{name}"] = value
        if score_summary is not None:
            for name in [
                "final_score",
                "cd_score",
                "cd_pred",
                "cd_noisy",
                "p2s_score",
                "p2s_pred",
                "p2s_noisy",
            ]:
                value = score_summary.get(name, None)
                if value is not None:
                    row[name] = value
            row["score_num_samples"] = score_summary.get("num_samples", "")
        row["validation_score_errors"] = len(self._validation_score_errors)
        row["best_epoch"] = "" if self.best_epoch is None else self.best_epoch
        row["best_score"] = "" if self.best_score is None else self.best_score
        current_lr = self._get_optimizer_lr()
        row["current_lr"] = "" if current_lr is None else current_lr
        row["epochs_since_best_checkpoint"] = self._epochs_since_best_checkpoint
        row["lr_decayed"] = int(self._last_lr_decayed)
        row["checkpoint_path"] = checkpoint_path

        log_path = os.path.abspath(self.training_log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._training_log_fields())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Epoch {epoch}, Training log updated: {log_path}")

    def _get_optimizer_lr(self):
        if self.optimizer is None:
            return None
        lr = getattr(self.optimizer, "lr", None)
        if lr is None:
            return None
        return float(_get_item(lr))

    def _set_optimizer_lr(self, lr):
        if self.optimizer is None:
            return
        if hasattr(self.optimizer, "lr"):
            self.optimizer.lr = lr
        for group in getattr(self.optimizer, "param_groups", []):
            if isinstance(group, dict) and "lr" in group:
                group["lr"] = lr

    def _update_lr_after_best_check(self):
        self._last_lr_decayed = False
        if self._best_checkpoint_updated:
            self._epochs_since_best_checkpoint = 0
            return
        self._epochs_since_best_checkpoint += 1
        if self.lr_decay_patience <= 0:
            return
        if self._epochs_since_best_checkpoint < self.lr_decay_patience:
            return

        current_lr = self._get_optimizer_lr()
        if current_lr is None:
            return
        new_lr = max(current_lr * self.lr_decay_factor, self.lr_decay_min)
        if new_lr >= current_lr:
            self._epochs_since_best_checkpoint = 0
            return
        self._set_optimizer_lr(new_lr)
        self._last_lr_decayed = True
        self._epochs_since_best_checkpoint = 0
        print(
            f"Epoch {self.current_epoch}, Learning rate decayed: "
            f"{current_lr:.8g} -> {new_lr:.8g} "
            f"(no best checkpoint update for {self.lr_decay_patience} epochs)"
        )

    def _save_best_checkpoint(self, summary):
        score = summary.get('final_score', None)
        if score is None:
            return False
        if self.best_score is not None and score <= self.best_score:
            return False

        self.best_score = score
        self.best_epoch = self.current_epoch
        self._best_checkpoint_updated = True
        os.makedirs(self.ckpt_save_dir, exist_ok=True)

        best_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_best.pkl')
        self.model.save(best_path)

        meta_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_best.txt')
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"best_epoch: {self.best_epoch}\n")
            f.write(f"best_score: {self.best_score:.8f}\n")
            for key in [
                "final_score",
                "cd_score",
                "cd_pred",
                "cd_noisy",
                "p2s_score",
                "p2s_pred",
                "p2s_noisy",
                "num_samples",
                "loss_sum",
                "loss_num_samples",
            ]:
                value = summary.get(key, None)
                if value is not None:
                    f.write(f"{key}: {value}\n")

        print(
            f"Epoch {self.current_epoch}, Best checkpoint saved: "
            f"{best_path} (score={self.best_score:.4f})"
        )
        return True

    def _should_log_score(self):
        if not self.log_score:
            return False
        if self.score_every_n_epochs <= 0:
            return False
        return (self.current_epoch + 1) % self.score_every_n_epochs == 0

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
                self.record_train_losses(self._last_train_loss_dict)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                pbar.set_description(f"Epoch {epoch}, Loss: {_get_item(loss)}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.on_train_batch_end()
            self.on_train_epoch_end()
            train_loss_summary = self._get_train_loss_summary()
            
            self.model.eval()
            validation_loss_summary = None
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
                validation_loss_summary = self._get_validation_loss_summary()
            
            checkpoint_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_{epoch}.pkl')
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            self.model.save(checkpoint_path)
            self._update_lr_after_best_check()
            self._append_training_log(
                epoch=epoch,
                train_summary=train_loss_summary,
                validation_summary=validation_loss_summary,
                score_summary=self._validation_score_summary,
                checkpoint_path=checkpoint_path,
            )
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
