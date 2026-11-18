"""Deterministic training of the learned predictors on a nested scene-level split (spec v4.15 section 5.4;
implementation parameters from docs/decisions/0001-p2d-protocol-decisions.md, G).

Order of work in `train_predictor`, all before anything is written:
  1. the split's windows are checked: non-empty training and validation partitions, one protocol, finite;
  2. the six section 11.7 leakage checks run on the split, the source scenes and the very input function the
     model is trained with (`learned.window_inputs`), with normalization declared unused because none is
     applied; `require_no_leakage` aborts on any failure or missing check;
  3. training uses the training partition only; early stopping, the learning-rate schedule and checkpoint
     selection use validation ADE only. The test partition is never read here.

Training: Adam, MSE on frame-to-frame displacements, gradient clipping, ReduceLROnPlateau on validation ADE,
early stopping on validation ADE, best epoch kept (ties keep the earlier epoch). Batches are reshuffled each
epoch by a seeded generator; the last, partial batch of an epoch is kept. Validation ADE is computed in
float64 on positions reconstructed from the predicted displacements. Randomness comes from purpose-specific
streams SeedSequence([seed, purpose]); purpose codes 101 and 102 avoid the corruption family codes 1 to 4.

The checks validate this pipeline's own inputs and arrays. They cannot stop other code from reading targets
outside it (predict.leakage).
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from trackvision.config import Config
from trackvision.io.atomic import atomic_write_bytes, atomic_write_text
from trackvision.io.manifest import file_sha256
from trackvision.metrics.displacement import ade_fde, displacement_errors
from trackvision.predict import learned
from trackvision.predict.displacements import future_displacements
from trackvision.predict.folds import NestedSplit
from trackvision.predict.leakage import NORMALIZATION_NOT_USED, require_no_leakage, run_leakage_checks
from trackvision.predict.windowing import Windows

DECISION = "docs/decisions/0001-p2d-protocol-decisions.md (G)"
CHECKPOINT_FORMAT = "trackvision-learned-checkpoint/1"
PURPOSE_INIT = 101
PURPOSE_SHUFFLE = 102
# ReduceLROnPlateau arguments not set by decision G are PyTorch's defaults; they are recorded, not chosen.
SCHEDULER_DEFAULTS = {"mode": "min", "threshold": 1e-4, "threshold_mode": "rel", "cooldown": 0}


class TrainingError(ValueError):
    """Raised for invalid or incomplete training data, invalid settings or a non-finite training step."""


class CheckpointError(RuntimeError):
    """Raised when a checkpoint fails its hash, format or state verification."""


@dataclass(frozen=True)
class TrainingSettings:
    learning_rate: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    grad_clip_norm: float
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-5

    @classmethod
    def from_config(cls, cfg: Config) -> "TrainingSettings":
        """Section 5.4 values from frozen.predictors.learned; scheduler values from decision G."""
        p = cfg.frozen["predictors"]["learned"]
        if (p["optimizer"], p["loss"], p["lr_scheduler"], p["early_stopping_metric"]) != (
                "adam", "mse_on_displacements", "reduce_lr_on_plateau", "validation_ade"):
            raise TrainingError("configuration names an optimizer, loss, scheduler or stopping metric this "
                                "pipeline does not implement")
        return cls(learning_rate=p["learning_rate"], batch_size=p["batch_size"], max_epochs=p["max_epochs"],
                   early_stopping_patience=p["early_stopping_patience"], grad_clip_norm=p["grad_clip_norm"])

    def __post_init__(self):
        for name in ("batch_size", "max_epochs", "early_stopping_patience", "lr_patience"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise TrainingError(f"{name} must be a positive integer, got {v!r}")
        for name in ("learning_rate", "grad_clip_norm", "min_lr"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise TrainingError(f"{name} must be a positive number, got {v!r}")
        if not 0 < self.lr_factor < 1:
            raise TrainingError(f"lr_factor must lie in (0, 1), got {self.lr_factor!r}")


@dataclass
class EarlyStopping:
    """Lowest validation ADE wins; an equal later value does not replace the earlier epoch."""
    patience: int
    best_epoch: int | None = None
    best_value: float = math.inf
    since_best: int = 0

    def update(self, epoch: int, value: float) -> bool:
        if value < self.best_value:
            self.best_epoch, self.best_value, self.since_best = epoch, value, 0
            return True
        self.since_best += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.since_best >= self.patience


@dataclass(frozen=True)
class TrainingResult:
    model: learned.DisplacementPredictor
    metadata: dict
    checkpoint_path: Path
    checkpoint_sha256: str
    history: list = field(default_factory=list)


def purpose_seed(seed: int, purpose: int) -> int:
    """An independent 64-bit seed for one purpose, from SeedSequence([seed, purpose])."""
    return int(np.random.SeedSequence([seed, purpose]).generate_state(1, dtype=np.uint64)[0])


def state_sha256(state_dict) -> str:
    """SHA-256 over parameter names, dtypes, shapes and bytes, in name order."""
    h = hashlib.sha256()
    for name in sorted(state_dict):
        t = state_dict[name].detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _windows_sha256(w: Windows) -> str:
    h = hashlib.sha256(f"{w.scene}|{w.history_len}|{w.horizon_len}|{w.stride}|".encode())
    for arr in (w.observed, w.target, w.track_ids, w.frames):
        a = np.ascontiguousarray(arr)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def split_identity(split: NestedSplit) -> dict:
    """Scene names per role and a content hash and window count for every scene."""
    scenes = {w.scene: {"windows": len(w), "sha256": _windows_sha256(w)}
              for w in (*split.train, split.validation, split.test)}
    return {"test_scene": split.test_scene, "validation_scene": split.validation_scene,
            "train_scenes": list(split.train_scenes), "scenes": scenes}


def _check_split(split: NestedSplit) -> tuple[int, int]:
    if not isinstance(split, NestedSplit):
        raise TrainingError(f"expected a NestedSplit, got {type(split).__name__}")
    groups = (*split.train, split.validation, split.test)
    protocols = {(w.history_len, w.horizon_len) for w in groups}
    if len(protocols) != 1:
        raise TrainingError(f"partitions use different (history_len, horizon_len): {sorted(protocols)}")
    if sum(len(w) for w in split.train) == 0:
        raise TrainingError("the training partition has no windows")
    if len(split.validation) == 0:
        raise TrainingError("the validation partition has no windows")
    for w in groups:
        if not (np.all(np.isfinite(w.observed)) and np.all(np.isfinite(w.target))):
            raise TrainingError(f"scene {w.scene!r} contains non-finite positions")
    return next(iter(protocols))


def _validation_ade(model, observed: np.ndarray, target: np.ndarray) -> float:
    ade, _ = ade_fde(displacement_errors(learned.predict_positions(model, observed), target))
    value = float(ade.mean())
    if not math.isfinite(value):
        raise TrainingError("validation ADE is not finite")
    return value


def _environment() -> dict:
    # Plain str: TorchVersion is not an allowed type for the safe (weights_only) checkpoint loader.
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda, "numpy": np.__version__, "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": True}


def train_predictor(split: NestedSplit, scenes, cell: str, seed: int, cfg: Config, out_dir: Path, scope: str,
                    settings: TrainingSettings | None = None) -> TrainingResult:
    """Train one predictor on `split` and save its best checkpoint in `out_dir`.

    `scenes` are the source scenes the split's windows were cut from (needed by the window-boundary check).
    `scope` is the label recorded with the checkpoint (for example, synthetic engineering feasibility only).
    """
    if cell not in learned.CELLS:
        raise TrainingError(f"cell must be one of {learned.CELLS}, got {cell!r}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise TrainingError(f"seed must be a non-negative integer, got {seed!r}")
    if not isinstance(scope, str) or not scope.strip():
        raise TrainingError("a non-empty scope label is required")
    settings = settings or TrainingSettings.from_config(cfg)
    history_len, horizon_len = _check_split(split)

    checks = run_leakage_checks(split, scenes, learned.window_inputs, None, NORMALIZATION_NOT_USED)
    require_no_leakage(checks)

    out_dir = Path(out_dir)
    path = out_dir / f"{cell.lower()}-seed{seed}.pt"
    if path.exists() or path.with_suffix(".pt.json").exists():
        raise TrainingError(f"refusing to overwrite an existing checkpoint: {path}")

    train_x = torch.from_numpy(np.concatenate([learned.window_inputs(w) for w in split.train if len(w)]))
    train_y = torch.from_numpy(np.concatenate([future_displacements(w.observed, w.target).astype(np.float32)
                                               for w in split.train if len(w)]))
    val_obs, val_target = split.validation.observed, split.validation.target

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        seeds = {"init": purpose_seed(seed, PURPOSE_INIT), "shuffle": purpose_seed(seed, PURPOSE_SHUFFLE)}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seeds["init"])
            model = learned.build_model(cell, history_len, horizon_len, cfg)
        init_hash = state_sha256(model.state_dict())
        order = torch.Generator().manual_seed(seeds["shuffle"])
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=settings.lr_factor, patience=settings.lr_patience, min_lr=settings.min_lr,
            **SCHEDULER_DEFAULTS)
        stopper = EarlyStopping(settings.early_stopping_patience)
        best_state, history, stop_reason = None, [], "max_epochs"
        n = len(train_x)
        for epoch in range(1, settings.max_epochs + 1):
            model.train()
            perm = torch.randperm(n, generator=order)
            total = 0.0
            for start in range(0, n, settings.batch_size):
                idx = perm[start:start + settings.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(train_x[idx]), train_y[idx])
                if not torch.isfinite(loss):
                    raise TrainingError(f"non-finite training loss at epoch {epoch}")
                loss.backward()
                try:
                    nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip_norm, error_if_nonfinite=True)
                except RuntimeError as exc:
                    raise TrainingError(f"non-finite gradient at epoch {epoch}: {exc}") from exc
                optimizer.step()
                total += float(loss.item()) * len(idx)
            val_ade = _validation_ade(model, val_obs, val_target)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_ade)
            history.append({"epoch": epoch, "train_mse": total / n, "validation_ade_m": val_ade, "lr": lr})
            if stopper.update(epoch, val_ade):
                best_state = copy.deepcopy(model.state_dict())
            if stopper.should_stop:
                stop_reason = "early_stopping"
                break
        model.load_state_dict(best_state)
    finally:
        torch.use_deterministic_algorithms(previous)

    best_hash = state_sha256(best_state)
    metadata = {
        "format": CHECKPOINT_FORMAT, "scope": scope, "decision": DECISION,
        "model": {"type": cell, "architecture": model.architecture()},
        "seed": seed, "purpose_seeds": seeds, "purpose_codes": {"init": PURPOSE_INIT, "shuffle": PURPOSE_SHUFFLE},
        "config_hash": cfg.hash, "settings": asdict(settings), "scheduler_defaults": SCHEDULER_DEFAULTS,
        "batches": "reshuffled every epoch by the seeded generator; last partial batch kept",
        "dtype": {"training": "float32", "metrics": "float64"},
        "normalization": NORMALIZATION_NOT_USED, "split": split_identity(split),
        "leakage_checks": [{"name": r.name, "passed": r.passed, "note": r.note} for r in checks],
        "best_epoch": stopper.best_epoch, "best_validation_ade_m": stopper.best_value,
        "epochs_run": len(history), "stop_reason": stop_reason, "history": history,
        "init_state_sha256": init_hash, "state_sha256": best_hash, "environment": _environment(),
    }
    buffer = io.BytesIO()
    torch.save({"format": CHECKPOINT_FORMAT, "state_dict": best_state, "metadata": metadata}, buffer)
    atomic_write_bytes(path, buffer.getvalue())
    digest = file_sha256(path)
    sidecar = {"checkpoint": path.name, "checkpoint_sha256": digest, **metadata}
    # The sidecar is written last and is the commit marker: a checkpoint without one is an interrupted run.
    atomic_write_text(path.with_suffix(".pt.json"), json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return TrainingResult(model, metadata, path, digest, history)


def checkpoint_mismatches(meta: dict, split: NestedSplit, cell: str, seed: int, cfg: Config,
                          settings: TrainingSettings) -> list[str]:
    """Every training input on which a stored checkpoint differs from the requested run."""
    arch = meta["model"]["architecture"]
    probe = next(iter(split.train))
    wanted = {"cell": cell, "seed": seed, "config_hash": cfg.hash, "settings": asdict(settings),
              "scheduler_defaults": SCHEDULER_DEFAULTS, "split": split_identity(split),
              "history_len": probe.history_len, "horizon_len": probe.horizon_len,
              "purpose_codes": {"init": PURPOSE_INIT, "shuffle": PURPOSE_SHUFFLE},
              "torch": str(torch.__version__), "torch_num_threads": torch.get_num_threads()}
    stored = {"cell": meta["model"]["type"], "seed": meta["seed"], "config_hash": meta["config_hash"],
              "settings": meta["settings"], "scheduler_defaults": meta["scheduler_defaults"], "split": meta["split"],
              "history_len": arch["history_len"], "horizon_len": arch["horizon_len"],
              "purpose_codes": meta["purpose_codes"], "torch": meta["environment"]["torch"],
              "torch_num_threads": meta["environment"]["torch_num_threads"]}
    return sorted(name for name in wanted if wanted[name] != stored[name])


def load_or_train(split: NestedSplit, scenes, cell: str, seed: int, cfg: Config, out_dir: Path, scope: str,
                  settings: TrainingSettings | None = None) -> tuple[TrainingResult, bool]:
    """A verified existing checkpoint for exactly this run, or a new one. Returns (result, reused).

    A checkpoint with no sidecar was interrupted and is replaced. One that exists but differs in any training
    input raises CheckpointError: it is never loaded, and never overwritten without being looked at.
    """
    settings = settings or TrainingSettings.from_config(cfg)
    path = Path(out_dir) / f"{cell.lower()}-seed{seed}.pt"
    sidecar = path.with_suffix(".pt.json")
    if sidecar.exists():
        digest = json.loads(sidecar.read_text(encoding="utf-8"))["checkpoint_sha256"]
        if not path.exists():
            raise CheckpointError(f"{sidecar.name} exists but its checkpoint is missing")
        model, meta = load_checkpoint(path, digest)
        differing = checkpoint_mismatches(meta, split, cell, seed, cfg, settings)
        if differing:
            raise CheckpointError(f"{path.name} was trained with different {differing}")
        return TrainingResult(model, meta, path, digest, meta["history"]), True
    if path.exists():
        path.unlink()
    return train_predictor(split, scenes, cell, seed, cfg, out_dir, scope, settings), False


def load_checkpoint(path: Path, expected_sha256: str) -> tuple[learned.DisplacementPredictor, dict]:
    """Load a checkpoint after verifying its file hash, format and parameter hash."""
    path = Path(path)
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise CheckpointError(f"{path.name}: SHA-256 {digest} does not match the expected {expected_sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"{path.name}: not a {CHECKPOINT_FORMAT} checkpoint")
    meta, state = payload["metadata"], payload["state_dict"]
    if state_sha256(state) != meta["state_sha256"]:
        raise CheckpointError(f"{path.name}: parameters do not match the recorded state hash")
    arch = meta["model"]["architecture"]
    model = learned.DisplacementPredictor(arch["cell"], arch["history_len"], arch["horizon_len"],
                                          arch["hidden_size"], arch["num_layers"])
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, meta
