"""Learned sequence predictors (spec v4.15 section 5.4): an LSTM or GRU encoder over frame-to-frame
observed displacements and a linear prediction head (decision 0001 G).

    input    observed displacements  (N, H-1, 2)    predict.displacements.observed_displacements
    output   future displacements    (N, F, 2)      the decoder targets of future_displacements
    positions p_hat_{H+k} = p_H + cumulative sum     predict.displacements.reconstruct

One recurrent layer (hidden size and layer count from frozen.predictors.learned) reads the H-1 input
displacements. A linear layer maps the final encoder hidden state to F x 2 values. There is no attention
and no autoregressive feedback: the head sees only the final hidden state, and no prediction is fed back.
The model never receives future positions; H and F are fixed per model and recorded, never hard-coded.
Weights and inputs are float32; reconstructed positions and all metrics are float64.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from trackvision.config import Config
from trackvision.predict.displacements import observed_displacements, reconstruct
from trackvision.predict.windowing import Windows

CELLS = ("LSTM", "GRU")
HEAD = "linear_from_final_hidden_state"
INPUT_SIZE = 2


class DisplacementPredictor(nn.Module):
    """Encoder (LSTM or GRU) over H-1 displacements; linear head from the final hidden state to F x 2."""

    def __init__(self, cell: str, history_len: int, horizon_len: int, hidden_size: int, num_layers: int):
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {CELLS}, got {cell!r}")
        for name, value, low in (("history_len", history_len, 2), ("horizon_len", horizon_len, 1),
                                 ("hidden_size", hidden_size, 1), ("num_layers", num_layers, 1)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < low:
                raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
        self.cell, self.history_len, self.horizon_len = cell, int(history_len), int(horizon_len)
        self.hidden_size, self.num_layers = int(hidden_size), int(num_layers)
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.encoder = rnn(INPUT_SIZE, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.head = nn.Linear(self.hidden_size, self.horizon_len * INPUT_SIZE)

    def forward(self, obs_disp: torch.Tensor) -> torch.Tensor:
        if obs_disp.dtype != torch.float32:
            raise ValueError(f"inputs must be float32, got {obs_disp.dtype}")
        if obs_disp.ndim != 3 or obs_disp.shape[1:] != (self.history_len - 1, INPUT_SIZE):
            raise ValueError(f"inputs must have shape (N, {self.history_len - 1}, 2), got {tuple(obs_disp.shape)}")
        _, state = self.encoder(obs_disp)
        hidden = state[0] if self.cell == "LSTM" else state                 # (layers, N, hidden)
        return self.head(hidden[-1]).view(-1, self.horizon_len, INPUT_SIZE)

    def architecture(self) -> dict:
        return {"cell": self.cell, "input_size": INPUT_SIZE, "hidden_size": self.hidden_size,
                "num_layers": self.num_layers, "head": HEAD, "attention": False, "autoregressive_feedback": False,
                "history_len": self.history_len, "horizon_len": self.horizon_len,
                "n_parameters": parameter_count(self)}


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_model(cell: str, history_len: int, horizon_len: int, cfg: Config) -> DisplacementPredictor:
    """A predictor with the frozen section 5.4 architecture for the given protocol lengths."""
    learned = cfg.frozen["predictors"]["learned"]
    if learned["attention"] or learned["autoregressive_feedback"]:
        raise ValueError("this predictor has no attention and no autoregressive feedback")
    return DisplacementPredictor(cell, history_len, horizon_len, learned["hidden_size"], learned["num_layers"])


def encoder_inputs(observed) -> np.ndarray:
    """The model input (N, H-1, 2) float32 for observed positions (N, H, 2): their displacements only."""
    return observed_displacements(observed).astype(np.float32)


def window_inputs(windows: Windows) -> np.ndarray:
    """Model inputs for a set of windows; reads `windows.observed` and nothing else."""
    return encoder_inputs(windows.observed)


def predict_displacements(model: DisplacementPredictor, observed) -> np.ndarray:
    """Predicted future displacements (N, F, 2) as float64."""
    inputs = encoder_inputs(observed)
    if inputs.shape[1] != model.history_len - 1:
        raise ValueError(f"the model was built for history_len {model.history_len}, got {inputs.shape[1] + 1}")
    if len(inputs) == 0:
        return np.zeros((0, model.horizon_len, INPUT_SIZE))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(torch.from_numpy(inputs))
    finally:
        model.train(was_training)
    return out.numpy().astype(np.float64)


def predict_positions(model: DisplacementPredictor, observed) -> np.ndarray:
    """Predicted absolute positions (N, F, 2), float64, reconstructed from the last observed position."""
    displacements = predict_displacements(model, observed)          # validates the observed array first
    return reconstruct(np.asarray(observed, dtype=np.float64)[:, -1], displacements)
