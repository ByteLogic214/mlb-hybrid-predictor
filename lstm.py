from __future__ import annotations

from typing import Any


class OptionalLSTM:
    """Experimental sequence model, deliberately isolated from the promoted pipeline.

    This module never runs unless the caller installs the `lstm` extra and invokes it directly.
    It is not used to produce the official XGBoost-first prediction.
    """

    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("Install the optional dependency with: pip install '.[lstm]'") from exc

        class Network(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.output = nn.Linear(hidden_size, 1)

            def forward(self, values: Any) -> Any:
                sequence, _state = self.lstm(values)
                return self.output(sequence[:, -1, :]).squeeze(-1)

        self.network = Network()

    def model(self) -> Any:
        return self.network
