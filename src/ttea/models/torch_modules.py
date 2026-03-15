from __future__ import annotations

from dataclasses import dataclass

from ..config import ModelBackendConfig
from ..integrations import import_torch


@dataclass(slots=True)
class TensorProjection:
    values: list[float]
    backend: str


class TorchCommunicationProjector:
    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config
        self._torch = import_torch() if config.use_torch else None
        self._network = None
        self._device = config.device
        self._build()

    @property
    def available(self) -> bool:
        return self._network is not None

    def _build(self) -> None:
        if self._torch is None:
            return
        torch = self._torch
        torch.manual_seed(7)
        hidden = max(8, self.config.hidden_size)
        network = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
        )
        with torch.no_grad():
            for _, parameter in network.named_parameters():
                parameter.copy_(torch.linspace(-0.2, 0.2, parameter.numel(), dtype=torch.float32).reshape(parameter.shape))
        network.eval()
        self._network = network

    def project(self, token_ids: list[int], numeric_features: list[float]) -> TensorProjection:
        if self._network is None or self._torch is None:
            values = self._fallback_projection(token_ids, numeric_features)
            return TensorProjection(values=values, backend="fallback")
        torch = self._torch
        hidden = self.config.hidden_size
        base = [0.0 for _ in range(hidden)]
        for index, token_id in enumerate(token_ids[:hidden]):
            base[index] = (token_id % 101) / 100.0
        for offset, value in enumerate(numeric_features):
            index = (len(token_ids) + offset) % hidden
            base[index] += value
        tensor = torch.tensor(base, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            projected = self._network(tensor).squeeze(0).tolist()
        return TensorProjection(values=projected, backend="torch")

    def _fallback_projection(self, token_ids: list[int], numeric_features: list[float]) -> list[float]:
        hidden = self.config.hidden_size
        values = [0.0 for _ in range(hidden)]
        for index, token_id in enumerate(token_ids[:hidden]):
            values[index] = (token_id % 53) / 52.0
        for offset, value in enumerate(numeric_features):
            values[offset % hidden] += value * 0.5
        return values


class TorchImpactNetwork:
    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config
        self._torch = import_torch() if config.use_torch else None
        self._linear = None
        self._build()

    @property
    def available(self) -> bool:
        return self._linear is not None

    def _build(self) -> None:
        if self._torch is None:
            return
        torch = self._torch
        torch.manual_seed(11)
        linear = torch.nn.Linear(8, 3)
        with torch.no_grad():
            linear.weight.copy_(
                torch.tensor(
                    [
                        [0.24, -0.18, -0.16, -0.25, 0.31, -0.12, -0.08, -0.27],
                        [0.16, -0.05, -0.09, -0.13, 0.42, -0.18, 0.24, -0.14],
                        [0.05, 0.33, 0.29, 0.38, -0.21, 0.14, 0.22, 0.18],
                    ],
                    dtype=torch.float32,
                )
            )
            linear.bias.zero_()
        linear.eval()
        self._linear = linear

    def predict(self, features: list[float]) -> list[float]:
        if self._linear is None or self._torch is None:
            return [sum(features) * 0.01, sum(features) * 0.02, max(0.0, sum(features) * 0.03)]
        torch = self._torch
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self._linear(tensor).squeeze(0).tolist()
        return output
