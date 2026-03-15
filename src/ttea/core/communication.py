from __future__ import annotations

"""TTEA-native communication stack.

This module folds the external communication ideas from the local
`coummunicaiton/` reference directory into the main TTEA runtime:
confidence gating, collaboration prompts, and multi-agent feature fusion.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..config import CommunicationConfig
from ..integrations import import_torch
from ..types import Observation

if TYPE_CHECKING:
    from ..agents.leader import CategoryLeader, GlobalLeader
    from .memory import GlobalMemoryPool


_TORCH = import_torch()
_ModuleBase = object if _TORCH is None else _TORCH.nn.Module


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass(slots=True)
class EncodedCommunicationBatch:
    vectors: list[list[float]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ConfidenceGatedCommunication:
    def __init__(self, config: CommunicationConfig) -> None:
        self.config = config
        self._torch = import_torch()
        self._gaussian_filter = None
        if self._torch is not None and config.gaussian_smooth:
            self._gaussian_filter = self._build_gaussian_filter(
                config.gaussian_kernel_size,
                config.gaussian_sigma,
            )

    def _build_gaussian_filter(self, kernel_size: int, sigma: float):
        torch = self._torch
        if torch is None:
            return None
        padding = max(0, (kernel_size - 1) // 2)
        layer = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        center = kernel_size // 2
        values = []
        for row in range(kernel_size):
            line = []
            for col in range(kernel_size):
                distance = (row - center) ** 2 + (col - center) ** 2
                line.append((1.0 / (2.0 * math.pi * sigma)) * math.exp(-distance / (2.0 * sigma * sigma)))
            values.append(line)
        kernel = torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            layer.weight.copy_(kernel)
            layer.bias.zero_()
        layer.requires_grad_(False)
        layer.eval()
        return layer

    def apply(self, confidence_maps, feature_maps, record_len):
        torch = self._torch
        if torch is None or confidence_maps is None:
            return feature_maps, None, 0.0
        smoothed_maps = confidence_maps
        if self._gaussian_filter is not None:
            smoothed_maps = self._gaussian_filter(confidence_maps)
        masks: list[Any] = []
        gated_features: list[Any] = []
        communication_rates: list[float] = []
        start = 0
        for length in record_len:
            group_len = int(length)
            group_confidence = smoothed_maps[start : start + group_len]
            group_features = feature_maps[start : start + group_len]
            group_masks = (group_confidence > self.config.confidence_threshold).to(group_features.dtype)
            if group_len > 0:
                group_masks[0] = torch.ones_like(group_masks[0])
                communication_rates.append(float(group_masks[0].mean().item()))
            masks.append(group_masks)
            gated_features.append(group_features * group_masks)
            start += group_len
        combined_masks = torch.cat(masks, dim=0) if masks else None
        combined_features = torch.cat(gated_features, dim=0) if gated_features else feature_maps
        mean_rate = sum(communication_rates) / max(1, len(communication_rates))
        return combined_features, combined_masks, mean_rate


class CollaborationPromptAdapter:
    def __init__(self, hidden_size: int, config: CommunicationConfig) -> None:
        self.hidden_size = hidden_size
        self.config = config
        self._torch = import_torch()
        self._module = None
        if self._torch is not None:
            self._module = _PromptAdapterModule(hidden_size, config)

    @property
    def available(self) -> bool:
        return self._module is not None

    def apply(self, feature_maps, record_len):
        if self._module is None:
            return feature_maps, []
        return self._module(feature_maps, record_len)


class _PromptAdapterModule(_ModuleBase):
    def __init__(self, hidden_size: int, config: CommunicationConfig) -> None:
        torch = import_torch()
        if torch is None:
            raise RuntimeError("torch is required for the collaboration prompt adapter")
        super().__init__()
        self.torch = torch
        self.hidden_size = hidden_size
        down_ratio = max(1, config.prompt_downsample_ratio)
        reduced_channels = max(1, hidden_size // down_ratio)
        self.scale = torch.nn.Parameter(torch.ones(hidden_size))
        self.shift = torch.nn.Parameter(torch.zeros(hidden_size))
        self.spatial_attention = torch.nn.Conv2d(hidden_size, 1, 1, bias=config.prompt_bias)
        self.down_layer = torch.nn.Conv2d(hidden_size, reduced_channels, 1, bias=config.prompt_bias)
        self.up_layer = torch.nn.Conv2d(reduced_channels, hidden_size, 1, bias=config.prompt_bias)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(config.prompt_dropout)
        self.prompt_projection = torch.nn.Linear(hidden_size, hidden_size, bias=config.prompt_bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch = self.torch
        for layer in [self.spatial_attention, self.down_layer, self.up_layer]:
            torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, feature_maps, record_len):
        torch = self.torch
        if feature_maps.numel() == 0:
            return feature_maps, []
        local_contexts = []
        start = 0
        for length in record_len:
            group_len = int(length)
            group = feature_maps[start : start + group_len]
            if group_len > 0:
                pooled = group.max(dim=0, keepdim=True)[0].repeat(group_len, 1, 1, 1)
                local_contexts.append(pooled)
            start += group_len
        context = torch.cat(local_contexts, dim=0) if local_contexts else feature_maps
        attention = torch.sigmoid(self.spatial_attention(context))
        projected = self.up_layer(self.dropout(self.activation(self.down_layer(feature_maps))))
        adapted = attention * projected + feature_maps

        con_prompt = self.scale.view(1, -1, 1, 1) * adapted + self.shift.view(1, -1, 1, 1)
        prompts: list[Any] = []
        start = 0
        for length in record_len:
            group_len = int(length)
            group_prompt = con_prompt[start : start + group_len]
            if group_len > 0:
                pooled = group_prompt.max(dim=0, keepdim=True)[0]
                prompt = self.prompt_projection(pooled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                prompts.append(prompt)
            start += group_len
        return adapted, prompts


class ScaledDotProductAttention:
    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self.scale = math.sqrt(max(1, feature_dim))
        self._torch = import_torch()

    def apply(self, query, key, value):
        torch = self._torch
        if torch is None:
            raise RuntimeError("torch is required for attention fusion")
        score = torch.bmm(query, key.transpose(1, 2)) / self.scale
        attention = torch.softmax(score, dim=-1)
        return torch.bmm(attention, value)


class MultiAgentFeatureFusion:
    def __init__(self, config: CommunicationConfig) -> None:
        self.config = config
        self._torch = import_torch()
        self._attention = ScaledDotProductAttention(config.encoder_dim)
        self._transformer = None
        if self._torch is not None and config.fusion_mode.lower() == "transformer":
            self._transformer = self._torch.nn.MultiheadAttention(
                embed_dim=config.encoder_dim,
                num_heads=max(1, config.fusion_heads),
                dropout=config.fusion_dropout,
                batch_first=True,
            )

    def fuse(self, feature_maps, record_len, confidence_maps=None):
        torch = self._torch
        if torch is None or feature_maps is None:
            return feature_maps
        fused_groups: list[Any] = []
        start = 0
        for length in record_len:
            group_len = int(length)
            group = feature_maps[start : start + group_len]
            group_conf = None if confidence_maps is None else confidence_maps[start : start + group_len]
            if group_len <= 1:
                fused_groups.append(group[:1])
                start += group_len
                continue
            mode = self.config.fusion_mode.lower()
            if mode == "max":
                fused = group.max(dim=0, keepdim=True)[0]
            elif mode == "mean":
                fused = group.mean(dim=0, keepdim=True)
            elif mode == "transformer" and self._transformer is not None:
                fused = self._transformer_fusion(group, group_conf)
            else:
                fused = self._attention_fusion(group, group_conf)
            fused_groups.append(fused)
            start += group_len
        return torch.cat(fused_groups, dim=0) if fused_groups else feature_maps

    def _attention_fusion(self, group, confidence_maps=None):
        cav_num, channels, height, width = group.shape
        flattened = group.view(cav_num, channels, -1).permute(2, 0, 1)
        if confidence_maps is not None:
            weights = confidence_maps.mean(dim=(2, 3)).view(1, cav_num, 1)
            flattened = flattened * weights
        attended = self._attention.apply(flattened, flattened, flattened)
        return attended.permute(1, 2, 0).view(cav_num, channels, height, width)[0:1]

    def _transformer_fusion(self, group, confidence_maps=None):
        cav_num, channels, height, width = group.shape
        tokens = group.view(cav_num, channels, -1).permute(2, 0, 1)
        if confidence_maps is not None:
            weights = confidence_maps.mean(dim=(2, 3)).view(1, cav_num, 1)
            tokens = tokens * weights
        attended, _ = self._transformer(tokens, tokens, tokens)
        ego_tokens = attended[:, 0, :]
        return ego_tokens.transpose(0, 1).reshape(1, channels, height, width)


class ObservationEncoder:
    def __init__(self, config: CommunicationConfig, text_backend=None, projector=None) -> None:
        self.config = config
        self.text_backend = text_backend
        self.projector = projector
        self._torch = import_torch()
        self.gating = ConfidenceGatedCommunication(config)
        self.prompt_adapter = CollaborationPromptAdapter(config.encoder_dim, config)
        self.fusion = MultiAgentFeatureFusion(config)

    def encode(self, observation: Observation) -> list[float]:
        return self._encode_base(observation)

    def encode_batch(
        self,
        observations: list[Observation],
        agent_signatures: list[str] | None = None,
    ) -> EncodedCommunicationBatch:
        base_vectors = [self._encode_base(observation) for observation in observations]
        if self._torch is None or len(base_vectors) <= 1:
            return EncodedCommunicationBatch(
                vectors=base_vectors,
                diagnostics={
                    "communication_rate": 0.0,
                    "prompt_count": 0,
                    "fusion_mode": "fallback",
                    "agent_signatures": list(agent_signatures or []),
                },
            )
        torch = self._torch
        feature_maps = self._vectors_to_feature_maps(base_vectors)
        confidence_maps = torch.sigmoid(feature_maps.mean(dim=1, keepdim=True))
        record_len = [len(base_vectors)]
        gated_maps, masks, communication_rate = self.gating.apply(confidence_maps, feature_maps, record_len)
        adapted_maps, prompts = self.prompt_adapter.apply(gated_maps, record_len)
        fused_maps = self.fusion.fuse(adapted_maps, record_len, confidence_maps=confidence_maps if masks is None else masks)
        adapted_vectors = adapted_maps.mean(dim=(2, 3))
        fused_vector = fused_maps.mean(dim=(2, 3))[0]
        prompt_vector = fused_vector
        if prompts:
            prompt_vector = prompts[0].mean(dim=(2, 3))[0]
        combined_vectors: list[list[float]] = []
        for index in range(adapted_vectors.shape[0]):
            mixed = adapted_vectors[index] * 0.5 + fused_vector * 0.35 + prompt_vector * 0.15
            combined_vectors.append(mixed.tolist())
        mask_density = 0.0 if masks is None else float(masks.mean().item())
        return EncodedCommunicationBatch(
            vectors=combined_vectors,
            diagnostics={
                "communication_rate": communication_rate,
                "mask_density": mask_density,
                "prompt_count": len(prompts),
                "fusion_mode": self.config.fusion_mode,
                "agent_signatures": list(agent_signatures or []),
            },
        )

    def _encode_base(self, observation: Observation) -> list[float]:
        if self.text_backend is not None and self.projector is not None:
            tokenized = self.text_backend.tokenize(observation.summary)
            numeric_features = list(observation.numeric_features.values())
            projection = self.projector.project(tokenized.token_ids, numeric_features)
            values = projection.values
            if len(values) >= self.config.encoder_dim:
                return values[: self.config.encoder_dim]
            return values + [0.0 for _ in range(self.config.encoder_dim - len(values))]
        vector = [0.0 for _ in range(self.config.encoder_dim)]
        tokens = observation.summary.lower().split()
        for token in tokens:
            index = sum(ord(char) for char in token) % self.config.encoder_dim
            vector[index] += 1.0
        for offset, (_, value) in enumerate(sorted(observation.numeric_features.items())):
            index = offset % self.config.encoder_dim
            vector[index] += value
        norm = max(1.0, sum(abs(value) for value in vector))
        return [value / norm for value in vector]

    def _vectors_to_feature_maps(self, vectors: list[list[float]]):
        torch = self._torch
        if torch is None:
            raise RuntimeError("torch is required to lift communication vectors into feature maps")
        tensor = torch.tensor(vectors, dtype=torch.float32)
        grid_size = max(1, self.config.feature_grid_size)
        feature_maps = tensor.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, grid_size, grid_size)
        axis = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32)
        x_grid = axis.unsqueeze(0).repeat(grid_size, 1)
        y_grid = axis.unsqueeze(1).repeat(1, grid_size)
        positional = (x_grid + y_grid).unsqueeze(0).unsqueeze(0)
        return feature_maps * (1.0 + 0.05 * positional)


class MacroAdapter:
    def __init__(self, config: CommunicationConfig) -> None:
        self.config = config

    def _down_convolution(self, vector: list[float]) -> list[float]:
        return [
            (vector[index] + vector[min(index + 1, len(vector) - 1)]) / 2.0
            for index in range(0, len(vector), 2)
        ]

    def _relu(self, vector: list[float]) -> list[float]:
        return [max(0.0, value) for value in vector]

    def _fractional_generation(self, vector: list[float]) -> list[float]:
        if not vector:
            return []
        generated: list[float] = []
        for index, value in enumerate(vector):
            left = vector[index - 1] if index > 0 else value
            right = vector[index + 1] if index + 1 < len(vector) else value
            generated.append((left + value + right) / 3.0)
        return generated

    def _collaborative_filter(self, vector: list[float]) -> list[float]:
        if not vector:
            return []
        mean_value = sum(vector) / len(vector)
        return [value - mean_value for value in vector]

    def _up_convolution(self, vector: list[float], target_dim: int) -> list[float]:
        if not vector:
            return [0.0 for _ in range(target_dim)]
        expanded: list[float] = []
        while len(expanded) < target_dim:
            for value in vector:
                expanded.append(value)
                if len(expanded) == target_dim:
                    break
        return expanded

    def apply(self, feature_vector: list[float]) -> list[float]:
        down = self._down_convolution(feature_vector)
        activated = self._relu(down)
        generated = self._fractional_generation(activated)
        filtered = self._collaborative_filter(generated)
        up = self._up_convolution(filtered, len(feature_vector))
        return [feature + self.config.macro_scale * value for feature, value in zip(feature_vector, up, strict=True)]


class MicroAdapter:
    def __init__(self, config: CommunicationConfig) -> None:
        self.config = config

    def apply(self, macro_vector: list[float], agent_signature: str) -> list[float]:
        shift_seed = sum(ord(char) for char in agent_signature)
        personalized: list[float] = []
        for index, value in enumerate(macro_vector):
            scale = 1.0 + self.config.micro_scale * (((shift_seed + index) % 7) / 10.0)
            shift = (((shift_seed // max(1, index + 1)) % 11) - 5) / 50.0
            personalized.append(_clip(value * scale + shift))
        return personalized


class VectorTextBridge:
    def __init__(self, config: CommunicationConfig) -> None:
        self.config = config

    def encode(self, vector: list[float]) -> str:
        chunks = [f"v{index}={value:.{self.config.text_precision}f}" for index, value in enumerate(vector)]
        return " | ".join(chunks)

    def decode(self, text: str) -> list[float]:
        vector: list[float] = []
        for chunk in text.split("|"):
            piece = chunk.strip()
            if "=" not in piece:
                continue
            _, value = piece.split("=", 1)
            try:
                vector.append(float(value))
            except ValueError:
                vector.append(0.0)
        return vector


class KnowledgeSynergyEngine:
    def __init__(self, config: CommunicationConfig) -> None:
        self.config = config

    def synchronize(
        self,
        global_leader: "GlobalLeader",
        category_leaders: dict[str, "CategoryLeader"],
        global_memory: "GlobalMemoryPool",
    ) -> dict[str, Any]:
        for category_name, leader in category_leaders.items():
            leader.refresh_capability_map()
            summary = leader.summary()
            global_memory.update_category(category_name, summary)
            global_memory.record_message(
                f"{leader.agent_id} shared category summary with {summary['active_agents']} active agents"
            )
        unified = global_memory.build_global_view()
        global_leader.cognitive_view = unified
        return unified
