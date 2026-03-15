from ttea.config import load_platform_config
from ttea.core.communication import ObservationEncoder
from ttea.types import Observation


def test_observation_encoder_batch_communication() -> None:
    platform = load_platform_config("configs/platform.json")
    encoder = ObservationEncoder(platform.communication)
    observations = [
        Observation(
            summary="agent one checks a form and a submit button",
            numeric_features={"priority": 0.7, "complexity": 0.5},
        ),
        Observation(
            summary="agent two verifies the page state and confirms text",
            numeric_features={"priority": 0.6, "complexity": 0.4},
        ),
    ]
    encoded = encoder.encode_batch(observations, agent_signatures=["agent_one", "agent_two"])
    assert len(encoded.vectors) == 2
    assert all(len(vector) == platform.communication.encoder_dim for vector in encoded.vectors)
    assert "communication_rate" in encoded.diagnostics
    assert encoded.diagnostics["fusion_mode"] in {"attention", "fallback", "transformer", "mean", "max"}
