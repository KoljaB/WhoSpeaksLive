"""SpeechBrain API compatibility for the managed embeddings service."""

from __future__ import annotations

from typing import Any


def load_speechbrain_encoder(model_id: str, savedir: str, device: str) -> Any:
    from speechbrain.inference.speaker import EncoderClassifier

    try:
        from speechbrain.inference.interfaces import Pretrained
    except ImportError:
        from speechbrain.inference.speaker import Pretrained

    if not hasattr(Pretrained, "device_type"):
        Pretrained.device_type = "cpu"
    return EncoderClassifier.from_hparams(
        source=model_id,
        savedir=savedir,
        run_opts={"device": device},
    )
