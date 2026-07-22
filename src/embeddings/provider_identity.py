"""Human-readable identities for speaker-embedding provider stacks."""

from __future__ import annotations

from typing import Any


SPEECHBRAIN_ECAPA_PROVIDER = "speechbrain_ecapa"
SINGLE_ESPNET_PROVIDER = "espnet_ecapa_wavlm_joint"
PUBLIC_PROVIDER = (
    "espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+"
    "speechbrain_resnet=0.38+resemblyzer=0.12"
)
PROMOTED_PUBLIC_PROVIDER = (
    "espnet_ecapa_wavlm_joint=1.0+speechbrain_resnet=0.28+wespeaker_campplus=0.37"
)
FAST_LIVE_PROVIDER = "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50"
PROMOTED_LIVE_PROVIDER = "speechbrain_resnet"


_KNOWN_STACK_LABELS = {
    SPEECHBRAIN_ECAPA_PROVIDER: "SpeechBrain ECAPA",
    SINGLE_ESPNET_PROVIDER: "ESPnet ECAPA-WavLM",
    PUBLIC_PROVIDER: "Public high quality",
    PROMOTED_PUBLIC_PROVIDER: "Promoted public stack",
    FAST_LIVE_PROVIDER: "Fast live stack",
}

_KNOWN_COMPONENT_LABELS = {
    "espnet_ecapa_wavlm_joint": "ESPnet ECAPA-WavLM",
    "pyannote_wespeaker_resnet34_lm": "pyannote WeSpeaker ResNet34",
    "resemblyzer": "Resemblyzer",
    "speechbrain_ecapa": "SpeechBrain ECAPA",
    "speechbrain_resnet": "SpeechBrain ResNet",
    "wespeaker_campplus": "WeSpeaker CAM++",
    "wespeaker_resnet34_lm_onnx": "WeSpeaker ResNet34 ONNX",
}


def _provider_components(provider: str) -> list[str]:
    return [part.split("=", 1)[0].strip() for part in provider.split("+") if part.strip()]


def _component_label(component: str) -> str:
    known = _KNOWN_COMPONENT_LABELS.get(component)
    if known:
        return known
    words = [word for word in component.replace("-", "_").split("_") if word]
    if not words:
        return "Custom provider"
    label = " ".join(word.upper() if len(word) <= 4 else word.capitalize() for word in words)
    return label if len(label) <= 42 else f"{label[:39].rstrip()}..."


def summarize_embedding_stack(provider: Any, dimensions: Any = 0) -> dict[str, Any]:
    """Return a compact label plus opt-in technical identity for one provider stack."""

    identifier = str(provider or "").strip()
    try:
        dimension_count = max(0, int(dimensions or 0))
    except (TypeError, ValueError):
        dimension_count = 0
    components = _provider_components(identifier)
    component_labels = [_component_label(component) for component in components]
    known_label = _KNOWN_STACK_LABELS.get(identifier)
    if known_label:
        label = known_label
        kind = "known"
    elif len(component_labels) == 1:
        label = component_labels[0]
        kind = "custom_provider"
    elif 1 < len(component_labels) <= 3 and sum(map(len, component_labels)) <= 58:
        label = " + ".join(component_labels)
        kind = "custom_ensemble"
    elif component_labels:
        label = f"Custom ensemble · {len(component_labels)} providers"
        kind = "custom_ensemble"
    else:
        label = "Unknown embedding stack"
        kind = "unknown"
    return {
        "label": label,
        "kind": kind,
        "provider_count": len(components),
        "components": component_labels,
        "dimensions": dimension_count,
        "identifier": identifier,
    }
