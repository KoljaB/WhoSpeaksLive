"""Shared constants for WhoSpeaks CLI runtime and installation workflows."""

PACKAGE_NAME = "whospeaks"
KROKO_INSTALL_MODULE = "RealtimeSTT.install_kroko"
KROKO_PREVIEW_VENV_ENV = "WHOSPEAKS_KROKO_PREVIEW_VENV"
TRANSLATION_VENV_ROOT_ENV = "WHOSPEAKS_TRANSLATION_VENV_ROOT"
TRANSLATION_MODEL_ROOT_ENV = "WHOSPEAKS_TRANSLATION_MODEL_ROOT"
TESTPYPI_SIMPLE_URL = "https://test.pypi.org/simple/"
PIP_INDEX_URL_ENV = "WHOSPEAKS_PIP_INDEX_URL"
PIP_EXTRA_INDEX_URL_ENV = "WHOSPEAKS_PIP_EXTRA_INDEX_URL"
PIP_FIND_LINKS_ENV = "WHOSPEAKS_PIP_FIND_LINKS"
TORCH_INSTALL_POLICY_ENV = "WHOSPEAKS_TORCH_INSTALL"
PYTORCH_CUDA_BUILD_ENV = "WHOSPEAKS_PYTORCH_CUDA_BUILD"
PYTORCH_CUDA_INDEX_URL_ENV = "WHOSPEAKS_PYTORCH_CUDA_INDEX_URL"
PYTORCH_CPU_INDEX_URL_ENV = "WHOSPEAKS_PYTORCH_CPU_INDEX_URL"
DEFAULT_PYTORCH_CUDA_BUILD = "cu128"
PYTORCH_CUDA_INDEX_URLS = {
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
}
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGE_SPECS = ("torch>=2.2", "torchaudio>=2.2")
KROKO_LANGUAGE_MENU_CODES = ("en", "de", "es", "fr", "it", "nl", "pt", "sv", "tr", "he")
TORCH_INSTALL_POLICY_CHOICES = ("auto", "cuda", "cpu", "skip")
STATUS_ORDER = {"ok": 0, "skip": 1, "warn": 2, "fail": 3}
STATUS_LABEL = {
    "ok": "OK",
    "skip": "SKIP",
    "warn": "WARN",
    "fail": "FAIL",
}
