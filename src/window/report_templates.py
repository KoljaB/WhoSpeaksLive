"""Validated, versioned report templates and bundled report presets."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Mapping
import unicodedata

from window.language_config import normalize_language_code


TEMPLATE_SCHEMA_VERSION = "report_template_v1"
STANDARD_TEMPLATE_ID = "builtin.standard-meeting"
_STORE_LOCKS_GUARD = threading.Lock()
_STORE_LOCKS: dict[str, threading.RLock] = {}

_PRESET_DIRECTORY = Path(__file__).with_name("report_template_presets")
_MAX_SECTIONS = 16
_RENDER_KINDS = frozenset({"cards", "table", "timeline", "quotes"})
_SORT_ORDERS = frozenset({"relevance", "chronological", "severity"})
_FIELD_TYPES = frozenset({"text", "enum", "speaker", "date", "timestamp", "boolean", "number"})
_PRIVACY_POLICIES = frozenset({"inherit", "local_only"})
_TEMPLATE_KEYS = frozenset({
    "schema_version",
    "template_id",
    "name",
    "description",
    "version",
    "builtin",
    "language_mode",
    "privacy_policy",
    "sections",
    "revision_hash",
})
_SECTION_KEYS = frozenset({
    "key",
    "title",
    "objective",
    "max_items",
    "evidence_required",
    "render_kind",
    "sort_order",
    "output_fields",
})
_FIELD_KEYS = frozenset({"key", "label", "type", "description", "options"})
_TEMPLATE_ID_RE = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")


def _unexpected_keys(payload: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(f"{context} contains unsupported fields: {', '.join(unexpected)}")


def _normalized_string(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = re.sub(r"\s+", " ", value).strip()
    if required and not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return normalized


def _normalized_key(value: Any, field_name: str) -> str:
    text = _normalized_string(value, field_name, max_length=80).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        raise ValueError(f"{field_name} must contain at least one letter or number")
    if len(text) > 64:
        raise ValueError(f"{field_name} must normalize to at most 64 characters")
    return text


def slugify_template_id(name: str) -> str:
    """Return a stable URL/file-safe slug for a human report name."""

    text = _normalized_string(name, "name", max_length=160).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text[:80].rstrip("-") or "report"


def _normalized_template_id(value: Any) -> str:
    text = _normalized_string(value, "template_id", max_length=120).lower()
    text = text.replace("_", "-").replace(" ", "-")
    text = re.sub(r"-{2,}", "-", text)
    if not _TEMPLATE_ID_RE.fullmatch(text):
        raise ValueError(
            "template_id must use lowercase letters, numbers, dots, and hyphens only"
        )
    return text


def _normalized_choice(value: Any, field_name: str, choices: frozenset[str]) -> str:
    text = _normalized_string(value, field_name, max_length=40).lower().replace("-", "_")
    if text not in choices:
        raise ValueError(f"{field_name} must be one of: {', '.join(sorted(choices))}")
    return text


def _normalize_output_field(payload: Any, section_key: str, index: int) -> dict[str, Any]:
    context = f"section {section_key!r} output_fields[{index}]"
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")
    _unexpected_keys(payload, _FIELD_KEYS, context)
    key = _normalized_key(payload.get("key"), f"{context}.key")
    field_type = _normalized_choice(payload.get("type"), f"{context}.type", _FIELD_TYPES)
    raw_options = payload.get("options", [])
    if not isinstance(raw_options, list):
        raise ValueError(f"{context}.options must be a list")
    options: list[str] = []
    seen_options: set[str] = set()
    for option_index, raw_option in enumerate(raw_options):
        option = _normalized_string(
            raw_option,
            f"{context}.options[{option_index}]",
            max_length=80,
        )
        option_identity = option.casefold()
        if option_identity in seen_options:
            raise ValueError(f"{context}.options contains duplicate value {option!r}")
        seen_options.add(option_identity)
        options.append(option)
    if field_type == "enum" and not options:
        raise ValueError(f"{context}.options must not be empty for an enum field")
    if field_type != "enum" and options:
        raise ValueError(f"{context}.options is only supported for enum fields")
    return {
        "key": key,
        "label": _normalized_string(payload.get("label"), f"{context}.label", max_length=100),
        "type": field_type,
        "description": _normalized_string(
            payload.get("description", ""),
            f"{context}.description",
            required=False,
            max_length=500,
        ),
        "options": options,
    }


def _normalize_section(payload: Any, index: int) -> dict[str, Any]:
    context = f"sections[{index}]"
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")
    _unexpected_keys(payload, _SECTION_KEYS, context)
    key = _normalized_key(payload.get("key"), f"{context}.key")
    max_items = payload.get("max_items", 8)
    if isinstance(max_items, bool) or not isinstance(max_items, int) or not 1 <= max_items <= 20:
        raise ValueError(f"{context}.max_items must be an integer from 1 to 20")
    evidence_required = payload.get("evidence_required", True)
    if not isinstance(evidence_required, bool):
        raise ValueError(f"{context}.evidence_required must be a boolean")
    raw_fields = payload.get("output_fields", [])
    if not isinstance(raw_fields, list):
        raise ValueError(f"{context}.output_fields must be a list")
    fields = [_normalize_output_field(field, key, field_index) for field_index, field in enumerate(raw_fields)]
    field_keys = [field["key"] for field in fields]
    duplicate_fields = sorted({item for item in field_keys if field_keys.count(item) > 1})
    if duplicate_fields:
        raise ValueError(
            f"section {key!r} contains duplicate output field keys: {', '.join(duplicate_fields)}"
        )
    return {
        "key": key,
        "title": _normalized_string(payload.get("title"), f"{context}.title", max_length=120),
        "objective": _normalized_string(
            payload.get("objective"), f"{context}.objective", max_length=1200
        ),
        "max_items": max_items,
        "evidence_required": evidence_required,
        "render_kind": _normalized_choice(
            payload.get("render_kind", "cards"), f"{context}.render_kind", _RENDER_KINDS
        ),
        "sort_order": _normalized_choice(
            payload.get("sort_order", "relevance"), f"{context}.sort_order", _SORT_ORDERS
        ),
        "output_fields": fields,
    }


def _calculate_revision_hash(template: Mapping[str, Any]) -> str:
    hash_payload = deepcopy(dict(template))
    hash_payload.pop("revision_hash", None)
    encoded = json.dumps(
        hash_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_template(
    payload: Any,
    *,
    allow_builtin: bool,
    include_revision_hash: bool,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("report template must be an object")
    _unexpected_keys(payload, _TEMPLATE_KEYS, "report template")
    schema_version = payload.get("schema_version", TEMPLATE_SCHEMA_VERSION)
    if schema_version != TEMPLATE_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {TEMPLATE_SCHEMA_VERSION!r}")
    template_id = _normalized_template_id(payload.get("template_id"))
    builtin = payload.get("builtin", False)
    if not isinstance(builtin, bool):
        raise ValueError("builtin must be a boolean")
    if builtin and not allow_builtin:
        raise ValueError("builtin report templates are read-only")
    if builtin and not template_id.startswith("builtin."):
        raise ValueError("builtin template_id must start with 'builtin.'")
    if not builtin and template_id.startswith("builtin."):
        raise ValueError("custom template_id must not use the reserved 'builtin.' namespace")
    version = payload.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("version must be a positive integer")
    language_mode_raw = _normalized_string(
        payload.get("language_mode", "inherit"), "language_mode", max_length=80
    )
    language_mode = (
        "inherit"
        if language_mode_raw.lower() == "inherit"
        else normalize_language_code(language_mode_raw)
    )
    privacy_policy = _normalized_choice(
        payload.get("privacy_policy", "inherit"), "privacy_policy", _PRIVACY_POLICIES
    )
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError("sections must be a non-empty list")
    if len(raw_sections) > _MAX_SECTIONS:
        raise ValueError(f"sections must contain at most {_MAX_SECTIONS} entries")
    sections = [_normalize_section(section, index) for index, section in enumerate(raw_sections)]
    section_keys = [section["key"] for section in sections]
    duplicate_sections = sorted({item for item in section_keys if section_keys.count(item) > 1})
    if duplicate_sections:
        raise ValueError(f"duplicate section keys: {', '.join(duplicate_sections)}")
    normalized: dict[str, Any] = {
        "schema_version": TEMPLATE_SCHEMA_VERSION,
        "template_id": template_id,
        "name": _normalized_string(payload.get("name"), "name", max_length=160),
        "description": _normalized_string(
            payload.get("description", ""), "description", required=False, max_length=1000
        ),
        "version": version,
        "builtin": builtin,
        "language_mode": language_mode,
        "privacy_policy": privacy_policy,
        "sections": sections,
    }
    if include_revision_hash:
        normalized["revision_hash"] = _calculate_revision_hash(normalized)
    return normalized


def validate_report_template(payload: Any, *, allow_builtin: bool = False) -> dict[str, Any]:
    """Validate and return a normalized, detached report template document."""

    return _normalize_template(
        payload,
        allow_builtin=allow_builtin,
        include_revision_hash=True,
    )


def template_revision_hash(template: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 revision of a normalized template."""

    normalized = _normalize_template(
        template,
        allow_builtin=True,
        include_revision_hash=False,
    )
    return _calculate_revision_hash(normalized)


def builtin_report_templates() -> list[dict[str, Any]]:
    """Discover and validate the inspectable JSON report presets bundled with WhoSpeaks."""

    if not _PRESET_DIRECTORY.is_dir():
        raise RuntimeError(f"Bundled report template directory is missing: {_PRESET_DIRECTORY}")
    templates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in sorted(_PRESET_DIRECTORY.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not load bundled report template {path.name}: {exc}") from exc
        try:
            template = validate_report_template(payload, allow_builtin=True)
        except ValueError as exc:
            raise RuntimeError(f"Invalid bundled report template {path.name}: {exc}") from exc
        if not template["builtin"]:
            raise RuntimeError(f"Bundled report template {path.name} must set builtin=true")
        template_id = template["template_id"]
        if template_id in seen_ids:
            raise RuntimeError(f"Duplicate bundled report template id: {template_id}")
        seen_ids.add(template_id)
        templates.append(template)
    return templates


def get_builtin_report_template(template_id: str) -> dict[str, Any] | None:
    """Return one fresh bundled preset by ID, or ``None`` when it does not exist."""

    try:
        normalized_id = _normalized_template_id(template_id)
    except ValueError:
        return None
    for template in builtin_report_templates():
        if template["template_id"] == normalized_id:
            return template
    return None


class ReportTemplateStore:
    """JSON-backed custom template store with bundled presets in its read view."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # One store owns the read/modify/write transaction used for versions,
        # clone IDs, deletion, and atomic file replacement.
        directory_key = str(self.directory.resolve())
        with _STORE_LOCKS_GUARD:
            self._lock = _STORE_LOCKS.setdefault(directory_key, threading.RLock())

    def _path_for_id(self, template_id: str) -> Path:
        normalized_id = _normalized_template_id(template_id)
        return self.directory / f"{normalized_id}.json"

    def _load_custom_path(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load report template {path.name}: {exc}") from exc
        template = validate_report_template(payload)
        if path != self._path_for_id(template["template_id"]):
            raise ValueError(
                f"Report template file {path.name} does not match id {template['template_id']!r}"
            )
        return template

    def list_templates(self) -> list[dict[str, Any]]:
        with self._lock:
            templates = builtin_report_templates()
            builtin_ids = {template["template_id"] for template in templates}
            custom_templates: list[dict[str, Any]] = []
            for path in sorted(self.directory.glob("*.json")):
                template = self._load_custom_path(path)
                if template["template_id"] in builtin_ids:
                    raise ValueError(f"Custom template shadows builtin id {template['template_id']!r}")
                custom_templates.append(template)
            custom_templates.sort(key=lambda item: (item["name"].casefold(), item["template_id"]))
            return [*templates, *custom_templates]

    def get_template(self, template_id: str) -> dict[str, Any] | None:
        with self._lock:
            builtin = get_builtin_report_template(template_id)
            if builtin is not None:
                return builtin
            try:
                path = self._path_for_id(template_id)
            except ValueError:
                return None
            if not path.is_file():
                return None
            return self._load_custom_path(path)

    def save_template(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._save_template_locked(payload)

    def _save_template_locked(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("report template must be an object")
        draft = deepcopy(dict(payload))
        if not draft.get("template_id"):
            name = _normalized_string(draft.get("name"), "name", max_length=160)
            draft["template_id"] = f"custom.{slugify_template_id(name)}"
        template_id = _normalized_template_id(draft["template_id"])
        if get_builtin_report_template(template_id) is not None or template_id.startswith("builtin."):
            raise ValueError("builtin report templates are immutable; clone the preset first")
        draft["template_id"] = template_id
        draft["builtin"] = False
        path = self._path_for_id(template_id)
        if path.is_file():
            existing = self._load_custom_path(path)
            draft["version"] = existing["version"] + 1
        else:
            draft["version"] = 1
        draft.pop("revision_hash", None)
        template = validate_report_template(draft)
        serialized = json.dumps(template, ensure_ascii=False, indent=2) + "\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{template_id}.",
                suffix=".tmp",
                dir=self.directory,
                delete=False,
            ) as temporary:
                temporary.write(serialized)
                temporary_name = temporary.name
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        return deepcopy(template)

    def delete_template(self, template_id: str) -> bool:
        with self._lock:
            normalized_id = _normalized_template_id(template_id)
            if get_builtin_report_template(normalized_id) is not None or normalized_id.startswith("builtin."):
                raise ValueError("builtin report templates are immutable")
            path = self._path_for_id(normalized_id)
            if not path.is_file():
                return False
            path.unlink()
            return True

    def clone_template(self, source_id: str, name: str) -> dict[str, Any]:
        with self._lock:
            source = self.get_template(source_id)
            if source is None:
                raise ValueError(f"Unknown report template: {source_id}")
            normalized_name = _normalized_string(name, "name", max_length=160)
            base_id = f"custom.{slugify_template_id(normalized_name)}"
            candidate_id = base_id
            suffix = 2
            while self.get_template(candidate_id) is not None:
                candidate_id = f"{base_id}-{suffix}"
                suffix += 1
            draft = deepcopy(source)
            draft.update({
                "template_id": candidate_id,
                "name": normalized_name,
                "version": 1,
                "builtin": False,
            })
            draft.pop("revision_hash", None)
            return self._save_template_locked(draft)
