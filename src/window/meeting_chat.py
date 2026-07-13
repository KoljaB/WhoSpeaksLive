"""Grounded single- and cross-session chat with durable hybrid retrieval."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import sqlite3
import tempfile
import threading
from typing import Any, Callable, Iterable, Iterator, Protocol
import urllib.error
import urllib.request
import uuid

import numpy as np

from window.meeting_intelligence import transcript_revision_id
from window.meeting_intelligence_pipeline import StructuredChatClient


MAX_SCOPE_MEETINGS = 20
FULL_CONTEXT_WORD_LIMIT = 8000
TARGET_CHUNK_WORDS = 180
MAX_CHUNK_WORDS = 260
OVERLAP_WORDS = 40
TOKEN_PATTERN = re.compile(r"[\w'-]+", flags=re.UNICODE)
QUERY_STOPWORDS = {
    "about", "after", "also", "been", "being", "could", "does", "from", "have", "meeting",
    "said", "says", "that", "their", "there", "these", "they", "this", "transcript", "what",
    "when", "where", "which", "with", "would",
    "aber", "auch", "dann", "dass", "dies", "diesem", "dieser", "gesagt", "hatte", "hier",
    "meeting", "sagte", "sind", "über", "wurde", "werden", "wird",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_hash(value: Any, length: int = 24) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(str(text or ""))]


def _row_id(row: dict[str, Any], index: int) -> str:
    explicit = row.get("row_id") or row.get("id") or row.get("sentence_id")
    if explicit:
        return str(explicit)
    try:
        transcript_index = int(row.get("index"))
    except (TypeError, ValueError):
        transcript_index = index
    return f"row_{transcript_index}"


def _speaker_name(row: dict[str, Any]) -> str:
    return str(row.get("speaker_name") or row.get("speaker") or row.get("speaker_label") or "Unknown").strip() or "Unknown"


def finalized_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict) or bool(raw.get("pending")):
            continue
        text = str(raw.get("text") or raw.get("sentence") or "").strip()
        if not text:
            continue
        row = dict(raw)
        if row.get("index") is None:
            row["index"] = index
        row["row_id"] = _row_id(row, index)
        row["text"] = text
        row["speaker_id"] = str(row.get("speaker_id") or row.get("assigned_speaker") or "")
        row["speaker_name"] = _speaker_name(row)
        row["start"] = float(row.get("start") or row.get("start_sec") or 0.0)
        row["end"] = float(row.get("end") or row.get("end_sec") or row["start"])
        result.append(row)
    return result


def scope_id_for(session_ids: Iterable[str]) -> str:
    normalized = sorted({str(value or "").strip() for value in session_ids if str(value or "").strip()})
    if not normalized:
        raise ValueError("Choose at least one session.")
    if len(normalized) > MAX_SCOPE_MEETINGS:
        raise ValueError(f"Select at most {MAX_SCOPE_MEETINGS} sessions.")
    return f"meetings_{_stable_hash(normalized, 20)}"


def chunk_transcript(
    session_id: str,
    title: str,
    rows: Iterable[dict[str, Any]],
    revision_id: str,
) -> list[dict[str, Any]]:
    clean_rows: list[dict[str, Any]] = []
    for row in finalized_rows(rows):
        words = str(row["text"]).split()
        if len(_tokens(row["text"])) <= MAX_CHUNK_WORDS:
            clean_rows.append(row)
            continue
        part: list[str] = []
        for word in words:
            candidate = " ".join([*part, word])
            if part and len(_tokens(candidate)) > MAX_CHUNK_WORDS:
                split_row = dict(row)
                split_row["text"] = " ".join(part)
                clean_rows.append(split_row)
                part = [word]
            else:
                part.append(word)
        if part:
            split_row = dict(row)
            split_row["text"] = " ".join(part)
            clean_rows.append(split_row)
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_words = 0

    def emit() -> None:
        nonlocal current, current_words
        if not current:
            return
        ordinal = len(chunks)
        text = "\n".join(f"{row['speaker_name']}: {row['text']}" for row in current)
        row_ids = [str(row["row_id"]) for row in current]
        speaker_ids = sorted({str(row.get("speaker_id") or "") for row in current if row.get("speaker_id")})
        speaker_names = sorted({str(row.get("speaker_name") or "Unknown") for row in current})
        content = {
            "session_id": session_id,
            "title": title,
            "row_ids": row_ids,
            "speaker_ids": speaker_ids,
            "speaker_names": speaker_names,
            "text": text,
            "start": float(current[0].get("start") or 0.0),
            "end": float(current[-1].get("end") or 0.0),
        }
        chunks.append({
            **content,
            "chunk_id": f"{session_id}:CH-{ordinal + 1:05d}",
            "ordinal": ordinal,
            "revision_id": revision_id,
            "content_hash": _stable_hash(content, 32),
            "word_count": sum(len(_tokens(row["text"])) for row in current),
        })

        overlap: list[dict[str, Any]] = []
        overlap_count = 0
        for row in reversed(current):
            overlap.insert(0, row)
            overlap_count += len(_tokens(row["text"]))
            if overlap_count >= OVERLAP_WORDS:
                break
        current = overlap
        current_words = overlap_count

    for row in clean_rows:
        row_words = max(1, len(_tokens(row["text"])))
        if current and current_words + row_words > MAX_CHUNK_WORDS:
            emit()
            # A complete row is preferred, but the retained overlap must not
            # make the next chunk exceed the hard maximum.
            if current and current_words + row_words > MAX_CHUNK_WORDS:
                current = []
                current_words = 0
        current.append(row)
        current_words += row_words
        if current_words >= TARGET_CHUNK_WORDS:
            emit()
    if current:
        # Do not emit a duplicate consisting only of the previous overlap.
        candidate_ids = [str(row["row_id"]) for row in current]
        if not chunks or candidate_ids != chunks[-1]["row_ids"]:
            emit()
    return chunks


@dataclass(frozen=True)
class TextEmbeddingConfig:
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    timeout_seconds: float = 120.0

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "").strip() if self.api_key_env else ""

    @property
    def configured(self) -> bool:
        return bool(self.base_url.strip() and self.model.strip())

    @property
    def identity(self) -> str:
        return _stable_hash([self.base_url.rstrip("/"), self.model], 20)

    def public(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key_configured": bool(self.api_key) if self.api_key_env else True,
        }


class TextEmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleTextEmbeddingClient:
    def __init__(self, config: TextEmbeddingConfig) -> None:
        self.config = config

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.config.configured:
            raise ValueError("Configure a text embedding URL and model in Meeting Intelligence.")
        url = self.config.base_url.rstrip("/")
        if not url.endswith("/embeddings"):
            url += "/embeddings"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps({"model": self.config.model, "input": texts}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Text embedding request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Text embedding endpoint is unavailable: {exc.reason}") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise RuntimeError("Text embedding endpoint returned no data array.")
        ordered = sorted((item for item in data if isinstance(item, dict)), key=lambda item: int(item.get("index") or 0))
        vectors = [[float(value) for value in item.get("embedding") or []] for item in ordered]
        if len(vectors) != len(texts) or not vectors or any(not vector for vector in vectors):
            raise RuntimeError("Text embedding endpoint returned incomplete vectors.")
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise RuntimeError("Text embedding endpoint returned inconsistent vector dimensions.")
        return vectors


class MockTextEmbeddingClient:
    """Deterministic token-hash embeddings for tests and the demo server."""

    dimensions = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in _tokens(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                slot = int.from_bytes(digest[:2], "little") % self.dimensions
                vector[slot] += -1.0 if digest[2] & 1 else 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            result.append([value / norm for value in vector])
        return result


def _pack_vector(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def _unpack_vector(blob: bytes, dimensions: int) -> np.ndarray:
    values = np.frombuffer(blob, dtype="<f4")
    if dimensions and int(values.size) != dimensions:
        raise ValueError("Stored embedding dimensions do not match metadata.")
    return values


class MeetingTextIndex:
    def __init__(
        self,
        database: Path,
        config: TextEmbeddingConfig,
        *,
        client_factory: Callable[[], TextEmbeddingClient] | None = None,
    ) -> None:
        self.database = Path(database).expanduser().resolve()
        self.config = config
        self.client_factory = client_factory
        self._write_lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS indexed_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    embedding_identity TEXT NOT NULL,
                    embedding_dimensions INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meeting_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    row_ids_json TEXT NOT NULL,
                    speaker_ids_json TEXT NOT NULL,
                    speaker_names_json TEXT NOT NULL,
                    start_seconds REAL NOT NULL,
                    end_seconds REAL NOT NULL,
                    revision_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding_identity TEXT NOT NULL,
                    embedding_dimensions INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    UNIQUE(session_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_meeting_chunks_session ON meeting_chunks(session_id, ordinal);
            """)

    def public_state(self, session_ids: Iterable[str]) -> dict[str, Any]:
        ids = sorted({str(value) for value in session_ids if str(value)})
        states: list[dict[str, Any]] = []
        with self._connect() as connection:
            for session_id in ids:
                row = connection.execute(
                    "SELECT revision_id, embedding_identity, embedding_dimensions, updated_at FROM indexed_sessions WHERE session_id=?",
                    (session_id,),
                ).fetchone()
                states.append({
                    "session_id": session_id,
                    "indexed": bool(row),
                    "revision_id": str(row["revision_id"]) if row else "",
                    "current_embedding_model": bool(row and row["embedding_identity"] == self.config.identity),
                    "embedding_dimensions": int(row["embedding_dimensions"]) if row else 0,
                    "updated_at": str(row["updated_at"]) if row else "",
                })
        return {"configured": self.config.configured, "sessions": states}

    def ensure_sessions(
        self,
        sessions: list[dict[str, Any]],
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not sessions:
            return {"indexed": 0, "chunks": 0}
        if not self.config.configured and self.client_factory is None:
            raise ValueError("Configure a text embedding URL and model to search long or multiple sessions.")
        callback = progress or (lambda _event: None)
        indexed = 0
        chunk_count = 0
        for position, session in enumerate(sessions, start=1):
            callback({
                "stage": "indexing",
                "message": f"Indexing {position} of {len(sessions)} sessions",
                "current": position,
                "total": len(sessions),
                "percent": int((position - 1) * 35 / max(1, len(sessions))),
            })
            if self._ensure_session(session):
                indexed += 1
            chunk_count += len(session.get("chunks") or [])
        return {"indexed": indexed, "chunks": chunk_count}

    def _ensure_session(self, session: dict[str, Any]) -> bool:
        session_id = str(session["session_id"])
        title = str(session.get("title") or session_id)
        revision_id = str(session["revision_id"])
        chunks = list(session.get("chunks") or [])
        with self._write_lock, self._connect() as connection:
            existing_session = connection.execute(
                "SELECT revision_id, embedding_identity FROM indexed_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if (
                existing_session
                and existing_session["revision_id"] == revision_id
                and existing_session["embedding_identity"] == self.config.identity
            ):
                return False
            existing = {
                str(row["chunk_id"]): row
                for row in connection.execute(
                    "SELECT chunk_id, content_hash, embedding_identity, embedding_dimensions, embedding FROM meeting_chunks WHERE session_id=?",
                    (session_id,),
                )
            }
            changed = [
                chunk for chunk in chunks
                if chunk["chunk_id"] not in existing
                or existing[chunk["chunk_id"]]["content_hash"] != chunk["content_hash"]
                or existing[chunk["chunk_id"]]["embedding_identity"] != self.config.identity
            ]
            vectors = self._new_client().embed([self._embedding_text(chunk) for chunk in changed]) if changed else []
            vector_map = {chunk["chunk_id"]: vector for chunk, vector in zip(changed, vectors)}
            valid_ids = {str(chunk["chunk_id"]) for chunk in chunks}
            for chunk_id in set(existing) - valid_ids:
                connection.execute("DELETE FROM meeting_chunks WHERE chunk_id=?", (chunk_id,))
            dimensions = 0
            for chunk in chunks:
                vector = vector_map.get(chunk["chunk_id"])
                if vector is None:
                    row = existing.get(chunk["chunk_id"])
                    if row is None:
                        raise RuntimeError("Missing embedding for a new transcript chunk.")
                    blob = bytes(row["embedding"])
                    dimensions = int(row["embedding_dimensions"])
                else:
                    blob = _pack_vector(vector)
                    dimensions = len(vector)
                connection.execute(
                    """INSERT OR REPLACE INTO meeting_chunks
                    (chunk_id, session_id, ordinal, title, text, row_ids_json, speaker_ids_json,
                     speaker_names_json, start_seconds, end_seconds, revision_id, content_hash,
                     embedding_identity, embedding_dimensions, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk["chunk_id"], session_id, int(chunk["ordinal"]), title, chunk["text"],
                        json.dumps(chunk["row_ids"], ensure_ascii=False),
                        json.dumps(chunk["speaker_ids"], ensure_ascii=False),
                        json.dumps(chunk["speaker_names"], ensure_ascii=False),
                        float(chunk["start"]), float(chunk["end"]), revision_id, chunk["content_hash"],
                        self.config.identity, dimensions, blob,
                    ),
                )
            connection.execute(
                "INSERT OR REPLACE INTO indexed_sessions VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, title, revision_id, self.config.identity, dimensions, _now()),
            )
        return True

    def delete_session(self, session_id: str) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute("DELETE FROM meeting_chunks WHERE session_id=?", (session_id,))
            connection.execute("DELETE FROM indexed_sessions WHERE session_id=?", (session_id,))

    def session_ids(self) -> set[str]:
        with self._connect() as connection:
            return {str(row[0]) for row in connection.execute("SELECT session_id FROM indexed_sessions")}

    def search(self, session_ids: list[str], question: str, *, limit: int = 30) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in session_ids)
        with self._connect() as connection:
            records = [dict(row) for row in connection.execute(
                f"SELECT * FROM meeting_chunks WHERE session_id IN ({placeholders}) AND embedding_identity=?",
                (*session_ids, self.config.identity),
            )]
        if not records:
            raise ValueError("The selected meetings have not been indexed yet.")
        query_vector = self._new_client().embed([question])[0]
        query = np.asarray(query_vector, dtype=np.float32)
        vectors = [_unpack_vector(bytes(row["embedding"]), int(row["embedding_dimensions"])) for row in records]
        if any(vector.size != query.size for vector in vectors):
            raise RuntimeError("Text embedding dimensions changed; reindex the selected meetings.")
        matrix = np.vstack(vectors)
        query_norm = float(np.linalg.norm(query))
        row_norms = np.linalg.norm(matrix, axis=1)
        denominators = row_norms * query_norm
        similarities = np.divide(
            matrix @ query,
            denominators,
            out=np.zeros(len(records), dtype=np.float32),
            where=denominators != 0,
        )
        semantic = sorted(
            ((float(score), row) for score, row in zip(similarities, records)),
            key=lambda item: item[0], reverse=True,
        )[:30]
        lexical = self._bm25(records, question)[:30]
        scores: dict[str, float] = {}
        lookup = {str(row["chunk_id"]): row for row in records}
        for rank, (_score, row) in enumerate(semantic, start=1):
            scores[str(row["chunk_id"])] = scores.get(str(row["chunk_id"]), 0.0) + 1.0 / (60 + rank)
        for rank, (_score, row) in enumerate(lexical, start=1):
            scores[str(row["chunk_id"])] = scores.get(str(row["chunk_id"]), 0.0) + 1.0 / (60 + rank)
        folded_question = question.casefold()
        for chunk_id, row in lookup.items():
            names = json.loads(row["speaker_names_json"])
            if any(len(name) > 2 and str(name).casefold() in folded_question for name in names):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 0.04
            title = str(row["title"])
            if len(title) > 3 and title.casefold() in folded_question:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 0.04
        selected = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:max(1, limit)]
        return [self._public_chunk(lookup[chunk_id], scores[chunk_id]) for chunk_id in selected]

    def _new_client(self) -> TextEmbeddingClient:
        return self.client_factory() if self.client_factory is not None else OpenAICompatibleTextEmbeddingClient(self.config)

    @staticmethod
    def _embedding_text(chunk: dict[str, Any]) -> str:
        names = ", ".join(chunk.get("speaker_names") or [])
        return f"Meeting: {chunk.get('title')}\nSpeakers: {names}\n{chunk.get('text')}"

    @staticmethod
    def _bm25(records: list[dict[str, Any]], question: str) -> list[tuple[float, dict[str, Any]]]:
        query = list(dict.fromkeys(_tokens(question)))
        documents = [_tokens(str(row["text"]) + " " + str(row["speaker_names_json"]) + " " + str(row["title"])) for row in records]
        average = sum(len(document) for document in documents) / max(1, len(documents))
        document_frequency = {token: sum(1 for document in documents if token in document) for token in query}
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row, document in zip(records, documents):
            counts = {token: document.count(token) for token in query}
            score = 0.0
            for token in query:
                count = counts[token]
                if not count:
                    continue
                frequency = document_frequency[token]
                inverse = math.log(1.0 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
                denominator = count + 1.5 * (1.0 - 0.75 + 0.75 * len(document) / max(1.0, average))
                score += inverse * count * 2.5 / denominator
            if score > 0:
                ranked.append((score, row))
        return sorted(ranked, key=lambda item: item[0], reverse=True)

    @staticmethod
    def _public_chunk(row: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "chunk_id": str(row["chunk_id"]), "session_id": str(row["session_id"]),
            "meeting_title": str(row["title"]), "ordinal": int(row["ordinal"]),
            "text": str(row["text"]), "row_ids": json.loads(row["row_ids_json"]),
            "speaker_ids": json.loads(row["speaker_ids_json"]),
            "speaker_names": json.loads(row["speaker_names_json"]),
            "start": float(row["start_seconds"]), "end": float(row["end_seconds"]),
            "revision_id": str(row["revision_id"]), "retrieval_score": float(score),
        }


class MeetingChatStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self._lock = threading.RLock()

    def load(self, session_ids: Iterable[str]) -> dict[str, Any]:
        ids = sorted({str(value) for value in session_ids if str(value)})
        scope_id = scope_id_for(ids)
        path = self.directory / f"{scope_id}.json"
        with self._lock:
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = {}
            else:
                payload = {}
        return {
            "schema_version": "meeting_chat_v1", "scope_id": scope_id,
            "session_ids": ids, "history": list(payload.get("history") or []),
            "updated_at": str(payload.get("updated_at") or ""),
        }

    def append(self, session_ids: Iterable[str], entry: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            payload = self.load(session_ids)
            payload["history"].append(entry)
            payload["updated_at"] = _now()
            self._write(payload)
            return payload

    def clear(self, session_ids: Iterable[str]) -> dict[str, Any]:
        with self._lock:
            payload = self.load(session_ids)
            payload["history"] = []
            payload["updated_at"] = _now()
            self._write(payload)
            return payload

    def delete_scopes_containing(self, session_id: str) -> None:
        with self._lock:
            if not self.directory.is_dir():
                return
            for path in self.directory.glob("meetings_*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if session_id in (payload.get("session_ids") or []):
                    path.unlink(missing_ok=True)

    def _write(self, payload: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{payload['scope_id']}.json"
        handle, temporary = tempfile.mkstemp(prefix=".meeting-chat-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


EVIDENCE_SELECTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string"},
        "chunk_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
    },
    "required": ["schema_version", "chunk_ids"],
}

ANSWER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string"},
        "status": {"type": "string", "enum": ["answered", "not_established", "needs_review"]},
        "answer": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["schema_version", "status", "answer", "evidence_ids"],
}


class MeetingChatEngine:
    def __init__(
        self,
        index: MeetingTextIndex,
        store: MeetingChatStore,
        *,
        session_loader: Callable[[str], dict[str, Any]],
        llm_client_factory: Callable[[], StructuredChatClient],
        report_loader: Callable[[str, str], dict[str, Any] | None] | None = None,
    ) -> None:
        self.index = index
        self.store = store
        self.session_loader = session_loader
        self.llm_client_factory = llm_client_factory
        self.report_loader = report_loader

    def scope(self, session_ids: Iterable[str]) -> dict[str, Any]:
        sessions = self._capture_sessions(session_ids)
        chat = self.store.load([session["session_id"] for session in sessions])
        index_state = self.index.public_state(chat["session_ids"])
        revisions = {session["session_id"]: session["revision_id"] for session in sessions}
        for state in index_state["sessions"]:
            state["current_revision"] = bool(
                state["indexed"] and state["revision_id"] == revisions.get(state["session_id"])
            )
        return {
            **chat,
            "meetings": [self._meeting_summary(session) for session in sessions],
            "index": index_state,
            "requires_index": self._requires_index(sessions),
        }

    def capture_sessions(self, session_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Capture immutable, revision-tagged session inputs for background indexing."""

        return self._capture_sessions(session_ids)

    def ensure_index(
        self,
        session_ids: Iterable[str],
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        sessions = self._capture_sessions(session_ids)
        if not self._requires_index(sessions):
            return {"indexed": 0, "chunks": 0}
        return self.index.ensure_sessions(sessions, progress=progress)

    def ask(
        self,
        session_ids: Iterable[str],
        question: str,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
        provisional: bool = False,
    ) -> dict[str, Any]:
        clean_question = str(question or "").strip()
        if not clean_question:
            raise ValueError("Question is required.")
        if len(clean_question) > 4000:
            raise ValueError("Question is too long.")
        callback = progress or (lambda _event: None)
        sessions = self._capture_sessions(session_ids)
        ids = [session["session_id"] for session in sessions]
        chat = self.store.load(ids)
        history = list(chat.get("history") or [])[-8:]
        if self._requires_index(sessions):
            self.index.ensure_sessions(sessions, progress=callback)
            callback({"stage": "retrieving", "message": "Searching selected sessions", "percent": 45})
            candidates = self.index.search(ids, clean_question, limit=30)
            callback({"stage": "selecting_evidence", "message": "Selecting transcript evidence", "percent": 62})
            selected_chunks = self._select_chunks(clean_question, candidates, history)
            evidence_rows = self._expand_evidence_rows(sessions, selected_chunks)
        else:
            callback({"stage": "selecting_evidence", "message": "Reading the session transcript", "percent": 55})
            evidence_rows = self._all_evidence_rows(sessions)
        if not evidence_rows:
            raise ValueError("Selected meetings contain no finalized transcript rows.")
        callback({"stage": "answering", "message": "Writing a grounded answer", "percent": 78})
        person_targets = self._person_targets(clean_question, sessions)
        answer = self._answer(
            clean_question,
            sessions,
            evidence_rows,
            history,
            person_targets,
            progress=callback,
        )
        revisions = {session["session_id"]: session["revision_id"] for session in sessions}
        transcript_end = max(
            (float(row.get("end") or 0.0) for session in sessions for row in session["rows"]),
            default=0.0,
        )
        entry = {
            "id": f"answer_{uuid.uuid4().hex[:16]}", "question": clean_question,
            "text": answer["answer"], "status": answer["status"],
            "grounding_status": answer["status"],
            "evidence": answer["evidence"], "scope_id": chat["scope_id"],
            "session_ids": ids, "meeting_revisions": revisions,
            "provisional": bool(provisional), "transcript_end_seconds": transcript_end,
            "created_at": _now(),
        }
        updated = self.store.append(ids, entry)
        callback({"stage": "succeeded", "message": "Answer ready", "percent": 100})
        return {"answer": entry, "history": updated["history"], "scope_id": chat["scope_id"]}

    def _capture_sessions(self, session_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = sorted({str(value or "").strip() for value in session_ids if str(value or "").strip()})
        scope_id_for(ids)
        sessions: list[dict[str, Any]] = []
        for session_id in ids:
            loaded = self.session_loader(session_id)
            rows = finalized_rows(loaded.get("transcript_rows") or [])
            if not rows:
                raise ValueError(f"Meeting {session_id} has no finalized transcript rows.")
            speaker_state = loaded.get("speaker_state") if isinstance(loaded.get("speaker_state"), dict) else {}
            summary = loaded.get("summary") if isinstance(loaded.get("summary"), dict) else {}
            revision_id = transcript_revision_id(rows, speaker_state)
            title = str(summary.get("title") or session_id)
            sessions.append({
                "session_id": session_id, "title": title, "summary": summary,
                "rows": rows, "speaker_state": speaker_state, "revision_id": revision_id,
                "chunks": chunk_transcript(session_id, title, rows, revision_id),
            })
        return sessions

    @staticmethod
    def _requires_index(sessions: list[dict[str, Any]]) -> bool:
        return len(sessions) > 1 or sum(len(_tokens(row["text"])) for session in sessions for row in session["rows"]) > FULL_CONTEXT_WORD_LIMIT

    @staticmethod
    def _meeting_summary(session: dict[str, Any]) -> dict[str, Any]:
        summary = session.get("summary") if isinstance(session.get("summary"), dict) else {}
        return {
            "id": session["session_id"], "title": session["title"],
            "revision_id": session["revision_id"], "row_count": len(session["rows"]),
            "duration_seconds": max((float(row.get("end") or 0.0) for row in session["rows"]), default=0.0),
            "started_at": str(summary.get("started_at") or summary.get("created_at") or ""),
            "ended_at": str(summary.get("ended_at") or ""),
            "updated_at": str(summary.get("updated_at") or ""),
            "status_label": str(summary.get("status_label") or ""),
            "speaker_count": int(summary.get("speaker_count") or 0),
        }

    def _select_chunks(self, question: str, candidates: list[dict[str, Any]], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        client = self.llm_client_factory()
        response = client.chat_json(
            schema_name="meeting_chat_evidence", schema=EVIDENCE_SELECTION_SCHEMA,
            system_prompt=(
                "Select transcript chunks that can answer the question. Prefer direct evidence, exact named speakers, "
                "numbers, negations, assignments, and decisions. Return only supplied chunk_ids and at most 12."
            ),
            user_payload={"question": question, "recent_history": history, "candidates": candidates},
            max_tokens=1200,
        )
        allowed = {chunk["chunk_id"]: chunk for chunk in candidates}
        chosen = [allowed[value] for value in response.get("chunk_ids") or [] if value in allowed]
        return chosen[:12] or candidates[:12]

    @staticmethod
    def _all_evidence_rows(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [MeetingChatEngine._evidence_row(session, row) for session in sessions for row in session["rows"]]

    @staticmethod
    def _expand_evidence_rows(sessions: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_session = {session["session_id"]: session for session in sessions}
        selected: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            session = by_session.get(chunk["session_id"])
            if session is None:
                continue
            rows = session["rows"]
            indexes = {str(row["row_id"]): index for index, row in enumerate(rows)}
            for row_id in chunk.get("row_ids") or []:
                index = indexes.get(str(row_id))
                if index is None:
                    continue
                for expanded in range(max(0, index - 1), min(len(rows), index + 2)):
                    row = MeetingChatEngine._evidence_row(session, rows[expanded])
                    selected[row["evidence_id"]] = row
        return sorted(selected.values(), key=lambda row: (row["meeting_id"], row["start"], row["evidence_id"]))

    @staticmethod
    def _evidence_row(session: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        row_id = str(row["row_id"])
        return {
            "evidence_id": f"{session['session_id']}::{row_id}", "meeting_id": session["session_id"],
            "meeting_title": session["title"], "row_id": row_id, "speaker_id": str(row.get("speaker_id") or ""),
            "row_index": int(row.get("index") or 0),
            "speaker_name": str(row.get("speaker_name") or "Unknown"), "start": float(row.get("start") or 0.0),
            "end": float(row.get("end") or 0.0), "quote": str(row.get("text") or ""),
        }

    def _answer(
        self,
        question: str,
        sessions: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
        history: list[dict[str, Any]],
        person_targets: set[str],
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        if self.report_loader is not None:
            for session in sessions:
                report = self.report_loader(session["session_id"], session["revision_id"])
                if report:
                    reports.append(report)
        exact_rows = self._exact_match_rows(question, evidence_rows, person_targets)
        client = self.llm_client_factory()
        response = self._request_answer(
            client,
            question=question,
            sessions=sessions,
            evidence_rows=evidence_rows,
            history=history,
            reports=reports,
            person_targets=person_targets,
            exact_rows=exact_rows,
            focused_retry=False,
        )
        answer = self._validated_answer(response, evidence_rows, person_targets)
        if answer["status"] == "answered" or answer["raw_status"] != "not_established" or not exact_rows:
            answer.pop("raw_status", None)
            return answer

        focused_rows = self._focused_evidence_rows(evidence_rows, exact_rows)
        if progress is not None:
            progress({
                "stage": "answering",
                "message": "Rechecking exact transcript matches",
                "percent": 88,
            })
        retry = self._request_answer(
            client,
            question=question,
            sessions=sessions,
            evidence_rows=focused_rows,
            history=history,
            reports=[],
            person_targets=person_targets,
            exact_rows=exact_rows,
            focused_retry=True,
        )
        retried = self._validated_answer(retry, focused_rows, person_targets)
        if retried["status"] == "answered":
            retried.pop("raw_status", None)
            return retried
        if retried["raw_status"] == "not_established":
            if progress is not None:
                progress({
                    "stage": "answering",
                    "message": "Building answer from matching transcript quotes",
                    "percent": 94,
                })
            return self._extractive_answer(question, exact_rows)
        retried.pop("raw_status", None)
        return retried

    def _request_answer(
        self,
        client: StructuredChatClient,
        *,
        question: str,
        sessions: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
        history: list[dict[str, Any]],
        reports: list[dict[str, Any]],
        person_targets: set[str],
        exact_rows: list[dict[str, Any]],
        focused_retry: bool,
    ) -> dict[str, Any]:
        prompt_evidence_rows = [
            {
                "evidence_id": row["evidence_id"],
                "meeting_title": row["meeting_title"],
                "speaker_name": row["speaker_name"],
                "start": row["start"],
                "quote": row["quote"],
            }
            for row in evidence_rows
        ]
        retry_instruction = (
            " A previous broad-context pass returned not_established even though exact query terms occur in the "
            "supplied transcript rows. Re-evaluate these focused rows carefully. If they state anything responsive "
            "to the question, return answered and cite the matching evidence_ids."
            if focused_retry else ""
        )
        return client.chat_json(
            schema_name="meeting_chat_answer", schema=ANSWER_SCHEMA,
            system_prompt=(
                "Answer in the user's language using transcript evidence only. Reports are orientation, never evidence. "
                "Return exact evidence_ids separately and do not print those IDs in the answer prose. Do not attribute "
                "UNKNOWN speech to a named person. If the evidence does not "
                "establish the answer, return status not_established and say so plainly. When named_speakers_in_question "
                "is non-empty, direct claims about that person require evidence attributed to that diarized speaker. "
                "exact_match_evidence_ids identify rows containing substantive query terms; inspect them first, but "
                "still base every claim on what the quoted row actually says."
                + retry_instruction
            ),
            user_payload={
                "question": question, "recent_history": history,
                "meetings": [self._meeting_summary(session) for session in sessions],
                "report_context": reports, "evidence_rows": prompt_evidence_rows,
                "exact_match_evidence_ids": [row["evidence_id"] for row in exact_rows],
                "named_speakers_in_question": sorted(person_targets),
            },
            max_tokens=1200 if focused_retry else 2200,
        )

    @staticmethod
    def _validated_answer(
        response: dict[str, Any],
        evidence_rows: list[dict[str, Any]],
        person_targets: set[str],
    ) -> dict[str, Any]:
        allowed = {row["evidence_id"]: row for row in evidence_rows}
        cited = [allowed[value] for value in response.get("evidence_ids") or [] if value in allowed]
        raw_status = str(response.get("status") or "not_established")
        status = raw_status if raw_status in {"answered", "not_established", "needs_review"} else "not_established"
        text = str(response.get("answer") or "").strip()
        if status == "answered" and not cited:
            status = "not_established"
            text = "The selected meeting transcript does not establish an answer."
        if status == "answered" and person_targets and not any(
            str(item.get("speaker_name") or "").casefold() in person_targets for item in cited
        ):
            status = "not_established"
            text = "The selected transcript does not establish that statement for the named speaker."
            cited = []
        return {
            "status": status,
            "raw_status": raw_status,
            "answer": text or "The selected meeting transcript does not establish an answer.",
            "evidence": cited,
        }

    @staticmethod
    def _query_terms(question: str) -> set[str]:
        return {
            token for token in _tokens(question)
            if len(token) >= 4 and token not in QUERY_STOPWORDS
        }

    @classmethod
    def _exact_match_rows(
        cls,
        question: str,
        evidence_rows: list[dict[str, Any]],
        person_targets: set[str],
    ) -> list[dict[str, Any]]:
        terms = cls._query_terms(question)
        if not terms:
            return []
        matches: list[tuple[int, float, dict[str, Any]]] = []
        for row in evidence_rows:
            if person_targets and str(row.get("speaker_name") or "").casefold() not in person_targets:
                continue
            row_terms = set(_tokens(str(row.get("quote") or "")))
            score = len(terms & row_terms)
            if score:
                matches.append((-score, float(row.get("start") or 0.0), row))
        matches.sort(key=lambda item: (item[0], item[1], str(item[2].get("evidence_id") or "")))
        return [item[2] for item in matches[:6]]

    @staticmethod
    def _focused_evidence_rows(
        evidence_rows: list[dict[str, Any]],
        exact_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        exact_ids = {str(row["evidence_id"]) for row in exact_rows}
        selected: set[int] = set()
        for index, row in enumerate(evidence_rows):
            if str(row.get("evidence_id") or "") not in exact_ids:
                continue
            selected.add(index)
            for neighbor in (index - 1, index + 1):
                if 0 <= neighbor < len(evidence_rows) and evidence_rows[neighbor]["meeting_id"] == row["meeting_id"]:
                    selected.add(neighbor)
        return [evidence_rows[index] for index in sorted(selected)[:12]]

    @staticmethod
    def _extractive_answer(question: str, exact_rows: list[dict[str, Any]]) -> dict[str, Any]:
        cited = exact_rows[:4]
        german_markers = {"gesagt", "wurde", "wird", "über", "welche", "welcher", "welches"}
        german = bool(set(_tokens(question)) & german_markers)
        heading = (
            "Die direkt passenden Transkriptstellen lauten:"
            if german else "The directly matching transcript passages say:"
        )
        lines = [f'- "{str(row.get("quote") or "").strip()}"' for row in cited]
        return {"status": "answered", "answer": "\n".join([heading, *lines]), "evidence": cited}

    @staticmethod
    def _person_targets(question: str, sessions: list[dict[str, Any]]) -> set[str]:
        folded = question.casefold()
        known: set[str] = set()
        for session in sessions:
            for row in session["rows"]:
                speaker_id = str(row.get("speaker_id") or "").strip().casefold()
                name = str(row.get("speaker_name") or "").strip()
                if speaker_id and speaker_id not in {"unknown", "unk"} and name.casefold() not in {"unknown", "speaker"}:
                    known.add(name.casefold())
        return {name for name in known if len(name) > 2 and name in folded}


@dataclass
class _ChatJob:
    job_id: str
    session_ids: list[str]
    question: str
    provisional: bool
    status: str = "queued"
    stage: str = "queued"
    message: str = "Question queued"
    percent: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    error: str = ""
    result: dict[str, Any] | None = None


class MeetingChatJobManager:
    _STOP = object()

    def __init__(self, engine: MeetingChatEngine, *, max_queue_size: int = 16) -> None:
        self.engine = engine
        self._queue: queue.Queue[_ChatJob | object] = queue.Queue(maxsize=max_queue_size)
        self._jobs: dict[str, _ChatJob] = {}
        self._lock = threading.RLock()
        self._thread = threading.Thread(target=self._worker, name="meeting-chat-worker", daemon=True)
        self._thread.start()

    def submit(self, session_ids: Iterable[str], question: str, *, provisional: bool = False) -> dict[str, Any]:
        ids = sorted({str(value) for value in session_ids if str(value)})
        scope_id_for(ids)
        job = _ChatJob(f"michat_{uuid.uuid4().hex[:16]}", ids, str(question or "").strip(), bool(provisional))
        if not job.question:
            raise ValueError("Question is required.")
        with self._lock:
            self._jobs[job.job_id] = job
        try:
            self._queue.put_nowait(job)
        except queue.Full as exc:
            with self._lock:
                self._jobs.pop(job.job_id, None)
            raise RuntimeError("Session chat queue is full.") from exc
        return self._snapshot(job)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(job_id or "").strip())
            if job is None:
                raise ValueError("Session chat job not found.")
            return self._snapshot(job)

    def close(self) -> None:
        self._queue.put(self._STOP)
        self._thread.join(timeout=10.0)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, _ChatJob)
                self._run(item)
            finally:
                self._queue.task_done()

    def _run(self, job: _ChatJob) -> None:
        self._update(job, status="running", stage="indexing", message="Preparing session search")
        try:
            result = self.engine.ask(
                job.session_ids, job.question, provisional=job.provisional,
                progress=lambda event: self._update(job, **event),
            )
        except Exception as exc:
            self._update(job, status="failed", stage="failed", message="Question failed", error=str(exc))
            return
        self._update(job, status="succeeded", stage="succeeded", message="Answer ready", percent=100, result=result)

    def _update(self, job: _ChatJob, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = _now()

    @staticmethod
    def _snapshot(job: _ChatJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id, "session_ids": list(job.session_ids), "status": job.status,
            "stage": job.stage, "message": job.message, "percent": job.percent,
            "created_at": job.created_at, "updated_at": job.updated_at,
            "error": job.error, "result": job.result,
        }


__all__ = [
    "FULL_CONTEXT_WORD_LIMIT", "MAX_SCOPE_MEETINGS", "MeetingChatEngine", "MeetingChatJobManager",
    "MeetingChatStore", "MeetingTextIndex", "MockTextEmbeddingClient", "OpenAICompatibleTextEmbeddingClient",
    "TextEmbeddingConfig", "chunk_transcript", "finalized_rows", "scope_id_for",
]
