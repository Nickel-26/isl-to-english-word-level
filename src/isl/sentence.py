"""Experimental sentence layer: buffer committed glosses, ask Gemini to rewrite.

This module is pure Python plus ``requests``: no OpenCV, no model, no camera, so
it can be unit-tested without a network connection.

The classifier stays **word-level**. Nothing here trains or evaluates a sentence
model: the LLM only reorders and drops glosses that the existing decision gate
already committed. Gemini is prompt-trusted; the smaller local Ollama model also
gets a lightweight grounding guard, because it can invent content words. The
``used`` / ``dropped`` lists are logged for inspection.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import requests
from dotenv import load_dotenv

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"

# The user-requested Ollama instruction, verbatim. No grammar rules, examples
# or article instructions are added; Gemini keeps build_prompt() unchanged.
OLLAMA_SYSTEM_PROMPT = (
    "following detected signs into small english sentences and remove stray "
    "signs which were probably detected on accident"
)

# Response-type/JSON instructions only. The transport also sets
# format="json" and stream=False.
OLLAMA_RESPONSE_INSTRUCTIONS = (
    "Reply with JSON only, exactly this shape:\n"
    '  {"sentence": "...", "used": ["..."], "dropped": ["..."]}\n'
    '  "used" lists the signs you kept; "dropped" lists the stray signs you ignored.'
)

# Allowed non-gloss words in an Ollama sentence. Deliberately excludes "it" and
# "I": those must come from the gloss list, not from the model's imagination.
OLLAMA_FUNCTION_WORDS = frozenset(
    {
        "a", "an", "am", "and", "are", "at", "be", "been", "being", "by",
        "for", "in", "is", "of", "on", "or", "the", "to", "was", "were",
        "with",
    }
)

# The public endpoint intermittently returns these when a model is busy.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def read_env_key(
    env_name: str,
    *,
    env_file: str | Path | None = None,
    root: str | Path | None = None,
) -> str:
    """Read a key from the environment or a project-root ``.env``.

    An already-exported variable always wins (``override=False``). The value is
    returned to the caller only; nothing here prints or logs it.
    """
    path = (
        Path(env_file).expanduser()
        if env_file is not None
        else Path(root or Path.cwd()) / ".env"
    )
    if path.is_file():
        load_dotenv(path, override=False)
    return str(os.environ.get(env_name, "")).strip()


def gloss_word(display: str) -> str:
    """Turn a display name into a single gloss for the prompt.

    ``"Big / Large" -> "big"``, ``"You (plural)" -> "you"``,
    ``"Cell phone" -> "cell phone"``. The alternative after a slash and any
    parenthetical qualifier are dropped: the model was trained on the first
    meaning, and the LLM prompt should stay short.
    """
    text = str(display).split("/", 1)[0]
    text = re.sub(r"\([^)]*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class Gloss:
    """One committed word-level prediction."""

    label: str
    display: str
    timestamp: float

    @property
    def word(self) -> str:
        return gloss_word(self.display)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "display": self.display,
            "word": self.word,
            "t": round(float(self.timestamp), 3),
        }


@dataclass
class Utterance:
    """A bounded run of glosses handed to the LLM as one unit."""

    glosses: list[Gloss]
    reason: str  # rest | gap | wait | eof
    started_at: float
    ended_at: float

    @property
    def words(self) -> list[str]:
        return [g.word for g in self.glosses]

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def __len__(self) -> int:
        return len(self.glosses)

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "started_at": round(float(self.started_at), 3),
            "ended_at": round(float(self.ended_at), 3),
            "glosses": [g.to_dict() for g in self.glosses],
        }


@dataclass
class RewriteResult:
    """LLM output plus the bookkeeping needed to audit it."""

    text: str = ""
    used: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    warning: str | None = None
    raw: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text)

    def to_dict(self) -> dict:
        return {
            "sentence": self.text,
            "used": list(self.used),
            "dropped": list(self.dropped),
            "warning": self.warning,
            "raw": self.raw,
        }


class UtteranceBuffer:
    """Collect committed glosses and flush on rest, silence, or a safety cap.

    Consecutive commits of the same label are collapsed: the gate keeps committing
    a held sign on every inference window, so without this a two-second sign would
    become six identical glosses. A word can appear again after any other gloss.
    """

    def __init__(
        self,
        *,
        gap_seconds: float = 1.2,
        max_glosses: int = 12,
        max_wait_seconds: float = 8.0,
    ):
        if gap_seconds <= 0:
            raise ValueError("gap_seconds must be > 0")
        if max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be > 0")
        if max_glosses < 1:
            raise ValueError("max_glosses must be >= 1")
        self.gap_seconds = float(gap_seconds)
        self.max_glosses = int(max_glosses)
        self.max_wait_seconds = float(max_wait_seconds)
        self.dropped_oldest = 0
        self.flush_count = 0
        self._glosses: list[Gloss] = []
        self._started_at: float | None = None
        self._last_sign_at: float | None = None

    @property
    def glosses(self) -> list[Gloss]:
        return list(self._glosses)

    @property
    def text(self) -> str:
        return " ".join(g.word for g in self._glosses)

    def __len__(self) -> int:
        return len(self._glosses)

    def _flush(self, reason: str, timestamp: float) -> Utterance | None:
        if not self._glosses:
            self._started_at = None
            self._last_sign_at = None
            return None
        utterance = Utterance(
            glosses=list(self._glosses),
            reason=reason,
            started_at=self._started_at
            if self._started_at is not None
            else self._glosses[0].timestamp,
            ended_at=float(timestamp),
        )
        self._glosses.clear()
        self._started_at = None
        self._last_sign_at = None
        self.flush_count += 1
        return utterance

    def push(self, label: str, display: str, timestamp: float) -> Utterance | None:
        """Add a committed non-rest gloss.

        Returns a completed utterance only when the wait cap fired *before* this
        gloss was added; the new gloss then starts the next utterance.
        """
        t = float(timestamp)
        flushed: Utterance | None = None
        if (
            self._glosses
            and self._started_at is not None
            and t - self._started_at >= self.max_wait_seconds
        ):
            flushed = self._flush("wait", t)

        if self._glosses and self._glosses[-1].label == label:
            self._last_sign_at = t
            return flushed

        if not self._glosses:
            self._started_at = t
        self._glosses.append(
            Gloss(label=str(label), display=str(display), timestamp=t)
        )
        if len(self._glosses) > self.max_glosses:
            excess = len(self._glosses) - self.max_glosses
            self.dropped_oldest += excess
            del self._glosses[:excess]
            self._started_at = self._glosses[0].timestamp
        self._last_sign_at = t
        return flushed

    def on_rest(self, timestamp: float) -> Utterance | None:
        """A committed 'No sign' is an explicit end of utterance."""
        return self._flush("rest", float(timestamp))

    def on_no_sign(self, timestamp: float) -> Utterance | None:
        """No committed sign at this instant; flush once the silence gap passes."""
        t = float(timestamp)
        if not self._glosses:
            self._last_sign_at = None
            return None
        if (
            self._started_at is not None
            and t - self._started_at >= self.max_wait_seconds
        ):
            return self._flush("wait", t)
        if self._last_sign_at is None:
            self._last_sign_at = t
            return None
        if t - self._last_sign_at >= self.gap_seconds:
            return self._flush("gap", t)
        return None

    def flush(
        self, reason: str = "eof", timestamp: float | None = None
    ) -> Utterance | None:
        """Flush explicitly, e.g. when the webcam session ends."""
        t = float(timestamp) if timestamp is not None else (
            self._last_sign_at if self._last_sign_at is not None else 0.0
        )
        return self._flush(reason, t)

    def reset(self) -> None:
        self._glosses.clear()
        self._started_at = None
        self._last_sign_at = None


def build_prompt(words: Sequence[str]) -> str:
    """The only guardrail: a prompt that asks for reordering, dropping and no invention."""
    glosses = ", ".join(str(w) for w in words if str(w).strip())
    return (
        "You rewrite short sequences of Indian Sign Language glosses into ONE "
        "short English sentence.\n\n"
        f"Glosses, in the order they were signed: {glosses}\n\n"
        "Rules:\n"
        "- Use only the information in the glosses. Never add a person, object, "
        "place, action, time, feeling or any other fact that is not there.\n"
        "- Reorder the glosses into natural English word order. You may add only "
        "function words (articles, prepositions, a form of 'to be' when needed) "
        "and punctuation.\n"
        "- Drop a gloss that does not fit instead of inventing a meaning for it.\n"
        "- Keep it to one short sentence. Do not explain anything.\n"
        "- Reply with JSON only, exactly this shape:\n"
        '  {"sentence": "...", "used": ["..."], "dropped": ["..."]}\n'
        '  "used" lists the glosses you kept; "dropped" lists the ones you ignored.\n\n'
        "Examples:\n"
        "  glosses: big cow\n"
        '  {"sentence": "the cow is big", "used": ["big", "cow"], "dropped": []}\n'
        "  glosses: loud paint bird\n"
        '  {"sentence": "the bird is loud", "used": ["loud", "bird"], '
        '"dropped": ["paint"]}'
    )


def build_ollama_prompt(words: Sequence[str]) -> str:
    """Minimal Ollama user prompt: only the glosses plus JSON response rules."""

    glosses = ", ".join(str(w) for w in words if str(w).strip())
    return (
        f"Glosses, in the order they were signed: {glosses}\n\n"
        + OLLAMA_RESPONSE_INSTRUCTIONS
    )


def parse_rewrite_response(text: str) -> tuple[str, list[str], list[str]]:
    """Return ``(sentence, used, dropped)`` from a model response.

    Handles raw JSON, a fenced JSON block, JSON surrounded by prose, and (as a
    last resort) a plain-text sentence. Malformed JSON never crashes the demo.
    """
    raw = str(text or "").strip()
    if not raw:
        return "", [], []

    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])
    candidates.append(raw)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        sentence = str(data.get("sentence") or data.get("text") or "").strip()
        if not sentence:
            continue
        used = [str(x) for x in _as_list(data.get("used"))]
        dropped = [str(x) for x in _as_list(data.get("dropped"))]
        return sentence, used, dropped

    # It looked like JSON but was broken: do not display the raw blob.
    if raw.startswith("{") or "```" in raw or '"sentence"' in raw:
        return "", [], []
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    return line, [], []


def _as_list(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return []


def _word_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in re.finditer(r"[A-Za-z]+", text or "")]


def _ollama_is_grounded(
    sentence: str, used: Sequence[str], glosses: Sequence[str]
) -> bool:
    """Reject Ollama output that invents content words or is a bare gloss list."""

    tokens = _word_tokens(sentence)
    if not tokens:
        return False
    gloss_set = {g.lower() for g in glosses}
    if not any(token in gloss_set for token in tokens):
        return False
    # A grounded sentence may start with a verb ("paint the window"). What we
    # still reject is a bare gloss list such as "big cow": it has no function
    # word and no glossed pronoun to make it a sentence.
    starts_with_glossed_pronoun = (
        tokens[0] in {"i", "you"} and tokens[0] in gloss_set
    )
    has_function_word = any(token in OLLAMA_FUNCTION_WORDS for token in tokens)
    if not starts_with_glossed_pronoun and not has_function_word:
        return False
    for token in tokens:
        if token not in gloss_set and token not in OLLAMA_FUNCTION_WORDS:
            return False
    for item in used:
        token = str(item).lower()
        if token not in gloss_set and token not in OLLAMA_FUNCTION_WORDS:
            return False
    return True


def _ollama_grounded_fallback(words: Sequence[str]) -> RewriteResult:
    """Build a minimal sentence from glosses when the LLM output is rejected."""

    unique: list[str] = []
    for word in words:
        if word not in unique:
            unique.append(word)
    if not unique:
        return RewriteResult(warning="empty utterance")

    first = unique[0].lower()
    if first == "i":
        head = "I"
    elif first == "you":
        head = "You"
    else:
        head = f"the {unique[0]}"
    rest = unique[1:]
    sentence = " ".join([head] + rest) + "."
    return RewriteResult(text=sentence, used=unique, dropped=[])


def _extract_text(data: Any) -> str:
    try:
        return str(data["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected Gemini response shape") from exc


def _extract_ollama_text(data: Any) -> str:
    try:
        return str(data["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected Ollama response shape") from exc


def _status_code(response: Any) -> int:
    """Read the status from a real requests.Response or a test double."""
    value = getattr(response, "status_code", getattr(response, "status", 200))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 200


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) in RETRYABLE_STATUS


class GeminiRewriter:
    """Call the Gemini ``generateContent`` endpoint; never raise on API errors."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        max_attempts: int = 3,
        retry_backoff: float = 0.6,
        post: Callable[..., Any] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ):
        if not str(api_key or "").strip():
            raise ValueError("Gemini API key is empty")
        if not str(model or "").strip():
            raise ValueError("Gemini model is empty")
        self._api_key = str(api_key)
        self.model = str(model)
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.max_attempts = max(1, int(max_attempts))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.url = GEMINI_ENDPOINT.format(model=self.model)
        self._post = post  # injectable for tests; defaults to requests.post
        self._sleep = sleep or time.sleep

    def _safe_error(self, exc: BaseException) -> str:
        message = f"{type(exc).__name__}: {exc}"
        message = message.replace(self._api_key, "<redacted>")
        return message[:300]

    def rewrite(self, utterance: Utterance | Sequence[str]) -> RewriteResult:
        words = (
            utterance.words
            if isinstance(utterance, Utterance)
            else [str(w) for w in utterance]
        )
        words = [w for w in words if w]
        if not words:
            return RewriteResult(warning="empty utterance")

        # Gemini 3.x Flash-Lite ignores custom sampling parameters and Google
        # plans to reject them with HTTP 400. The default 0.0 therefore means
        # "do not send temperature": determinism comes from the explicit prompt
        # rules. A non-default value is still passed through for other models.
        generation_config: dict[str, Any] = {"responseMimeType": "application/json"}
        if self.temperature != 0.0:
            generation_config["temperature"] = self.temperature
        payload = {
            "contents": [{"role": "user", "parts": [{"text": build_prompt(words)}]}],
            "generationConfig": generation_config,
        }
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        post = self._post or requests.post
        text: str | None = None
        for attempt in range(self.max_attempts):
            try:
                response = post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                status = _status_code(response)
                if status in RETRYABLE_STATUS and attempt + 1 < self.max_attempts:
                    # Busy/rate-limited: wait and try again instead of failing.
                    self._sleep(self.retry_backoff * (2**attempt))
                    continue
                response.raise_for_status()
                text = _extract_text(response.json())
                break
            except Exception as exc:  # network, HTTP, JSON or schema failure
                if attempt + 1 < self.max_attempts and _is_retryable(exc):
                    self._sleep(self.retry_backoff * (2**attempt))
                    continue
                return RewriteResult(warning=self._safe_error(exc))

        if text is None:
            return RewriteResult(warning="Gemini request failed after retries")

        sentence, used, dropped = parse_rewrite_response(text)
        result = RewriteResult(
            text=sentence,
            used=used,
            dropped=dropped,
            raw=text[:500] if not sentence else None,
        )
        if not sentence:
            result.warning = "Gemini returned no usable sentence"
        return result


class OllamaRewriter:
    """Call a local Ollama ``/api/chat`` model; never raise on API errors.

    Uses a minimal ``build_ollama_prompt`` and the shared
    ``parse_rewrite_response``, plus Ollama's JSON mode. Intended for
    ``qwen2.5:3b-instruct`` so no API key or internet is needed.
    """

    def __init__(
        self,
        *,
        model: str = "qwen2.5:3b-instruct",
        base_url: str = OLLAMA_DEFAULT_URL,
        timeout_seconds: float = 60.0,
        temperature: float = 0.0,
        seed: int = 42,
        post: Callable[..., Any] | None = None,
    ):
        if not str(model or "").strip():
            raise ValueError("Ollama model is empty")
        if not str(base_url or "").strip():
            raise ValueError("Ollama base URL is empty")
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.url = f"{self.base_url}/api/chat"
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self._post = post  # injectable for tests; defaults to requests.post

    @staticmethod
    def _safe_error(exc: BaseException) -> str:
        return f"{type(exc).__name__}: {exc}"[:300]

    def rewrite(self, utterance: Utterance | Sequence[str]) -> RewriteResult:
        words = (
            utterance.words
            if isinstance(utterance, Utterance)
            else [str(w) for w in utterance]
        )
        words = [w for w in words if w]
        if not words:
            return RewriteResult(warning="empty utterance")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
                {"role": "user", "content": build_ollama_prompt(words)},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 1.0,
                "seed": self.seed,
                "num_predict": 128,
            },
        }
        post = self._post or requests.post
        try:
            response = post(
                self.url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            text = _extract_ollama_text(response.json())
        except Exception as exc:  # network, HTTP, JSON or schema failure
            return RewriteResult(warning=self._safe_error(exc))

        sentence, used, dropped = parse_rewrite_response(text)
        if not _ollama_is_grounded(sentence, used, words):
            # Qwen 2.5 3B occasionally invents a content word or returns a bare
            # gloss list. Never show that: rebuild from the glosses.
            result = _ollama_grounded_fallback(words)
            result.raw = text[:500]
            result.warning = "Ollama output rejected as ungrounded; used gloss fallback"
            return result
        return RewriteResult(
            text=sentence,
            used=used,
            dropped=dropped,
            raw=text[:500] if not sentence else None,
        )


class EchoRewriter:
    """Offline debug rewriter. Never calls the network; not a grammar model."""

    def rewrite(self, utterance: Utterance | Sequence[str]) -> RewriteResult:
        words = (
            utterance.words
            if isinstance(utterance, Utterance)
            else [str(w) for w in utterance]
        )
        words = [w for w in words if w]
        if not words:
            return RewriteResult(warning="empty utterance")
        return RewriteResult(text=", ".join(words), used=list(words), dropped=[])


class AsyncRewriter:
    """Run a rewriter on one background worker so a slow LLM cannot stall capture.

    The live demo must keep reading frames while Gemini thinks: the public
    endpoint intermittently takes 5-30 s or returns 503. ``submit`` only queues
    work; ``poll`` returns any finished results. ``close`` stops the daemon
    worker after a grace period so quitting the demo never hangs forever.
    """

    def __init__(self, rewriter: Any):
        self._rewriter = rewriter
        self._requests: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="sentence-rewriter", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._requests.get()
            try:
                if item is None:
                    return
                utterance, tag = item
                try:
                    result = self._rewriter.rewrite(utterance)
                except Exception as exc:  # a bad backend must not kill the demo
                    result = RewriteResult(
                        warning=f"{type(exc).__name__}: {exc}"[:300]
                    )
                self._results.put((utterance, tag, result))
            finally:
                self._requests.task_done()

    def submit(self, utterance: Utterance | None, tag: Any = None) -> bool:
        """Queue one utterance. Returns False for None or after ``close``."""
        if utterance is None or self._closed:
            return False
        self._requests.put((utterance, tag))
        return True

    def poll(self) -> list[tuple[Utterance, Any, RewriteResult]]:
        """Return every result that finished since the last poll (non-blocking)."""
        done: list[tuple[Utterance, Any, RewriteResult]] = []
        while True:
            try:
                done.append(self._results.get_nowait())
            except queue.Empty:
                return done

    def close(self, timeout: float | None = None) -> list[
        tuple[Utterance, Any, RewriteResult]
    ]:
        """Stop accepting work, wait up to ``timeout``, return finished results."""
        if not self._closed:
            self._closed = True
            self._requests.put(None)
        self._thread.join(timeout)
        return self.poll()

    @property
    def pending(self) -> int:
        """Outstanding requests still queued or in flight."""
        return max(0, self._requests.unfinished_tasks - (1 if self._closed else 0))
