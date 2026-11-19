"""Sentence-buffer and Gemini-rewrite tests. No camera, no network.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl.config import Config, ConfigError  # noqa: E402
from isl.sentence import (  # noqa: E402
    OLLAMA_RESPONSE_INSTRUCTIONS,
    OLLAMA_SYSTEM_PROMPT,
    AsyncRewriter,
    EchoRewriter,
    GeminiRewriter,
    Gloss,
    OllamaRewriter,
    RewriteResult,
    Utterance,
    UtteranceBuffer,
    build_ollama_prompt,
    build_prompt,
    gloss_word,
    parse_rewrite_response,
    read_env_key,
)

CONFIG = ROOT / "configs" / "include50.yaml"
SECRET = "test-secret-key-do-not-log"


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload=None, status: int = 200, error: Exception | None = None):
        self.payload = payload if payload is not None else {}
        self.status = status
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status} from Gemini")

    def json(self):
        return self.payload


def gemini_payload(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def make_post(payload=None, status: int = 200, error: Exception | None = None):
    calls: list[dict] = []

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse(payload, status=status, error=error)

    return post, calls


class TestGlossWord(unittest.TestCase):
    def test_display_to_gloss(self) -> None:
        self.assertEqual(gloss_word("Big / Large"), "big")
        self.assertEqual(gloss_word("Small / Little"), "small")
        self.assertEqual(gloss_word("You (plural)"), "you")
        self.assertEqual(gloss_word("Cell phone"), "cell phone")
        self.assertEqual(gloss_word("Cow"), "cow")


class TestUtteranceBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = UtteranceBuffer(gap_seconds=1.2, max_glosses=12, max_wait_seconds=8.0)

    def test_accumulates_and_collapses_held_sign(self) -> None:
        self.assertIsNone(self.buffer.push("cow", "Cow", 0.0))
        self.assertIsNone(self.buffer.push("cow", "Cow", 0.33))
        self.assertIsNone(self.buffer.push("big", "Big / Large", 0.66))
        self.assertEqual(self.buffer.text, "cow big")
        self.assertEqual(len(self.buffer), 2)

    def test_rest_flushes_immediately(self) -> None:
        self.buffer.push("big", "Big / Large", 0.0)
        self.buffer.push("cow", "Cow", 0.5)
        utterance = self.buffer.on_rest(1.0)
        self.assertIsNotNone(utterance)
        assert utterance is not None
        self.assertEqual(utterance.reason, "rest")
        self.assertEqual(utterance.words, ["big", "cow"])
        self.assertEqual(len(self.buffer), 0)
        self.assertIsNone(self.buffer.on_rest(1.1))

    def test_silence_gap_flushes(self) -> None:
        self.buffer.push("loud", "Loud", 0.0)
        self.buffer.push("bird", "Bird", 0.5)
        self.assertIsNone(self.buffer.on_no_sign(1.0))  # 0.5 s of silence
        utterance = self.buffer.on_no_sign(1.8)  # 1.3 s of silence
        self.assertIsNotNone(utterance)
        assert utterance is not None
        self.assertEqual(utterance.reason, "gap")
        self.assertEqual(utterance.words, ["loud", "bird"])

    def test_held_sign_refreshes_the_gap(self) -> None:
        self.buffer.push("cow", "Cow", 0.0)
        self.buffer.push("cow", "Cow", 1.0)
        self.assertIsNone(self.buffer.on_no_sign(2.0))
        self.assertIsNotNone(self.buffer.on_no_sign(2.3))

    def test_empty_buffer_never_flushes(self) -> None:
        self.assertIsNone(self.buffer.on_no_sign(50.0))
        self.assertIsNone(self.buffer.flush("eof", 51.0))

    def test_wait_cap_flushes_before_new_gloss(self) -> None:
        small = UtteranceBuffer(gap_seconds=1.0, max_glosses=12, max_wait_seconds=2.0)
        small.push("cow", "Cow", 0.0)
        small.push("big", "Big / Large", 1.0)
        flushed = small.push("bird", "Bird", 2.2)
        self.assertIsNotNone(flushed)
        assert flushed is not None
        self.assertEqual(flushed.reason, "wait")
        self.assertEqual(flushed.words, ["cow", "big"])
        self.assertEqual(small.text, "bird")

    def test_max_glosses_drops_oldest(self) -> None:
        tiny = UtteranceBuffer(gap_seconds=1.0, max_glosses=2, max_wait_seconds=8.0)
        tiny.push("cow", "Cow", 0.0)
        tiny.push("big", "Big / Large", 0.3)
        tiny.push("bird", "Bird", 0.6)
        self.assertEqual(tiny.text, "big bird")
        self.assertEqual(tiny.dropped_oldest, 1)

    def test_eof_flush(self) -> None:
        self.buffer.push("cow", "Cow", 0.0)
        utterance = self.buffer.flush("eof", 0.6)
        self.assertIsNotNone(utterance)
        assert utterance is not None
        self.assertEqual(utterance.reason, "eof")
        self.assertEqual(utterance.glosses[0].label, "cow")


class TestPromptAndParsing(unittest.TestCase):
    def test_prompt_has_ordered_glosses_and_examples(self) -> None:
        prompt = build_prompt(["big", "cow"])
        self.assertIn("big, cow", prompt)
        self.assertIn("the cow is big", prompt)
        self.assertIn("the bird is loud", prompt)
        self.assertIn("Do not explain", prompt)
        self.assertIn('"dropped"', prompt)

    def test_parse_json(self) -> None:
        text = '{"sentence": "the cow is big", "used": ["big", "cow"], "dropped": []}'
        sentence, used, dropped = parse_rewrite_response(text)
        self.assertEqual(sentence, "the cow is big")
        self.assertEqual(used, ["big", "cow"])
        self.assertEqual(dropped, [])

    def test_parse_fenced_json(self) -> None:
        text = '```json\n{"sentence": "the bird is loud", "used": ["loud"]}\n```'
        sentence, used, dropped = parse_rewrite_response(text)
        self.assertEqual(sentence, "the bird is loud")
        self.assertEqual(used, ["loud"])
        self.assertEqual(dropped, [])

    def test_parse_json_surrounded_by_prose(self) -> None:
        text = 'Sure!\n{"sentence": "the cow is big", "used": [], "dropped": []}\nDone.'
        sentence, _, _ = parse_rewrite_response(text)
        self.assertEqual(sentence, "the cow is big")

    def test_parse_plain_text_fallback(self) -> None:
        sentence, used, dropped = parse_rewrite_response("The cow is big.")
        self.assertEqual(sentence, "The cow is big.")
        self.assertEqual((used, dropped), ([], []))

    def test_parse_broken_json_does_not_show_raw_blob(self) -> None:
        sentence, used, dropped = parse_rewrite_response(
            '{"sentence": "the cow is big", "used": '
        )
        self.assertEqual((sentence, used, dropped), ("", [], []))


class TestGeminiRewriter(unittest.TestCase):
    def make_rewriter(self, post, **kwargs):
        kwargs.setdefault("sleep", lambda _: None)
        return GeminiRewriter(
            api_key=SECRET,
            model="gemini-3.5-flash-lite",
            timeout_seconds=5.0,
            temperature=0.0,
            post=post,
            **kwargs,
        )

    def test_success_uses_expected_endpoint_headers_and_body(self) -> None:
        text = '{"sentence": "the cow is big", "used": ["big", "cow"], "dropped": []}'
        post, calls = make_post(gemini_payload(text))
        result = self.make_rewriter(post).rewrite(["big", "cow"])
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "the cow is big")
        self.assertEqual(result.used, ["big", "cow"])
        self.assertEqual(result.dropped, [])
        self.assertIsNone(result.warning)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(
            call["url"],
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-3.5-flash-lite:generateContent",
        )
        self.assertEqual(call["headers"]["x-goog-api-key"], SECRET)
        self.assertEqual(call["timeout"], 5.0)
        self.assertIn("big, cow", call["json"]["contents"][0]["parts"][0]["text"])
        # Default 0.0 is deliberately not sent: Gemini 3.5 Flash-Lite deprecates
        # custom sampling parameters; the prompt carries the determinism rules.
        self.assertNotIn("temperature", call["json"]["generationConfig"])
        self.assertEqual(
            call["json"]["generationConfig"]["responseMimeType"], "application/json"
        )
        self.assertNotIn(SECRET, call["url"])

    def test_non_default_temperature_is_passed_through(self) -> None:
        post, calls = make_post(gemini_payload('{"sentence": "the cow is big"}'))
        rewriter = GeminiRewriter(
            api_key=SECRET,
            model="older-model",
            temperature=0.5,
            post=post,
        )
        result = rewriter.rewrite(["cow"])
        self.assertTrue(result.ok)
        self.assertEqual(calls[0]["json"]["generationConfig"]["temperature"], 0.5)

    def test_http_error_returns_warning_without_raising(self) -> None:
        post, calls = make_post(status=429)
        result = self.make_rewriter(post).rewrite(["cow"])
        self.assertFalse(result.ok)
        self.assertEqual(result.text, "")
        self.assertIn("HTTP 429", result.warning or "")
        self.assertEqual(len(calls), 3)  # retried, then gave up

    def test_transient_503_is_retried_then_succeeds(self) -> None:
        ok_text = '{"sentence": "the cow is big", "used": ["big", "cow"], "dropped": []}'
        calls: list[dict] = []

        def post(url, **kwargs):
            calls.append({"url": url, **kwargs})
            if len(calls) == 1:
                return FakeResponse({"error": "busy"}, status=503)
            return FakeResponse(gemini_payload(ok_text))

        sleeps: list[float] = []
        rewriter = self.make_rewriter(post, sleep=sleeps.append)
        result = rewriter.rewrite(["big", "cow"])
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "the cow is big")
        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(len(sleeps), 1)

    def test_timeout_error_is_retried_then_reported(self) -> None:
        import requests

        post, calls = make_post(error=requests.Timeout("slow"))
        result = self.make_rewriter(post).rewrite(["cow"])
        self.assertFalse(result.ok)
        self.assertIn("Timeout", result.warning or "")
        self.assertEqual(len(calls), 3)

    def test_malformed_response_returns_warning(self) -> None:
        post, _ = make_post({"unexpected": "shape"})
        result = self.make_rewriter(post).rewrite(["cow"])
        self.assertFalse(result.ok)
        self.assertIn("unexpected Gemini response shape", result.warning or "")

    def test_no_hard_validator_on_the_sentence(self) -> None:
        # The LLM may add "the"; we report it, we do not reject it.
        text = '{"sentence": "the happy cow is big", "used": ["big", "cow"], "dropped": []}'
        post, _ = make_post(gemini_payload(text))
        result = self.make_rewriter(post).rewrite(["big", "cow"])
        self.assertEqual(result.text, "the happy cow is big")

    def test_api_key_is_redacted_from_error_messages(self) -> None:
        post, _ = make_post(error=RuntimeError(f"request rejected for {SECRET}"))
        result = self.make_rewriter(post).rewrite(["cow"])
        self.assertNotIn(SECRET, result.warning or "")
        self.assertIn("<redacted>", result.warning or "")

    def test_empty_key_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GeminiRewriter(api_key="", model="gemini-3.5-flash-lite", post=make_post({})[0])

    def test_empty_utterance_does_not_call_network(self) -> None:
        post, calls = make_post(gemini_payload('{"sentence": "x"}'))
        result = self.make_rewriter(post).rewrite([])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls), 0)

    def test_echo_rewriter_is_offline(self) -> None:
        result = EchoRewriter().rewrite(["big", "cow"])
        self.assertEqual(result.text, "big, cow")
        self.assertEqual(result.used, ["big", "cow"])


class TestOllamaRewriter(unittest.TestCase):
    def make_rewriter(self, post, **kwargs):
        return OllamaRewriter(
            model="qwen2.5:3b-instruct",
            base_url="http://127.0.0.1:11434/",
            timeout_seconds=12.0,
            post=post,
            **kwargs,
        )

    def test_success_posts_chat_payload(self) -> None:
        text = '{"sentence": "the cow is big", "used": ["big", "cow"], "dropped": []}'
        post, calls = make_post({"message": {"content": text}})
        result = self.make_rewriter(post).rewrite(["big", "cow"])
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "the cow is big")
        self.assertEqual(result.used, ["big", "cow"])
        self.assertIsNone(result.warning)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(call["timeout"], 12.0)
        body = call["json"]
        self.assertEqual(body["model"], "qwen2.5:3b-instruct")
        self.assertTrue(body["format"])
        self.assertEqual(body["format"], "json")
        self.assertFalse(body["stream"])
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][0]["content"], OLLAMA_SYSTEM_PROMPT)
        self.assertEqual(body["messages"][1]["role"], "user")
        self.assertEqual(
            body["messages"][1]["content"], build_ollama_prompt(["big", "cow"])
        )
        self.assertIn("big, cow", body["messages"][1]["content"])
        self.assertTrue(
            body["messages"][1]["content"].endswith(OLLAMA_RESPONSE_INSTRUCTIONS)
        )
        self.assertNotIn("Examples:", body["messages"][1]["content"])
        self.assertNotIn("Rules:", body["messages"][1]["content"])
        self.assertEqual(
            body["options"],
            {"temperature": 0.0, "top_p": 1.0, "seed": 42, "num_predict": 128},
        )

    def test_hallucinated_adjective_falls_back_to_glosses(self) -> None:
        text = '{"sentence": "the cow is big", "used": ["cow", "big"], "dropped": []}'
        post, _ = make_post({"message": {"content": text}})
        result = self.make_rewriter(post).rewrite(["cow"])
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "the cow.")
        self.assertEqual(result.used, ["cow"])
        self.assertIn("ungrounded", result.warning or "")

    def test_bare_gloss_falls_back_to_glosses(self) -> None:
        text = '{"sentence": "cow", "used": ["cow"], "dropped": []}'
        post, _ = make_post({"message": {"content": text}})
        result = self.make_rewriter(post).rewrite(["cow"])
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "the cow.")
        self.assertIn("ungrounded", result.warning or "")

    def test_grounded_verb_first_sentence_is_accepted(self) -> None:
        text = (
            '{"sentence": "paint the window", '
            '"used": ["paint", "window"], "dropped": []}'
        )
        post, _ = make_post({"message": {"content": text}})
        result = self.make_rewriter(post).rewrite(["paint", "window"])
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "paint the window")
        self.assertEqual(result.used, ["paint", "window"])
        self.assertIsNone(result.warning)

    def test_bare_gloss_list_still_falls_back(self) -> None:
        text = '{"sentence": "big cow", "used": ["big", "cow"], "dropped": []}'
        post, _ = make_post({"message": {"content": text}})
        result = self.make_rewriter(post).rewrite(["big", "cow"])
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "the big cow.")
        self.assertIn("ungrounded", result.warning or "")

    def test_http_error_returns_warning_without_raising(self) -> None:
        post, _ = make_post(status=500)
        result = self.make_rewriter(post).rewrite(["cow"])
        self.assertFalse(result.ok)
        self.assertIn("HTTP 500", result.warning or "")

    def test_unexpected_response_shape_returns_warning(self) -> None:
        post, _ = make_post({"unexpected": "shape"})
        result = self.make_rewriter(post).rewrite(["cow"])
        self.assertFalse(result.ok)
        self.assertIn("unexpected Ollama response shape", result.warning or "")

    def test_empty_utterance_does_not_call_network(self) -> None:
        post, calls = make_post({"message": {"content": '{"sentence": "x"}'}})
        result = self.make_rewriter(post).rewrite([])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls), 0)


class TestAsyncRewriter(unittest.TestCase):
    @staticmethod
    def make_utterance(*words: str) -> Utterance:
        glosses = [
            Gloss(label=w, display=w.capitalize(), timestamp=float(i))
            for i, w in enumerate(words)
        ]
        return Utterance(
            glosses=glosses,
            reason="gap",
            started_at=0.0,
            ended_at=float(len(words)),
        )

    def test_submit_returns_immediately_while_backend_is_slow(self) -> None:
        release = threading.Event()

        class Slow:
            def rewrite(self, utterance):
                release.wait(2.0)
                return RewriteResult(text="the cow is big", used=utterance.words)

        runner = AsyncRewriter(Slow())
        try:
            started = time.perf_counter()
            self.assertTrue(runner.submit(self.make_utterance("big", "cow"), 1.5))
            self.assertLess(time.perf_counter() - started, 0.5)
            self.assertEqual(runner.poll(), [])

            release.set()
            deadline = time.perf_counter() + 2.0
            results = []
            while not results and time.perf_counter() < deadline:
                results = runner.poll()
                if not results:
                    time.sleep(0.01)
            self.assertEqual(len(results), 1)
            utterance, tag, result = results[0]
            self.assertEqual(tag, 1.5)
            self.assertEqual(result.text, "the cow is big")
        finally:
            release.set()
            runner.close(timeout=2.0)

    def test_close_drains_queued_work(self) -> None:
        runner = AsyncRewriter(EchoRewriter())
        runner.submit(self.make_utterance("big", "cow"), 1.0)
        runner.submit(self.make_utterance("loud", "bird"), 2.0)
        results = runner.close(timeout=2.0)
        self.assertEqual(len(results), 2)
        self.assertEqual(runner.pending, 0)
        self.assertEqual({r[2].text for r in results}, {"big, cow", "loud, bird"})

    def test_backend_exception_is_returned_as_warning(self) -> None:
        class Broken:
            def rewrite(self, utterance):
                raise RuntimeError("backend exploded")

        runner = AsyncRewriter(Broken())
        runner.submit(self.make_utterance("cow"), 1.0)
        results = runner.close(timeout=2.0)
        self.assertEqual(len(results), 1)
        self.assertIn("backend exploded", results[0][2].warning or "")
        self.assertFalse(results[0][2].ok)

    def test_submit_after_close_is_ignored(self) -> None:
        runner = AsyncRewriter(EchoRewriter())
        runner.close(timeout=2.0)
        self.assertFalse(runner.submit(self.make_utterance("cow"), 1.0))


class TestEnvKey(unittest.TestCase):
    NAME = "SENTENCE_TEST_ENV_KEY"

    def test_reads_key_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".env").write_text(
                f"# comment\n{self.NAME}='file-secret'\n", encoding="utf-8"
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(read_env_key(self.NAME, root=tmp), "file-secret")

    def test_exported_variable_wins_over_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".env").write_text(
                f"{self.NAME}=file-secret\n", encoding="utf-8"
            )
            with patch.dict(
                os.environ, {self.NAME: "exported-secret"}, clear=True
            ):
                self.assertEqual(
                    read_env_key(self.NAME, root=tmp), "exported-secret"
                )

    def test_missing_key_returns_empty_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(read_env_key(self.NAME, root=tmp), "")


class TestSentenceConfig(unittest.TestCase):
    def test_shipped_config_values(self) -> None:
        settings = Config(CONFIG).sentence
        self.assertEqual(settings.gap_seconds, 1.2)
        self.assertEqual(settings.max_glosses, 12)
        self.assertEqual(settings.max_wait_seconds, 8.0)
        self.assertEqual(settings.gemini_model, "gemini-3.1-flash-lite")
        self.assertEqual(settings.api_key_env, "GEMINI_API_KEY")
        self.assertEqual(settings.timeout_seconds, 30.0)
        self.assertEqual(settings.temperature, 0.0)
        self.assertEqual(settings.ollama_model, "qwen2.5:3b-instruct")
        self.assertEqual(settings.ollama_url, "http://127.0.0.1:11434")
        self.assertEqual(settings.ollama_timeout_seconds, 60.0)

    def test_invalid_gap_rejected(self) -> None:
        raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        raw["sentence"]["gap_seconds"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "configs" / "bad_gap.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ConfigError):
                Config(path)

    def test_missing_block_uses_defaults(self) -> None:
        raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        del raw["sentence"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "configs" / "no_sentence.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            settings = Config(path).sentence
        self.assertEqual(settings.gap_seconds, 1.2)
        self.assertEqual(settings.api_key_env, "GEMINI_API_KEY")


if __name__ == "__main__":
    unittest.main()
