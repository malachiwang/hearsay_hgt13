from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import NewAttempt.Deepfake as deepfake
from grok_interpretation import (
    GrokAPIError,
    GrokInterpretation,
    GrokNotConfiguredError,
    ask_grok_about_analysis,
    build_grok_context,
)


ANALYSIS = {
    "file": "/private/data/example.wav",
    "eliya_mean": 0.35,
    "eliya_max": 0.91,
    "eliya_top3_mean": 0.78,
    "eliya_p90": 0.84,
    "eliya_high_window_fraction": 0.4,
    "n_windows": 3,
    "windows": [
        {"start": 0.0, "end": 5.0, "score": 0.11},
        {"start": 0.5, "end": 5.5, "score": 0.91},
        {"start": 1.0, "end": 6.0, "score": 0.42},
    ],
    "fake_probability": 0.78,
    "bonafide_score": 0.22,
    "verdict": "FAKE",
}


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._body


class GrokContextTests(unittest.TestCase):
    def test_context_contains_only_supplied_measurements_and_safe_file_name(
        self,
    ) -> None:
        context = build_grok_context(ANALYSIS)

        self.assertEqual(context["file"], "example.wav")
        self.assertNotIn("/private/data", json.dumps(context))
        self.assertEqual(context["overall_result"]["synthetic_score"], 0.78)
        self.assertEqual(context["detector_outputs"]["eliya"]["eliya_top3_mean"], 0.78)
        self.assertEqual(
            context["detector_outputs"]["eliya"]["highest_scoring_windows"][0],
            {"start": 0.5, "end": 5.5, "score": 0.91},
        )
        self.assertEqual(
            context["suspicious_characteristics"]["highest_eliya_window_scores"][0],
            {"start": 0.5, "end": 5.5, "score": 0.91},
        )
        self.assertEqual(context["relevant_metadata"], "not supplied")
        self.assertIn("only one detector", context["detector_disagreement"])

    def test_available_detector_metadata_and_disagreement_are_preserved(self) -> None:
        analysis = {
            **ANALYSIS,
            "detector_outputs": {"cpps": {"value": 12.4}},
            "suspicious_characteristics": ["localized high Eliya score"],
            "metadata": {"duration_seconds": 6.0},
            "detector_disagreement": "Eliya was high while CPPS was inconclusive",
        }

        context = build_grok_context(analysis)

        self.assertEqual(context["detector_outputs"]["cpps"]["value"], 12.4)
        self.assertEqual(
            context["suspicious_characteristics"],
            ["localized high Eliya score"],
        )
        self.assertEqual(context["relevant_metadata"]["duration_seconds"], 6.0)
        self.assertEqual(
            context["detector_disagreement"],
            "Eliya was high while CPPS was inconclusive",
        )


class GrokAPITests(unittest.TestCase):
    def test_missing_api_key_fails_with_configuration_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GrokNotConfiguredError, "XAI_API_KEY"):
                ask_grok_about_analysis(ANALYSIS)

    def test_request_uses_xai_responses_api_and_extracts_nested_text(self) -> None:
        observed = {}

        def transport(request, *, timeout):
            observed["url"] = request.full_url
            observed["authorization"] = request.get_header("Authorization")
            observed["body"] = json.loads(request.data.decode("utf-8"))
            observed["timeout"] = timeout
            return FakeResponse(
                {
                    "id": "response-123",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Measured: Eliya was elevated. Interpretation: review the flagged interval.",
                                }
                            ],
                        }
                    ],
                }
            )

        result = ask_grok_about_analysis(
            ANALYSIS,
            "Why suspicious?",
            api_key="test-secret-key",
            model="grok-test",
            timeout_seconds=7.5,
            transport=transport,
        )

        self.assertIsInstance(result, GrokInterpretation)
        self.assertEqual(result.model, "grok-test")
        self.assertEqual(result.response_id, "response-123")
        self.assertIn("Measured:", result.text)
        self.assertEqual(observed["url"], "https://api.x.ai/v1/responses")
        self.assertEqual(observed["authorization"], "Bearer test-secret-key")
        self.assertEqual(observed["timeout"], 7.5)
        self.assertEqual(observed["body"]["model"], "grok-test")
        self.assertIn("Why suspicious?", observed["body"]["input"])
        self.assertIn("eliya_top3_mean", observed["body"]["input"])
        self.assertNotIn("test-secret-key", json.dumps(observed["body"]))

    def test_unavailable_api_becomes_contextual_grok_error(self) -> None:
        def transport(_request, *, timeout):
            raise urllib.error.URLError("offline")

        with self.assertRaisesRegex(GrokAPIError, "currently unavailable"):
            ask_grok_about_analysis(
                ANALYSIS,
                api_key="test-key",
                transport=transport,
            )


class GrokCLIIntegrationTests(unittest.TestCase):
    def test_existing_cli_can_ask_grok_after_detection(self) -> None:
        interpretation = GrokInterpretation(
            text="The elevated score is localized; inspect the strongest window.",
            model="grok-test",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch.object(sys, "argv", ["Deepfake.py", "clip.wav", "--ask-grok"]),
            patch.object(
                deepfake, "collect_audio_files", return_value=[Path("clip.wav")]
            ),
            patch.object(deepfake, "get_detector", return_value=object()),
            patch.object(deepfake, "run_inference", return_value=dict(ANALYSIS)),
            patch(
                "grok_interpretation.ask_grok_about_analysis",
                return_value=interpretation,
            ) as ask_grok,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            deepfake.main()

        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("eliya_top3_mean=0.7800", stdout.getvalue())
        self.assertIn("Grok interpretation", stdout.getvalue())
        self.assertIn("not a detector result", stdout.getvalue())
        ask_grok.assert_called_once()
        self.assertEqual(ask_grok.call_args.args[0]["eliya_top3_mean"], 0.78)

    def test_grok_failure_does_not_erase_detector_result(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch.object(sys, "argv", ["Deepfake.py", "clip.wav", "--ask-grok"]),
            patch.object(
                deepfake, "collect_audio_files", return_value=[Path("clip.wav")]
            ),
            patch.object(deepfake, "get_detector", return_value=object()),
            patch.object(deepfake, "run_inference", return_value=dict(ANALYSIS)),
            patch(
                "grok_interpretation.ask_grok_about_analysis",
                side_effect=GrokNotConfiguredError("XAI_API_KEY is not configured"),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            deepfake.main()

        self.assertIn("eliya_top3_mean=0.7800", stdout.getvalue())
        self.assertIn("GROK UNAVAILABLE", stderr.getvalue())
        self.assertIn("XAI_API_KEY", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
