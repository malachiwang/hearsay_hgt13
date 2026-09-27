"""xAI/Grok interpretation of already-measured HEARSAY analysis results.

This module never analyzes audio and never changes a detector score.  It sends
a bounded, structured summary of results produced by HEARSAY to xAI's
Responses API and asks Grok to explain those measurements in plain language.
"""

from __future__ import annotations

import json
import math
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


XAI_API_URL = "https://api.x.ai/v1/responses"
DEFAULT_GROK_MODEL = "grok-4.7"
DEFAULT_GROK_QUESTION = (
    "Why was this audio classified this way? Which measured signals contributed "
    "most, did the detectors disagree, what should I look for, and how confident "
    "should I be in the result?"
)
MAX_CONTEXT_WINDOWS = 10


class GrokError(RuntimeError):
    """Base class for user-facing Grok integration failures."""


class GrokNotConfiguredError(GrokError):
    """Raised when a Grok interpretation is requested without an API key."""


class GrokAPIError(GrokError):
    """Raised when xAI rejects a request or returns an unusable response."""


@dataclass(frozen=True)
class GrokInterpretation:
    """Natural-language interpretation returned by xAI."""

    text: str
    model: str
    response_id: str | None = None


def _finite_float(value: Any) -> float | None:
    """Return a finite built-in float for numeric values, otherwise ``None``."""

    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert supplied result data to bounded JSON-safe values."""

    if depth > 5:
        return "[omitted: nesting limit]"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "[omitted: nonfinite value]"
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:100]]

    # NumPy scalar types and similar numeric wrappers expose item().
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method(), depth=depth + 1)
        except (TypeError, ValueError):
            pass
    return str(value)


def _eliya_measurements(analysis: Mapping[str, Any]) -> dict[str, Any]:
    numeric_fields = (
        "eliya_mean",
        "eliya_max",
        "eliya_top3_mean",
        "eliya_p90",
        "eliya_high_window_fraction",
    )
    result: dict[str, Any] = {}
    for field in numeric_fields:
        numeric = _finite_float(analysis.get(field))
        if numeric is not None:
            result[field] = numeric

    n_windows = analysis.get("n_windows")
    if isinstance(n_windows, int) and not isinstance(n_windows, bool):
        result["n_windows"] = n_windows

    windows: list[dict[str, float]] = []
    raw_windows = analysis.get("windows")
    if isinstance(raw_windows, Sequence) and not isinstance(
        raw_windows, (str, bytes, bytearray)
    ):
        for window in raw_windows:
            if not isinstance(window, Mapping):
                continue
            start = _finite_float(window.get("start"))
            end = _finite_float(window.get("end"))
            score = _finite_float(window.get("score"))
            if start is not None and end is not None and score is not None:
                windows.append({"start": start, "end": end, "score": score})

    if windows:
        strongest = sorted(
            windows,
            key=lambda window: (-window["score"], window["start"]),
        )[:MAX_CONTEXT_WINDOWS]
        result["highest_scoring_windows"] = strongest
        result["window_count_supplied"] = len(windows)
        result["window_score_direction"] = "0 = bona fide, 1 = synthetic"
    return result


def build_grok_context(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Build the explicit measured-results context sent to Grok.

    Only fields present in ``analysis`` are included. Missing detector evidence,
    metadata, and cross-detector comparisons are called out instead of guessed.
    """

    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be a mapping of HEARSAY results")

    source = analysis.get("file") or analysis.get("filename") or "unknown"
    source_name = Path(str(source)).name
    synthetic_score = _finite_float(
        analysis.get("eliya_top3_mean", analysis.get("fake_probability"))
    )

    overall: dict[str, Any] = {
        "score_direction": "0 = bona fide/real, 1 = synthetic/fake",
        "calibration_warning": (
            "The score is detector output, not a guaranteed calibrated probability."
        ),
    }
    if synthetic_score is not None:
        overall["synthetic_score"] = synthetic_score
    verdict = analysis.get("verdict")
    if isinstance(verdict, str) and verdict:
        overall["diagnostic_prediction"] = verdict
        overall["prediction_warning"] = (
            "This label comes from HEARSAY's diagnostic threshold; Grok did not "
            "produce it."
        )

    detector_outputs: dict[str, Any] = {}
    eliya = _eliya_measurements(analysis)
    if eliya:
        detector_outputs["eliya"] = eliya
    for field in ("detector_outputs", "detectors"):
        supplied = analysis.get(field)
        if isinstance(supplied, Mapping):
            for name, measurements in supplied.items():
                detector_outputs[str(name)] = _json_safe(measurements)

    suspicious = analysis.get("suspicious_characteristics")
    if suspicious is None and "highest_scoring_windows" in eliya:
        suspicious = {
            "highest_eliya_window_scores": eliya["highest_scoring_windows"],
            "note": (
                "These are measured Eliya synthetic-score locations, not an "
                "independent acoustic diagnosis."
            ),
        }
    metadata = analysis.get("metadata")
    disagreement = analysis.get("detector_disagreement")
    if disagreement is None:
        disagreement = (
            "not assessable: only one detector result was supplied"
            if len(detector_outputs) <= 1
            else "not supplied: do not infer disagreement from undocumented fields"
        )

    return {
        "context_type": "HEARSAY measured analysis results",
        "file": source_name,
        "overall_result": overall,
        "detector_outputs": detector_outputs,
        "suspicious_characteristics": (
            _json_safe(suspicious) if suspicious is not None else "not supplied"
        ),
        "relevant_metadata": (
            _json_safe(metadata) if metadata is not None else "not supplied"
        ),
        "detector_disagreement": _json_safe(disagreement),
    }


def _prompt(context: Mapping[str, Any], question: str) -> str:
    return (
        "You are the explanation layer for HEARSAY, an audio-authentication tool.\n"
        "The JSON below contains measured HEARSAY results. You did not inspect the "
        "audio. Interpret only the supplied values. Never invent a detector, audio "
        "characteristic, metadata field, causal claim, or confidence estimate. "
        "Treat absent evidence as unavailable. Clearly separate measured findings "
        "from your interpretation, mention detector disagreement only when supplied, "
        "and explain that a detector score is not certainty. Be concise and practical.\n\n"
        f"HEARSAY_RESULTS_JSON:\n{json.dumps(context, sort_keys=True)}\n\n"
        f"USER_QUESTION:\n{question.strip()}"
    )


def _extract_response_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    pieces: list[str] = []
    output = payload.get("output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, Sequence) or isinstance(
                content, (str, bytes, bytearray)
            ):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    pieces.append(text.strip())
    if pieces:
        return "\n".join(pieces)
    raise GrokAPIError("xAI returned no natural-language interpretation")


def _api_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(2048).decode("utf-8", errors="replace")
        payload = json.loads(body)
        error = payload.get("error") if isinstance(payload, Mapping) else None
        if isinstance(error, Mapping) and isinstance(error.get("message"), str):
            return error["message"][:500]
        if isinstance(error, str):
            return error[:500]
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    return exc.reason or "unknown xAI API error"


def ask_grok_about_analysis(
    analysis: Mapping[str, Any],
    question: str = DEFAULT_GROK_QUESTION,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 30.0,
    transport: Callable[..., Any] | None = None,
) -> GrokInterpretation:
    """Ask Grok to interpret a completed HEARSAY result.

    ``analysis`` is never changed, and the returned prose is not fed back into
    the detector or its score.
    """

    key = (api_key if api_key is not None else os.getenv("XAI_API_KEY", "")).strip()
    if not key:
        raise GrokNotConfiguredError(
            "Grok interpretation is unavailable because XAI_API_KEY is not configured"
        )
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Grok question must be a nonempty string")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")

    selected_model = (model or os.getenv("XAI_MODEL") or DEFAULT_GROK_MODEL).strip()
    if not selected_model:
        raise ValueError("Grok model name must not be empty")

    context = build_grok_context(analysis)
    body = json.dumps(
        {"model": selected_model, "input": _prompt(context, question)},
        allow_nan=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        XAI_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    open_request = transport or urllib.request.urlopen
    try:
        with open_request(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _api_error_detail(exc)
        raise GrokAPIError(
            f"xAI request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        raise GrokAPIError(f"xAI is currently unavailable: {reason}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GrokAPIError(f"could not read the xAI response: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise GrokAPIError("xAI returned an unexpected response format")
    text = _extract_response_text(payload)
    response_id = payload.get("id")
    return GrokInterpretation(
        text=text,
        model=selected_model,
        response_id=response_id if isinstance(response_id, str) else None,
    )
