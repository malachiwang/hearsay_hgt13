"""Decode HEARSAY audio into a strict, detector-neutral waveform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import soundfile as sf


EXPECTED_SAMPLE_RATE = 16_000
_FFMPEG_EXECUTABLE = shutil.which("ffmpeg")
_ASHOWINFO_RATE = re.compile(r"\brate:(\d+)\b")
_ASHOWINFO_CHANNELS = re.compile(r"\bchannels:(\d+)\b")


class AudioLoadError(Exception):
    """Base exception for audio-ingestion failures."""


class AudioDecodeError(AudioLoadError):
    """Raised when no configured decoder can decode an input file."""


class AudioValidationError(AudioLoadError):
    """Raised when decoded audio violates the competition contract."""


@dataclass(frozen=True)
class AudioRecord:
    """A validated waveform that can be shared by downstream detectors."""

    source_path: Path
    waveform: np.ndarray
    sample_rate: int
    duration_seconds: float
    source_channel_count: int
    decode_backend: str
    warnings: tuple[str, ...] = ()


def _decode_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    """Decode at the source sample rate without changing channel count."""

    return sf.read(path, dtype="float32", always_2d=True)


def _parse_ashowinfo(stderr: bytes, path: Path) -> tuple[int, int]:
    """Read native sample rate and channel count reported by FFmpeg."""

    text = stderr.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if "ashowinfo" not in line:
            continue
        rate_match = _ASHOWINFO_RATE.search(line)
        channels_match = _ASHOWINFO_CHANNELS.search(line)
        if rate_match and channels_match:
            return int(rate_match.group(1)), int(channels_match.group(1))

    raise AudioDecodeError(
        f"FFmpeg decoded {path} but did not report its sample rate and channels"
    )


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode directly to interleaved float32 PCM on stdout.

    No output file, resampling option, or channel-conversion option is used.
    ``ashowinfo`` observes the decoded frames so the native rate and channel
    count can be validated without a separate metadata probe.
    """

    if _FFMPEG_EXECUTABLE is None:
        raise AudioDecodeError(
            f"SoundFile could not decode {path}, and FFmpeg is not available on PATH"
        )

    command = [
        _FFMPEG_EXECUTABLE,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        "ashowinfo",
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioDecodeError(f"FFmpeg failed while decoding {path}: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        if len(detail) > 1_000:
            detail = detail[-1_000:]
        raise AudioDecodeError(
            f"FFmpeg could not decode {path}"
            + (f": {detail}" if detail else "")
        )

    if not result.stdout:
        raise AudioDecodeError(f"FFmpeg decoded no audio samples from {path}")
    if len(result.stdout) % np.dtype("<f4").itemsize != 0:
        raise AudioDecodeError(f"FFmpeg returned malformed float32 PCM for {path}")

    sample_rate, channel_count = _parse_ashowinfo(result.stderr, path)
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if samples.size % channel_count != 0:
        raise AudioDecodeError(
            f"FFmpeg returned a partial {channel_count}-channel frame for {path}"
        )

    return samples.reshape(-1, channel_count), sample_rate


def _validate_and_build_record(
    path: Path,
    decoded: np.ndarray,
    sample_rate: int,
    decode_backend: str,
    warnings: tuple[str, ...],
) -> AudioRecord:
    if decoded.ndim != 2:
        raise AudioValidationError(
            f"Decoded audio for {path} has shape {decoded.shape}; expected frames x channels"
        )

    source_channel_count = decoded.shape[1]
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise AudioValidationError(
            f"Unexpected sample rate for {path}: {sample_rate} Hz; "
            f"expected {EXPECTED_SAMPLE_RATE} Hz (resampling is disabled)"
        )
    if source_channel_count != 1:
        raise AudioValidationError(
            f"Unexpected channel count for {path}: {source_channel_count}; "
            "expected mono audio (downmixing is disabled)"
        )
    if decoded.shape[0] == 0:
        raise AudioValidationError(f"Decoded audio is empty: {path}")

    # This is a numeric representation conversion only. It does not normalize,
    # clip, resample, denoise, pad, crop, or otherwise alter the waveform.
    waveform = np.array(decoded[:, 0], dtype=np.float32, order="C", copy=True)

    if waveform.dtype != np.float32:
        raise AudioValidationError(f"Waveform for {path} is not float32")
    if waveform.ndim != 1:
        raise AudioValidationError(f"Waveform for {path} is not one-dimensional")
    if not waveform.flags.c_contiguous:
        raise AudioValidationError(f"Waveform for {path} is not C-contiguous")
    if not np.isfinite(waveform).all():
        raise AudioValidationError(f"Waveform for {path} contains NaN or Inf samples")

    return AudioRecord(
        source_path=path,
        waveform=waveform,
        sample_rate=sample_rate,
        duration_seconds=waveform.size / sample_rate,
        source_channel_count=source_channel_count,
        decode_backend=decode_backend,
        warnings=warnings,
    )


def load_audio(path: str | Path) -> AudioRecord:
    """Decode and validate one 16-kHz mono audio file.

    SoundFile is deterministic first choice. If libsndfile cannot decode the
    source container, FFmpeg is used only as a decoder and streams uncompressed
    float32 PCM back to this process. No intermediate audio file is created.
    """

    try:
        source_path = Path(path).expanduser()
    except (TypeError, ValueError) as exc:
        raise AudioValidationError(f"Invalid audio path {path!r}") from exc

    if not source_path.exists():
        raise AudioValidationError(f"Audio path does not exist: {source_path}")
    if not source_path.is_file():
        raise AudioValidationError(f"Audio path is not a regular file: {source_path}")

    try:
        decoded, sample_rate = _decode_with_soundfile(source_path)
        backend = "soundfile"
        warnings: tuple[str, ...] = ()
    except (OSError, RuntimeError, ValueError, sf.SoundFileError) as soundfile_error:
        try:
            decoded, sample_rate = _decode_with_ffmpeg(source_path)
        except AudioDecodeError as ffmpeg_error:
            raise AudioDecodeError(
                f"Unable to decode {source_path} with SoundFile or FFmpeg. "
                f"SoundFile error: {soundfile_error}. FFmpeg error: {ffmpeg_error}"
            ) from ffmpeg_error
        backend = "ffmpeg"
        warnings = ("SoundFile decode failed; used the FFmpeg PCM fallback.",)

    return _validate_and_build_record(
        source_path,
        decoded,
        int(sample_rate),
        backend,
        warnings,
    )
