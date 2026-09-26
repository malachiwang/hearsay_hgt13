from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np

import ProcessFiles


class AudioIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _write_wav(
        self,
        path: Path,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        frame_count: int = 320,
    ) -> None:
        samples = (
            np.sin(2 * np.pi * 440 * np.arange(frame_count) / sample_rate)
            * 12_000
        ).astype("<i2")
        if channels > 1:
            samples = np.repeat(samples[:, None], channels, axis=1).reshape(-1)

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

    def test_loads_16khz_mono_wav_with_canonical_contract(self) -> None:
        path = self.directory / "sample.wav"
        self._write_wav(path, frame_count=800)

        record = ProcessFiles.load_audio(path)

        self.assertEqual(record.source_path, path)
        self.assertEqual(record.sample_rate, 16_000)
        self.assertEqual(record.source_channel_count, 1)
        self.assertEqual(record.decode_backend, "soundfile")
        self.assertEqual(record.waveform.dtype, np.float32)
        self.assertEqual(record.waveform.shape, (800,))
        self.assertTrue(record.waveform.flags.c_contiguous)
        self.assertTrue(np.isfinite(record.waveform).all())
        self.assertAlmostEqual(record.duration_seconds, 800 / 16_000)
        self.assertAlmostEqual(
            float(np.max(np.abs(record.waveform))), 12_000 / 32_768, places=4
        )

    def test_loader_accepts_str_and_path_with_complex_case_insensitive_name(self) -> None:
        path = self.directory / "speaker.session.001.WaV"
        self._write_wav(path)
        original_files = set(self.directory.iterdir())

        for value in (path, str(path)):
            with self.subTest(input_type=type(value).__name__):
                record = ProcessFiles.load_audio(value)
                self.assertEqual(record.waveform.shape, (320,))

        self.assertEqual(set(self.directory.iterdir()), original_files)

    def test_rejects_unexpected_sample_rate_without_resampling(self) -> None:
        path = self.directory / "eight_khz.wav"
        self._write_wav(path, sample_rate=8_000)

        with self.assertRaisesRegex(
            ProcessFiles.AudioValidationError, "Unexpected sample rate.*8000 Hz"
        ):
            ProcessFiles.load_audio(path)

    def test_rejects_stereo_without_downmixing(self) -> None:
        path = self.directory / "stereo.wav"
        self._write_wav(path, channels=2)

        with self.assertRaisesRegex(
            ProcessFiles.AudioValidationError, "Unexpected channel count.*2"
        ):
            ProcessFiles.load_audio(path)

    def test_rejects_empty_audio(self) -> None:
        path = self.directory / "empty.wav"
        self._write_wav(path, frame_count=0)

        with self.assertRaisesRegex(ProcessFiles.AudioValidationError, "empty"):
            ProcessFiles.load_audio(path)

    def test_invalid_audio_fails_with_context(self) -> None:
        path = self.directory / "invalid.wav"
        path.write_bytes(b"not an audio file")

        with mock.patch.object(ProcessFiles, "_FFMPEG_EXECUTABLE", None):
            with self.assertRaisesRegex(
                ProcessFiles.AudioDecodeError, "Unable to decode.*invalid.wav"
            ):
                ProcessFiles.load_audio(path)

    def test_missing_path_fails_clearly(self) -> None:
        path = self.directory / "missing.wav"

        with self.assertRaisesRegex(
            ProcessFiles.AudioValidationError, "does not exist.*missing.wav"
        ):
            ProcessFiles.load_audio(path)

    def test_directory_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ProcessFiles.AudioValidationError, "not a regular file"
        ):
            ProcessFiles.load_audio(self.directory)

    def test_non_finite_samples_are_rejected(self) -> None:
        path = self.directory / "nonfinite.wav"
        path.write_bytes(b"placeholder")
        decoded = np.array([[0.0], [np.nan]], dtype=np.float32)

        with mock.patch.object(ProcessFiles.sf, "read", return_value=(decoded, 16_000)):
            with self.assertRaisesRegex(
                ProcessFiles.AudioValidationError, "NaN or Inf"
            ):
                ProcessFiles.load_audio(path)

    def test_ffmpeg_fallback_streams_raw_pcm_without_creating_a_file(self) -> None:
        path = self.directory / "clip.with.periods.M4A"
        path.write_bytes(b"placeholder")
        expected = np.array([0.25, -0.25, 0.0, 0.5], dtype="<f4")
        stderr = (
            b"[Parsed_ashowinfo_0 @ 0x0] n:0 pts:0 fmt:flt channels:1 "
            b"chlayout:mono rate:16000 nb_samples:4\n"
        )
        original_files = set(self.directory.iterdir())

        def fake_run(command, **kwargs):
            self.assertEqual(command[0], "/test/bin/ffmpeg")
            self.assertIn("pcm_f32le", command)
            self.assertIn("f32le", command)
            self.assertEqual(command[-1], "pipe:1")
            self.assertNotIn("-ar", command)
            self.assertNotIn("-ac", command)
            self.assertEqual(kwargs["stdout"], subprocess.PIPE)
            self.assertEqual(kwargs["stderr"], subprocess.PIPE)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=expected.tobytes(),
                stderr=stderr,
            )

        with (
            mock.patch.object(
                ProcessFiles.sf, "read", side_effect=RuntimeError("unsupported")
            ),
            mock.patch.object(
                ProcessFiles, "_FFMPEG_EXECUTABLE", "/test/bin/ffmpeg"
            ),
            mock.patch.object(ProcessFiles.subprocess, "run", side_effect=fake_run),
        ):
            record = ProcessFiles.load_audio(path)

        self.assertEqual(record.decode_backend, "ffmpeg")
        self.assertEqual(record.sample_rate, 16_000)
        self.assertEqual(record.source_channel_count, 1)
        self.assertEqual(record.waveform.dtype, np.float32)
        np.testing.assert_array_equal(record.waveform, expected)
        self.assertEqual(set(self.directory.iterdir()), original_files)


if __name__ == "__main__":
    unittest.main()
