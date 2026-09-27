from __future__ import annotations

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

from run_hearsay import HearsayRunnerError, main, run_batch


class FakeDetector:
    def __init__(
        self,
        scores: dict[str, float],
        *,
        failure: str | None = None,
    ) -> None:
        self.scores = scores
        self.failure = failure
        self.calls: list[str] = []

    def analyze_file(self, path: Path) -> dict[str, float]:
        self.calls.append(path.name)
        if path.name == self.failure:
            raise RuntimeError("mock inference failure")
        return {"eliya_top3_mean": self.scores[path.name]}


class HearsayRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_directory = self.root / "input"
        self.input_directory.mkdir()
        self.output_path = self.root / "output" / "predictions.tsv"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def touch(self, relative_path: str) -> Path:
        path = self.input_directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return path

    def test_exact_header_filenames_scores_and_deterministic_order(self) -> None:
        self.touch("zeta.MP3")
        self.touch("nested/beta.wav")
        self.touch("Alpha.flac")
        detector = FakeDetector(
            {"zeta.MP3": 0.8234, "beta.wav": 0.1021, "Alpha.flac": 0.5}
        )

        rows = run_batch(self.input_directory, self.output_path, detector=detector)

        self.assertEqual(
            rows,
            [("Alpha.flac", 0.5), ("beta.wav", 0.1021), ("zeta.MP3", 0.8234)],
        )
        self.assertEqual(detector.calls, ["Alpha.flac", "beta.wav", "zeta.MP3"])
        self.assertEqual(
            self.output_path.read_text(encoding="utf-8"),
            "filename\tcm-score\n"
            "Alpha.flac\t0.5\n"
            "beta.wav\t0.1021\n"
            "zeta.MP3\t0.8234\n",
        )

    def test_all_required_extensions_produce_one_row_each(self) -> None:
        filenames = [
            "a.wav",
            "b.mp3",
            "c.m4a",
            "d.mp4",
            "e.ogg",
            "f.opus",
            "g.flac",
        ]
        for filename in filenames:
            self.touch(filename)
        self.touch("ignored.txt")
        detector = FakeDetector({filename: 0.25 for filename in filenames})

        rows = run_batch(self.input_directory, self.output_path, detector=detector)

        self.assertEqual(len(rows), len(filenames))
        self.assertEqual(len(self.output_path.read_text().splitlines()), len(filenames) + 1)
        self.assertEqual({filename for filename, _ in rows}, set(filenames))

    def test_invalid_scores_are_rejected_without_writing_output(self) -> None:
        self.touch("bad.wav")
        for score in (math.nan, math.inf, -0.01, 1.01):
            with self.subTest(score=score):
                detector = FakeDetector({"bad.wav": score})
                with self.assertRaisesRegex(HearsayRunnerError, "invalid cm-score"):
                    run_batch(
                        self.input_directory,
                        self.output_path,
                        detector=detector,
                    )
                self.assertFalse(self.output_path.exists())

    def test_file_failure_is_non_silent_and_output_is_atomic(self) -> None:
        self.touch("a.wav")
        self.touch("b.wav")
        detector = FakeDetector(
            {"a.wav": 0.1, "b.wav": 0.9},
            failure="b.wav",
        )

        with self.assertRaisesRegex(HearsayRunnerError, "failed to score.*b.wav"):
            run_batch(self.input_directory, self.output_path, detector=detector)

        self.assertFalse(self.output_path.exists())
        self.assertEqual(detector.calls, ["a.wav", "b.wav"])

    def test_empty_directory_fails_clearly(self) -> None:
        with self.assertRaisesRegex(HearsayRunnerError, "no supported audio"):
            run_batch(
                self.input_directory,
                self.output_path,
                detector=FakeDetector({}),
            )

    def test_cli_failure_returns_nonzero_status(self) -> None:
        standard_error = io.StringIO()
        with contextlib.redirect_stderr(standard_error):
            status = main([str(self.input_directory), str(self.output_path)])

        self.assertEqual(status, 1)
        self.assertIn("ERROR:", standard_error.getvalue())
        self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
