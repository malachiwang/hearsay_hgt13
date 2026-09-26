from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fusion.build_features import FusionDataError, build_feature_table


class FusionFeatureTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = pd.DataFrame(
            {
                "file_id": ["a.wav", "b.wav", "c.wav", "d.wav"],
                "label": [0, 1, "real", "synthetic"],
                "fold": [2, 2, 5, 5],
            }
        )

    def test_strict_merge_aligns_by_file_id_and_preserves_manifest_order(self) -> None:
        cpps = pd.DataFrame(
            {"file_id": ["d.wav", "b.wav", "a.wav", "c.wav"], "cpps": [4, 2, 1, 3]}
        )
        splice = pd.DataFrame(
            {
                "file_id": ["c.wav", "a.wav", "d.wav", "b.wav"],
                "splice_max_score": [30.0, 10.0, 40.0, 20.0],
                "splice_n_candidates": [3, 1, 4, 2],
            }
        )

        result = build_feature_table(
            self.manifest, {"cpps": cpps, "splice": splice}
        )

        self.assertEqual(
            result.columns.tolist(),
            [
                "file_id",
                "label",
                "fold",
                "cpps",
                "splice_max_score",
                "splice_n_candidates",
            ],
        )
        self.assertEqual(result["file_id"].tolist(), self.manifest["file_id"].tolist())
        self.assertEqual(result["label"].tolist(), [0, 1, 0, 1])
        self.assertEqual(result["cpps"].tolist(), [1, 2, 3, 4])
        self.assertEqual(result["splice_max_score"].tolist(), [10, 20, 30, 40])

    def test_missing_detector_file_is_rejected(self) -> None:
        source = pd.DataFrame(
            {"file_id": ["a.wav", "b.wav", "c.wav"], "score": [0.1, 0.2, 0.3]}
        )
        with self.assertRaisesRegex(FusionDataError, "missing 1 manifest file.*d.wav"):
            build_feature_table(self.manifest, {"RFP": source})

    def test_extra_detector_file_is_rejected(self) -> None:
        source = pd.DataFrame(
            {
                "file_id": ["a.wav", "b.wav", "c.wav", "d.wav", "extra.wav"],
                "score": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        with self.assertRaisesRegex(FusionDataError, "1 unexpected row.*extra.wav"):
            build_feature_table(self.manifest, {"WavLM": source})

    def test_duplicate_detector_file_is_rejected(self) -> None:
        source = pd.DataFrame(
            {
                "file_id": ["a.wav", "a.wav", "b.wav", "c.wav", "d.wav"],
                "score": [0.1, 0.1, 0.2, 0.3, 0.4],
            }
        )
        with self.assertRaisesRegex(FusionDataError, "duplicate file_id.*a.wav"):
            build_feature_table(self.manifest, {"RFP": source})

    def test_nonfinite_features_are_rejected(self) -> None:
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(value=invalid):
                source = pd.DataFrame(
                    {
                        "file_id": self.manifest["file_id"],
                        "score": [0.1, invalid, 0.3, 0.4],
                    }
                )
                with self.assertRaisesRegex(FusionDataError, "NaN or Inf"):
                    build_feature_table(self.manifest, {"detector": source})

    def test_nonnumeric_feature_is_rejected(self) -> None:
        source = pd.DataFrame(
            {
                "file_id": self.manifest["file_id"],
                "score": ["0.1", "0.2", "0.3", "0.4"],
            }
        )
        with self.assertRaisesRegex(FusionDataError, "numeric"):
            build_feature_table(self.manifest, {"detector": source})

    def test_duplicate_feature_names_across_sources_are_rejected(self) -> None:
        first = pd.DataFrame(
            {"file_id": self.manifest["file_id"], "score": [1, 2, 3, 4]}
        )
        second = pd.DataFrame(
            {"file_id": self.manifest["file_id"], "score": [4, 3, 2, 1]}
        )
        with self.assertRaisesRegex(FusionDataError, "duplicates existing feature"):
            build_feature_table(self.manifest, {"first": first, "second": second})

    def test_duplicate_source_column_names_are_rejected(self) -> None:
        source = pd.DataFrame(
            [["a.wav", 1.0, 2.0], ["b.wav", 2.0, 3.0], ["c.wav", 3.0, 4.0], ["d.wav", 4.0, 5.0]],
            columns=["file_id", "score", "score"],
        )
        with self.assertRaisesRegex(FusionDataError, "duplicate column"):
            build_feature_table(self.manifest, {"detector": source})

    def test_manifest_validation_rejects_invalid_rows(self) -> None:
        cases = {
            "duplicate manifest file": pd.DataFrame(
                {"file_id": ["a", "a"], "label": [0, 1], "fold": [0, 1]}
            ),
            "missing fold": pd.DataFrame(
                {"file_id": ["a", "b"], "label": [0, 1], "fold": [0, np.nan]}
            ),
            "unsupported string label": pd.DataFrame(
                {"file_id": ["a", "b"], "label": ["human", "spoof"], "fold": [0, 1]}
            ),
            "invalid numeric label": pd.DataFrame(
                {"file_id": ["a", "b"], "label": [2, 1], "fold": [0, 1]}
            ),
        }
        for name, manifest in cases.items():
            with self.subTest(name=name):
                source = pd.DataFrame(
                    {"file_id": manifest["file_id"], "score": np.arange(len(manifest))}
                )
                with self.assertRaises(FusionDataError):
                    build_feature_table(manifest, {"detector": source})

    def test_detector_cannot_override_manifest_label_or_fold(self) -> None:
        for reserved in ("label", "fold"):
            with self.subTest(column=reserved):
                source = pd.DataFrame(
                    {
                        "file_id": self.manifest["file_id"],
                        reserved: [0, 1, 0, 1],
                        "score": [0.1, 0.9, 0.2, 0.8],
                    }
                )
                with self.assertRaisesRegex(FusionDataError, "manifest-owned"):
                    build_feature_table(self.manifest, {"detector": source})

    def test_csv_and_tsv_paths_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.csv"
            feature_path = root / "features.tsv"
            self.manifest.to_csv(manifest_path, index=False)
            pd.DataFrame(
                {"file_id": self.manifest["file_id"], "score": [1, 2, 3, 4]}
            ).to_csv(feature_path, sep="\t", index=False)

            result = build_feature_table(manifest_path, {"detector": feature_path})

        self.assertEqual(result["score"].tolist(), [1, 2, 3, 4])

    def test_duplicate_columns_in_file_source_are_rejected_before_pandas_renames_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "features.csv"
            source_path.write_text(
                "file_id,score,score\n"
                "a.wav,1,2\n"
                "b.wav,2,3\n"
                "c.wav,3,4\n"
                "d.wav,4,5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FusionDataError, "duplicate column"):
                build_feature_table(self.manifest, {"detector": source_path})


if __name__ == "__main__":
    unittest.main()
