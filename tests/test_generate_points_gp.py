import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import generate_points as POINTS


class GaussianAdaptivePointTests(unittest.TestCase):
    def test_gp_ucb_prefers_high_error_or_uncertain_regions(self) -> None:
        regions = [
            POINTS.ErrorRegion(
                source_index=str(index),
                unit_point=[x, y],
                score=0.05 + 2.0 * x * x,
                worst_sparam="S21",
                worst_sparam_score=0.05 + 2.0 * x * x,
                row_count=1,
            )
            for index, (x, y) in enumerate(
                ((0.05, 0.1), (0.2, 0.8), (0.45, 0.35), (0.75, 0.7), (0.95, 0.2)),
                start=1,
            )
        ]
        candidates = POINTS.maximin_lhs_points(
            count=80,
            dimensions=2,
            rng=__import__("random").Random(19),
            candidates=6,
        )
        suggestions, model = POINTS.select_gp_ucb_points(
            candidates,
            regions,
            [region.unit_point for region in regions],
            count=4,
            exploration_weight=2.0,
            novelty_power=1.0,
            min_distance=0.02,
            length_scale=None,
            noise_variance=1e-6,
            error_floor=1e-12,
        )
        self.assertEqual(len(suggestions), 4)
        self.assertGreater(model.length_scale, 0.0)
        self.assertTrue(all(item.predicted_error is not None for item in suggestions))
        self.assertTrue(
            all(item.gp_log_uncertainty is not None for item in suggestions)
        )
        self.assertTrue(
            all(item.gp_upper_confidence_error is not None for item in suggestions)
        )
        self.assertEqual(len({tuple(item.unit_point) for item in suggestions}), 4)

    def test_gp_cli_uses_maximin_lhs_and_writes_auditable_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics = root / "verification_metrics.csv"
            output = root / "additional.csv"
            parameter_names = [f"p{index}" for index in range(1, 7)]
            with metrics.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["source_index", "sparam", "evm_pct", *parameter_names],
                )
                writer.writeheader()
                for index in range(12):
                    coordinate = (index + 0.5) / 12.0
                    writer.writerow(
                        {
                            "source_index": index + 1,
                            "sparam": "S21",
                            "evm_pct": 0.1 + 4.0 * coordinate * coordinate,
                            **{
                                name: (coordinate + dim * 0.137) % 1.0
                                for dim, name in enumerate(parameter_names)
                            },
                        }
                    )

            arguments = ["suggest-additional"]
            for name in parameter_names:
                arguments.extend(["--parameter", f"{name}=0:1"])
            arguments.extend(
                [
                    "--count",
                    "4",
                    "--verification-metrics",
                    str(metrics),
                    "--acquisition",
                    "gp-ucb",
                    "--candidate-count",
                    "96",
                    "--lhs-candidates",
                    "6",
                    "--include-normalized",
                    "--out",
                    str(output),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(POINTS.main(arguments), 0)

            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["acquisition_method"] == "gp-ucb" for row in rows))
            self.assertTrue(
                all(row["method"] == "gp-ucb-maximin-lhs" for row in rows)
            )
            self.assertTrue(all(float(row["predicted_error"]) > 0.0 for row in rows))
            self.assertTrue(
                all(float(row["gp_upper_confidence_error"]) > 0.0 for row in rows)
            )
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["candidate_method"], "maximin-lhs")
            self.assertEqual(metadata["acquisition_method"], "gp-ucb")
            self.assertEqual(metadata["gp"]["kernel"], "matern52_isotropic")
            self.assertEqual(metadata["gp"]["observation_count"], 12)

    def test_initial_generation_still_defaults_to_maximin_lhs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "initial.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "a=0:1",
                            "--parameter",
                            "b=0:1",
                            "--count",
                            "12",
                            "--lhs-candidates",
                            "4",
                            "--out",
                            str(output),
                        ]
                    ),
                    0,
                )
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["method"], "maximin-lhs")

    def test_minimax_lhs_alias_is_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "alias.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "a=0:1",
                            "--parameter",
                            "b=0:1",
                            "--count",
                            "8",
                            "--method",
                            "minimax-lhs",
                            "--lhs-candidates",
                            "3",
                            "--out",
                            str(output),
                        ]
                    ),
                    0,
                )
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["method"], "maximin-lhs")

    def test_error_distance_acquisition_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics = root / "verification_metrics.csv"
            output = root / "legacy_additional.csv"
            with metrics.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["source_index", "sparam", "evm_pct", "a", "b"],
                )
                writer.writeheader()
                for index, (a, b, error) in enumerate(
                    ((0.1, 0.2, 0.5), (0.5, 0.8, 1.0), (0.9, 0.3, 2.0)),
                    start=1,
                ):
                    writer.writerow(
                        {
                            "source_index": index,
                            "sparam": "S21",
                            "evm_pct": error,
                            "a": a,
                            "b": b,
                        }
                    )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "suggest-additional",
                            "--parameter",
                            "a=0:1",
                            "--parameter",
                            "b=0:1",
                            "--count",
                            "2",
                            "--verification-metrics",
                            str(metrics),
                            "--candidate-count",
                            "24",
                            "--lhs-candidates",
                            "3",
                            "--out",
                            str(output),
                        ]
                    ),
                    0,
                )
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertTrue(
                all(row["method"] == "targeted-maximin-lhs" for row in rows)
            )
            self.assertTrue(
                all(row["acquisition_method"] == "error-distance" for row in rows)
            )
            self.assertTrue(all(row["predicted_error"] == "" for row in rows))


if __name__ == "__main__":
    unittest.main()
