import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import generate_points as POINTS


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        raise OSError("in-memory terminal has no file descriptor")


class GaussianAdaptivePointTests(unittest.TestCase):
    def assert_geometry_splits_are_disjoint(
        self,
        combined_path: Path,
        parameter_names: list[str],
    ) -> None:
        training_path = POINTS.split_output_path(combined_path, "train")
        verification_path = POINTS.split_output_path(combined_path, "verification")
        with combined_path.open(newline="") as stream:
            combined_rows = list(csv.DictReader(stream))
        with training_path.open(newline="") as stream:
            training_rows = list(csv.DictReader(stream))
        with verification_path.open(newline="") as stream:
            verification_rows = list(csv.DictReader(stream))

        self.assertTrue(all(row["dataset"] == "train" for row in training_rows))
        self.assertTrue(
            all(row["dataset"] == "verification" for row in verification_rows)
        )
        training_keys = {
            tuple(row[name] for name in parameter_names) for row in training_rows
        }
        verification_keys = {
            tuple(row[name] for name in parameter_names) for row in verification_rows
        }
        combined_keys = {
            tuple(row[name] for name in parameter_names) for row in combined_rows
        }
        self.assertFalse(training_keys & verification_keys)
        self.assertEqual(len(training_keys), len(training_rows))
        self.assertEqual(len(verification_keys), len(verification_rows))
        self.assertEqual(combined_keys, training_keys | verification_keys)
        self.assertEqual(len(combined_keys), len(combined_rows))

    def test_hybrid_is_the_default_acquisition(self) -> None:
        parser = POINTS.build_suggest_parser()
        args = parser.parse_args(["--count", "2"])
        self.assertEqual(args.acquisition, "hybrid")
        self.assertEqual(args.verification_policy, "auto")
        self.assertEqual(args.target_dataset, "train")

    def test_rational_hybrid_builds_response_gp_from_training_blocks_only(self) -> None:
        sample = Path(__file__).resolve().parents[1] / (
            "neuro_tf_sample_training_verification.mdif"
        )
        parameters = [
            POINTS.parse_parameter_spec("W=0.40mm:0.55mm"),
            POINTS.parse_parameter_spec("L=1.20mm:1.45mm"),
        ]
        surrogate = POINTS.build_rational_response_surrogate(
            [str(sample)],
            parameters,
            split_var="dataset",
            bare_values="auto",
            order=2,
            pole_placement="adaptive",
            pole_damping=0.18,
            pole_iterations=3,
            ridge=1e-8,
            variance_fraction=0.99,
            max_components=3,
            frequency_weight_spec=None,
            noise_variance=1e-6,
            ard_mode="off",
        )
        self.assertEqual(
            surrogate.diagnostics["distinct_training_geometries"],
            4,
        )
        self.assertTrue(surrogate.diagnostics["verification_responses_excluded"])
        self.assertGreaterEqual(surrogate.diagnostics["retained_components"], 1)
        uncertainty, change = POINTS.rational_response_scores(
            surrogate,
            [[0.0, 0.0], [0.5, 0.5]],
        )
        self.assertTrue(np.all(np.isfinite(uncertainty)))
        self.assertTrue(np.all(np.isfinite(change)))
        self.assertLess(uncertainty[0], uncertainty[1])

        regions = [
            POINTS.ErrorRegion("v1", [0.2, 0.2], 1.0, "S21", 1.0, 1),
            POINTS.ErrorRegion("v2", [0.8, 0.8], 2.0, "S21", 2.0, 1),
        ]
        selected, _error_gp, diagnostics = POINTS.select_rational_hybrid_points(
            [[0.1, 0.5], [0.5, 0.5], [0.9, 0.5], [0.5, 0.1], [0.5, 0.9]],
            regions,
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            surrogate,
            count=3,
            allocation=POINTS.HybridAllocation(1, 1, 1, "test"),
            exploration_weight=2.0,
            novelty_power=1.0,
            min_distance=0.0,
            length_scale=0.4,
            noise_variance=1e-6,
            error_floor=1e-12,
            ard_mode="off",
        )
        self.assertEqual(len(selected), 3)
        self.assertIn(
            "rational-uncertainty",
            {point.selection_component for point in selected},
        )
        self.assertEqual(
            diagnostics["response_surrogate"]["training_response_blocks"],
            4,
        )

    def test_rational_hybrid_cli_writes_response_diagnostics(self) -> None:
        sample = Path(__file__).resolve().parents[1] / (
            "neuro_tf_sample_training_verification.mdif"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics = root / "verification_metrics.csv"
            with metrics.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["source_index", "sparam", "evm_pct", "W", "L"],
                )
                writer.writeheader()
                for index, (width, length, error) in enumerate(
                    (
                        ("0.425mm", "1.25mm", 2.0),
                        ("0.525mm", "1.25mm", 4.0),
                        ("0.425mm", "1.40mm", 3.0),
                        ("0.525mm", "1.40mm", 6.0),
                    ),
                    start=1,
                ):
                    writer.writerow(
                        {
                            "source_index": index,
                            "sparam": "S21",
                            "evm_pct": error,
                            "W": width,
                            "L": length,
                        }
                    )
            output = root / "rational_round.csv"
            combined = root / "rational_round_all_geometries.csv"
            command = [
                "suggest-additional",
                "--parameter",
                "W=0.40mm:0.55mm",
                "--parameter",
                "L=1.20mm:1.45mm",
                "--verification-metrics",
                str(metrics),
                "--existing-mdif",
                str(sample),
                "--acquisition",
                "rational-hybrid",
                "--count",
                "3",
                "--verification-policy",
                "off",
                "--candidate-count",
                "48",
                "--lhs-candidates",
                "3",
                "--rational-order",
                "2",
                "--rational-pole-placement",
                "fixed",
                "--rational-components",
                "2",
                "--out",
                str(output),
                "--combined-out",
                str(combined),
            ]
            progress_output = TTYBuffer()
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(progress_output):
                    self.assertEqual(POINTS.main(command), 0)

            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertIn("rational_response_uncertainty", rows[0])
            self.assertIn("rational_response_change", rows[0])
            metadata = json.loads(output.with_suffix(".json").read_text())
            rational = metadata["rational_hybrid"]["response_surrogate"]
            self.assertEqual(rational["method"], "common-pole-rational-pca-gp")
            self.assertEqual(rational["distinct_training_geometries"], 4)
            self.assertEqual(rational["pole_placement"], "fixed")
            self.assertTrue(combined.is_file())
            progress_text = progress_output.getvalue()
            self.assertIn(
                "Additional-point selection: fitting common-pole rational response helper",
                progress_text,
            )
            self.assertIn("selecting rational-uncertainty points", progress_text)
            self.assertTrue(progress_text.endswith("\r\033[2K"))

    def test_six_dimensional_verification_growth_policy_catches_up(self) -> None:
        before_trigger = POINTS.automatic_verification_plan(
            dimensions=6,
            existing_training_count=24,
            verification_observation_count=8,
            requested_training_count=8,
            enabled=True,
        )
        self.assertEqual(before_trigger["projected_training_count"], 32)
        self.assertEqual(before_trigger["additional_verification_count"], 0)
        self.assertEqual(before_trigger["next_training_trigger"], 36)

        first_growth = POINTS.automatic_verification_plan(
            dimensions=6,
            existing_training_count=32,
            verification_observation_count=8,
            requested_training_count=8,
            enabled=True,
        )
        self.assertEqual(first_growth["target_verification_count"], 12)
        self.assertEqual(first_growth["additional_verification_count"], 4)

        catch_up = POINTS.automatic_verification_plan(
            dimensions=6,
            existing_training_count=40,
            verification_observation_count=8,
            requested_training_count=8,
            enabled=True,
        )
        self.assertEqual(catch_up["projected_training_count"], 48)
        self.assertEqual(catch_up["target_verification_count"], 16)
        self.assertEqual(catch_up["additional_verification_count"], 8)

    def test_gp_suggest_automatically_writes_rfpro_verification_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            geometries = root / "geometries.csv"
            metrics = root / "verification_metrics.csv"
            output = root / "round_3.csv"
            generate_args = ["generate"]
            for index in range(1, 7):
                generate_args.extend(["--parameter", f"P{index}=0:1"])
            generate_args.extend(
                [
                    "--count",
                    "48",
                    "--verification-count",
                    "8",
                    "--lhs-candidates",
                    "4",
                    "--out",
                    str(geometries),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(POINTS.main(generate_args), 0)
            with geometries.open(newline="") as stream:
                geometry_rows = list(csv.DictReader(stream))
            verification_rows = [
                row for row in geometry_rows if row["dataset"] == "verification"
            ]
            with metrics.open("w", newline="") as stream:
                fields = [
                    "source_index",
                    "sparam",
                    "evm_pct",
                    *(f"P{index}" for index in range(1, 7)),
                ]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index, row in enumerate(verification_rows, start=1):
                    writer.writerow(
                        {
                            "source_index": index,
                            "sparam": "S21",
                            "evm_pct": 0.5 + index / 10.0,
                            **{f"P{item}": row[f"P{item}"] for item in range(1, 7)},
                        }
                    )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    POINTS.main(
                        [
                            "suggest-additional",
                            "--count",
                            "8",
                            "--verification-metrics",
                            str(metrics),
                            "--existing-points",
                            str(geometries),
                            "--candidate-count",
                            "96",
                            "--lhs-candidates",
                            "4",
                            "--target-dataset",
                            "train",
                            "--out",
                            str(output),
                        ]
                    ),
                    0,
                )

            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            self.assertEqual(sum(row["dataset"] == "train" for row in rows), 8)
            self.assertEqual(
                sum(row["dataset"] == "verification" for row in rows),
                8,
            )
            training_queue = root / "round_3_training.csv"
            verification_queue = root / "round_3_verification.csv"
            self.assertTrue(training_queue.is_file())
            self.assertTrue(verification_queue.is_file())
            self.assertFalse(training_queue.with_suffix(".json").exists())
            self.assertFalse(verification_queue.with_suffix(".json").exists())
            cumulative = root / "round_3_all_geometries.csv"
            cumulative_training = root / "round_3_all_geometries_training.csv"
            cumulative_verification = (
                root / "round_3_all_geometries_verification.csv"
            )
            for path in (cumulative, cumulative_training, cumulative_verification):
                self.assertTrue(path.is_file())
            with training_queue.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            with verification_queue.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            with cumulative_training.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 48)
            with cumulative_verification.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 16)
            with cumulative.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 64)
            single_coverage_plot = (
                root / "round_3_all_geometries_parameter_coverage.png"
            )
            self.assertTrue(single_coverage_plot.is_file())
            self.assertFalse((root / "round_3_parameter_coverage.png").exists())
            self.assertFalse(
                (root / "round_3_training_parameter_coverage.png").exists()
            )
            self.assertFalse(
                (root / "round_3_verification_parameter_coverage.png").exists()
            )
            self.assertFalse(
                (
                    root
                    / "round_3_all_geometries_training_parameter_coverage.png"
                ).exists()
            )
            self.assertFalse(
                (
                    root
                    / "round_3_all_geometries_verification_parameter_coverage.png"
                ).exists()
            )
            with Image.open(
                single_coverage_plot
            ) as image:
                combined_colors = {
                    color
                    for _, color in image.convert("RGB").getcolors(
                        maxcolors=image.width * image.height
                    )
                    or []
                }
            self.assertIn((37, 99, 235), combined_colors)
            self.assertIn((249, 115, 22), combined_colors)
            self.assertIn((22, 163, 74), combined_colors)
            self.assertIn((168, 85, 247), combined_colors)
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(
                metadata["parameter_coverage_plot"],
                single_coverage_plot.name,
            )
            self.assertEqual(
                metadata["output_files"]["new_points_only"]["verification"],
                str(verification_queue),
            )
            self.assertEqual(
                metadata["output_files"]["cumulative_all_points"][
                    "verification"
                ],
                str(cumulative_verification),
            )
            self.assertEqual(
                metadata["output_files"]["coverage_plot_all_points"],
                str(single_coverage_plot),
            )
            policy = metadata["automatic_verification"]
            self.assertEqual(policy["projected_training_count"], 48)
            self.assertEqual(policy["target_verification_count"], 16)
            self.assertEqual(policy["selected_additional_verification_count"], 8)
            self.assertIn("automatic verification: 8 point(s) added", stdout.getvalue())

    def test_next_round_recovers_cumulative_inventory_from_new_only_csv(self) -> None:
        """A prior new-only queue must never replace the accumulated history."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            initial = root / "initial.csv"
            metrics = root / "verification_metrics.csv"
            round_one = root / "round_one.csv"
            round_two = root / "round_two.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=0:1",
                            "--parameter",
                            "L=0:1",
                            "--count",
                            "18",
                            "--verification-count",
                            "6",
                            "--lhs-candidates",
                            "4",
                            "--out",
                            str(initial),
                        ]
                    ),
                    0,
                )
            with initial.open(newline="") as stream:
                initial_rows = list(csv.DictReader(stream))
            initial_verification = [
                row for row in initial_rows if row["dataset"] == "verification"
            ]
            with metrics.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["source_index", "sparam", "evm_pct", "W", "L"],
                )
                writer.writeheader()
                for index, row in enumerate(initial_verification, start=1):
                    writer.writerow(
                        {
                            "source_index": index,
                            "sparam": "S21",
                            "evm_pct": 0.25 + index / 10.0,
                            "W": row["W"],
                            "L": row["L"],
                        }
                    )

            common = [
                "--count",
                "2",
                "--verification-metrics",
                str(metrics),
                "--candidate-count",
                "64",
                "--lhs-candidates",
                "4",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "suggest-additional",
                            *common,
                            "--existing-points",
                            str(initial),
                            "--out",
                            str(round_one),
                        ]
                    ),
                    0,
                )

            # Deliberately supply the prior round's *new-only training split*.
            # Its combined companion JSON must redirect the command to the
            # complete round_one_all_geometries inventory.
            prior_partial = POINTS.split_output_path(round_one, "train")
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self.assertEqual(
                    POINTS.main(
                        [
                            "suggest-additional",
                            *common,
                            "--existing-points",
                            str(prior_partial),
                            "--out",
                            str(round_two),
                        ]
                    ),
                    0,
                )
            self.assertIn("replaced partial/new-only CSV", stderr.getvalue())

            new_training = POINTS.split_output_path(round_two, "train")
            new_verification = POINTS.split_output_path(round_two, "verification")
            cumulative = POINTS.accumulated_geometry_path(round_two)
            all_training = POINTS.split_output_path(cumulative, "train")
            all_verification = POINTS.split_output_path(cumulative, "verification")
            for path in (
                new_training,
                new_verification,
                cumulative,
                all_training,
                all_verification,
            ):
                self.assertTrue(path.is_file(), path)

            def rows(path: Path) -> list[dict[str, str]]:
                with path.open(newline="") as stream:
                    return list(csv.DictReader(stream))

            self.assertEqual(len(rows(new_training)), 2)
            self.assertEqual(len(rows(new_verification)), 2)
            self.assertEqual(len(rows(all_training)), 16)
            self.assertEqual(len(rows(all_verification)), 8)
            self.assertEqual(len(rows(cumulative)), 24)
            self.assertTrue(
                all(row["dataset"] == "train" for row in rows(new_training))
            )
            self.assertTrue(
                all(
                    row["dataset"] == "verification"
                    for row in rows(new_verification)
                )
            )
            self.assert_geometry_splits_are_disjoint(cumulative, ["W", "L"])

            cumulative_rows = rows(cumulative)
            role_counts = {
                role: sum(
                    POINTS.coverage_point_group(row, "dataset") == role
                    for row in cumulative_rows
                )
                for role in POINTS.COVERAGE_GROUP_COLORS
            }
            self.assertEqual(
                role_counts,
                {
                    "training": 14,
                    "verification": 6,
                    "additional_training": 2,
                    "additional_verification": 2,
                },
            )
            metadata = json.loads(round_two.with_suffix(".json").read_text())
            cumulative_metadata = json.loads(cumulative.with_suffix(".json").read_text())
            self.assertEqual(cumulative_metadata["coverage_role_counts"], role_counts)
            self.assertEqual(
                Path(
                    metadata["existing_geometry_resolution"]["resolved"][0]
                ),
                POINTS.accumulated_geometry_path(round_one),
            )
            coverage_plot = POINTS.geometry_coverage_plot_path(cumulative)
            with Image.open(coverage_plot) as image:
                rgb_image = image.convert("RGB")
                colors = set(
                    rgb_image.get_flattened_data()
                    if hasattr(rgb_image, "get_flattened_data")
                    else rgb_image.getdata()
                )
            for color in POINTS.COVERAGE_GROUP_COLORS.values():
                self.assertIn(color, colors)

    def test_split_points_csv_uses_single_combined_geometry_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            geometries = root / "geometries.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=0.4mm:0.8mm",
                            "--parameter",
                            "L=1mm:2mm",
                            "--count",
                            "8",
                            "--verification-count",
                            "2",
                            "--lhs-candidates",
                            "4",
                            "--write-split-files",
                            "--out",
                            str(geometries),
                        ]
                    ),
                    0,
                )

            train_points = root / "geometries_training.csv"
            verification_points = root / "geometries_verification.csv"
            coverage_plot = root / "geometries_parameter_coverage.png"
            train_coverage_plot = (
                root / "geometries_training_parameter_coverage.png"
            )
            verification_coverage_plot = (
                root / "geometries_verification_parameter_coverage.png"
            )
            self.assertTrue(train_points.is_file())
            self.assertTrue(verification_points.is_file())
            self.assertTrue(geometries.with_suffix(".json").is_file())
            self.assertTrue(coverage_plot.is_file())
            self.assertFalse(train_points.with_suffix(".json").exists())
            self.assertFalse(verification_points.with_suffix(".json").exists())
            self.assertFalse(train_coverage_plot.is_file())
            self.assertFalse(verification_coverage_plot.is_file())
            self.assertFalse(
                (root / "geometries_parameter_coverage.svg").exists()
            )
            with Image.open(coverage_plot) as image:
                coverage_colors = {
                    color
                    for _, color in image.convert("RGB").getcolors(
                        maxcolors=image.width * image.height
                    )
                    or []
                }
            self.assertIn((37, 99, 235), coverage_colors)
            self.assertIn((249, 115, 22), coverage_colors)
            self.assertNotIn((22, 163, 74), coverage_colors)
            self.assertNotIn((168, 85, 247), coverage_colors)
            metadata = json.loads(geometries.with_suffix(".json").read_text())
            self.assertEqual(
                metadata["parameter_coverage_plot"],
                coverage_plot.name,
            )
            self.assertEqual(
                metadata["geometry_integrity"],
                {
                    "unique_point_count": 8,
                    "training_point_count": 6,
                    "verification_point_count": 2,
                    "training_verification_overlap_count": 0,
                    "duplicates_present": False,
                },
            )
            parser = POINTS.build_suggest_parser()
            args = parser.parse_args(
                [
                    "--count",
                    "2",
                    "--existing-points",
                    str(train_points),
                ]
            )
            parameters = POINTS.resolve_suggest_parameters(parser, args)
            self.assertEqual(
                [parameter.name for parameter in parameters],
                ["W", "L"],
            )
            self.assertEqual(
                args.parameter_metadata_source,
                str(geometries.with_suffix(".json")),
            )

    def test_split_filename_classifies_rfpro_rows_without_dataset_column(self) -> None:
        parameters = [
            POINTS.parse_parameter_spec("W=0.4mm:0.8mm"),
            POINTS.parse_parameter_spec("L=1mm:2mm"),
        ]
        rows = [
            {"W": "0.45mm", "L": "1.2mm"},
            {"W": "0.75mm", "L": "1.8mm"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            verification_path = Path(temp_dir) / "rfpro_verification.csv"
            plot_path = POINTS.write_parameter_coverage_png(
                verification_path,
                parameters,
                rows,
                "dataset",
            )
            with Image.open(plot_path) as image:
                colors = {
                    color
                    for _, color in image.convert("RGB").getcolors(
                        maxcolors=image.width * image.height
                    )
                    or []
                }
            self.assertIn((249, 115, 22), colors)
            self.assertNotIn((37, 99, 235), colors)
            self.assertEqual(
                POINTS.geometry_file_split_group(verification_path),
                "verification",
            )
            self.assertEqual(
                POINTS.geometry_file_split_group(
                    Path(temp_dir) / "rfpro_train.csv"
                ),
                "training",
            )
            self.assertEqual(
                POINTS.geometry_file_split_group(
                    Path(temp_dir) / "gp_round_training_geometries.csv"
                ),
                "training",
            )

    def test_added_coverage_markers_are_twice_normal_diameter(self) -> None:
        parameters = [
            POINTS.parse_parameter_spec("W=0:1"),
            POINTS.parse_parameter_spec("L=0:1"),
        ]
        rows = [
            {
                "dataset": "train",
                "point_origin": "existing",
                "W": "0.25",
                "L": "0.25",
            },
            {
                "dataset": "train",
                "point_origin": "additional",
                "W": "0.75",
                "L": "0.75",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = POINTS.write_parameter_coverage_png(
                Path(temp_dir) / "all_geometries.csv",
                parameters,
                rows,
                "dataset",
            )
            with Image.open(plot_path) as image:
                color_counts = {
                    color: count
                    for count, color in image.convert("RGB").getcolors(
                        maxcolors=image.width * image.height
                    )
                    or []
                }

        normal_radius = POINTS.COVERAGE_GROUP_MARKER_RADII["training"]
        added_radius = POINTS.COVERAGE_GROUP_MARKER_RADII["additional_training"]
        self.assertEqual(added_radius, 2.0 * normal_radius)
        self.assertGreater(
            color_counts[POINTS.COVERAGE_GROUP_COLORS["additional_training"]],
            2.5 * color_counts[POINTS.COVERAGE_GROUP_COLORS["training"]],
        )

    def test_combined_geometry_outputs_reject_split_role_words(self) -> None:
        with self.assertRaisesRegex(ValueError, "combined geometry output"):
            POINTS.require_combined_geometry_path(
                Path("round_training_geometries.csv"),
                "--combined-out",
            )
        with self.assertRaisesRegex(ValueError, "combined geometry output"):
            POINTS.require_combined_geometry_path(
                Path("round_verification_geometries.csv"),
                "--combined-out",
            )
        POINTS.require_combined_geometry_path(
            Path("round_all_geometries.csv"),
            "--combined-out",
        )

    def test_output_rounding_cannot_duplicate_training_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "geometries.csv"
            with self.assertRaisesRegex(
                ValueError,
                "across training and verification",
            ):
                POINTS.write_points_csv(
                    output,
                    "maximin-lhs",
                    [[0.1], [0.2]],
                    [POINTS.parse_parameter_spec("W=0:1")],
                    verification_count=1,
                    split_var="dataset",
                    include_normalized=False,
                    write_split_files=True,
                    decimal_places=0,
                )
            self.assertFalse(output.exists())

    def test_duplicate_identity_uses_declared_unit_target_digits(self) -> None:
        parameter = POINTS.parse_parameter_spec("W=0um:10um")
        rows = [
            {"dataset": "train", "W": "1.2344um"},
            {"dataset": "train", "W": "1.23449um"},
        ]
        with self.assertRaisesRegex(ValueError, "within train"):
            POINTS.validate_geometry_output_rows(
                Path("geometries.csv"),
                rows,
                [parameter],
                "dataset",
                bare_values="parameter-units",
                decimal_places=3,
            )

    def test_generation_refills_points_collapsed_by_target_digits(self) -> None:
        parameter = POINTS.parse_parameter_spec("W=0:1")
        generated_batches = [
            [[0.1], [0.2]],
            [[0.6], [0.7]],
        ]
        with mock.patch.object(
            POINTS,
            "generate_unit_points",
            side_effect=generated_batches,
        ) as generate:
            points = POINTS.generate_unique_output_points(
                "maximin-lhs",
                count=2,
                sampling_parameters=[parameter],
                output_parameters=[parameter],
                decimal_places=0,
                seed=7,
                lhs_candidates=3,
                scramble=True,
                skip=0,
            )
        self.assertEqual(generate.call_count, 2)
        keys = {
            POINTS.geometry_output_key_from_unit_point(point, [parameter], 0)
            for point in points
        }
        self.assertEqual(keys, {("0",), ("1",)})

    def test_generation_rejects_count_above_target_digit_capacity(self) -> None:
        parameter = POINTS.parse_parameter_spec("W=0:1")
        with self.assertRaisesRegex(ValueError, "only 2 unique point"):
            POINTS.generate_unique_output_points(
                "maximin-lhs",
                count=3,
                sampling_parameters=[parameter],
                output_parameters=[parameter],
                decimal_places=0,
                seed=7,
                lhs_candidates=3,
                scramble=True,
                skip=0,
            )

    def test_candidate_filter_excludes_target_digit_duplicates(self) -> None:
        parameter = POINTS.parse_parameter_spec("W=0um:10um")
        existing = POINTS.geometry_output_key_from_values(
            [1.2344e-6],
            [parameter],
            3,
        )
        candidates = POINTS.filter_unique_output_candidates(
            [[0.123449], [0.123451], [0.8], [0.80001]],
            [parameter],
            3,
            excluded_keys={existing},
        )
        keys = [
            POINTS.geometry_output_key_from_unit_point(point, [parameter], 3)
            for point in candidates
        ]
        self.assertEqual(keys, [("1.235um",), ("8um",)])

    def test_range_extension_refills_target_digit_boundary_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            initial = root / "initial.csv"
            extended = root / "extended.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=0:1",
                            "--count",
                            "2",
                            "--decimal-places",
                            "0",
                            "--lhs-candidates",
                            "2",
                            "--out",
                            str(initial),
                        ]
                    ),
                    0,
                )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=0:1",
                            "--existing-points",
                            str(initial),
                            "--extend-range",
                            "W=0:2",
                            "--count",
                            "1",
                            "--verification-count",
                            "0",
                            "--method",
                            "halton",
                            "--skip",
                            "1",
                            "--decimal-places",
                            "0",
                            "--out",
                            str(extended),
                        ]
                    ),
                    0,
                )
            with extended.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            keys = [row["W"] for row in rows]
            self.assertEqual(len(keys), 3)
            self.assertEqual(set(keys), {"0", "1", "2"})

    def test_range_extension_preserves_disjoint_combined_and_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            initial = root / "initial.csv"
            extended = root / "extended.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=0:1",
                            "--parameter",
                            "L=0:1",
                            "--count",
                            "8",
                            "--verification-count",
                            "2",
                            "--lhs-candidates",
                            "3",
                            "--write-split-files",
                            "--out",
                            str(initial),
                        ]
                    ),
                    0,
                )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=0:1",
                            "--parameter",
                            "L=0:1",
                            "--existing-points",
                            str(initial),
                            "--extend-range",
                            "W=0:1.2",
                            "--count",
                            "4",
                            "--verification-count",
                            "1",
                            "--lhs-candidates",
                            "3",
                            "--write-split-files",
                            "--out",
                            str(extended),
                        ]
                    ),
                    0,
                )
            self.assert_geometry_splits_are_disjoint(extended, ["W", "L"])
            metadata = json.loads(extended.with_suffix(".json").read_text())
            self.assertEqual(
                metadata["input_geometry_cleanup"][
                    "cross_split_duplicates_removed"
                ],
                0,
            )

    def test_repeated_gp_rounds_keep_combined_and_split_outputs_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            initial = root / "initial.csv"
            metrics = root / "verification_metrics.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=0:1",
                            "--parameter",
                            "L=0:1",
                            "--count",
                            "18",
                            "--verification-count",
                            "6",
                            "--lhs-candidates",
                            "4",
                            "--write-split-files",
                            "--out",
                            str(initial),
                        ]
                    ),
                    0,
                )
            self.assert_geometry_splits_are_disjoint(initial, ["W", "L"])

            with initial.open(newline="") as stream:
                verification_rows = [
                    row
                    for row in csv.DictReader(stream)
                    if row["dataset"] == "verification"
                ]
            with metrics.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["source_index", "sparam", "evm_pct", "W", "L"],
                )
                writer.writeheader()
                for index, row in enumerate(verification_rows, start=1):
                    writer.writerow(
                        {
                            "source_index": index,
                            "sparam": "S21",
                            "evm_pct": 0.1 + 0.1 * index,
                            "W": row["W"],
                            "L": row["L"],
                        }
                    )

            round_one = root / "round_one.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "suggest-additional",
                            "--count",
                            "2",
                            "--verification-metrics",
                            str(metrics),
                            "--existing-points",
                            str(initial),
                            "--candidate-count",
                            "64",
                            "--lhs-candidates",
                            "4",
                            "--out",
                            str(round_one),
                        ]
                    ),
                    0,
                )
            round_one_all = root / "round_one_all_geometries.csv"
            self.assert_geometry_splits_are_disjoint(round_one_all, ["W", "L"])
            self.assertTrue((root / "round_one_training.csv").is_file())
            self.assertFalse((root / "round_one_verification.csv").exists())

            round_two = root / "round_two.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "suggest-additional",
                            "--count",
                            "2",
                            "--verification-metrics",
                            str(metrics),
                            "--existing-points",
                            str(round_one_all),
                            "--candidate-count",
                            "96",
                            "--lhs-candidates",
                            "4",
                            "--out",
                            str(round_two),
                        ]
                    ),
                    0,
                )
            round_two_all = root / "round_two_all_geometries.csv"
            self.assert_geometry_splits_are_disjoint(round_two, ["W", "L"])
            self.assert_geometry_splits_are_disjoint(round_two_all, ["W", "L"])
            with round_two.open(newline="") as stream:
                round_two_rows = list(csv.DictReader(stream))
            self.assertEqual(
                {row["dataset"] for row in round_two_rows},
                {"train", "verification"},
            )

    def test_legacy_targeted_and_cross_split_duplicates_are_migrated_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed = root / "seed.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=0:1",
                            "--parameter",
                            "L=0:1",
                            "--count",
                            "4",
                            "--lhs-candidates",
                            "3",
                            "--out",
                            str(seed),
                        ]
                    ),
                    0,
                )
            legacy = root / "legacy_training_geometries.csv"
            legacy.write_text(
                "point_index,dataset,W,L\n"
                "1,train,0.1,0.1\n"
                "2,verification,0.1,0.1\n"
                "3,targeted,0.3,0.3\n"
                "4,verification,0.7,0.7\n",
                encoding="utf-8",
            )
            legacy.with_suffix(".json").write_text(
                seed.with_suffix(".json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            metrics = root / "verification_metrics.csv"
            metrics.write_text(
                "source_index,sparam,evm_pct,W,L\n"
                "1,S21,0.5,0.7,0.7\n"
                "2,S21,0.7,0.8,0.8\n",
                encoding="utf-8",
            )
            output = root / "migrated.csv"
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    POINTS.main(
                        [
                            "suggest-additional",
                            "--count",
                            "1",
                            "--verification-policy",
                            "off",
                            "--verification-metrics",
                            str(metrics),
                            "--existing-points",
                            str(legacy),
                            "--candidate-count",
                            "32",
                            "--lhs-candidates",
                            "3",
                            "--out",
                            str(output),
                        ]
                    ),
                    0,
                )
            cumulative = root / "migrated_all_geometries.csv"
            self.assert_geometry_splits_are_disjoint(cumulative, ["W", "L"])
            with cumulative.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["dataset"] for row in rows}, {"train", "verification"})
            self.assertNotIn("targeted", {row["dataset"] for row in rows})
            metadata = json.loads(cumulative.with_suffix(".json").read_text())
            self.assertEqual(metadata["cross_split_duplicates_removed"], 1)
            self.assertEqual(metadata["legacy_dataset_rows_normalized"], 1)
            self.assertEqual(metadata["cross_split_conflict_resolution"], "training_wins")
            self.assertEqual(len(metadata["cross_split_conflicts"]), 1)
            self.assertIn("retained each as training", stderr.getvalue())

    def test_cross_split_duplicate_resolution_is_independent_of_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = root / "points_training.csv"
            verification = root / "points_verification.csv"
            training.write_text("W,L\n0.25,0.75\n", encoding="utf-8")
            verification.write_text("W,L\n0.25,0.75\n", encoding="utf-8")
            parameters = [
                POINTS.parse_parameter_spec("W=0:1"),
                POINTS.parse_parameter_spec("L=0:1"),
            ]
            for paths in (
                [str(training), str(verification)],
                [str(verification), str(training)],
            ):
                assignments = POINTS.existing_csv_dataset_assignments(
                    paths,
                    parameters,
                    "dataset",
                    "parameter-units",
                )
                self.assertEqual(list(assignments.values()), ["training"])

    def test_suggest_infers_parameters_from_existing_points_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            geometries = root / "geometries.csv"
            metrics = root / "verification_metrics.csv"
            output = root / "additional.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=0.4mm:0.8mm",
                            "--parameter",
                            "R=1:100:log",
                            "--count",
                            "8",
                            "--verification-count",
                            "2",
                            "--lhs-candidates",
                            "4",
                            "--out",
                            str(geometries),
                        ]
                    ),
                    0,
                )
            with geometries.open(newline="") as stream:
                geometry_rows = list(csv.DictReader(stream))
            with metrics.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["source_index", "sparam", "evm_pct", "W", "R"],
                )
                writer.writeheader()
                for index, row in enumerate(geometry_rows, start=1):
                    writer.writerow(
                        {
                            "source_index": index,
                            "sparam": "S21",
                            "evm_pct": 0.2 + index / 10.0,
                            "W": row["W"],
                            "R": row["R"],
                        }
                    )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    POINTS.main(
                        [
                            "suggest-additional",
                            "--count",
                            "3",
                            "--verification-metrics",
                            str(metrics),
                            "--existing-points",
                            str(geometries),
                            "--acquisition",
                            "gp-ucb",
                            "--candidate-count",
                            "48",
                            "--lhs-candidates",
                            "4",
                            "--out",
                            str(output),
                        ]
                    ),
                    0,
                )
            self.assertTrue(output.is_file())
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(
                metadata["parameter_metadata_source"],
                str(geometries.with_suffix(".json")),
            )
            self.assertEqual(
                [parameter["name"] for parameter in metadata["parameters"]],
                ["W", "R"],
            )
            self.assertEqual(metadata["parameters"][0]["range"]["unit"], "mm")
            self.assertEqual(metadata["parameters"][1]["scale"], "log")
            self.assertEqual(
                metadata["automatic_verification"][
                    "training_verification_overlap_count"
                ],
                6,
            )
            self.assertFalse(
                (root / "additional_parameter_coverage.png").is_file()
            )
            self.assertEqual(
                metadata["parameter_coverage_plot"],
                "additional_all_geometries_parameter_coverage.png",
            )
            with output.open(newline="") as stream:
                additional_rows = list(csv.DictReader(stream))
            self.assertTrue(
                all(row["point_origin"] == "additional" for row in additional_rows)
            )
            combined = root / "additional_all_geometries.csv"
            combined_json = combined.with_suffix(".json")
            combined_plot = root / "additional_all_geometries_parameter_coverage.png"
            self.assertTrue(combined.is_file())
            self.assertTrue(combined_json.is_file())
            self.assertTrue(combined_plot.is_file())
            with combined.open(newline="") as stream:
                combined_rows = list(csv.DictReader(stream))
            self.assertEqual(len(combined_rows), 11)
            self.assertEqual(
                len({(row["W"], row["R"]) for row in combined_rows}),
                11,
            )
            self.assertEqual(
                [row["dataset"] for row in combined_rows].count("train"),
                9,
            )
            self.assertEqual(
                [row["dataset"] for row in combined_rows].count("verification"),
                2,
            )
            self.assertEqual(
                [row["dataset"] for row in combined_rows].count("targeted"),
                0,
            )
            self.assertEqual(
                [row["point_origin"] for row in combined_rows].count("additional"),
                3,
            )
            with Image.open(combined_plot) as image:
                combined_plot_colors = {
                    color
                    for _, color in image.convert("RGB").getcolors(
                        maxcolors=image.width * image.height
                    )
                    or []
                }
            self.assertIn((37, 99, 235), combined_plot_colors)
            self.assertIn((249, 115, 22), combined_plot_colors)
            self.assertIn((22, 163, 74), combined_plot_colors)
            self.assertNotIn((168, 85, 247), combined_plot_colors)
            self.assertEqual(
                POINTS.coverage_point_group(
                    {"dataset": "train", "point_origin": "additional"},
                    "dataset",
                ),
                "additional_training",
            )
            self.assertEqual(
                POINTS.coverage_point_group(
                    {
                        "dataset": "verification",
                        "point_origin": "additional",
                    },
                    "dataset",
                ),
                "additional_verification",
            )
            self.assertEqual(
                POINTS.coverage_point_group(
                    {"dataset": "targeted", "point_origin": "existing"},
                    "dataset",
                ),
                "training",
            )
            combined_metadata = json.loads(combined_json.read_text())
            self.assertEqual(
                combined_metadata["generation_kind"],
                "accumulated_geometries",
            )
            self.assertEqual(combined_metadata["point_count"], 11)
            self.assertEqual(combined_metadata["new_point_count"], 3)
            self.assertEqual(
                combined_metadata["geometry_integrity"][
                    "training_verification_overlap_count"
                ],
                0,
            )
            self.assertEqual(
                combined_metadata["split_counts"],
                {"train": 9, "verification": 2},
            )
            self.assertEqual(
                combined_metadata["next_gp_existing_points"],
                str(combined),
            )
            next_parser = POINTS.build_suggest_parser()
            next_args = next_parser.parse_args(
                ["--count", "2", "--existing-points", str(combined)]
            )
            next_parameters = POINTS.resolve_suggest_parameters(
                next_parser,
                next_args,
            )
            self.assertEqual(
                [parameter.name for parameter in next_parameters],
                ["W", "R"],
            )
            round_two = root / "additional_round_two.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "suggest-additional",
                            "--count",
                            "2",
                            "--verification-metrics",
                            str(metrics),
                            "--existing-points",
                            str(combined),
                            "--candidate-count",
                            "48",
                            "--lhs-candidates",
                            "4",
                            "--out",
                            str(round_two),
                        ]
                    ),
                    0,
                )
            round_two_combined = root / "additional_round_two_all_geometries.csv"
            with round_two_combined.open(newline="") as stream:
                round_two_rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["point_origin"] for row in round_two_rows].count("existing"),
                11,
            )
            self.assertEqual(
                [row["point_origin"] for row in round_two_rows].count("additional"),
                2,
            )
            self.assertIn("parameter domain:", stdout.getvalue())
            self.assertIn(
                f"next GP round: --existing-points {combined}",
                stdout.getvalue(),
            )

    def test_suggest_auto_detects_unitless_base_unit_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            geometries = root / "geometries.csv"
            metrics = root / "verification_metrics.csv"
            output = root / "additional.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=400um:800um",
                            "--count",
                            "6",
                            "--verification-count",
                            "2",
                            "--lhs-candidates",
                            "3",
                            "--out",
                            str(geometries),
                        ]
                    ),
                    0,
                )
            # Geometry points remain expressed as unitless micrometre-scaled
            # values even though the post-fit metrics below use SI metres.
            with geometries.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["point_index", "dataset", "W"],
                )
                writer.writeheader()
                writer.writerow({"point_index": 1, "dataset": "train", "W": 500})
                writer.writerow({"point_index": 2, "dataset": "verification", "W": 700})
            with metrics.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["source_index", "sparam", "evm_pct", "W"],
                )
                writer.writeheader()
                writer.writerow(
                    {"source_index": 1, "sparam": "S21", "evm_pct": 0.5, "W": 0.00045}
                )
                writer.writerow(
                    {"source_index": 2, "sparam": "S21", "evm_pct": 1.5, "W": 0.00075}
                )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    POINTS.main(
                        [
                            "suggest-additional",
                            "--verification-metrics",
                            str(metrics),
                            "--existing-points",
                            str(geometries),
                            "--count",
                            "1",
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
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["bare_values_mode"], "auto")
            self.assertEqual(metadata["bare_values_interpretation"], "base-units")
            self.assertIn(
                "unitless input interpretation: base-units",
                stdout.getvalue(),
            )
            self.assertIn(
                "detected independently for each source",
                stdout.getvalue(),
            )
            with self.assertRaises(ValueError) as caught:
                POINTS.load_error_regions(
                    metrics,
                    [POINTS.parse_parameter_spec("W=400um:800um")],
                    metric_name="evm_pct",
                    bare_values="parameter-units",
                )
            self.assertIn(
                "alternate --bare-values base-units interpretation would accept",
                str(caught.exception),
            )

    def test_unusable_metric_rows_report_the_exact_parameter_problem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = Path(temp_dir) / "verification_metrics.csv"
            metrics.write_text(
                "source_index,sparam,evm_pct,W\n1,S21,0.5,0.0005\n",
                encoding="utf-8",
            )
            parameters = [
                POINTS.parse_parameter_spec("W=0.4mm:0.8mm"),
                POINTS.parse_parameter_spec("L=1mm:2mm"),
            ]
            with self.assertRaises(ValueError) as caught:
                POINTS.load_error_regions(
                    metrics,
                    parameters,
                    metric_name="evm_pct",
                    bare_values="auto",
                )
            message = str(caught.exception)
            self.assertIn("Missing requested parameter column(s): L", message)
            self.assertIn("Available columns:", message)
            self.assertIn("Requested domains:", message)

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

    def test_point_recommendation_scales_with_dimension_accuracy_and_progress(self) -> None:
        regions = [
            POINTS.ErrorRegion(
                source_index=str(index),
                unit_point=[(index + dimension) % 11 / 10.0 for dimension in range(6)],
                score=4.0,
                worst_sparam="S21",
                worst_sparam_score=4.0,
                row_count=1,
            )
            for index in range(18)
        ]
        far = POINTS.recommend_additional_point_count(
            dimensions=6,
            regions=regions,
            existing_training_count=56,
            target_error=1.0,
        )
        plateau = POINTS.recommend_additional_point_count(
            dimensions=6,
            regions=regions,
            existing_training_count=56,
            target_error=1.0,
            previous_error_rms=[4.1],
        )
        met_regions = [
            POINTS.ErrorRegion(
                source_index="1",
                unit_point=[0.5] * 6,
                score=0.5,
                worst_sparam="S21",
                worst_sparam_score=0.5,
                row_count=1,
            )
        ]
        met = POINTS.recommend_additional_point_count(
            dimensions=6,
            regions=met_regions,
            existing_training_count=80,
            target_error=1.0,
        )

        self.assertEqual(far.recommended_count, 18)
        self.assertEqual(far.stage, "far-from-target")
        self.assertEqual(plateau.recommended_count, 12)
        self.assertEqual(plateau.stage, "plateau")
        self.assertEqual(met.recommended_count, 0)
        self.assertEqual(met.stage, "target-met")

        for dimensions, expected_near, expected_above, expected_far in (
            (2, 4, 4, 6),
            (3, 5, 6, 9),
            (4, 6, 8, 12),
            (6, 9, 12, 18),
            (8, 12, 16, 24),
        ):
            with self.subTest(dimensions=dimensions):
                observation_count = max(3 * dimensions, 12)
                inventory = [
                    POINTS.ErrorRegion(
                        source_index=str(index),
                        unit_point=[
                            ((index * (axis + 1)) % observation_count + 0.5)
                            / observation_count
                            for axis in range(dimensions)
                        ],
                        score=1.5,
                        worst_sparam="S21",
                        worst_sparam_score=1.5,
                        row_count=1,
                    )
                    for index in range(observation_count)
                ]
                counts = []
                for score in (1.5, 2.5, 4.5):
                    scaled = [
                        POINTS.ErrorRegion(
                            source_index=region.source_index,
                            unit_point=region.unit_point,
                            score=score,
                            worst_sparam=region.worst_sparam,
                            worst_sparam_score=score,
                            row_count=1,
                        )
                        for region in inventory
                    ]
                    counts.append(
                        POINTS.recommend_additional_point_count(
                            dimensions=dimensions,
                            regions=scaled,
                            existing_training_count=max(4 * dimensions, 12),
                            target_error=1.0,
                        ).recommended_count
                    )
                self.assertEqual(
                    counts,
                    [expected_near, expected_above, expected_far],
                )

    def test_hybrid_allocation_and_selection_cover_all_three_roles(self) -> None:
        regions = [
            POINTS.ErrorRegion(
                source_index=str(index),
                unit_point=[x, y],
                score=0.2 + 2.0 * x,
                worst_sparam="S21",
                worst_sparam_score=0.2 + 2.0 * x,
                row_count=1,
            )
            for index, (x, y) in enumerate(
                ((0.05, 0.1), (0.2, 0.8), (0.45, 0.35), (0.75, 0.7), (0.95, 0.2)),
                start=1,
            )
        ]
        allocation = POINTS.hybrid_component_allocation(
            6,
            dimensions=2,
            observation_count=len(regions),
            target_ratio=3.0,
            latest_improvement_fraction=None,
        )
        candidates = POINTS.maximin_lhs_points(
            count=80,
            dimensions=2,
            rng=__import__("random").Random(23),
            candidates=6,
        )
        suggestions, model, diagnostics = POINTS.select_hybrid_points(
            candidates,
            regions,
            [region.unit_point for region in regions],
            count=6,
            allocation=allocation,
            exploration_weight=2.0,
            novelty_power=1.0,
            min_distance=0.01,
            length_scale=None,
            noise_variance=1e-6,
            error_floor=1e-12,
        )

        self.assertEqual(len(suggestions), 6)
        self.assertEqual(
            {item.selection_component for item in suggestions},
            {"exploitation", "uncertainty", "coverage"},
        )
        self.assertEqual(sum(diagnostics["selected_components"].values()), 6)
        self.assertEqual(
            diagnostics["batch_posterior_update"],
            "kriging-believer-posterior-mean",
        )
        self.assertEqual(len(model.length_scales), 2)

    def test_ard_uses_one_length_scale_per_dimension_when_observations_allow(self) -> None:
        regions = [
            POINTS.ErrorRegion(
                source_index=str(index),
                unit_point=[
                    (index + 0.5) / 18.0,
                    ((index * 5) % 18 + 0.5) / 18.0,
                    ((index * 11) % 18 + 0.5) / 18.0,
                ],
                score=0.2 + 3.0 * ((index + 0.5) / 18.0) ** 2,
                worst_sparam="S21",
                worst_sparam_score=1.0,
                row_count=1,
            )
            for index in range(18)
        ]
        model = POINTS.fit_error_gaussian_process(
            regions,
            length_scale=None,
            noise_variance=1e-6,
            error_floor=1e-12,
            ard_mode="auto",
        )
        self.assertEqual(len(model.length_scales), 3)
        self.assertEqual(model.length_scale_selection, "ard-coordinate-likelihood")
        self.assertTrue(all(value > 0.0 for value in model.length_scales))

    def test_auto_count_and_hybrid_metadata_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics = root / "verification_metrics.csv"
            output = root / "hybrid.csv"
            parameter_names = [f"p{index}" for index in range(1, 4)]
            with metrics.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["source_index", "sparam", "evm_pct", *parameter_names],
                )
                writer.writeheader()
                for index in range(12):
                    writer.writerow(
                        {
                            "source_index": index + 1,
                            "sparam": "S21",
                            "evm_pct": 4.0 + index / 10.0,
                            **{
                                name: ((index + dim * 3) % 12 + 0.5) / 12.0
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
                    "auto",
                    "--target-error",
                    "1.0",
                    "--verification-metrics",
                    str(metrics),
                    "--candidate-count",
                    "96",
                    "--lhs-candidates",
                    "4",
                    "--out",
                    str(output),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(POINTS.main(arguments), 0)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            metadata = json.loads(output.with_suffix(".json").read_text())

        self.assertEqual(len(rows), 9)
        self.assertEqual(
            {row["selection_component"] for row in rows},
            {"exploitation", "uncertainty", "coverage"},
        )
        recommendation = metadata["point_count_recommendation"]
        self.assertEqual(recommendation["count_mode"], "auto")
        self.assertEqual(recommendation["recommended_primary_count"], 9)
        self.assertEqual(recommendation["resolved_primary_count"], 9)
        self.assertEqual(metadata["acquisition_method"], "hybrid")
        self.assertIn("allocation", metadata["hybrid"])

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
                    "--candidate-count",
                    "96",
                    "--lhs-candidates",
                    "6",
                    "--acquisition",
                    "gp-ucb",
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
            progress_output = TTYBuffer()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                progress_output
            ):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "a=0:1",
                            "--parameter",
                            "b=0:1",
                            "--parameter",
                            "c=1:100:log",
                            "--count",
                            "12",
                            "--verification-count",
                            "3",
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
            progress_text = progress_output.getvalue()
            self.assertIn(
                "Point generation: maximin-lhs: testing maximin-LHS designs",
                progress_text,
            )
            self.assertIn(
                "Point generation: writing maximin-lhs CSV, metadata, and coverage plot",
                progress_text,
            )
            self.assertTrue(progress_text.endswith("\r\033[2K"))
            coverage_path = Path(temp_dir) / "initial_parameter_coverage.png"
            self.assertTrue(coverage_path.is_file())
            self.assertEqual(
                metadata["parameter_coverage_plot"],
                coverage_path.name,
            )
            self.assertEqual(coverage_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with Image.open(coverage_path) as coverage_image:
                self.assertEqual(coverage_image.format, "PNG")
                self.assertGreaterEqual(coverage_image.width, 1302)
                self.assertEqual(coverage_image.height, 1372)
                color_counts = coverage_image.convert("RGB").getcolors(
                    maxcolors=coverage_image.width * coverage_image.height
                )
                self.assertIsNotNone(color_counts)
                colors = {color for _, color in color_counts or []}
            self.assertIn((37, 99, 235), colors)
            self.assertIn((249, 115, 22), colors)

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
                            "--acquisition",
                            "error-distance",
                            "--target-dataset",
                            "targeted",
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
            self.assertTrue(all(row["dataset"] == "train" for row in rows))
            self.assertTrue(all(row["predicted_error"] == "" for row in rows))

    def test_sweep_root_resolves_promoted_best_model_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_dir = Path(temp_dir) / "sweep"
            metrics = sweep_dir / "best_model" / "verification_metrics.csv"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(
                "source_index,evm_pct,p\n1,0.2,0.25\n2,0.4,0.75\n"
            )
            parser = POINTS.build_suggest_parser()
            args = parser.parse_args(
                ["--count", "1", "--fit-dir", str(sweep_dir)]
            )
            self.assertEqual(
                POINTS.verification_metrics_path(args, parser),
                metrics,
            )
            self.assertIsNone(args.nonpassive_source)

    def test_missing_promoted_metrics_are_recovered_without_refitting(self) -> None:
        import surrogate_common

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sweep_dir = root / "sweep"
            best_model_dir = sweep_dir / "best_model"
            best_model_dir.mkdir(parents=True)
            source_mdif = Path(__file__).resolve().parents[1] / (
                "dnn_sample_training_verification.mdif"
            )
            source_blocks = surrogate_common.read_mdif(source_mdif)
            split = surrogate_common.split_blocks(
                source_blocks,
                split_var="dataset",
                train_values={"train", "training"},
                verify_values={"verify", "verification", "test", "validation"},
                holdout_fraction=0.2,
                seed=1234,
            )
            labels = surrogate_common.common_sparameter_labels(source_blocks)
            surrogate_common.write_mdif(
                best_model_dir / "predicted_verification.mdif",
                split.verify,
                labels,
            )
            (best_model_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "parameter_names": ["L", "W"],
                        "sparam_labels": labels,
                        "sparam_weights": {label: 1.0 for label in labels},
                        "frequency_weights": None,
                    }
                )
            )
            (sweep_dir / "dnn_best_config.json").write_text(
                json.dumps(
                    {
                        "trial": 3,
                        "best_model_dir": str(best_model_dir),
                        "reproduction_command": (
                            "python3 surrogate.py --model dnn train "
                            f"--mdif {source_mdif} --split-var dataset "
                            "--train-values train,training "
                            "--verify-values verify,verification,test,validation "
                            "--holdout-fraction 0.2 --seed 1234"
                        ),
                    }
                )
            )

            parser = POINTS.build_suggest_parser()
            args = parser.parse_args(
                ["--count", "1", "--fit-dir", str(sweep_dir)]
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                recovered = POINTS.verification_metrics_path(args, parser)

            self.assertEqual(
                recovered,
                best_model_dir / "verification_metrics.csv",
            )
            self.assertTrue(recovered.is_file())
            with recovered.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertIn("L", rows[0])
            self.assertIn("W", rows[0])
            self.assertIn(
                "recovered missing promoted verification metrics",
                stderr.getvalue(),
            )

    def test_missing_neurotf_prediction_and_metrics_are_rebuilt_from_model(self) -> None:
        import surrogate_common

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sweep_dir = root / "sweep"
            best_model_dir = sweep_dir / "best_model"
            best_model_dir.mkdir(parents=True)
            source_mdif = Path(__file__).resolve().parents[1] / (
                "neuro_tf_sample_training_verification.mdif"
            )
            source_blocks = surrogate_common.read_mdif(source_mdif)
            split = surrogate_common.split_blocks(
                source_blocks,
                split_var="dataset",
                train_values={"train", "training"},
                verify_values={"verify", "verification", "test", "validation"},
                holdout_fraction=0.2,
                seed=1234,
            )
            labels = surrogate_common.common_sparameter_labels(source_blocks)
            (best_model_dir / "model.npz").write_text("model placeholder")
            (best_model_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "parameter_names": ["L", "W"],
                        "sparam_labels": labels,
                        "verification_blocks": len(split.verify),
                        "sparam_weights": {label: 1.0 for label in labels},
                        "frequency_weights": None,
                    }
                )
            )
            (sweep_dir / "neurotf_best_config.json").write_text(
                json.dumps(
                    {
                        "trial": 2,
                        "best_model_dir": str(best_model_dir),
                        "fit_data": {
                            "mdif": str(source_mdif),
                            "split_var": "dataset",
                            "train_values": "train,training",
                            "verify_values": "verify,verification,test,validation",
                            "holdout_fraction": 0.2,
                            "seed": 1234,
                        },
                    }
                )
            )
            fake_model = mock.Mock()
            fake_model.predict_blocks.return_value = split.verify
            fake_neurotf = mock.Mock()
            fake_neurotf.load.return_value = fake_model
            fake_module = mock.Mock(NeuroTF=fake_neurotf)

            parser = POINTS.build_suggest_parser()
            args = parser.parse_args(
                ["--count", "1", "--fit-dir", str(sweep_dir)]
            )
            with mock.patch.dict("sys.modules", {"neuro_tf": fake_module}):
                recovered = POINTS.verification_metrics_path(args, parser)

            self.assertEqual(
                recovered,
                best_model_dir / "verification_metrics.csv",
            )
            self.assertTrue(recovered.is_file())
            self.assertTrue(
                (best_model_dir / "predicted_verification.mdif").is_file()
            )
            fake_neurotf.load.assert_called_once_with(best_model_dir)

    def test_moved_sweep_uses_local_best_config_relationship(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_dir = Path(temp_dir) / "copied_sweep"
            best_model_dir = sweep_dir / "best_model"
            best_model_dir.mkdir(parents=True)
            retained_metrics = (
                sweep_dir
                / "trials"
                / "trial_0004"
                / "verification_metrics.csv"
            )
            retained_metrics.parent.mkdir(parents=True)
            retained_metrics.write_text(
                "source_index,evm_pct,p\n1,0.2,0.25\n"
            )
            (sweep_dir / "neurotf_best_config.json").write_text(
                json.dumps(
                    {
                        "trial": 4,
                        "best_model_dir": "/old/location/neurotf/best_model",
                    }
                )
            )
            parser = POINTS.build_suggest_parser()
            args = parser.parse_args(
                ["--count", "1", "--fit-dir", str(sweep_dir)]
            )
            self.assertEqual(
                POINTS.verification_metrics_path(args, parser),
                retained_metrics,
            )

    def test_missing_metrics_reports_zero_recognized_verification_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fit_dir = Path(temp_dir) / "neurotf_model"
            fit_dir.mkdir(parents=True)
            (fit_dir / "metadata.json").write_text(
                json.dumps({"verification_blocks": 0})
            )
            (fit_dir / "verification_summary.json").write_text(
                json.dumps({"warning": "No verification blocks were available"})
            )
            parser = POINTS.build_suggest_parser()
            args = parser.parse_args(
                ["--count", "1", "--fit-dir", str(fit_dir)]
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    POINTS.verification_metrics_path(args, parser)
            message = stderr.getvalue()
            self.assertIn("zero recognized verification blocks", message)
            self.assertIn("--verification-mdif", message)

    def test_missing_optimize_metrics_reports_trial_split_problem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_dir = Path(temp_dir) / "neurotf_optimize"
            trial_dir = sweep_dir / "trials" / "trial_0001"
            trial_dir.mkdir(parents=True)
            (trial_dir / "verification_summary.json").write_text(
                json.dumps({"warning": "No verification blocks were available"})
            )
            parser = POINTS.build_suggest_parser()
            args = parser.parse_args(
                ["--count", "1", "--fit-dir", str(sweep_dir)]
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    POINTS.verification_metrics_path(args, parser)
            message = stderr.getvalue()
            self.assertIn("optimize trials report", message)
            self.assertIn("--split-var/--verify-values", message)

    def test_missing_promoted_metrics_resolve_from_retained_selected_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_dir = Path(temp_dir) / "sweep"
            best_model_dir = sweep_dir / "best_model"
            best_model_dir.mkdir(parents=True)
            retained_metrics = (
                sweep_dir
                / "trials"
                / "trial_0007"
                / "verification_metrics.csv"
            )
            retained_metrics.parent.mkdir(parents=True)
            retained_metrics.write_text(
                "source_index,evm_pct,p\n1,0.2,0.25\n"
            )
            (sweep_dir / "dnn_best_config.json").write_text(
                json.dumps(
                    {
                        "trial": 7,
                        "best_model_dir": str(best_model_dir),
                    }
                )
            )
            parser = POINTS.build_suggest_parser()
            args = parser.parse_args(
                ["--count", "1", "--fit-dir", str(best_model_dir)]
            )
            self.assertEqual(
                POINTS.verification_metrics_path(args, parser),
                retained_metrics,
            )

    def test_nonpassive_sweep_fallback_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_dir = Path(temp_dir) / "sweep"
            fallback_dir = sweep_dir / "point_generation_fallback"
            fallback_dir.mkdir(parents=True)
            metrics = fallback_dir / "verification_metrics.csv"
            metrics.write_text(
                "source_index,evm_pct,p\n1,0.2,0.25\n2,0.4,0.75\n"
            )
            source = {
                "status": "passivity_ineligible",
                "purpose": "gp_point_generation_only",
                "eligible_for_export": False,
                "source_trial": 7,
                "selection_metric": "evm_pct",
                "metric": 0.2,
            }
            (fallback_dir / "point_generation_source.json").write_text(
                json.dumps(source)
            )
            parser = POINTS.build_suggest_parser()
            blocked_args = parser.parse_args(
                ["--count", "1", "--fit-dir", str(sweep_dir)]
            )
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    POINTS.verification_metrics_path(blocked_args, parser)

            allowed_args = parser.parse_args(
                [
                    "--count",
                    "1",
                    "--fit-dir",
                    str(sweep_dir),
                    "--allow-nonpassive",
                ]
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                resolved = POINTS.verification_metrics_path(allowed_args, parser)
            self.assertEqual(resolved, metrics)
            self.assertEqual(allowed_args.nonpassive_source["source_trial"], 7)
            self.assertIn("point selection only", stderr.getvalue())

            legacy_best_args = parser.parse_args(
                [
                    "--count",
                    "1",
                    "--fit-dir",
                    str(sweep_dir / "best_model"),
                    "--allow-nonpassive",
                ]
            )
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    POINTS.verification_metrics_path(legacy_best_args, parser),
                    metrics,
                )

            output = Path(temp_dir) / "additional.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        POINTS.main(
                            [
                                "suggest-additional",
                                "--parameter",
                                "p=0:1",
                                "--count",
                                "1",
                                "--fit-dir",
                                str(sweep_dir),
                                "--allow-nonpassive",
                                "--candidate-count",
                                "12",
                                "--lhs-candidates",
                                "2",
                                "--out",
                                str(output),
                            ]
                        ),
                        0,
                    )
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(
                metadata["nonpassive_point_generation_source"]["source_trial"],
                7,
            )
            self.assertEqual(metadata["verification_metrics_source"], str(metrics))

    def test_best_and_nonpassive_fit_dirs_write_complete_equivalent_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            geometries = root / "geometries.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    POINTS.main(
                        [
                            "generate",
                            "--parameter",
                            "W=0:1",
                            "--parameter",
                            "L=0:1",
                            "--count",
                            "8",
                            "--verification-count",
                            "2",
                            "--lhs-candidates",
                            "3",
                            "--out",
                            str(geometries),
                        ]
                    ),
                    0,
                )
            with geometries.open(newline="") as stream:
                original_rows = list(csv.DictReader(stream))
            verification_rows = [
                row for row in original_rows if row["dataset"] == "verification"
            ]

            best_sweep = root / "best_sweep"
            best_metrics = best_sweep / "best_model" / "verification_metrics.csv"
            best_metrics.parent.mkdir(parents=True)
            fallback_sweep = root / "fallback_sweep"
            fallback_dir = fallback_sweep / "point_generation_fallback"
            fallback_metrics = fallback_dir / "verification_metrics.csv"
            fallback_dir.mkdir(parents=True)
            for metrics_path in (best_metrics, fallback_metrics):
                with metrics_path.open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=["source_index", "sparam", "evm_pct", "W", "L"],
                    )
                    writer.writeheader()
                    for index, row in enumerate(verification_rows, start=1):
                        writer.writerow(
                            {
                                "source_index": index,
                                "sparam": "S21",
                                "evm_pct": float(index),
                                "W": row["W"],
                                "L": row["L"],
                            }
                        )
            fallback_source = {
                "status": "passivity_ineligible",
                "purpose": "gp_point_generation_only",
                "eligible_for_export": False,
                "source_trial": 4,
                "selection_metric": "evm_pct",
                "metric": 1.0,
            }
            (fallback_dir / "point_generation_source.json").write_text(
                json.dumps(fallback_source)
            )

            generated_new_points: list[set[tuple[str, str]]] = []
            cases = (
                ("best", best_sweep, False, "best_model", True),
                (
                    "fallback",
                    fallback_sweep,
                    True,
                    "nonpassive_optimization_fallback",
                    False,
                ),
            )
            for name, fit_dir, allow_nonpassive, source_kind, export_eligible in cases:
                output = root / f"{name}_additional.csv"
                stale_training = POINTS.split_output_path(output, "train")
                stale_training.write_text("stale\n")
                stale_plot = POINTS.geometry_coverage_plot_path(output)
                stale_plot.write_bytes(b"stale")
                command = [
                    "suggest-additional",
                    "--count",
                    "2",
                    "--fit-dir",
                    str(fit_dir),
                    "--existing-points",
                    str(geometries),
                    "--target-dataset",
                    "verification",
                    "--verification-policy",
                    "off",
                    "--acquisition",
                    "error-distance",
                    "--candidate-count",
                    "48",
                    "--lhs-candidates",
                    "3",
                    "--out",
                    str(output),
                ]
                if allow_nonpassive:
                    command.append("--allow-nonpassive")
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(POINTS.main(command), 0)

                new_verification = POINTS.split_output_path(
                    output, "verification"
                )
                cumulative = POINTS.accumulated_geometry_path(output)
                cumulative_training = POINTS.split_output_path(
                    cumulative, "train"
                )
                cumulative_verification = POINTS.split_output_path(
                    cumulative, "verification"
                )
                coverage_plot = POINTS.geometry_coverage_plot_path(cumulative)
                self.assertFalse(stale_training.exists())
                self.assertFalse(stale_plot.exists())
                for path in (
                    output,
                    new_verification,
                    cumulative,
                    cumulative_training,
                    cumulative_verification,
                    coverage_plot,
                ):
                    self.assertTrue(path.is_file(), path)

                with output.open(newline="") as stream:
                    new_rows = list(csv.DictReader(stream))
                self.assertEqual(len(new_rows), 2)
                self.assertEqual(
                    {row["dataset"] for row in new_rows},
                    {"verification"},
                )
                generated_new_points.append(
                    {(row["W"], row["L"]) for row in new_rows}
                )
                with cumulative_training.open(newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 6)
                with cumulative_verification.open(newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 4)
                with cumulative.open(newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 10)

                metadata = json.loads(output.with_suffix(".json").read_text())
                resolution = metadata["verification_metrics_resolution"]
                self.assertEqual(resolution["kind"], source_kind)
                self.assertEqual(resolution["export_eligible"], export_eligible)
                self.assertEqual(
                    metadata["output_files"]["new_points_only"]["verification"],
                    str(new_verification),
                )
                self.assertEqual(
                    metadata["output_files"]["cumulative_all_points"][
                        "training"
                    ],
                    str(cumulative_training),
                )
                if allow_nonpassive:
                    self.assertEqual(
                        metadata["nonpassive_point_generation_source"][
                            "source_trial"
                        ],
                        4,
                    )
                else:
                    self.assertNotIn(
                        "nonpassive_point_generation_source", metadata
                    )

            self.assertEqual(generated_new_points[0], generated_new_points[1])


if __name__ == "__main__":
    unittest.main()
