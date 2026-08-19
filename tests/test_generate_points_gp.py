import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import generate_points as POINTS


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

    def test_gp_ucb_is_the_default_acquisition(self) -> None:
        parser = POINTS.build_suggest_parser()
        args = parser.parse_args(["--count", "2"])
        self.assertEqual(args.acquisition, "gp-ucb")
        self.assertEqual(args.verification_policy, "auto")
        self.assertEqual(args.target_dataset, "train")

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
            self.assertTrue(
                (root / "round_3_training_parameter_coverage.png").is_file()
            )
            self.assertTrue(
                (root / "round_3_verification_parameter_coverage.png").is_file()
            )
            with Image.open(
                root / "round_3_parameter_coverage.png"
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
            with Image.open(
                root / "round_3_training_parameter_coverage.png"
            ) as image:
                training_colors = {
                    color
                    for _, color in image.convert("RGB").getcolors(
                        maxcolors=image.width * image.height
                    )
                    or []
                }
            self.assertIn((37, 99, 235), training_colors)
            self.assertIn((22, 163, 74), training_colors)
            self.assertNotIn((249, 115, 22), training_colors)
            with Image.open(
                root / "round_3_verification_parameter_coverage.png"
            ) as image:
                verification_colors = {
                    color
                    for _, color in image.convert("RGB").getcolors(
                        maxcolors=image.width * image.height
                    )
                    or []
                }
            self.assertIn((249, 115, 22), verification_colors)
            self.assertIn((22, 163, 74), verification_colors)
            self.assertNotIn((37, 99, 235), verification_colors)
            metadata = json.loads(output.with_suffix(".json").read_text())
            policy = metadata["automatic_verification"]
            self.assertEqual(policy["projected_training_count"], 48)
            self.assertEqual(policy["target_verification_count"], 16)
            self.assertEqual(policy["selected_additional_verification_count"], 8)
            self.assertIn("automatic verification: 8 point(s) added", stdout.getvalue())

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
            self.assertTrue(train_coverage_plot.is_file())
            self.assertTrue(verification_coverage_plot.is_file())
            self.assertFalse(
                (root / "geometries_parameter_coverage.svg").exists()
            )
            with Image.open(train_coverage_plot) as image:
                train_colors = {
                    color
                    for _, color in image.convert("RGB").getcolors(
                        maxcolors=image.width * image.height
                    )
                    or []
                }
            self.assertIn((37, 99, 235), train_colors)
            self.assertNotIn((249, 115, 22), train_colors)
            self.assertNotIn((22, 163, 74), train_colors)
            with Image.open(verification_coverage_plot) as image:
                verification_colors = {
                    color
                    for _, color in image.convert("RGB").getcolors(
                        maxcolors=image.width * image.height
                    )
                    or []
                }
            self.assertIn((249, 115, 22), verification_colors)
            self.assertNotIn((37, 99, 235), verification_colors)
            self.assertNotIn((22, 163, 74), verification_colors)
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
            self.assertTrue(
                (root / "additional_parameter_coverage.png").is_file()
            )
            self.assertEqual(
                metadata["parameter_coverage_plot"],
                "additional_parameter_coverage.png",
            )
            with output.open(newline="") as stream:
                additional_rows = list(csv.DictReader(stream))
            self.assertTrue(
                all(row["point_origin"] == "additional" for row in additional_rows)
            )
            with Image.open(root / "additional_parameter_coverage.png") as image:
                additional_plot_colors = {
                    color
                    for _, color in image.convert("RGB").getcolors(
                        maxcolors=image.width * image.height
                    )
                    or []
                }
            self.assertIn((22, 163, 74), additional_plot_colors)
            self.assertIn((37, 99, 235), additional_plot_colors)
            self.assertIn((249, 115, 22), additional_plot_colors)
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
            self.assertEqual(
                POINTS.coverage_point_group(
                    {"dataset": "train", "point_origin": "additional"},
                    "dataset",
                ),
                "additional",
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
            coverage_path = Path(temp_dir) / "initial_parameter_coverage.png"
            self.assertTrue(coverage_path.is_file())
            self.assertEqual(
                metadata["parameter_coverage_plot"],
                coverage_path.name,
            )
            self.assertEqual(coverage_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with Image.open(coverage_path) as coverage_image:
                self.assertEqual(coverage_image.format, "PNG")
                self.assertEqual(coverage_image.size, (1302, 1372))
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


if __name__ == "__main__":
    unittest.main()
