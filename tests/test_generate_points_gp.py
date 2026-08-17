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
    def test_gp_ucb_is_the_default_acquisition(self) -> None:
        parser = POINTS.build_suggest_parser()
        args = parser.parse_args(["--count", "2"])
        self.assertEqual(args.acquisition, "gp-ucb")

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

            train_points = root / "geometries_train.csv"
            coverage_plot = root / "geometries_parameter_coverage.png"
            self.assertTrue(train_points.is_file())
            self.assertTrue(geometries.with_suffix(".json").is_file())
            self.assertTrue(coverage_plot.is_file())
            self.assertFalse(train_points.with_suffix(".json").exists())
            self.assertFalse(
                (root / "geometries_train_parameter_coverage.png").exists()
            )
            self.assertFalse(
                (root / "geometries_parameter_coverage.svg").exists()
            )
            metadata = json.loads(geometries.with_suffix(".json").read_text())
            self.assertEqual(
                metadata["parameter_coverage_plot"],
                coverage_plot.name,
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
            combined = root / "additional_training_geometries.csv"
            combined_json = combined.with_suffix(".json")
            combined_plot = root / "additional_training_geometries_parameter_coverage.png"
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
                6,
            )
            self.assertEqual(
                [row["dataset"] for row in combined_rows].count("verification"),
                2,
            )
            self.assertEqual(
                [row["dataset"] for row in combined_rows].count("targeted"),
                3,
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
            combined_metadata = json.loads(combined_json.read_text())
            self.assertEqual(
                combined_metadata["generation_kind"],
                "accumulated_training_geometries",
            )
            self.assertEqual(combined_metadata["point_count"], 11)
            self.assertEqual(combined_metadata["new_point_count"], 3)
            self.assertEqual(
                combined_metadata["split_counts"],
                {"train": 6, "verification": 2, "targeted": 3},
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
