# Model Extraction Plugin API

This document describes the rc2 module API for adding a new model extraction
plugin. In rc2, a plugin is a model package that uses the shared infrastructure
in `common/surrogate_common.py` while keeping model-specific fitting and
prediction logic in its own module.

There is no dynamic plugin discovery layer yet. A new plugin is added by
creating a new directory beside `dnn/`, `kbnn/`, and `neuro_tf/`, then wiring a
thin entry-point script to that directory's `model.py`.

## Directory Layout

Use this shape for a new plugin named `my_model`:

```text
outputs/rc2/
  common/
    surrogate_common.py
  my_model/
    my_model.py
    model.py
    README.md
    sample_training_verification.mdif
```

`my_model.py` should be a thin wrapper:

```python
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from model import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
```

`model.py` should add the rc2 root to `sys.path` and import shared APIs:

```python
from pathlib import Path
import sys

RC2_ROOT = Path(__file__).resolve().parents[1]
if str(RC2_ROOT) not in sys.path:
    sys.path.insert(0, str(RC2_ROOT))

from common.surrogate_common import (
    MDIFBlock,
    MLP,
    Standardizer,
    common_sparameter_labels,
    infer_parameter_names,
    make_training_progress_callback,
    parse_csv_set,
    progress_interval_from_args,
    read_mdif,
    run_sweep_command,
    split_blocks,
    summary_metric,
    trial_plot_paths,
    sweep_trial_seed,
    write_history,
    write_training_markdown,
    write_training_verification_artifacts,
)
```

## Shared Data Types

`MDIFBlock` is the common in-memory response container:

```python
@dataclass
class MDIFBlock:
    params: dict[str, str]
    freq_hz: np.ndarray
    sparams: dict[str, np.ndarray]
    source_index: int = 0
```

Rules:

- `params` holds MDIF `VAR` values as strings.
- `freq_hz` is a one-dimensional frequency array in Hz.
- `sparams[label]` is a complex array with the same length as `freq_hz`.
- Prediction functions must preserve `params`, `freq_hz`, and `source_index`
  unless the plugin intentionally changes the evaluation grid.

## Model Class Contract

The model class can be named however you like. Existing examples use `DNN`,
`KBNN`, and `NeuroTF`.

Minimum recommended methods:

```python
class MyModel:
    parameter_names: list[str]
    sparam_labels: list[str]

    def predict_blocks(self, blocks: Sequence[MDIFBlock]) -> list[MDIFBlock]:
        ...

    def save(self, out_dir: Path, metadata: dict[str, object]) -> None:
        ...

    @staticmethod
    def load(model_dir: Path) -> "MyModel":
        ...
```

For models that need additional prediction inputs, such as KBNN coarse blocks,
extend `predict_blocks` as needed and keep that wiring local to the plugin.

The saved model directory should contain:

- `model.npz` or equivalent model parameters
- `metadata.json`
- enough metadata to reconstruct `parameter_names`, `sparam_labels`, model
  dimensions, training domain, and export assumptions

## Training Command Contract

Each plugin should provide:

```python
def train_model(args: argparse.Namespace):
    ...

def command_train(args: argparse.Namespace) -> int:
    ...
```

`train_model` is plugin-specific. It usually:

1. Reads and splits MDIF data with `read_mdif()` and `split_blocks()`.
2. Infers or validates `parameter_names` and `sparam_labels`.
3. Builds model-specific feature/target arrays.
4. Trains the model.
5. Returns the trained model, verification blocks, labels, history, and
   metadata needed by `command_train`.

If the plugin uses the shared `MLP`, connect the standard progress callback so
single training runs and sweep trials report live fitting progress:

```python
progress_interval = progress_interval_from_args(args)
history = mlp.train(
    x_train_scaled,
    y_train_scaled,
    x_verify_scaled,
    y_verify_scaled,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    patience=args.patience,
    seed=args.seed + 17,
    loss_interval=getattr(args, "loss_interval", 1),
    progress_callback=make_training_progress_callback(
        getattr(args, "progress_label", "MyModel fit"),
        args.epochs,
        progress_interval,
    ),
    progress_interval=progress_interval,
)
```

`--loss-interval` controls how often full train/verification losses are
computed and stored in `training_history.csv`. `--progress-interval` controls
how often console heartbeats are printed. When both intervals align, the
progress line includes the latest loss values; otherwise it prints epoch count
and elapsed time without extra full-dataset scoring.

`command_train` should handle shared output generation:

```python
def command_train(args: argparse.Namespace) -> int:
    model, verify_blocks, parameter_names, labels, history, metadata = train_model(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir, metadata=metadata)
    write_history(out_dir / "training_history.csv", history)

    training_config = {
        "training_blocks": metadata["training_blocks"],
        "verification_blocks": metadata["verification_blocks"],
        "parameters": parameter_names,
        "sparameters": labels,
        "progress_interval": progress_interval_from_args(args),
        "seed": args.seed,
    }

    if verify_blocks:
        pred_blocks = model.predict_blocks(verify_blocks)
        summary = write_training_verification_artifacts(
            out_dir,
            verify_blocks,
            pred_blocks,
            labels,
            parameter_names,
            max_worst_plots=getattr(args, "worst_plots", 6),
            sparam_weights=None,
            y_z0=50.0,
        )
    else:
        summary = {"warning": "No verification blocks were available"}
        (out_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2))

    write_training_markdown(
        out_dir / "training_summary.md",
        model_kind="MyModel",
        config=training_config,
        summary=summary,
        history=history,
    )
    return 0
```

`write_training_verification_artifacts()` writes:

- `predicted_verification.mdif`
- `verification_metrics.csv`
- `verification_summary.json`
- `worst_case_plots/*.pdf`
- `worst_case_y_plots/*.pdf`

## Sweep API

To use the common sweep orchestration, a plugin supplies three hooks and calls
`run_sweep_command()`. The common runner handles candidate evaluation,
parallel execution, passivity-aware ranking, live best-model promotion,
optional best-model retraining, CSV/Markdown summaries, and post-sweep
diagnostics.

Required constants:

```python
MY_MODEL_SWEEP_RESULT_COLUMNS = ["hidden_layers", "activation", "learning_rate"]
```

These column names must match keys emitted by `sweep_candidate_grid()`. They
are used to reconstruct the best candidate, write the best config, and group
sweep diagnostics.

Candidate hook:

```python
def sweep_candidate_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {
            "hidden_layers": "64,64",
            "activation": "tanh",
            "learning_rate": 0.002,
        },
    ]
```

Trial namespace hook:

```python
def namespace_for_trial(
    args: argparse.Namespace,
    candidate: dict[str, object],
    out_dir: Path,
    trial_index: int,
    plots: int,
) -> argparse.Namespace:
    trial_seed = sweep_trial_seed(args.seed, trial_index, args.trial_seed_mode)
    return argparse.Namespace(
        mdif=args.mdif,
        out_dir=str(out_dir),
        hidden_layers=str(candidate["hidden_layers"]),
        activation=str(candidate["activation"]),
        learning_rate=float(candidate["learning_rate"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        loss_interval=args.loss_interval,
        progress_interval=args.progress_interval,
        progress_label=f"MyModel trial {trial_index}",
        seed=trial_seed,
        worst_plots=plots,
        quiet=True,
    )
```

Worker hook:

```python
def my_model_sweep_trial_worker(
    payload: tuple[dict[str, object], dict[str, object], str, int, int],
) -> dict[str, object]:
    args_values, candidate, out_dir_text, trial_index, plots = payload
    args = argparse.Namespace(**args_values)
    out_dir = Path(out_dir_text)
    trial_dir = out_dir / "trials" / f"trial_{trial_index:04d}"
    trial_seed = sweep_trial_seed(args.seed, trial_index, args.trial_seed_mode)

    try:
        status = command_train(
            namespace_for_trial(args, candidate, trial_dir, trial_index, plots=plots)
        )
        error_message = None
    except Exception as exc:
        status = 2
        error_message = str(exc)

    summary_path = trial_dir / "verification_summary.json"
    if status != 0 or not summary_path.exists():
        summary = {"error": error_message or "trial failed"}
        metric_value = None
    else:
        summary = json.loads(summary_path.read_text())
        metric_value = summary_metric(summary, args.selection_metric)

    return {
        "trial": trial_index,
        "candidate": candidate,
        "summary": summary,
        "metric": metric_value,
        "trial_seed": trial_seed,
        "plot_paths": trial_plot_paths(summary, trial_dir, out_dir),
    }
```

Sweep command:

```python
def command_sweep(args: argparse.Namespace) -> int:
    return run_sweep_command(
        args,
        sweep_candidate_grid(args),
        worker_func=my_model_sweep_trial_worker,
        namespace_for_trial_func=namespace_for_trial,
        train_func=command_train,
        result_columns=MY_MODEL_SWEEP_RESULT_COLUMNS,
        results_filename="my_model_sweep_results.csv",
        best_config_filename="my_model_best_config.json",
        summary_filename="my_model_sweep_summary.md",
        diagnostics_prefix="my_model",
    )
```

The common sweep runner handles:

- serial or process-parallel trial execution
- trial cleanup
- passivity-aware reranking
- best-model retraining
- sweep results CSV
- best config JSON
- sweep summary Markdown
- error-vs-swept-parameter PDF/CSV diagnostics

## Required CLI Surface

Every plugin should expose a parser and `main()`:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="My model MDIF trainer")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--mdif", required=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--worst-plots", type=int, default=6)
    train.set_defaults(func=command_train)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--mdif", required=True)
    sweep.add_argument("--out-dir", required=True)
    sweep.add_argument("--selection-metric", default="rmse_abs")
    sweep.add_argument("--trial-worst-plots", type=int, default=1)
    sweep.add_argument("--worst-plots", type=int, default=6)
    sweep.add_argument("--jobs", type=int, default=1)
    sweep.add_argument("--keep-trial-models", action="store_true")
    sweep.add_argument("--require-passive", action="store_true")
    sweep.add_argument("--max-passivity-violations", type=int)
    sweep.add_argument("--max-passivity-sigma", type=float)
    sweep.add_argument("--trial-seed-mode", choices=["fixed", "indexed"], default="fixed")
    sweep.set_defaults(func=command_sweep)

    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.func(args)
```

Optional commands normally include:

- `predict`
- `inspect-mdif`
- `rerank-sweep`
- `export-ads-mdif`
- `export-ads`
- `export-ads-ann`
- `export-veriloga`

## Shared Utilities Worth Reusing

Common data and parsing:

- `read_mdif(path)`
- `write_mdif(path, blocks, labels)`
- `split_blocks(blocks, split_var, train_values, verify_values, holdout_fraction, seed)`
- `infer_parameter_names(blocks, requested, split_var)`
- `common_sparameter_labels(blocks)`
- `parameter_matrix(blocks, parameter_names)`

Metrics and weighting:

- `verification_metrics(truth_blocks, pred_blocks, labels, parameter_names, sparam_weights)`
- `parse_sparam_weights(labels, spec)`
- `normalize_sparam_weights(labels, weights)`
- `output_weights_from_sparam_weights(labels, weights)`
- `passivity_summary(blocks, labels)`
- `summary_metric(summary, metric_name)`

Training helpers:

- `MLP`
- `Standardizer`
- `mse(pred, truth, output_weights=None)`
- `write_history(path, history)`

Plots and summaries:

- `plot_worst_case_fits(...)`
- `plot_worst_case_y_fits(...)`
- `plot_sweep_diagnostics(...)`
- `write_training_markdown(...)`
- `write_sweep_markdown(...)`

Export helpers:

- `build_ads_export_blocks(...)`
- `write_ads_export_package(...)`
- `write_ads_ann_package(...)`
- `write_veriloga_package(...)`

## Artifact Expectations

A successful `train` command should write:

```text
model.npz
metadata.json
training_history.csv
training_summary.md
verification_summary.json
predicted_verification.mdif
verification_metrics.csv
worst_case_plots/
worst_case_y_plots/
```

A successful `sweep` command should write:

```text
trials/
best_model/
<plugin>_sweep_results.csv
<plugin>_best_config.json
<plugin>_sweep_summary.md
sweep_diagnostics/
```

If `--keep-trial-models` is false, the common sweep runner removes large
per-trial model artifacts after each trial while keeping lightweight summaries
and plots.

`best_model/` is updated on the fly. After each completed trial, the common
runner reranks all completed rows using the active selection metric and
passivity constraints. If the new trial is the best valid completed model, its
trial directory is copied into `best_model/` before cleanup removes large
per-trial artifacts. The end of the sweep therefore reuses the promoted model
instead of fitting the same configuration again. Add a sweep parser option named
`--retrain-best` when you want to expose the older behavior where the winning
configuration is trained again after all trials finish.

## Implementation Checklist

1. Create `outputs/rc2/<plugin>/`.
2. Add a thin `<plugin>.py` wrapper.
3. Implement `model.py` with `train_model`, `command_train`, `predict_blocks`,
   `save`, and `load`.
4. Add `sweep_candidate_grid`, `namespace_for_trial`, a sweep worker, and
   `command_sweep` using `run_sweep_command`.
5. Add `build_arg_parser` and `main`.
6. Add `--loss-interval` and `--progress-interval` to neural training/sweep
   parsers, and add `--retrain-best` to sweep parsers.
7. Reuse `write_training_verification_artifacts` for verification outputs.
8. Add a plugin README and one small sample MDIF.
9. Verify with `python3 -m py_compile` and a short `train` run.
10. Verify `sweep --max-trials 2 --worst-plots 0 --trial-worst-plots 0`.
