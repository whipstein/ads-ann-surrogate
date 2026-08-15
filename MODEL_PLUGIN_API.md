# Model Extraction Plugin API

This document describes the flat module API for adding a new model extraction
front end. A model script imports shared infrastructure from
`surrogate_common.py` while keeping its fitting, prediction, and CLI logic in
one root-level file.

There is no dynamic plugin discovery layer. Add a uniquely named Python backend
beside `dnn.py`, `kbnn.py`, and `neuro_tf.py`, register its public model type in
`surrogate.py`, then add its workflow and command reference to the integrated
`README.md`.

## Flat Layout

For a new model named `my_model`, add:

```text
.
|-- surrogate.py
|-- surrogate_common.py
|-- my_model.py
|-- my_model_sample_training_verification.mdif
`-- README.md
```

`my_model.py` is the implementation and executable backend. Import the shared
APIs directly:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from surrogate_common import (
    MDIFBlock,
    MLP,
    Standardizer,
    add_dc_port_paths_argument,
    apply_distinct_dc_response,
    common_sparameter_labels,
    extract_average_dc_resistance,
    frequency_weights_from_blocks,
    infer_parameter_names,
    make_training_progress_callback,
    normalize_frequency_weights,
    parse_csv_set,
    progress_interval_from_args,
    positive_frequency_blocks,
    read_mdif,
    run_sweep_command,
    split_blocks,
    summary_metric,
    sweep_trial_seed,
    trial_plot_paths,
    write_history,
    write_training_markdown,
    write_training_verification_artifacts,
)

# Define the model, command handlers, and build_arg_parser() here.

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
```

Register the backend in `surrogate.py` so users keep one stable entry point:

```python
MODEL_SCRIPTS = {
    "dnn": "dnn.py",
    "kbnn": "kbnn.py",
    "neuro-tf": "neuro_tf.py",
    "my-model": "my_model.py",
}
```

The resulting command form is:

```bash
python3 surrogate.py --model my-model train --mdif input.mdif --out-dir outputs/my_model
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
- `dc_port_paths`, `dc_port_resistances_ohm`, `dc_sparameters`, the aggregate
  `dc_equivalent_resistance_ohm` summary, and the accompanying DC extraction
  metadata described below

## Required Distinct DC Contract

Every model family must keep exact DC separate from its fitted frequency
response. Before creating training features or rational coefficients:

```python
dc_metadata = extract_average_dc_resistance(
    train_blocks,
    labels,
    z0=50.0,
    port_paths=args.dc_port_paths,
)
fit_train_blocks = positive_frequency_blocks(train_blocks)
fit_verify_blocks = positive_frequency_blocks(verify_blocks) if verify_blocks else []
```

Store `dc_metadata` in `metadata.json` and place
`dc_equivalent_resistance_ohm`, `dc_resistance_source_kind`, and
`dc_port_resistances_ohm` on the loaded model. `predict_blocks()` must use
`apply_distinct_dc_response()` after forming its normal S-parameter values and
pass all three saved fields. This guarantees that zero-Hz input data cannot
affect fitted weights or poles and that older fallback-derived models cannot
supply DC.

`extract_average_dc_resistance()` uses exact-zero-Hz rows only. It rejects
non-passive S-matrices using the shared singular-value tolerance and ignores
non-finite or electrically invalid DC rows. Blocks without DC are skipped, but
extraction fails when no exact DC row exists or no usable passive row remains.
Only paths selected by `port_paths` are evaluated. Each selected path's valid
resistances are converted to conductances and averaged independently. Open
samples contribute zero conductance, so they cannot dominate a connected sample
merely because the finite open sentinel is large. A selected path above the
configured open threshold is replaced by the configured finite open resistance
(defaults: `1e12` and `1e19` ohm). Undeclared paths remain unstamped and open.
Extraction never falls back to positive-frequency data.

`build_ads_export_blocks()` automatically adds zero Hz to sampled exports.
Direct Verilog-A and ADS HB writers must receive the saved
`dc_equivalent_resistance_ohm`, `dc_resistance_source_kind`, and
`dc_port_resistances_ohm`; the shared writers validate exact-DC provenance,
stamp only the selected resistor paths at zero Hz, and bypass the fitted
response. Export CLIs must expose the shared `--dc-mdif`, `--dc-port-paths`,
`--dc-open-threshold`, and `--dc-open-resistance` arguments and use
`resolve_export_dc_metadata()`. This lets an older RF model derive DC during
export without changing or refitting its weights or poles.

## Training Command Contract

Each plugin should provide:

```python
def train_model(args: argparse.Namespace):
    ...

def command_train(args: argparse.Namespace) -> int:
    ...
```

Training and sweep commands should accept `--frequency-weights`. Parse the
positive-frequency training rows with `frequency_weights_from_blocks()` and
normalize them with `normalize_frequency_weights()` before applying them to
the fitted loss. Supported selectors are `all`/`default`/`*`, exact
engineering-unit frequencies, and inclusive `start:stop` ranges. Store the
original specification, raw mean, minimum, maximum, and normalization note in
`metadata.json`. If a model has a separate intermediate frequency-domain fit,
such as Neuro-TF rational coefficient extraction, apply the weights at that
stage.

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

Sweep CLI candidate axes must use the same naming contract as the built-in
models:

- The train option selects one value, such as `--activation relu` or
  `--learning-rate 0.002`.
- The standardized plural selects several values, such as
  `--activations tanh,relu` or `--learning-rates 0.001,0.002`.
- The sweep parser must also accept the singular train option for a one-value
  candidate set.
- Use `--search-mode grid|random` for the sweep strategy. Do not overload a
  model's train-time `--mode`; legacy aliases may be retained separately.
- Existing `*-options` names may remain compatibility aliases, but should not
  be the primary documented spelling.

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
        train_command_prefix=[
            sys.executable,
            "surrogate.py",
            "--model",
            "my-model",
            "train",
        ],
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
- a shell-copyable standalone training command in the terminal, best config
  JSON, and Markdown summary when `train_command_prefix` is supplied
- error-vs-swept-parameter PDF/CSV diagnostics

## Required CLI Surface

Every plugin should expose a parser and `main()`:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=os.environ.get("ADS_SURROGATE_CLI_PROG"),
        description="My model MDIF trainer",
    )
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
- `export-ads-hb`
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

- `verification_metrics(truth_blocks, pred_blocks, labels, parameter_names, sparam_weights, frequency_weights)`
- `parse_sparam_weights(labels, spec)`
- `normalize_sparam_weights(labels, weights)`
- `output_weights_from_sparam_weights(labels, weights)`
- `frequency_weights_from_blocks(blocks, spec)`
- `normalize_frequency_weights(weights)`
- `passivity_summary(blocks, labels)`
- `summary_metric(summary, metric_name)`

Training helpers:

- `MLP`
- `Standardizer`
- `mse(pred, truth, output_weights=None, sample_weights=None)`
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
- `write_ads_hb_mlp_package(...)`
- `write_ads_hb_neurotf_package(...)`
- `write_veriloga_package(...)`

The ADS HB writers generate a self-contained linear SDD subnetwork. They use
frequency-domain weighting so ADS evaluates the fitted S- or Y-matrix at every
HB spectral component. These exports must remain power independent: do not add
input-power features, compression curves, or nonlinear amplitude terms for a
passive structure. Exact DC is selected only at `freq=0`; positive frequencies
use only the fitted RF response. Negative frequencies must use the complex
conjugate of the fitted response evaluated at the corresponding positive
frequency so the frequency weights preserve real-waveform symmetry.

### Composite Verilog-A and ADS HB Models

`write_veriloga_package(...)` accepts an optional `embedded_coarse_model`
mapping for a model whose runtime calculation depends on a coarse response.
The common exporter evaluates that DNN first, maps its outputs to complex
coarse S-parameters, evaluates the primary model, and emits one Verilog-A
N-port. The mapping contains:

```python
embedded_coarse_model = {
    "source_model_dir": str(coarse_model_dir),
    "parameter_names": coarse_model.parameter_names,
    "sparam_labels": coarse_model.sparam_labels,
    "freq_transform": coarse_model.freq_transform,
    "activation": coarse_model.mlp.activation,
    "layer_sizes": coarse_model.mlp.layer_sizes,
    "weights": coarse_model.mlp.weights,
    "biases": coarse_model.mlp.biases,
    "x_mean": coarse_model.x_scaler.mean,
    "x_std": coarse_model.x_scaler.std,
    "y_mean": coarse_model.y_scaler.mean,
    "y_std": coarse_model.y_scaler.std,
    "output_domain": "s",
}
```

The coarse model must be S-domain and must use exactly the same parameter-name
order and S-parameter-label order as the primary model. Its inputs are geometry
or process parameters plus its own frequency features; it cannot itself require
a coarse response. Set `uses_coarse_inputs=True` when the primary model consumes
the coarse S-parameters and `adds_coarse_to_output=True` when it predicts a
residual that must be added to them. A plugin should reject a missing embedded
model by default whenever either flag is true, unless it deliberately exposes a
legacy hook-based export mode.

`write_ads_hb_mlp_package(...)` accepts the same embedded-coarse mapping and
flags. Unlike the optional legacy Verilog-A hook mode, a residual or
prior-input HB export must embed the matching frozen coarse DNN so that the
subnetwork is self-contained and reproduces the actual fine-model fit.

## Artifact Expectations

A successful `train` command should write:

```text
model.npz
metadata.json
training_history.csv
training_history.pdf
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

The best-config JSON and Markdown summary include a standalone `train` command
that uses the selected trial seed and effective training arguments. The command
targets `<sweep-dir>/reproduced_model` so it does not overwrite `best_model/`.

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

1. Add a unique root-level `<plugin>.py` model script.
2. Import reusable infrastructure directly from `surrogate_common.py`.
3. Implement the script with `train_model`, `command_train`, `predict_blocks`,
   `save`, and `load`.
4. Add `sweep_candidate_grid`, `namespace_for_trial`, a sweep worker, and
   `command_sweep` using `run_sweep_command`.
5. Add `build_arg_parser` and `main`.
6. Add `--loss-interval` and `--progress-interval` to neural training/sweep
   parsers, and add `--retrain-best` to sweep parsers.
7. Reuse `write_training_verification_artifacts` for verification outputs.
8. Add the plugin workflow to `README.md` and add one uniquely named sample MDIF.
9. Verify with `python3 -m py_compile` and a short `train` run.
10. Verify `sweep --max-trials 2 --worst-plots 0 --trial-worst-plots 0`.
