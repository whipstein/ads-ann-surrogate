# Deep Neural Network MDIF Trainer

This is the direct DNN companion to the Neuro-TF and KBNN prototypes. It trains
a deep multilayer perceptron from parameterized S-parameter MDIF data.

Model structure:

```text
geometry/process VARs + frequency features -> deep neural network -> S-parameters or Y-parameters
```

The DNN treats frequency as an input and predicts real/imaginary response
values directly. The default target is S-parameters. For direct Verilog-A use,
`--output-domain y` trains the same network against converted admittance
targets, which lets the exported ADS model stamp Y directly and skip a runtime
S-to-Y matrix inversion at every simulator evaluation.

The trainer automatically floors zero-variance output scaler columns to a
representative response scale, which prevents constant terms such as an exactly
zero isolation path from becoming large admittance errors in direct-Y models.

## RC2 Layout

The command entry point is `dnn.py`. The DNN fitting calculation lives in
`model.py`; shared MDIF I/O, plotting, metrics, sweep diagnostics, summaries,
and export helpers live in `../common/surrogate_common.py`.

## Expected MDIF Shape

Each block should contain numeric geometry variables as `VAR` values and an
ACDATA table. Use a split variable such as `dataset=train` or
`dataset=verification` when training and verification data are in one file.

```text
VAR dataset=train
VAR W=0.40mm
VAR L=1.20mm
BEGIN ACDATA
% Freq S11 S12 S21 S22
# Hz S RI R 50
1.0e9  0.1 0.0  0.0 0.0  0.8 -0.1  0.1 0.0
2.0e9  0.2 0.0  0.0 0.0  0.6 -0.2  0.2 0.0
END
```

Supported pair formats in the option line are `RI`, `MA`, and `DB`. Header
names may be logical names (`S11`) or explicit columns (`S11R S11I`).

## Inspect MDIF

Use `inspect-mdif` first when you want to confirm block count, S-parameter
labels, inferred numeric variables, split values, and frequency span.

```bash
python3 dnn.py inspect-mdif \
  --mdif train_verify.mdif
```

### Options

| Option Name | Description | Example |
| ------------------------------- | --- | ------------------------------------------------ |
| <nobr><code>--mdif PATH</code></nobr> | Required. MDIF file to inspect. | <nobr><code>--mdif train_verify.mdif</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | Split variable name to count in the summary. Default: `dataset`. | <nobr><code>--split-var dataset</code></nobr> |

## Usage

Train one DNN model with `train`:

```bash
python3 dnn.py train \
  --mdif train_verify.mdif \
  --out-dir dnn_model \
  --parameter-names W,L \
  --hidden-layers 128,128,64
```

Outputs:

- `model.npz` and `metadata.json`: trained DNN model
- `predicted_verification.mdif`: model predictions at verification points
- `verification_metrics.csv`: per-block and per-S-parameter errors, including EVM
- `verification_summary.json`: global error, passivity summary, and plot paths
- `training_history.csv`: neural training loss history
- `worst_case_plots/*.pdf`: multi-page worst verification case plots
- `worst_case_y_plots/*.pdf`: matching Y-parameter implementation-view plots

For plots that match the Matplotlib styling used by the reference example,
install Matplotlib into the Python environment used to run the trainer:

```bash
python3 -m pip install matplotlib
```

If Matplotlib is unavailable, the trainer still writes PDFs with a lightweight
built-in renderer, but those plots are intended as a fallback rather than an
exact visual match to the example.

Training writes multi-page PDF plots for the worst verification cases by
default. Each plot contains a magnitude grid for all S-parameters, an unwrapped
phase grid, and an error-focus page with a max magnitude-error heatmap plus the
worst S-parameter overlay. The same cases are also converted to Y-parameters
with `--target-z0` and written under `worst_case_y_plots/`, which shows the
admittance response used by the direct Verilog-A implementation. Use
`--worst-plots 0` to skip both plot sets during large experiments.
Each single `train` run also writes `training_summary.md`, which collects the
chosen settings, final loss values, verification metrics, passivity summary,
and links to the generated S- and Y-parameter worst-case plots.

### Options

| Option Name | Description | Example |
| ------------------------------- | --- | ------------------------------------------------ |
| <nobr><code>--activation {tanh,relu}</code></nobr> | Hidden-layer activation. `tanh` is smoother for small microwave datasets; `relu` can help larger datasets. Default: `tanh`. | <nobr><code>--activation tanh</code></nobr> |
| <nobr><code>--batch-size INT</code></nobr> | Number of frequency-sample rows per Adam update. Default: `256`. | <nobr><code>--batch-size 256</code></nobr> |
| <nobr><code>--epochs INT</code></nobr> | Maximum Adam training epochs. Early stopping may stop before this value. Default: `2000`. | <nobr><code>--epochs 2000</code></nobr> |
| <nobr><code>--freq-transform {log,linear,log-linear}</code></nobr> | Frequency input transform. `log` uses `log10(freq_hz)`, `linear` uses raw Hz, and `log-linear` uses both. Default: `log`. | <nobr><code>--freq-transform log-linear</code></nobr> |
| <nobr><code>--hidden-layers LIST</code></nobr> | Comma-separated hidden layer sizes. More entries create a deeper model. Default: `128,128,64`. | <nobr><code>--hidden-layers 128,128,64</code></nobr> |
| <nobr><code>--holdout-fraction FLOAT</code></nobr> | Fraction of blocks to reserve for verification when no split values are found in a combined MDIF. Default: `0.2`. | <nobr><code>--holdout-fraction 0.25</code></nobr> |
| <nobr><code>--learning-rate FLOAT</code></nobr> | Adam optimizer step size. Lower values are safer; higher values may converge faster but can overshoot. Default: `0.002`. | <nobr><code>--learning-rate 0.002</code></nobr> |
| <nobr><code>--loss-interval INT</code></nobr> | Full train/verification loss check interval in epochs. Increasing this reduces full-dataset scoring overhead during long runs while early stopping still uses epoch-based patience. Default: `1`. | <nobr><code>--loss-interval 5</code></nobr> |
| <nobr><code>--mdif PATH</code></nobr> | Required. Input MDIF. If `--verification-mdif` is not supplied, this file should contain both training and verification blocks, typically separated by a `VAR` such as `dataset=train` or `dataset=verification`. | <nobr><code>--mdif train_verify.mdif</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | Required. Output directory for `model.npz`, `metadata.json`, `training_summary.md`, `predicted_verification.mdif`, metrics CSV/JSON files, training history, and S/Y worst-case plot PDFs. | <nobr><code>--out-dir dnn_model</code></nobr> |
| <nobr><code>--output-domain {s,y}</code></nobr> | Training target domain. `s` predicts S-parameters and is compatible with every export path. `y` converts the MDIF S-data to admittance targets using `--target-z0`; this is the fastest formulation for direct Verilog-A solve speed. Default: `s`. | <nobr><code>--output-domain y</code></nobr> |
| <nobr><code>--parameter-names LIST</code></nobr> | Comma-separated geometry/process variable names to use as DNN inputs. If omitted, the trainer infers numeric `VAR`s common to all blocks, excluding the split variable. | <nobr><code>--parameter-names W,L,H</code></nobr> |
| <nobr><code>--patience INT</code></nobr> | Early-stopping patience measured in epochs without validation-loss improvement. Use `0` to disable early stopping. Default: `200`. | <nobr><code>--patience 200</code></nobr> |
| <nobr><code>--progress-interval INT</code></nobr> | Console progress update interval in epochs. Updates redraw one terminal status line and include epoch count, elapsed time, and loss values when that epoch also matches `--loss-interval`. Use `0` to disable. Default: `25`. | <nobr><code>--progress-interval 10</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | Random seed for holdout splitting and neural-network initialization. Default: `1234`. | <nobr><code>--seed 1234</code></nobr> |
| <nobr><code>--sparam-weights SPEC</code></nobr> | Optional S-parameter loss weights. Supports grouped selectors such as `diag`, `offdiag`, `row1`, `col2`, wildcards, and comma-separated explicit labels. Later rules override earlier rules. Weights are normalized to mean 1 internally. | <nobr><code>--sparam-weights 'diag=1;offdiag=0.2'</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | Name of the `VAR` used to split a combined MDIF. Default: `dataset`. | <nobr><code>--split-var dataset</code></nobr> |
| <nobr><code>--target-z0 FLOAT</code></nobr> | Reference impedance used only when `--output-domain y` converts S-parameters into Y-parameter training targets. Use the same value as the MDIF option line reference impedance. Default: `50.0`. | <nobr><code>--target-z0 50</code></nobr> |
| <nobr><code>--train-values LIST</code></nobr> | Comma-separated values of `--split-var` that identify training blocks. Default: `train,training`. | <nobr><code>--train-values train,training</code></nobr> |
| <nobr><code>--verification-mdif PATH</code></nobr> | Optional. Separate MDIF containing verification blocks. When supplied, every block in `--mdif` is treated as training data and every block in this file is treated as verification data. | <nobr><code>--verification-mdif verify.mdif</code></nobr> |
| <nobr><code>--verify-values LIST</code></nobr> | Comma-separated values of `--split-var` that identify verification blocks. Default: `verify,verification,test,validation`. | <nobr><code>--verify-values verification,test</code></nobr> |
| <nobr><code>--worst-plots INT</code></nobr> | Number of worst verification fits to render as PDFs. Each selected case gets an S-parameter plot and a Y-parameter implementation-view plot. Use `0` to skip plot generation. Default: `6`. | <nobr><code>--worst-plots 6</code></nobr> |

## Sweeping / Optimizing

Use `sweep` or its alias `optimize` to try multiple DNN configurations. The
command writes `dnn_sweep_results.csv` and `dnn_sweep_summary.md`, chooses the
best trial using `--selection-metric`, and keeps the current best completed
trial in `best_model/` as the sweep runs. This avoids a final refit after all
trials finish.

```bash
python3 dnn.py optimize \
  --mdif train_verify.mdif \
  --out-dir dnn_sweep \
  --parameter-names W,L,H \
  --freq-transform-options log,log-linear \
  --hidden-layer-options '64,64;128,128,64;256,128,64' \
  --activation-options tanh,relu \
  --learning-rates 0.001,0.002,0.005 \
  --sparam-weights 'diag=1;offdiag=0.2' \
  --output-domain y \
  --target-z0 50 \
  --mode random \
  --max-trials 40 \
  --selection-metric weighted_evm_pct \
  --require-passive
```

Use `--sparam-weights` to make some S-parameters matter less during training
and sweep selection. The same weight is applied to the real and imaginary
columns for each selected S-parameter. Rules are applied left to right, so later
rules override earlier broad rules. Weights are normalized internally so their
average across S-parameter labels is 1.0 before they are applied to the
training loss and scale-sensitive weighted metrics.

Examples:

```bash
--sparam-weights 'diag=1;offdiag=0.2'
--sparam-weights 'all=0.2;S21=1;S12=0.8'
--sparam-weights 'offdiag=0.1;S11,S22,S33,S44=1;S21,S12=0.8'
--sparam-weights 'default=0.2;row1=1;col1=1'
--sparam-weights 'offdiag=0.25;S1*=1;S*1=1'
```

Available selectors:

- `all` or `default`: every S-parameter
- `diag`, `diagonal`, `return`, or `reflection`: `Sii`
- `offdiag`, `off-diagonal`, or `transmission`: `Sij` where `i != j`
- `upper`: all `Sij` where `i < j`
- `lower`: all `Sij` where `i > j`
- `rowN`, `outN`, or `outputN`: all `SNj`
- `colN`, `columnN`, `inN`, or `inputN`: all `SiN`
- Wildcards such as `S1*` or `S*1`
- Explicit groups such as `S11,S22,S33,S44`

Useful selection metrics:

- `evm_db`: EVM in dB
- `evm_pct`: EVM as a percentage
- `evm_rms`: RMS error vector magnitude normalized to RMS measured magnitude
- `max_abs`: worst absolute complex error
- `max_abs_db`: worst dB magnitude error, ignoring near-zero magnitudes
- `passivity.max_singular_value`: worst predicted S-matrix singular value
- `passivity.violating_points`: number of sampled frequency points with singular value above 1
- `rmse_abs`: global complex S-parameter RMSE
- `rmse_db`: dB magnitude RMSE, ignoring near-zero magnitudes
- `weighted_evm_db`: S-parameter-weighted EVM in dB
- `weighted_evm_pct`: S-parameter-weighted EVM as a percentage
- `weighted_evm_rms`: S-parameter-weighted EVM ratio
- `weighted_rmse_abs`: S-parameter-weighted complex RMSE

Set `--mode grid` to exhaustively test all combinations, or keep the default
`--mode random --max-trials N` for direct hyperparameter optimization over a
larger search space.

Use `--require-passive` when passivity is a hard acceptance criterion. The
sweep still records every trial, but `best_model/` is selected from the passive
trials only. For softer limits, use `--max-passivity-violations` or
`--max-passivity-sigma` with any error-based `--selection-metric`.

For faster sweeps, use `--jobs N` to train independent trials in parallel and
set `--trial-worst-plots 0` to skip per-trial PDFs. Because `best_model/` is
promoted from the best completed trial by default, its plot set comes from
`--trial-worst-plots`. Use `--retrain-best` when you want the older behavior:
the winning configuration is fit again after the sweep and `--worst-plots`
controls that final model's verification plots. After each sweep, the summary
links a diagnostics PDF and CSV under `sweep_diagnostics/` comparing error
metrics against each swept option. Passivity-failing trials are shown in red on
those plots and are excluded from the grouped mean values in the diagnostic
CSV/PDF; the CSV records how many samples were excluded for each setting. If the
goal is fastest direct
Verilog-A simulation, add `--output-domain y` to the sweep command so the
winning model can be exported as a direct admittance-stamping n-port. During a
DNN sweep, parsed MDIF blocks and prepared feature/target matrices are cached
inside each process, so repeated trials with the same data and frequency
transform do not rebuild the same training arrays.

### Options

| Option Name | Description | Example |
| ------------------------------- | --- | ------------------------------------------------ |
| <nobr><code>--activation-options LIST</code></nobr> | Comma-separated activation functions to try. Available values are `tanh` and `relu`. Default: `tanh,relu`. | <nobr><code>--activation-options tanh,relu</code></nobr> |
| <nobr><code>--batch-size INT</code></nobr> | Batch size per trial. Default: `256`. | <nobr><code>--batch-size 256</code></nobr> |
| <nobr><code>--epochs INT</code></nobr> | Maximum epochs per trial and for the final best-model retrain. Default: `2000`. | <nobr><code>--epochs 1200</code></nobr> |
| <nobr><code>--freq-transform-options LIST</code></nobr> | Comma-separated frequency transforms to try. Available values are `log`, `linear`, and `log-linear`. Default: `log,log-linear`. | <nobr><code>--freq-transform-options log,log-linear</code></nobr> |
| <nobr><code>--hidden-layer-options LIST</code></nobr> | Semicolon-separated DNN layouts to try. Use commas inside one layout and semicolons between layouts. Default: `64,64;128,128,64;128,128,128;256,128,64`. | <nobr><code>--hidden-layer-options '64,64;128,128,64'</code></nobr> |
| <nobr><code>--holdout-fraction FLOAT</code></nobr> | Verification holdout fraction if split values are absent. Default: `0.2`. | <nobr><code>--holdout-fraction 0.2</code></nobr> |
| <nobr><code>--jobs INT</code></nobr> | Number of sweep trials to train in parallel. Use up to the number of physical cores and lower it if memory use gets high. Default: `1`. | <nobr><code>--jobs 4</code></nobr> |
| <nobr><code>--keep-trial-models</code></nobr> | Keep full per-trial model directories under `trials/`. By default, each trial keeps lightweight summary and plot artifacts while large model files are removed. | <nobr><code>--keep-trial-models</code></nobr> |
| <nobr><code>--learning-rates LIST</code></nobr> | Comma-separated Adam learning rates to try. Default: `0.001,0.002,0.005`. | <nobr><code>--learning-rates 0.001,0.002,0.005</code></nobr> |
| <nobr><code>--loss-interval INT</code></nobr> | Full train/verification loss check interval in epochs for each trial. Higher values can speed large sweeps because validation is scored less often. Default: `1`. | <nobr><code>--loss-interval 5</code></nobr> |
| <nobr><code>--max-passivity-sigma FLOAT</code></nobr> | Only consider trials whose worst predicted S-matrix singular value is at or below this value when selecting `best_model/`. | <nobr><code>--max-passivity-sigma 1.000001</code></nobr> |
| <nobr><code>--max-passivity-violations INT</code></nobr> | Only consider trials with this many or fewer passivity-violating frequency points when selecting `best_model/`. | <nobr><code>--max-passivity-violations 0</code></nobr> |
| <nobr><code>--max-trials INT</code></nobr> | Maximum number of candidate configurations to evaluate. In `random` mode this limits the random sample; in `grid` mode it truncates the product list. Default: `24`. | <nobr><code>--max-trials 40</code></nobr> |
| <nobr><code>--mdif PATH</code></nobr> | Required. Input MDIF. Same meaning as in `train`. | <nobr><code>--mdif train_verify.mdif</code></nobr> |
| <nobr><code>--mode {grid,random}</code></nobr> | Search strategy. `grid` evaluates combinations in deterministic product order; `random` samples combinations from the full grid. Default: `random`. | <nobr><code>--mode random</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | Required. Sweep output directory. Contains `dnn_sweep_results.csv`, `dnn_best_config.json`, and `best_model/`. | <nobr><code>--out-dir dnn_sweep</code></nobr> |
| <nobr><code>--output-domain {s,y}</code></nobr> | Fixed training target domain for every trial. Use `y` when the sweep is intended to produce a faster direct Verilog-A model. Default: `s`. | <nobr><code>--output-domain y</code></nobr> |
| <nobr><code>--parameter-names LIST</code></nobr> | Comma-separated model input variable names. Same meaning as in `train`. | <nobr><code>--parameter-names W,L,H</code></nobr> |
| <nobr><code>--patience INT</code></nobr> | Early-stopping patience per trial. Default: `200`. | <nobr><code>--patience 150</code></nobr> |
| <nobr><code>--progress-interval INT</code></nobr> | Console progress update interval in epochs for each trial and for the optional final best-model retrain. Updates redraw one terminal status line. Use `0` to disable. Default: `25`. | <nobr><code>--progress-interval 10</code></nobr> |
| <nobr><code>--require-passive</code></nobr> | Only consider trials with zero passivity-violating frequency points when selecting `best_model/`. Equivalent to `--max-passivity-violations 0` unless a stricter value is supplied. | <nobr><code>--require-passive</code></nobr> |
| <nobr><code>--retrain-best</code></nobr> | Retrain the selected best configuration at the end of the sweep instead of using the best completed trial model promoted during the sweep. Use this when you want `--worst-plots` to apply only to the final model. | <nobr><code>--retrain-best</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | Base random seed for holdout splitting, random candidate selection, initialization, and minibatch order. With the default `--trial-seed-mode fixed`, every trial uses this exact seed for apples-to-apples comparisons. Default: `1234`. | <nobr><code>--seed 1234</code></nobr> |
| <nobr><code>--selection-metric NAME</code></nobr> | Metric minimized when choosing the best trial. Options include `evm_pct`, `rmse_abs`, passivity metrics, and weighted metrics such as `weighted_evm_pct` and `weighted_rmse_abs`. Default: `rmse_abs`. | <nobr><code>--selection-metric weighted_evm_pct</code></nobr> |
| <nobr><code>--sparam-weights SPEC</code></nobr> | Optional S-parameter loss and ranking weights used by every sweep trial. Use this with weighted selection metrics to rank by the same priorities used during training. Weights are normalized to mean 1 internally. | <nobr><code>--sparam-weights 'all=0.2;S21=1;S12=0.8'</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | Split `VAR` name for combined MDIF files. Default: `dataset`. | <nobr><code>--split-var dataset</code></nobr> |
| <nobr><code>--target-z0 FLOAT</code></nobr> | Reference impedance used when `--output-domain y` converts S-parameter MDIF data to Y targets. Use the MDIF reference impedance. Default: `50.0`. | <nobr><code>--target-z0 50</code></nobr> |
| <nobr><code>--train-values LIST</code></nobr> | Comma-separated training split values. Default: `train,training`. | <nobr><code>--train-values train,training</code></nobr> |
| <nobr><code>--trial-seed-mode {fixed,indexed}</code></nobr> | Controls the seed used inside each sweep trial. `fixed` uses `--seed` for every trial so repeated candidates compare directly across sweeps. `indexed` restores the older `--seed + trial_number` behavior. Default: `fixed`. | <nobr><code>--trial-seed-mode fixed</code></nobr> |
| <nobr><code>--trial-worst-plots INT</code></nobr> | Number of lightweight worst-case S/Y PDF pairs generated and linked for each sweep trial. Default: `1`. | <nobr><code>--trial-worst-plots 1</code></nobr> |
| <nobr><code>--verification-mdif PATH</code></nobr> | Optional separate verification MDIF. Same meaning as in `train`. | <nobr><code>--verification-mdif verify.mdif</code></nobr> |
| <nobr><code>--verify-values LIST</code></nobr> | Comma-separated verification split values. Default: `verify,verification,test,validation`. | <nobr><code>--verify-values verification,test</code></nobr> |
| <nobr><code>--worst-plots INT</code></nobr> | Number of worst-case S/Y PDF pairs generated for the final `best_model/` only when `--retrain-best` is used. Without `--retrain-best`, `best_model/` is copied from a completed trial and uses `--trial-worst-plots`. Default: `6`. | <nobr><code>--worst-plots 6</code></nobr> |

## Post-Run Sweep Reranking

Passivity is computed and saved for every sweep trial, regardless of the
selection metric used during the original run. If you later decide that the
best model should be the lowest-error passive candidate, rerank the existing
sweep instead of rerunning the whole optimization:

```bash
python3 dnn.py rerank-sweep \
  --sweep-dir dnn_sweep \
  --selection-metric weighted_evm_pct \
  --require-passive
```

This writes `dnn_reranked_sweep_results.csv`,
`dnn_reranked_sweep_summary.md`, `dnn_reranked_best_config.json`, and refreshed
diagnostic artifacts under `sweep_diagnostics/`. The reranker accepts both
current `dnn_sweep_results.csv` folders and older `sweep_results.csv` folders.

If the original sweep used `--keep-trial-models`, the selected model can be
copied without retraining:

```bash
python3 dnn.py rerank-sweep \
  --sweep-dir dnn_sweep \
  --selection-metric weighted_evm_pct \
  --require-passive \
  --promote-best
```

Without kept trial models, reranking still identifies the winning
configuration, but the script cannot copy deleted `model.npz` files. In that
case, retrain only the selected configuration rather than rerunning the full
sweep.

### Options

| Option Name | Description | Example |
| ------------------------------------------------------ | --- | ------------------------------------------------------------------------ |
| <nobr><code>--best-model-dir PATH</code></nobr> | Destination for `--promote-best`. Default: `<sweep-dir>/best_model_reranked`. | <nobr><code>--best-model-dir dnn_sweep/best_model_passive</code></nobr> |
| <nobr><code>--max-passivity-sigma FLOAT</code></nobr> | Only consider trials whose worst predicted S-matrix singular value is at or below this value. | <nobr><code>--max-passivity-sigma 1.000001</code></nobr> |
| <nobr><code>--max-passivity-violations INT</code></nobr> | Only consider trials with this many or fewer passivity-violating frequency points. | <nobr><code>--max-passivity-violations 0</code></nobr> |
| <nobr><code>--overwrite</code></nobr> | Allow `--promote-best` to replace an existing `--best-model-dir`. | <nobr><code>--overwrite</code></nobr> |
| <nobr><code>--promote-best</code></nobr> | Copy the selected trial model to `--best-model-dir` if that trial still contains `model.npz` and `metadata.json`. Requires the original sweep to have used `--keep-trial-models`. | <nobr><code>--promote-best</code></nobr> |
| <nobr><code>--replace-current-best</code></nobr> | Overwrite `<sweep-dir>/best_model` with the selected trial model if the trial model files are available. | <nobr><code>--replace-current-best</code></nobr> |
| <nobr><code>--require-passive</code></nobr> | Only consider trials with zero passivity-violating frequency points. Equivalent to `--max-passivity-violations 0` unless a stricter value is supplied. | <nobr><code>--require-passive</code></nobr> |
| <nobr><code>--selection-metric NAME</code></nobr> | Metric minimized after filtering. Use this to choose the lowest-error passive model, such as `weighted_evm_pct` with `--require-passive`. | <nobr><code>--selection-metric weighted_evm_pct</code></nobr> |
| <nobr><code>--sweep-dir PATH</code></nobr> | Required. Existing DNN sweep or optimize output directory. | <nobr><code>--sweep-dir dnn_sweep</code></nobr> |

## Predict

Predict new parameter blocks after training:

```bash
python3 dnn.py predict \
  --model-dir dnn_model \
  --mdif new_parameter_blocks.mdif \
  --out-mdif predicted.mdif
```

For prediction, the input MDIF must provide the geometry `VAR`s and frequency
grid. Placeholder S-parameter columns are acceptable; their values are ignored.

### Options

| Option Name | Description | Example |
| ------------------------------- | --- | ------------------------------------------------ |
| <nobr><code>--mdif PATH</code></nobr> | Required. MDIF containing geometry/process `VAR`s and frequency grids for prediction. S-parameter values in this file are ignored except for parsing the frequency table shape. | <nobr><code>--mdif new_parameter_blocks.mdif</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | Required. Directory containing a trained `model.npz` and `metadata.json`. | <nobr><code>--model-dir dnn_model</code></nobr> |
| <nobr><code>--out-mdif PATH</code></nobr> | Required. Output MDIF containing predicted S-parameters. | <nobr><code>--out-mdif predicted.mdif</code></nobr> |

## ADS MDIF Export

After training, export a parameterized S-parameter table that ADS can use
directly through an MDIF-capable data-based n-port or data access component.

The safest export is template driven: provide an MDIF containing the exact
geometry `VAR`s and frequency grids you want available in ADS. Placeholder
S-parameter values are accepted and ignored.

```bash
python3 dnn.py export-ads-mdif \
  --model-dir dnn_model \
  --out-dir ads_export \
  --template-mdif ads_sweep_template.mdif
```

You can also generate a rectangular parameter/frequency grid directly:

```bash
python3 dnn.py export-ads-mdif \
  --model-dir dnn_model \
  --out-dir ads_export \
  --parameter-grid W=0.40mm:0.80mm:9 \
  --parameter-grid L=1.00mm:1.60mm:7 \
  --freqs 1GHz:20GHz:401
```

Exported files:

- `surrogate_ads.mdif`: predicted parameterized S-parameter MDIF
- `ads_model_manifest.json`: model and export metadata
- `ADS_README.md`: ADS usage note for the exported package

Copy the MDIF into the ADS workspace data area, point the data-based component
at that file, and drive the same schematic variable names as the MDIF `VAR`s`.
For optimization, constrain ADS variables inside the exported grid; ADS will
interpolate between sampled points rather than evaluate the neural network.

### Options

| Option Name | Description | Example |
| ------------------------------------------------------ | --- | ------------------------------------------------------------------------ |
| <nobr><code>--freqs SPEC</code></nobr> | Frequency grid used with `--parameter-grid`. `SPEC` can be a comma list or `start:stop:count`. | <nobr><code>--freqs 1GHz:20GHz:401</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | Required. Directory containing a trained `model.npz` and `metadata.json`. | <nobr><code>--model-dir dnn_model</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | Required. Output directory for `surrogate_ads.mdif`, `ads_model_manifest.json`, and `ADS_README.md`. | <nobr><code>--out-dir ads_export</code></nobr> |
| <nobr><code>--output-name NAME</code></nobr> | Output MDIF file name. Default: `surrogate_ads.mdif`. | <nobr><code>--output-name dnn_ads.mdif</code></nobr> |
| <nobr><code>--parameter-grid NAME=SPEC</code></nobr> | Optional repeatable grid definition. `SPEC` can be a comma list or `start:stop:count`. Repeat once for every model parameter when not using `--template-mdif`. | <nobr><code>--parameter-grid W=0.40mm:0.80mm:9</code></nobr> |
| <nobr><code>--template-mdif PATH</code></nobr> | Optional. MDIF containing the exact geometry and frequency blocks to evaluate for ADS. S-parameter values are ignored. Use this when you already know the ADS optimization grid. | <nobr><code>--template-mdif ads_sweep_template.mdif</code></nobr> |

## ADS ANN Export

Use `export-ads-ann` when you want ADS ANN to train/extract the neural model
natively and emit the ADS ANN artifacts, including Verilog-A-oriented `.inc`,
C `.c`, text equation `.equation`, `.struc`, and `.scale` files.

```bash
python3 dnn.py export-ads-ann \
  --mdif train_verify.mdif \
  --model-dir dnn_sweep/best_model \
  --out-dir dnn_ads_ann \
  --ads-iterations 1000 \
  --ads-output-format all
```

The export writes `ads_ann_training.csv`, optional
`ads_ann_verification.csv`, `ads_ann_manifest.json`, `train_ads_ann.py`, and
`ADS_ANN_README.md`. Run `train_ads_ann.py` with the ADS Python interpreter on
a licensed ADS machine. This path retrains the network in ADS ANN; it does not
import the local NumPy `model.npz` weights.

ADS reference used:

- Primary example: Keysight ADS 2026 Update 2.1 ANN Python Documentation,
  `doc/ann/examples/inmemory_extraction.py`
- HTML page: `doc/ann/html/examples/ex_inmemory_extraction.html`
- API references: `doc/ann/html/reference/ann/annsetup.html`,
  `doc/ann/html/reference/ann/index.html`, `outputformat.html`,
  `modeleroptimizer.html`, `networktrainingtype.html`, and
  `neuronactivationfunctiontype.html` under the same reference folder

That example establishes the DataFrame-based flow used by `train_ads_ann.py`:
configure `keysight.ads.ann.AnnSetup`, call `ann.configure_setup(setup)`, train
with `ann.auxiliary_functions.extract_inmemory(...)`, then verify with
`ann.auxiliary_functions.simulate_inmemory(...)` after loading the generated
`.struc`/`.scale` files. If your ADS machine uses a different release, compare
the same pages under its installed `$HPEESOF_DIR/doc/ann` tree.

For weighted fitting context, Keysight's
`doc/ann/examples/training_error_weighting.py` demonstrates sample/row error
weighting. The documented example does not establish direct per-output
S-parameter loss weights for a multi-output ANN, so the export records
S-parameter weights in the manifest but does not apply them inside ADS ANN.

Portable workflow:

1. Run `train` or `optimize` on any machine.
2. Run `export-ads-ann` on that same machine with `--model-dir` pointed at the
   trained model directory, or at an optimize run's `best_model/` directory.
3. Copy the entire `--out-dir` folder to the ADS machine.
4. On the ADS machine, run `train_ads_ann.py` with ADS Python to generate the
   native ADS ANN files.

Schematic use:

1. Create a small ADS wrapper cell. For Verilog-A, include/call the generated
   `.inc` file. For an SDD or equation-based wrapper, use the generated
   `.equation` file as the ANN equation source/reference.
2. Feed the wrapper inputs from the `input_columns` in `ads_ann_manifest.json`.
   For example, `freq_log10_hz` means the wrapper must pass `log10(freq_hz)`.
3. Interpret `output_columns` as the final fine S-parameters, with all real
   columns followed by the matching imaginary columns.
4. Convert the final complex S-matrix to a circuit relation before driving the
   schematic pins. For reference impedance `Z0`, use
   `Y = (I - S) * inverse(I + S) / Z0`, then `Iport = Y * Vport`.
5. Validate the wrapper in an S-parameter or AC simulation before circuit
   optimization.

### Options

| Option Name | Description | Example |
| ------------------------------------------------------ | --- | ------------------------------------------------------------------------ |
| <nobr><code>--activation {tanh,relu}</code></nobr> | Hidden-layer activation requested for ADS ANN. `tanh` maps to `HYPERBOLIC_TANGENT`; `relu` maps to `RELU`. Default: `tanh`. | <nobr><code>--activation tanh</code></nobr> |
| <nobr><code>--ads-hidden-layers INT</code></nobr> | Override ADS `AnnSetup.num_hidden_layers`. If omitted, this is derived from `--hidden-layers`. | <nobr><code>--ads-hidden-layers 3</code></nobr> |
| <nobr><code>--ads-iterations INT</code></nobr> | ADS ANN maximum training iterations. Default: `500`. | <nobr><code>--ads-iterations 1000</code></nobr> |
| <nobr><code>--ads-network-training-type {standard,adjoint,classification}</code></nobr> | ADS ANN training type. Use `standard` for normal S-parameter regression. Default: `standard`. | <nobr><code>--ads-network-training-type standard</code></nobr> |
| <nobr><code>--ads-neurons-per-layer INT</code></nobr> | Override ADS `AnnSetup.num_neurons_per_layer`. If omitted, this is derived from the average of `--hidden-layers`. | <nobr><code>--ads-neurons-per-layer 128</code></nobr> |
| <nobr><code>--ads-optimizer {quasi-newton,bayesian-regularization}</code></nobr> | ADS ANN modeler optimizer. `bayesian-regularization` can improve generalization at additional training cost. Default: `quasi-newton`. | <nobr><code>--ads-optimizer bayesian-regularization</code></nobr> |
| <nobr><code>--ads-output-format {all,verilog-a,c-code,equation,struct-scale}</code></nobr> | ADS ANN native artifact format. `all` requests every documented output. Default: `all`. | <nobr><code>--ads-output-format all</code></nobr> |
| <nobr><code>--ads-training-stop-tolerance FLOAT</code></nobr> | ADS ANN RMSE stop tolerance. Use `0` to rely on the iteration limit. Default: `0.0`. | <nobr><code>--ads-training-stop-tolerance 0</code></nobr> |
| <nobr><code>--freq-transform {log,linear,log-linear}</code></nobr> | Frequency input transform used in the ADS ANN training CSV. Default: `log`. | <nobr><code>--freq-transform log-linear</code></nobr> |
| <nobr><code>--hidden-layers LIST</code></nobr> | Local layout used to derive ADS's uniform hidden-layer count and width when ADS-specific overrides are omitted. | <nobr><code>--hidden-layers 128,128,64</code></nobr> |
| <nobr><code>--holdout-fraction FLOAT</code></nobr> | Fraction of blocks reserved for verification when split values are absent. Default: `0.2`. | <nobr><code>--holdout-fraction 0.2</code></nobr> |
| <nobr><code>--mdif PATH</code></nobr> | Required. Input MDIF. If `--verification-mdif` is not supplied, this file should contain both training and verification blocks. | <nobr><code>--mdif train_verify.mdif</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | Optional trained model directory, or an optimize run's `best_model/` directory. The exporter uses `metadata.json` for parameter names, S-parameter labels, frequency transform, activation, and hidden-layer layout. Weights are not imported into ADS ANN. | <nobr><code>--model-dir dnn_sweep/best_model</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | Required. Output directory for the ADS ANN package files. | <nobr><code>--out-dir dnn_ads_ann</code></nobr> |
| <nobr><code>--output-prefix NAME</code></nobr> | Prefix for native ADS ANN outputs such as `.inc`, `.c`, `.equation`, `.scale`, and `.struc`. Default: `dnn_ads_ann`. | <nobr><code>--output-prefix dnn_filter_ann</code></nobr> |
| <nobr><code>--parameter-names LIST</code></nobr> | Comma-separated geometry/process variables used as ADS ANN inputs. If omitted, common numeric `VAR`s are inferred. | <nobr><code>--parameter-names W,L,H</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | ADS ANN seed plus local holdout split seed. Default: `1234`. | <nobr><code>--seed 1234</code></nobr> |
| <nobr><code>--sparam-weights SPEC</code></nobr> | Optional S-parameter weights to record in the ADS ANN manifest. Defaults to `metadata.json` values when `--model-dir` is supplied. The generated ADS ANN script records but does not apply per-output weights because the documented ADS ANN API does not expose them. | <nobr><code>--sparam-weights 'diag=1;offdiag=0.2'</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | `VAR` used to split combined MDIF files. Default: `dataset`. | <nobr><code>--split-var dataset</code></nobr> |
| <nobr><code>--train-values LIST</code></nobr> | Comma-separated split values that identify training blocks. Default: `train,training`. | <nobr><code>--train-values train,training</code></nobr> |
| <nobr><code>--verification-mdif PATH</code></nobr> | Optional separate verification MDIF. When supplied, every block in `--mdif` is treated as training data. | <nobr><code>--verification-mdif verify.mdif</code></nobr> |
| <nobr><code>--verify-values LIST</code></nobr> | Comma-separated split values that identify verification blocks. Default: `verify,verification,test,validation`. | <nobr><code>--verify-values verification,test</code></nobr> |

## Direct Verilog-A Export

Use `export-veriloga` when you want to embed the trained local DNN weights
directly in a Verilog-A n-port instead of exporting a sampled MDIF table or
retraining with ADS ANN.

For ADS solve speed, first train or optimize the DNN with `--output-domain y`,
then export it normally. That formulation stores the learned response as
Y-parameters, so the generated Verilog-A stamps admittance directly instead of
performing a complex S-to-Y matrix inversion during every simulator evaluation:

```bash
python3 dnn.py train \
  --mdif train_verify.mdif \
  --out-dir dnn_y_model \
  --parameter-names W,L \
  --output-domain y \
  --target-z0 50
```

```bash
python3 dnn.py export-veriloga \
  --model-dir dnn_y_model \
  --out-dir dnn_veriloga \
  --module-name my_dnn_4port
```

The export writes `<module>.va`, `veriloga_manifest.json`, and
`VERILOGA_README.md`. The generated module evaluates the neural network at the
simulator frequency and contributes the corresponding port currents. For
S-output models it reconstructs the complex S-matrix and converts it to
admittance with `Y = (I - S) * inverse(I + S) / Z0`. For Y-output models it
stamps the predicted admittance directly. In both cases, the model must contain
a complete square S-parameter matrix, such as all 16 terms for a four-port.

By default, the exporter folds input standardization into the first neural
layer and output scaling into the final neural layer. This reduces explicit
arithmetic in the ADS Verilog-A analog block without changing the evaluated
model.

This path is intended for S-parameter and small-signal AC use in ADS. It uses
`$freq` as the default frequency expression; if your ADS Verilog-A environment
uses a different frequency symbol, set `--frequency-expression` during export.
Validate the generated component against `predicted_verification.mdif` before
using it in optimization.

Only the electrical ports and geometry/process parameters need to be provided
for normal ADS use. Generated scale parameters named like `*_input_scale` are
unit-conversion constants with export-time defaults; leave them unchanged unless
you intentionally change the ADS unit convention. The `freq_hz` and
`freq_log10_hz` feature names in `veriloga_manifest.json` or
`VERILOGA_README.md` are computed inside the Verilog-A module from simulator
frequency; do not add external pins or parameters for them.

If the MDIF training parameters were scaled dimensionless values but the ADS
schematic uses base units, export with `--parameter-input-scales`. For example,
if the MDIF used `W=0.4` to mean 0.4 microns and ADS will pass `W=0.4e-6`
meters, use `--parameter-input-scales W=1um`. The generated Verilog-A then
feeds `W / W_input_scale` to the neural network, so the ADS-facing parameter can
remain in meters.

### Options

| Option Name | Description | Example |
| ------------------------------------------------------ | --- | ------------------------------------------------------------------------ |
| <nobr><code>--frequency-expression EXPR</code></nobr> | Verilog-A expression for simulator frequency in Hz. Default: `$freq`. Change this only if your ADS Verilog-A release requires a different frequency expression. | <nobr><code>--frequency-expression '$freq'</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | Required. Directory containing a trained `model.npz` and `metadata.json`. | <nobr><code>--model-dir dnn_sweep/best_model</code></nobr> |
| <nobr><code>--module-name NAME</code></nobr> | Optional Verilog-A module name. If omitted, the exporter derives one from the output directory. | <nobr><code>--module-name my_dnn_4port</code></nobr> |
| <nobr><code>--no-fold-scalers</code></nobr> | Debug option. Keep input/output standardization as explicit Verilog-A arithmetic instead of folding it into the first and final neural layers. Leaving this unset is faster. | <nobr><code>--no-fold-scalers</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | Required. Output directory for `<module>.va`, `veriloga_manifest.json`, and `VERILOGA_README.md`. | <nobr><code>--out-dir dnn_veriloga</code></nobr> |
| <nobr><code>--parameter-input-scales SPEC</code></nobr> | Optional ADS/base-unit scale for each geometry/process parameter before the value is fed to the trained model. Use `NAME=SCALE` entries separated by commas or semicolons, `all=SCALE` for every parameter, or a bare scale such as `1um` for every parameter. Default: `1.0` for all parameters. | <nobr><code>--parameter-input-scales W=1um,L=1um</code></nobr> |
| <nobr><code>--z0 FLOAT</code></nobr> | Reference impedance used when exporting an S-output model and converting predicted S-parameters to admittance. Direct-Y models use the saved training `--target-z0` metadata instead. Default: `50.0`. | <nobr><code>--z0 50</code></nobr> |

## ADS Note

The `export-ads-mdif` command is the lowest-risk direct ADS handoff. It exports
the trained DNN response onto a dense parameter/frequency table, so ADS can use
normal MDIF interpolation during circuit optimization without embedding Python
or NumPy in the simulator. The `export-veriloga` command embeds the trained
local DNN weights into a Verilog-A n-port, which avoids ADS ANN retraining but
should be validated in the target ADS Verilog-A compiler. The `export-ads-ann`
command is the native ADS ANN handoff for generating ADS ANN
Verilog-A/C/equation artifacts on an ADS machine.
