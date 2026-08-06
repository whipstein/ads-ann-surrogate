# Neuro-TF MDIF Trainer

This is a self-contained prototype for training a Neuro-transfer-function
surrogate from parameterized S-parameter MDIF data.

Model structure:

```text
geometry/process VARs -> small neural network -> rational TF coefficients -> S-parameters
```

The rational transfer functions use fixed stable poles, so coefficient
extraction for each geometry is linear least squares. The neural network then
learns the geometry-to-coefficients map.

## RC2 Layout

The command entry point is `neuro_tf.py`. The Neuro-TF fitting calculation lives
in `model.py`; shared MDIF I/O, plotting, metrics, sweep diagnostics,
summaries, and export helpers live in `../common/surrogate_common.py`.

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
python3 neuro_tf.py inspect-mdif \
  --mdif train_verify.mdif
```

### Options

| Option Name | Description | Example |
| ------------------------------- | --- | ------------------------------------------------ |
| <nobr><code>--mdif PATH</code></nobr> | Required. MDIF file to inspect. | <nobr><code>--mdif train_verify.mdif</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | Split variable name to count in the summary. Default: `dataset`. | <nobr><code>--split-var dataset</code></nobr> |

## Usage

Train one Neuro-TF model with `train`:

```bash
python3 neuro_tf.py train \
  --mdif train_verify.mdif \
  --out-dir neuro_tf_model \
  --parameter-names W,L \
  --order 10 \
  --hidden-layers 64,64
```

Outputs:

- `model.npz` and `metadata.json`: trained Neuro-TF model
- `predicted_verification.mdif`: model predictions at verification points
- `verification_metrics.csv`: per-block and per-S-parameter errors, including EVM
- `verification_summary.json`: global error, passivity summary, and plot paths
- `training_history.csv`: neural training loss history
- `training_history.pdf`: train/verification loss versus epoch convergence plot
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
default. Each S-parameter plot contains a Smith/complex response grid, a
magnitude grid, an unwrapped phase grid, and an error-focus page. The same cases
are also converted to Y-parameters and written under `worst_case_y_plots/`,
where modeled and measured admittance are shown as real/imaginary frequency
plots. Use `--worst-plots 0` to skip both plot sets during large experiments.
Each single `train` run also writes `training_summary.md`, which collects the
chosen settings, final loss values, verification metrics, passivity summary,
and links to the generated S- and Y-parameter worst-case plots.

### Options

| Option Name | Description | Example |
| ------------------------------- | --- | ------------------------------------------------ |
| <nobr><code>--activation {tanh,relu}</code></nobr> | Hidden-layer activation. `tanh` is usually smoother for microwave response fitting; `relu` can be useful for larger datasets. Default: `tanh`. | <nobr><code>--activation tanh</code></nobr> |
| <nobr><code>--batch-size INT</code></nobr> | Number of training geometries per Adam update. The implementation clamps this to the number of available training blocks. Default: `64`. | <nobr><code>--batch-size 64</code></nobr> |
| <nobr><code>--debug</code></nobr> | Print common diagnostics and show Python tracebacks for failed commands. | <nobr><code>--debug</code></nobr> |
| <nobr><code>--epochs INT</code></nobr> | Maximum Adam training epochs. Early stopping may stop before this value. Default: `2000`. | <nobr><code>--epochs 2000</code></nobr> |
| <nobr><code>--hidden-layers LIST</code></nobr> | Comma-separated hidden layer sizes for the coefficient neural network. Default: `64,64`. | <nobr><code>--hidden-layers 64,64</code></nobr> |
| <nobr><code>--holdout-fraction FLOAT</code></nobr> | Fraction of blocks to reserve for verification when no split values are found in a combined MDIF. Default: `0.2`. | <nobr><code>--holdout-fraction 0.25</code></nobr> |
| <nobr><code>--learning-rate FLOAT</code></nobr> | Adam optimizer step size. Lower values are safer; higher values may converge faster but can overshoot. Default: `0.002`. | <nobr><code>--learning-rate 0.002</code></nobr> |
| <nobr><code>--loss-interval INT</code></nobr> | Full train/verification loss check interval in epochs. Increasing this reduces full-dataset scoring overhead during long runs while early stopping still uses epoch-based patience. Default: `1`. | <nobr><code>--loss-interval 5</code></nobr> |
| <nobr><code>--mdif PATH</code></nobr> | Required. Input MDIF. If `--verification-mdif` is not supplied, this file should contain both training and verification blocks, typically separated by a `VAR` such as `dataset=train` or `dataset=verification`. | <nobr><code>--mdif train_verify.mdif</code></nobr> |
| <nobr><code>--order INT</code></nobr> | Number of fixed stable rational poles used for each S-parameter transfer function. Higher values can fit sharper frequency behavior but increase coefficient count and NN output dimension. Default: `10`. | <nobr><code>--order 12</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | Required. Output directory for `model.npz`, `metadata.json`, `training_summary.md`, `predicted_verification.mdif`, metrics CSV/JSON files, training history, and S/Y worst-case plot PDFs. | <nobr><code>--out-dir neuro_tf_model</code></nobr> |
| <nobr><code>--parameter-names LIST</code></nobr> | Comma-separated geometry/process variable names to use as neural-network inputs. If omitted, the trainer infers numeric `VAR`s common to all blocks, excluding the split variable. | <nobr><code>--parameter-names W,L,H</code></nobr> |
| <nobr><code>--patience INT</code></nobr> | Early-stopping patience measured in epochs without validation-loss improvement. Use `0` to disable early stopping. Default: `200`. | <nobr><code>--patience 200</code></nobr> |
| <nobr><code>--pole-damping FLOAT</code></nobr> | Real-part damping factor for the fixed pole grid. Larger values make poles more damped and smoother; smaller values can follow sharper resonances but may be more sensitive. Default: `0.18`. | <nobr><code>--pole-damping 0.18</code></nobr> |
| <nobr><code>--progress-interval INT</code></nobr> | Console progress update interval in epochs. Updates redraw one terminal status line and include epoch count, elapsed time, and loss values when that epoch also matches `--loss-interval`. Use `0` to disable. Default: `25`. | <nobr><code>--progress-interval 10</code></nobr> |
| <nobr><code>--ridge FLOAT</code></nobr> | Ridge regularization used during linear least-squares TF coefficient fitting. Increase this if coefficient fits become noisy or ill-conditioned. Default: `1e-8`. | <nobr><code>--ridge 1e-8</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | Random seed for holdout splitting and neural-network initialization. Default: `1234`. | <nobr><code>--seed 1234</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | Name of the `VAR` used to split a combined MDIF. Default: `dataset`. | <nobr><code>--split-var dataset</code></nobr> |
| <nobr><code>--train-values LIST</code></nobr> | Comma-separated values of `--split-var` that identify training blocks. Default: `train,training`. | <nobr><code>--train-values train,training</code></nobr> |
| <nobr><code>--verification-mdif PATH</code></nobr> | Optional. Separate MDIF containing verification blocks. When supplied, every block in `--mdif` is treated as training data and every block in this file is treated as verification data. | <nobr><code>--verification-mdif verify.mdif</code></nobr> |
| <nobr><code>--verify-values LIST</code></nobr> | Comma-separated values of `--split-var` that identify verification blocks. Default: `verify,verification,test,validation`. | <nobr><code>--verify-values verification,test</code></nobr> |
| <nobr><code>--worst-plots INT</code></nobr> | Number of worst verification fits to render as PDFs. Each selected case gets an S-parameter plot and a Y-parameter implementation-view plot. Ranking uses max absolute complex response error, with RMSE also reported in the title and plot index CSV. Use `0` to skip plot generation. Default: `6`. | <nobr><code>--worst-plots 6</code></nobr> |

## Sweeping / Optimizing

Use `sweep` or its alias `optimize` to try multiple rational orders and neural
network settings. The command writes `neurotf_sweep_results.csv` and
`neurotf_sweep_summary.md`, chooses the best trial using `--selection-metric`,
and keeps the current best completed trial in `best_model/` as the sweep runs.
This avoids a final refit after all trials finish.

```bash
python3 neuro_tf.py optimize \
  --mdif train_verify.mdif \
  --out-dir neuro_tf_sweep \
  --parameter-names W,L,H \
  --orders 8,10,12,16 \
  --hidden-layer-options '32;64;64,64;128,64' \
  --activation-options tanh,relu \
  --learning-rates 0.001,0.002,0.005 \
  --mode random \
  --max-trials 40 \
  --selection-metric rmse_abs \
  --require-passive
```

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
CSV/PDF; the CSV records how many samples were excluded for each setting.

### Options

| Option Name | Description | Example |
| ------------------------------- | --- | ------------------------------------------------ |
| <nobr><code>--activation-options LIST</code></nobr> | Comma-separated activation functions to try. Available values are `tanh` and `relu`. Default: `tanh,relu`. | <nobr><code>--activation-options tanh,relu</code></nobr> |
| <nobr><code>--batch-size INT</code></nobr> | Batch size per trial. Default: `64`. | <nobr><code>--batch-size 64</code></nobr> |
| <nobr><code>--debug</code></nobr> | Print the selected candidate list, show tracebacks for failed trials, and include tracebacks in failed trial summaries. Use `--jobs 1` for the cleanest trace. | <nobr><code>--debug --jobs 1</code></nobr> |
| <nobr><code>--epochs INT</code></nobr> | Maximum epochs per trial and for the final best-model retrain. Default: `1200`. | <nobr><code>--epochs 1200</code></nobr> |
| <nobr><code>--hidden-layer-options LIST</code></nobr> | Semicolon-separated neural-network layouts to try. Use commas inside one layout and semicolons between layouts. Default: `32;64;64,64`. | <nobr><code>--hidden-layer-options '32;64,64;128,64'</code></nobr> |
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
| <nobr><code>--orders LIST</code></nobr> | Comma-separated rational pole counts to try. Default: `6,10,14`. | <nobr><code>--orders 8,10,12,16</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | Required. Sweep output directory. Contains `neurotf_sweep_results.csv`, `neurotf_best_config.json`, and `best_model/`. | <nobr><code>--out-dir neuro_tf_sweep</code></nobr> |
| <nobr><code>--parameter-names LIST</code></nobr> | Comma-separated model input variable names. Same meaning as in `train`. | <nobr><code>--parameter-names W,L,H</code></nobr> |
| <nobr><code>--patience INT</code></nobr> | Early-stopping patience per trial. Default: `150`. | <nobr><code>--patience 150</code></nobr> |
| <nobr><code>--pole-dampings LIST</code></nobr> | Comma-separated pole damping values to try. Default: `0.12,0.18,0.28`. | <nobr><code>--pole-dampings 0.12,0.18,0.28</code></nobr> |
| <nobr><code>--progress-interval INT</code></nobr> | Console progress update interval in epochs for each trial and for the optional final best-model retrain. Updates redraw one terminal status line. Use `0` to disable. Default: `25`. | <nobr><code>--progress-interval 10</code></nobr> |
| <nobr><code>--ridge-values LIST</code></nobr> | Comma-separated coefficient-fit ridge values to try. Default: `1e-10,1e-8,1e-6`. | <nobr><code>--ridge-values 1e-10,1e-8,1e-6</code></nobr> |
| <nobr><code>--require-passive</code></nobr> | Only consider trials with zero passivity-violating frequency points when selecting `best_model/`. Equivalent to `--max-passivity-violations 0` unless a stricter value is supplied. | <nobr><code>--require-passive</code></nobr> |
| <nobr><code>--retrain-best</code></nobr> | Retrain the selected best configuration at the end of the sweep instead of using the best completed trial model promoted during the sweep. Use this when you want `--worst-plots` to apply only to the final model. | <nobr><code>--retrain-best</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | Base random seed for holdout splitting, random candidate selection, initialization, and minibatch order. With the default `--trial-seed-mode fixed`, every trial uses this exact seed for apples-to-apples comparisons. Default: `1234`. | <nobr><code>--seed 1234</code></nobr> |
| <nobr><code>--selection-metric NAME</code></nobr> | Metric minimized when choosing the best trial. Options: `evm_db`, `evm_pct`, `evm_rms`, `max_abs`, `max_abs_db`, `passivity.max_singular_value`, `passivity.violating_points`, `rmse_abs`, and `rmse_db`. Default: `rmse_abs`. | <nobr><code>--selection-metric evm_pct</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | Split `VAR` name for combined MDIF files. Default: `dataset`. | <nobr><code>--split-var dataset</code></nobr> |
| <nobr><code>--train-values LIST</code></nobr> | Comma-separated training split values. Default: `train,training`. | <nobr><code>--train-values train,training</code></nobr> |
| <nobr><code>--trial-seed-mode {fixed,indexed}</code></nobr> | Controls the seed used inside each sweep trial. `fixed` uses `--seed` for every trial so repeated candidates compare directly across sweeps. `indexed` restores the older `--seed + trial_number` behavior. Default: `fixed`. | <nobr><code>--trial-seed-mode fixed</code></nobr> |
| <nobr><code>--trial-worst-plots INT</code></nobr> | Number of lightweight worst-case S/Y PDF pairs generated and linked for each sweep trial. Default: `1`. | <nobr><code>--trial-worst-plots 1</code></nobr> |
| <nobr><code>--verification-mdif PATH</code></nobr> | Optional separate verification MDIF. Same meaning as in `train`. | <nobr><code>--verification-mdif verify.mdif</code></nobr> |
| <nobr><code>--verify-values LIST</code></nobr> | Comma-separated verification split values. Default: `verify,verification,test,validation`. | <nobr><code>--verify-values verification,test</code></nobr> |
| <nobr><code>--worst-plots INT</code></nobr> | Number of worst-case S/Y PDF pairs generated for the final `best_model/` only when `--retrain-best` is used. Without `--retrain-best`, `best_model/` is copied from a completed trial and uses `--trial-worst-plots`. Default: `6`. | <nobr><code>--worst-plots 6</code></nobr> |

## Predict

Predict new parameter blocks after training:

```bash
python3 neuro_tf.py predict \
  --model-dir neuro_tf_model \
  --mdif new_parameter_blocks.mdif \
  --out-mdif predicted.mdif
```

### Options

| Option Name | Description | Example |
| ------------------------------- | --- | ------------------------------------------------ |
| <nobr><code>--mdif PATH</code></nobr> | Required. MDIF containing geometry/process `VAR`s and frequency grids for prediction. S-parameter values in this file are ignored except for parsing the frequency table shape. | <nobr><code>--mdif new_parameter_blocks.mdif</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | Required. Directory containing a trained `model.npz` and `metadata.json`. | <nobr><code>--model-dir neuro_tf_model</code></nobr> |
| <nobr><code>--out-mdif PATH</code></nobr> | Required. Output MDIF containing predicted S-parameters. | <nobr><code>--out-mdif predicted.mdif</code></nobr> |

## ADS Note

This is an offline trainer and verifier. It produces a differentiable
coefficient model plus MDIF predictions. For direct in-circuit ADS optimization,
the next integration step is to emit an ADS equation/SDD or Verilog-A wrapper
that evaluates the neural coefficient map and rational transfer functions for
the ADS simulator context you want to target.
