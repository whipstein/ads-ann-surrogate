# ADS ANN Surrogate Models

This repository contains offline Python tools for building surrogate models from
parameterized RF/microwave S-parameter MDIF data. The typical workflow is:

1. Export or assemble MDIF blocks for a swept device geometry or process corner.
2. Train a compact surrogate model that maps geometry/process variables and
   frequency to complex network response.
3. Verify the model against held-out MDIF blocks with numeric metrics,
   passivity checks, and worst-case plots.
4. Use the trained model to predict new MDIF data or export an ADS-ready
   package for circuit-level simulation and optimization.

The code is organized as a release-candidate refactor (`rc2`): each modeling
approach owns its fitting logic, while shared MDIF parsing, metrics, plotting,
sweep orchestration, and ADS export helpers live in `common/`.

## What It Builds

The repository provides three surrogate-model front ends:

| Model | Entry point | Best fit for | Basic idea |
| --- | --- | --- | --- |
| DNN | `dnn/dnn.py` | General parameterized S-parameter fitting when you want a direct neural response model. | A multilayer perceptron predicts S-parameters, or optionally Y-parameters, from geometry/process variables plus frequency features. |
| KBNN | `kbnn/kbnn.py` | Cases where a fast coarse model or lower-fidelity EM result is available. | A neural network learns the correction from coarse/prior response to fine/target response, or uses the coarse response as an input. |
| Neuro-TF | `neuro_tf/neuro_tf.py` | Smooth frequency responses where a rational transfer-function structure is useful. | Fixed stable poles define rational transfer functions; a neural network maps geometry/process variables to the fitted coefficients. |

All three tools read MDIF, train models, run sweeps, write verification
artifacts, and predict new response blocks. DNN and KBNN also include direct ADS
handoff commands for sampled MDIF export, native ADS ANN package export, and
Verilog-A n-port export.

## Repository Layout

```text
.
|-- common/
|   |-- surrogate_common.py      Shared MDIF, metrics, plotting, sweep, and export utilities
|   `-- README.md                Common support layer notes
|-- generate_points.py           Geometry/process point-set generator
|-- dnn/
|   |-- dnn.py                   Thin CLI wrapper
|   |-- model.py                 Direct DNN trainer, predictor, sweeper, and ADS exporters
|   |-- README.md                Full DNN command reference
|   `-- sample_training_verification.mdif
|-- kbnn/
|   |-- kbnn.py                  Thin CLI wrapper
|   |-- model.py                 KBNN trainer, predictor, sweeper, and ADS exporters
|   |-- README.md                Full KBNN command reference
|   |-- sample_fine.mdif
|   `-- sample_coarse.mdif
|-- neuro_tf/
|   |-- neuro_tf.py              Thin CLI wrapper
|   |-- model.py                 Neuro-TF trainer, predictor, and sweeper
|   |-- README.md                Full Neuro-TF command reference
|   `-- sample_training_verification.mdif
`-- MODEL_PLUGIN_API.md          Guide for adding another model family
```

## Requirements

The local trainers are script-first and currently do not use a packaging file.
Use a Python environment with NumPy installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy
```

Matplotlib is optional but recommended for publication-quality verification
plots:

```bash
python3 -m pip install matplotlib
```

ADS is not required for local training, inspection, prediction, MDIF export, or
Verilog-A file generation. The generated ADS ANN training package must be run on
a licensed ADS machine with the ADS Python environment because it imports
`keysight.ads.ann`.

`generate_points.py` is pure Python for `maximin-lhs`, `latin-hypercube`, and
`halton`. Its `sobol` method uses SciPy's Sobol implementation when SciPy is
available.

## Input Data

Input data is expected to be MDIF with one block per parameter point. Each block
should provide numeric geometry or process values as `VAR` entries and an
`ACDATA` table containing frequency and complex S-parameters:

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

The parser supports `RI`, `MA`, and `DB` complex pair formats. Header names may
use logical S-parameter labels such as `S11` or explicit real/imaginary columns
such as `S11R S11I`.

For combined training and verification files, use a split variable such as
`dataset=train` and `dataset=verification`. If no split values are present, the
trainers can reserve a holdout fraction of blocks for verification. KBNN also
accepts a separate coarse/prior MDIF; fine and coarse blocks are matched by
numeric geometry variables and the coarse response is interpolated onto the fine
frequency grid when needed.

## Point Generation

Use `generate_points.py` to create geometry/process sample CSVs before running
EM simulations or assembling MDIF. The default method is `maximin-lhs`, a
maximin Latin hypercube. For finite surrogate-training campaigns, this is often
more appropriate than a raw Sobol prefix because every parameter is stratified
and the script chooses the candidate design with the largest minimum point
spacing. Sobol remains useful when you want a low-discrepancy sequence that can
grow naturally in power-of-two batches.

```bash
python3 generate_points.py \
  --parameter W=0.40mm:0.80mm \
  --parameter L=1.00mm:1.60mm \
  --count 80 \
  --verification-count 16 \
  --method maximin-lhs \
  --out geometries.csv
```

Each generated CSV contains `point_index`, `dataset`, `split_sequence`,
`train_sequence`, `verification_sequence`, `method`, and one column per
parameter. Add `--include-normalized` when you also want the underlying
unit-cube coordinates. Add `--write-split-files` to also write separate
`*_train.csv` and `*_verification.csv` files for tools that consume the two
simulation queues independently.

For expensive EM campaigns, treat each point as one geometry/process setting
with a full frequency sweep. A practical initial design size is:

| Geometry parameters | Training points | Verification points |
| ---: | ---: | ---: |
| 2 | 20-35 | 8-12 |
| 3 | 35-60 | 12-18 |
| 4 | 60-100 | 16-25 |
| 5 | 100-160 | 25-40 |
| 6 | 160-250 | 35-55 |
| 7-8 | 250-450 | 50-90 |

For residual KBNN fits with a useful coarse model, start near the low-to-middle
end of each range because the neural network is learning `fine - coarse`
instead of the full response. A good staged workflow is to start with roughly
`15*d` training points, with a minimum of about 30, and `4*d` to `6*d`
verification points, with a minimum of about 12. Keep the verification set
fixed across model comparisons, then grow the training set in targeted batches
of about `3*d` to `5*d` points using the current worst-fit regions.

To compare the current Sobol-style workflow with the recommended space-filling
design, ask for both methods. The `{method}` placeholder is replaced in the
output path:

```bash
python3 generate_points.py \
  --parameter W=0.40mm:0.80mm \
  --parameter L=1.00mm:1.60mm \
  --count 64 \
  --method sobol \
  --method maximin-lhs \
  --out geometries_{method}.csv
```

After a first model fit, use the `suggest-additional` command to target the
next expensive EM simulations toward the current worst-fit regions. The command
reads `verification_metrics.csv`, ranks verification points by error, scores a
candidate pool by proximity to high-error regions and distance from existing
points, then writes a new CSV for the next training batch:

```bash
python3 generate_points.py suggest-additional \
  --parameter W=0.40mm:0.80mm \
  --parameter L=1.00mm:1.60mm \
  --count 12 \
  --fit-dir outputs/dnn_model \
  --existing-points geometries.csv \
  --out targeted_additional_points.csv
```

The suggested-point CSV uses `dataset=targeted` by default and includes the
nearest high-error verification source, distance from existing points, and
acquisition score. A companion `*_fit_error_regions.csv` file ranks the current
verification points by the selected metric, which defaults to `evm_pct`.

## Quick Start

From the repository root, inspect a sample MDIF before training:

```bash
python3 dnn/dnn.py inspect-mdif --mdif dnn/sample_training_verification.mdif
python3 kbnn/kbnn.py inspect-mdif --mdif kbnn/sample_fine.mdif
python3 neuro_tf/neuro_tf.py inspect-mdif --mdif neuro_tf/sample_training_verification.mdif
```

Train a direct DNN model:

```bash
python3 dnn/dnn.py train \
  --mdif dnn/sample_training_verification.mdif \
  --out-dir outputs/dnn_model \
  --parameter-names W,L \
  --hidden-layers 128,128,64
```

Train a KBNN residual model with fine and coarse MDIF data:

```bash
python3 kbnn/kbnn.py train \
  --mdif kbnn/sample_fine.mdif \
  --coarse-mdif kbnn/sample_coarse.mdif \
  --out-dir outputs/kbnn_model \
  --parameter-names W,L \
  --mode residual
```

Train a Neuro-TF model:

```bash
python3 neuro_tf/neuro_tf.py train \
  --mdif neuro_tf/sample_training_verification.mdif \
  --out-dir outputs/neuro_tf_model \
  --parameter-names W,L \
  --order 10
```

Each module README contains the complete option list:

- [DNN command reference](dnn/README.md)
- [KBNN command reference](kbnn/README.md)
- [Neuro-TF command reference](neuro_tf/README.md)

## Common Workflows

Run a hyperparameter sweep and keep the best completed model:

```bash
python3 dnn/dnn.py optimize \
  --mdif train_verify.mdif \
  --out-dir outputs/dnn_sweep \
  --parameter-names W,L,H \
  --mode random \
  --max-trials 40 \
  --selection-metric weighted_evm_pct \
  --require-passive
```

Predict a new set of parameter/frequency blocks after training:

```bash
python3 dnn/dnn.py predict \
  --model-dir outputs/dnn_model \
  --mdif new_parameter_blocks.mdif \
  --out-mdif predicted.mdif
```

Export a trained DNN or KBNN as a sampled ADS MDIF package:

```bash
python3 dnn/dnn.py export-ads-mdif \
  --model-dir outputs/dnn_model \
  --out-dir outputs/dnn_ads_mdif \
  --template-mdif ads_sweep_template.mdif
```

Export a trained DNN or KBNN as a direct Verilog-A n-port:

```bash
python3 dnn/dnn.py export-veriloga \
  --model-dir outputs/dnn_model \
  --out-dir outputs/dnn_veriloga \
  --module-name my_dnn_4port
```

Export a DNN or KBNN dataset for native ADS ANN extraction:

```bash
python3 dnn/dnn.py export-ads-ann \
  --mdif train_verify.mdif \
  --model-dir outputs/dnn_model \
  --out-dir outputs/dnn_ads_ann \
  --ads-output-format all
```

The ADS ANN export writes a portable package plus `train_ads_ann.py`; run that
script with ADS Python on the ADS machine to produce the native `.inc`, `.c`,
`.equation`, `.struc`, and `.scale` artifacts.

## Output Artifacts

A normal `train` run writes:

- `model.npz` and `metadata.json` with the trained model state and assumptions.
- `predicted_verification.mdif` for held-out verification blocks.
- `verification_metrics.csv` with per-block and per-S-parameter errors,
  including EVM.
- `verification_summary.json` with global errors and passivity summary data.
- `training_history.csv` and `training_history.pdf` with train/verification
  loss history and convergence plots.
- `training_summary.md` with a human-readable run summary.
- `worst_case_plots/*.pdf` with S-parameter Smith/complex, magnitude, phase,
  and error views.
- `worst_case_y_plots/*.pdf` with real/imaginary Y-parameter diagnostics.

Sweep runs add result CSVs, best-configuration JSON, Markdown summaries,
per-trial loss-vs-epoch plots, diagnostic plots, and a promoted `best_model/`
directory. DNN and KBNN sweep results can also be reranked after the fact to
choose a different passive or weighted-error winner without repeating every
trial.

## ADS Integration Paths

Choose the ADS handoff based on the level of simulator integration you need:

| Export path | Commands | Use when |
| --- | --- | --- |
| Sampled MDIF | `export-ads-mdif` | You want the lowest-risk ADS integration. ADS interpolates a dense generated MDIF table using normal data-based components. |
| Native ADS ANN package | `export-ads-ann` | You want ADS to retrain/extract the neural network and emit native ANN artifacts on a licensed ADS machine. |
| Direct Verilog-A | `export-veriloga` | You want to embed local trained weights in a generated n-port model and validate it with the target ADS Verilog-A compiler. |

For direct DNN Verilog-A, training with `--output-domain y` can improve solve
speed because the generated model stamps admittance directly instead of
converting S to Y at every simulator evaluation.

## Extending The Repository

New model families can reuse the common support layer by following
[MODEL_PLUGIN_API.md](MODEL_PLUGIN_API.md). A plugin keeps its own `model.py`
and thin CLI wrapper, then calls shared helpers for MDIF I/O, splitting,
metrics, plots, sweep orchestration, summaries, and ADS package generation.
