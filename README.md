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

The code uses a flat script-first layout: each modeling approach has one
root-level entry point, while shared MDIF parsing, metrics, plotting, sweep
orchestration, and ADS export helpers live in `surrogate_common.py`.

## What It Builds

The repository provides three surrogate-model front ends:

| Model | Entry point | Best fit for | Basic idea |
| --- | --- | --- | --- |
| DNN | `dnn.py` | General parameterized S-parameter fitting when you want a direct neural response model. | A multilayer perceptron predicts S-parameters, or optionally Y-parameters, from geometry/process variables plus frequency features. |
| KBNN | `kbnn.py` | Cases where a fast coarse model or lower-fidelity EM result is available. | A neural network learns the correction from coarse/prior response to fine/target response, or uses the coarse response as an input. |
| Neuro-TF | `neuro_tf.py` | Smooth frequency responses where a rational transfer-function structure is useful. | Fixed stable poles define rational transfer functions; a neural network maps geometry/process variables to the fitted coefficients. |

All three tools read MDIF, train models, run sweeps, write verification
artifacts, predict new response blocks, and export sampled ADS MDIF packages or
self-contained Verilog-A n-ports. DNN and KBNN additionally support native ADS
ANN package generation.

## Repository Layout

```text
.
|-- dnn.py                                Direct DNN CLI and implementation
|-- kbnn.py                               KBNN CLI and implementation
|-- neuro_tf.py                           Neuro-TF CLI and implementation
|-- surrogate_common.py                   Shared training, reporting, and ADS export utilities
|-- generate_points.py                    Geometry/process point-set generator
|-- dnn_sample_training_verification.mdif
|-- kbnn_sample_fine.mdif
|-- kbnn_sample_coarse.mdif
|-- neuro_tf_sample_training_verification.mdif
|-- MODEL_PLUGIN_API.md                   Guide for adding another model family
`-- README.md                              Integrated workflow and command reference
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
trainers can reserve a holdout fraction of blocks for verification. For KBNN,
pass the coarse/prior MDIF together with the fine MDIF. The integrated workflow
first fits and saves an S-domain coarse DNN, then evaluates that frozen DNN at
every fine training and verification point. This matches the two-network model
that will be embedded in a self-contained export.

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

Every geometry CSV also gets an automatic same-stem JSON file. For example,
`geometries.csv` produces `geometries.json`. The JSON records the generation
method, point and dataset counts, and each parameter's lower bound, upper bound,
unit, base-unit bounds, and linear/log scale. Separate train/verification CSVs
and targeted additional-point CSVs receive their own JSON files as well.

### Extending an Existing Parameter Range

To extend one side of an existing design, keep the original bounds in
`--parameter`, provide the new overall bounds with `--extend-range`, and pass
the original CSV with `--existing-points`. This example changes only the upper
`W` bound from `0.80mm` to `1.00mm`:

```bash
python3 generate_points.py \
  --parameter W=0.40mm:0.80mm \
  --parameter L=1.00mm:1.60mm \
  --extend-range W=0.40mm:1.00mm \
  --existing-points geometries.csv \
  --out geometries_extended.csv
```

The new sampler covers only the added slab: `W=0.80mm:1.00mm` across the full
original `L` range. The output contains the original rows first and the new
rows afterward, with continued point/split sequences and normalized columns
recomputed for the new overall range. The unchanged bound must exactly match
the corresponding original bound. To extend the lower side instead, enter a
lower new bound and retain the old upper bound. You may set `--out` to the same
path as `--existing-points` for an in-place combined result; a separate output
is safer until the new EM batch has been checked.

The extended geometry's JSON uses the new overall parameter ranges and also
includes `range_extension` details containing the original ranges, the slab
used to sample only the new points, the extended side, and the original and
added point counts.

#### How Many New Points to Add

When `--count` is omitted, the script prints and uses a density-based point
recommendation. Let `r` be the added design-space volume divided by the old
volume. For a one-variable linear extension this is the added width divided by
the old width; log variables use log-width.

| New point group | Recommended count |
| --- | --- |
| Training | `max(ceil(old_training_points * r), 4*d)` |
| Verification, when the original set contains verification points | `max(ceil(old_verification_points * r), 2*d)` |

Here, `d` is the number of geometry parameters. For example, extending one
range by 50% from an 80-point, two-parameter design containing 64 training and
16 verification points recommends 32 new training and 8 new verification
points. This maintains roughly the original sampling density while ensuring
basic coverage of the new boundary. It is a practical lower target, not a
mathematical guarantee; after refitting, use `suggest-additional` if errors are
still concentrated in the new slab. An explicit `--count` overrides the total,
and the original train/verification ratio is retained unless
`--verification-count` is also supplied.

`--range-factor NAME=FACTOR` remains available when you want a new independent
point set with a symmetrically wider range. It does not perform the one-sided
append workflow.

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

The explicit `generate` subcommand is optional; invoking `generate_points.py`
without a subcommand uses it automatically.

### Parameter Ranges

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--extend-range SPEC</code></nobr> | <code>generate</code> | Optional one-sided append workflow. Supplies the new overall bounds as <code>NAME=NEW_LOW:NEW_HIGH</code>; exactly one bound must match the original <code>--parameter</code> range and the other must move outward. Requires <code>--existing-points</code>. | <nobr><code>--extend-range W=0.40mm:1.00mm</code></nobr> |
| <nobr><code>--parameter SPEC</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Required and repeatable. Defines an existing parameter as <code>NAME=LOW:HIGH[:linear\|log]</code>. Matching bound units are retained in the output. | <nobr><code>--parameter W=0.40mm:0.80mm</code></nobr> |
| <nobr><code>--range-factor SPEC</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Optional and repeatable. Expands the named parameter's total span around its existing center. The finite factor must be greater than 1. | <nobr><code>--range-factor W=1.5</code></nobr> |

### Sampling

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--candidate-method METHOD</code></nobr> | <code>suggest-additional</code> | Candidate-pool method: <code>halton</code>, <code>latin-hypercube</code>, <code>maximin-lhs</code>, or <code>sobol</code>. Default: <code>latin-hypercube</code>. | <nobr><code>--candidate-method sobol</code></nobr> |
| <nobr><code>--lhs-candidates INT</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Candidate Latin hypercubes tried when using <code>maximin-lhs</code>. Default: <code>64</code>. | <nobr><code>--lhs-candidates 128</code></nobr> |
| <nobr><code>--method METHOD</code></nobr> | <code>generate</code> | Repeat or comma-separate point-set methods. Choices: <code>halton</code>, <code>latin-hypercube</code>, <code>maximin-lhs</code>, and <code>sobol</code>. Default: <code>maximin-lhs</code>. | <nobr><code>--method sobol,maximin-lhs</code></nobr> |
| <nobr><code>--no-scramble</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Disables Sobol scrambling, which is enabled by default. | <nobr><code>--no-scramble</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Random seed used by randomized sampling methods. Default: <code>1234</code>. | <nobr><code>--seed 42</code></nobr> |
| <nobr><code>--skip INT</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Non-negative number of leading Sobol or Halton points to skip. Default: <code>0</code>. | <nobr><code>--skip 64</code></nobr> |

### Output and Dataset Splits

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--analysis-out PATH</code></nobr> | <code>suggest-additional</code> | Ranked fit-error-region CSV. Default: <code>&lt;out&gt;_fit_error_regions.csv</code>. | <nobr><code>--analysis-out error_regions.csv</code></nobr> |
| <nobr><code>--count INT</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Positive number of new points. Required except for <code>generate --extend-range</code>, which calculates and uses a recommendation when omitted. | <nobr><code>--count 80</code></nobr> |
| <nobr><code>--existing-points PATH</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | With <code>generate --extend-range</code>, the original CSV retained at the start of the combined output. With <code>suggest-additional</code>, a repeatable CSV of simulated points to avoid. | <nobr><code>--existing-points geometries.csv</code></nobr> |
| <nobr><code>--include-normalized</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Adds each parameter's normalized <code>u_NAME</code> coordinate to the output. | <nobr><code>--include-normalized</code></nobr> |
| <nobr><code>--out PATH</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Output CSV path; a same-stem JSON containing parameter ranges is written automatically. For multiple generation methods, use <code>{method}</code> or let the script add a method suffix. A range extension defaults to <code>&lt;existing&gt;_extended.csv</code>. | <nobr><code>--out geometries_{method}.csv</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | CSV column used for dataset labels. Default: <code>dataset</code>. | <nobr><code>--split-var dataset</code></nobr> |
| <nobr><code>--target-dataset NAME</code></nobr> | <code>suggest-additional</code> | Dataset label assigned to suggested points. Default: <code>targeted</code>. | <nobr><code>--target-dataset train</code></nobr> |
| <nobr><code>--verification-count INT</code></nobr> | <code>generate</code> | Number of new tail points labeled verification; must be smaller than <code>--count</code>. Default: <code>0</code>, or the original split ratio during a range extension. | <nobr><code>--verification-count 16</code></nobr> |
| <nobr><code>--write-split-files</code></nobr> | <code>generate</code> | Also writes separate <code>*_train.csv</code> and, when applicable, <code>*_verification.csv</code> files. | <nobr><code>--write-split-files</code></nobr> |

### Existing Input and Targeted Selection

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--bare-values MODE</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Interprets unitless values from existing input rows as <code>parameter-units</code> or <code>base-units</code>. Default: <code>parameter-units</code>. | <nobr><code>--bare-values base-units</code></nobr> |
| <nobr><code>--candidate-count INT</code></nobr> | <code>suggest-additional</code> | Positive candidate-pool size. Default: the greater of 1000 and <code>count * candidate-factor</code>. | <nobr><code>--candidate-count 5000</code></nobr> |
| <nobr><code>--candidate-factor INT</code></nobr> | <code>suggest-additional</code> | Positive candidate multiplier used when <code>--candidate-count</code> is omitted. Default: <code>200</code>. | <nobr><code>--candidate-factor 300</code></nobr> |
| <nobr><code>--existing-mdif PATH</code></nobr> | <code>suggest-additional</code> | Repeatable MDIF containing previously simulated parameter points to avoid. | <nobr><code>--existing-mdif training.mdif</code></nobr> |
| <nobr><code>--fit-dir PATH</code></nobr> | <code>suggest-additional</code> | Fit directory containing <code>verification_metrics.csv</code>. Ignored when <code>--verification-metrics</code> is given. | <nobr><code>--fit-dir outputs/dnn_model</code></nobr> |
| <nobr><code>--focus-power FLOAT</code></nobr> | <code>suggest-additional</code> | Non-negative exponent applied to verification-error scores. Default: <code>1.0</code>. | <nobr><code>--focus-power 1.5</code></nobr> |
| <nobr><code>--focus-radius FLOAT</code></nobr> | <code>suggest-additional</code> | Positive unit-cube radius around high-error verification points. Default: <code>0.25</code>. | <nobr><code>--focus-radius 0.2</code></nobr> |
| <nobr><code>--metric NAME</code></nobr> | <code>suggest-additional</code> | Verification-metrics column used to target errors; <code>auto</code> selects a known available metric. Default: <code>evm_pct</code>. | <nobr><code>--metric auto</code></nobr> |
| <nobr><code>--min-distance FLOAT</code></nobr> | <code>suggest-additional</code> | Rejects candidates closer than this non-negative normalized distance to existing or already suggested points. Default: <code>0.0</code>. | <nobr><code>--min-distance 0.05</code></nobr> |
| <nobr><code>--novelty-power FLOAT</code></nobr> | <code>suggest-additional</code> | Non-negative exponent applied to distance from existing and suggested points. Default: <code>1.0</code>. | <nobr><code>--novelty-power 2</code></nobr> |
| <nobr><code>--verification-metrics PATH</code></nobr> | <code>suggest-additional</code> | Direct path to <code>verification_metrics.csv</code>; overrides <code>--fit-dir</code>. | <nobr><code>--verification-metrics trial/verification_metrics.csv</code></nobr> |

## Quick Start

From the repository root, inspect a sample MDIF before training:

```bash
python3 dnn.py inspect-mdif --mdif dnn_sample_training_verification.mdif
python3 kbnn.py inspect-mdif --mdif kbnn_sample_fine.mdif
python3 neuro_tf.py inspect-mdif --mdif neuro_tf_sample_training_verification.mdif
```

Train a direct DNN model:

```bash
python3 dnn.py train \
  --mdif dnn_sample_training_verification.mdif \
  --out-dir outputs/dnn_model \
  --parameter-names W,L \
  --hidden-layers 128,128,64
```

Fit the coarse DNN and fine KBNN together as one residual-model workflow:

```bash
python3 kbnn.py train \
  --mdif kbnn_sample_fine.mdif \
  --coarse-mdif kbnn_sample_coarse.mdif \
  --out-dir outputs/kbnn_model \
  --parameter-names W,L \
  --mode residual
```

Train a Neuro-TF model:

```bash
python3 neuro_tf.py train \
  --mdif neuro_tf_sample_training_verification.mdif \
  --out-dir outputs/neuro_tf_model \
  --parameter-names W,L \
  --order 10
```

Complete DNN, KBNN, and Neuro-TF command references are integrated below.

## Common Workflows

### Train and optimize option naming

Optimize/sweep commands use plural names for candidate lists and accept the
matching train option for a single candidate. This makes a train command easy
to reuse: change `train` to `optimize`, keep singular options when their value
should stay fixed, and pluralize only the settings that should be swept.
Existing `*-options` spellings remain supported as compatibility aliases.

| Train or one optimize value | Multiple optimize values |
| --- | --- |
| `--activation relu` | `--activations tanh,relu` |
| `--learning-rate 0.002` | `--learning-rates 0.001,0.002,0.005` |
| `--freq-transform log` | `--freq-transforms log,linear` |
| `--hidden-layers 64,64` | `--hidden-layers '32;64;64,64'` |
| `--order 10` | `--orders 6,10,14` |
| `--pole-damping 0.18` | `--pole-dampings 0.12,0.18,0.28` |
| `--ridge 1e-8` | `--ridges 1e-10,1e-8,1e-6` |
| KBNN `--mode residual` | KBNN `--modes residual,prior-input` |

Use `--search-mode grid|random` for the optimize search strategy. Legacy
`--mode grid|random` commands remain valid; on KBNN optimize commands,
`--mode plain|residual|prior-input` now has the same model meaning as it does
for `train`.

Run a hyperparameter sweep and keep the best completed model:

```bash
python3 dnn.py optimize \
  --mdif train_verify.mdif \
  --out-dir outputs/dnn_sweep \
  --parameter-names W,L,H \
  --search-mode random \
  --max-trials 40 \
  --selection-metric weighted_evm_pct \
  --require-passive
```

Predict a new set of parameter/frequency blocks after training:

```bash
python3 dnn.py predict \
  --model-dir outputs/dnn_model \
  --mdif new_parameter_blocks.mdif \
  --out-mdif predicted.mdif
```

Export any trained model family as a sampled ADS MDIF package:

```bash
python3 dnn.py export-ads-mdif \
  --model-dir outputs/dnn_model \
  --out-dir outputs/dnn_ads_mdif \
  --template-mdif ads_sweep_template.mdif
```

Export any trained model family as a direct Verilog-A n-port:

```bash
python3 dnn.py export-veriloga \
  --model-dir outputs/dnn_model \
  --out-dir outputs/dnn_veriloga \
  --module-name my_dnn_4port \
  --parameter-input-scales 1.0
```

Residual and prior-input KBNNs use a frozen coarse DNN during fitting and at
runtime. The integrated KBNN command fits the coarse model once, saves it under
`coarse_model/`, fits the fine network from its predictions, and retains both
models for one self-contained Verilog-A component:

```bash
python3 kbnn.py train \
  --mdif fine_train_verify.mdif \
  --coarse-mdif coarse_train_verify.mdif \
  --out-dir outputs/kbnn_model \
  --parameter-names W,L \
  --mode residual

python3 kbnn.py export-veriloga \
  --model-dir outputs/kbnn_model \
  --out-dir outputs/kbnn_veriloga \
  --module-name my_kbnn_4port
```

The generated KBNN module computes both networks and the S-to-Y conversion
internally. ADS supplies only the electrical ports, geometry/process instance
parameters, and simulator frequency. The exporter automatically finds the
packaged `coarse_model/` directory and verifies its saved file hashes.

Export a DNN or KBNN dataset for native ADS ANN extraction:

```bash
python3 dnn.py export-ads-ann \
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
- `training_summary.md` with a human-readable run summary and copyable export
  commands using paths relative to the repository root.
- `worst_case_plots/*.pdf` with S-parameter Smith/complex, magnitude, phase,
  and error views.
- `worst_case_y_plots/*.pdf` with real/imaginary Y-parameter diagnostics.

An integrated residual or prior-input KBNN run also writes a complete coarse
DNN package under `coarse_model/` and a `composite_model_manifest.json` that
identifies and hashes both saved networks for later Verilog-A extraction.
The reported Verilog-A commands include an explicit default module name and a
single `--parameter-input-scales 1.0` value applied to every fitted parameter,
making either value easy to edit before export.

### Distinct DC Point

Every DNN, KBNN, and Neuro-TF fit now stores a parameter-independent DC
equivalent resistance in `metadata.json` and reports it in
`training_summary.md`. The corresponding real/imaginary S-point is stored under
`dc_sparameters`. It is deliberately separate from the fitted response:

> **Exact DC data is required.** Every training block must contain exactly one
> zero-Hz row. A missing or duplicate DC row stops training with an error. The
> lowest positive frequency is never substituted for DC, and the RF fit is
> never extrapolated to create DC.

1. The exact zero-Hz S-matrix from every training block is converted to Y using
   the model reference impedance.
2. A one-amp balanced current is applied between each port pair with all other
   ports open; the real differential voltage gives that pair's equivalent
   resistance.
3. Every extracted pair value must be positive and finite. Their arithmetic
   mean is saved as `dc_equivalent_resistance_ohm`.
4. Zero-frequency rows are excluded from neural/rational fitting, so changing
   an input MDIF's DC samples cannot change the fitted weights or poles.

At exactly zero Hz, prediction and sampled-MDIF export use an equal-resistance
port network whose equivalent resistance between any two ports is the saved
average. Sampled ADS exports prepend this zero-Hz point automatically. Direct
Verilog-A exports electrically disable the fitted-response stamps at DC and
enable the resistor network instead; positive frequencies use the fitted
response. The exporter selects DC or fitted Y coefficients before an
unconditional current contribution, so `ddt()` is never placed in a conditional
and the generated source remains legal for ADS Verilog-A. Export also verifies
that the saved model records `dc_resistance_source_kind=exact_zero_frequency`;
older models created from RF fallback data are rejected and must be retrained.

For an integrated residual or prior-input KBNN, the coarse DNN and fine KBNN
store their own independently extracted resistance values, so both fine and
coarse training data must contain exact zero-Hz rows. The composite
Verilog-A component uses the fine-data resistance and bypasses both networks at
DC. Native ADS ANN retraining excludes zero-Hz rows, but its generated ANN alone
does not implement the distinct resistor branch; use the direct Verilog-A or
sampled-MDIF handoff when DC behavior is required.

Sweep runs add result CSVs, best-configuration JSON, Markdown summaries,
per-trial loss-vs-epoch plots, diagnostic plots, and a promoted `best_model/`
directory. At completion, each sweep prints a copyable standalone `train`
command for the winning configuration; the same command is saved in the
best-configuration JSON and Markdown summary. The sweep summary and the
promoted model's `training_summary.md` also contain export commands resolved to
`best_model/`. DNN and KBNN sweep results can be reranked after the fact to
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
[MODEL_PLUGIN_API.md](MODEL_PLUGIN_API.md). Add one root-level model script and
call shared helpers for MDIF I/O, splitting, metrics, plots, sweep
orchestration, summaries, and ADS package generation.

---

# Integrated Command Reference

The following sections contain the complete command references for all three
model types. Every command is intended to run from the repository root.

## DNN

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

### Expected MDIF Shape

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

### Inspect MDIF

Use `inspect-mdif` first when you want to confirm block count, S-parameter
labels, inferred numeric variables, split values, and frequency span.

```bash
python3 dnn.py inspect-mdif \
  --mdif train_verify.mdif
```

### Usage

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
are also converted to Y-parameters with `--target-z0` and written under
`worst_case_y_plots/`, where modeled and measured admittance are shown as
real/imaginary frequency plots. Use `--worst-plots 0` to skip both plot sets
during large experiments.
Each single `train` run also writes `training_summary.md`, which collects the
chosen settings, final loss values, verification metrics, passivity summary,
links to the generated S- and Y-parameter worst-case plots, and copyable
self-contained Verilog-A and sampled ADS MDIF export commands.

### Sweeping / Optimizing

Use `sweep` or its alias `optimize` to try multiple DNN configurations. The
command writes `dnn_sweep_results.csv` and `dnn_sweep_summary.md`, chooses the
best trial using `--selection-metric`, and keeps the current best completed
trial in `best_model/` as the sweep runs. This avoids a final refit after all
trials finish. When the sweep completes, it prints a copyable standalone
`train` command for the winning configuration and records that command in
`dnn_best_config.json` and `dnn_sweep_summary.md`.

```bash
python3 dnn.py optimize \
  --mdif train_verify.mdif \
  --out-dir dnn_sweep \
  --parameter-names W,L,H \
  --freq-transforms log,log-linear \
  --hidden-layers '64,64;128,128,64;256,128,64' \
  --activations tanh,relu \
  --learning-rates 0.001,0.002,0.005 \
  --sparam-weights 'diag=1;offdiag=0.2' \
  --output-domain y \
  --target-z0 50 \
  --search-mode random \
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

#### Frequency weighting

Use `--frequency-weights` with DNN, KBNN, or Neuro-TF training and sweep
commands to prioritize particular frequencies or bands. Rules are separated by
semicolons, applied left to right, and normalized over the positive-frequency
training samples so their mean is 1.0. Zero Hz remains the separate
data-derived DC point and is never part of the fitted loss.

```bash
--frequency-weights 'default=1;1GHz=5'
--frequency-weights 'default=0.25;2GHz:4GHz=2'
--frequency-weights 'all=1;900MHz,1GHz,1.1GHz=4;5GHz:8GHz=2'
```

Selectors accept engineering units understood by the MDIF parser. A single
value matches that sampled frequency; `start:stop` matches an inclusive band;
`all`, `default`, or `*` matches every fitted frequency. The normalized
frequency weight multiplies the normalized S-parameter weight, so both options
can be used together. Weighted verification metrics and weighted sweep
selection metrics use the same combined priority.

DNN and KBNN apply the weight directly to each neural-network sample. Neuro-TF
applies it to the frequency-domain rational least-squares coefficient fit. For
an integrated KBNN, the coarse DNN inherits `--frequency-weights`; use
`--coarse-frequency-weights` when the coarse fit needs different priorities.
The resulting saved weights and coefficients are used unchanged by the
self-contained Verilog-A and sampled-MDIF exports. Native ADS ANN export
re-trains through the ADS API, which does not expose per-sample loss weights;
use the local Verilog-A or sampled-MDIF path when these frequency priorities
must be preserved exactly.

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
- `weighted_rmse_abs`, `weighted_evm_pct`, and the other `weighted_*`
  variants: the same metrics using the configured S-parameter and frequency
  weights
- `weighted_evm_db`: S-parameter-weighted EVM in dB
- `weighted_evm_pct`: S-parameter-weighted EVM as a percentage
- `weighted_evm_rms`: S-parameter-weighted EVM ratio
- `weighted_rmse_abs`: S-parameter-weighted complex RMSE

Set `--search-mode grid` to exhaustively test all combinations, or keep the default
`--search-mode random --max-trials N` for direct hyperparameter optimization over a
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

### Post-Run Sweep Reranking

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

### Predict

Predict new parameter blocks after training:

```bash
python3 dnn.py predict \
  --model-dir dnn_model \
  --mdif new_parameter_blocks.mdif \
  --out-mdif predicted.mdif
```

For prediction, the input MDIF must provide the geometry `VAR`s and frequency
grid. Placeholder S-parameter columns are acceptable; their values are ignored.

### ADS MDIF Export

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

### ADS ANN Export

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

### Direct Verilog-A Export

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

If all MDIF training parameters were scaled dimensionless values but the ADS
schematic uses base units, export with one common scale. For example, if the
MDIF parameter values are expressed in microns and ADS passes them in meters,
use `--parameter-input-scales 1um`. The generated Verilog-A divides every
fitted parameter by its generated input-scale parameter before evaluating the
network.

### ADS Note

The `export-ads-mdif` command is the lowest-risk direct ADS handoff. It exports
the trained DNN response onto a dense parameter/frequency table, so ADS can use
normal MDIF interpolation during circuit optimization without embedding Python
or NumPy in the simulator. The `export-veriloga` command embeds the trained
local DNN weights into a Verilog-A n-port, which avoids ADS ANN retraining but
should be validated in the target ADS Verilog-A compiler. The `export-ads-ann`
command is the native ADS ANN handoff for generating ADS ANN
Verilog-A/C/equation artifacts on an ADS machine.

### Options Reference

Options are grouped by purpose below. Rows are alphabetical within each table;
the **Subcommands** column includes accepted command aliases.

#### Files, data, and outputs

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--mdif PATH</code></nobr> | <code>inspect-mdif</code>, <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>predict</code>, <code>export-ads-ann</code> | Input MDIF to inspect, fit, predict, or use as an ADS ANN retraining source, depending on the subcommand. | <nobr><code>--mdif train_verify.mdif</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | <code>predict</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-ann</code>, <code>export-veriloga</code> | Directory containing the trained <code>model.npz</code> and <code>metadata.json</code> used for prediction or export. | <nobr><code>--model-dir dnn_model</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-ann</code>, <code>export-veriloga</code> | Destination directory for the model, sweep, or export artifacts generated by the selected command. | <nobr><code>--out-dir dnn_model</code></nobr> |
| <nobr><code>--out-mdif PATH</code></nobr> | <code>predict</code> | Required. Output MDIF containing predicted S-parameters. | <nobr><code>--out-mdif predicted.mdif</code></nobr> |
| <nobr><code>--output-name NAME</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Output MDIF file name. Default: `surrogate_ads.mdif`. | <nobr><code>--output-name dnn_ads.mdif</code></nobr> |
| <nobr><code>--output-prefix NAME</code></nobr> | <code>export-ads-ann</code> | Prefix for native ADS ANN outputs such as `.inc`, `.c`, `.equation`, `.scale`, and `.struc`. Default: `dnn_ads_ann`. | <nobr><code>--output-prefix dnn_filter_ann</code></nobr> |
| <nobr><code>--template-mdif PATH</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Optional. MDIF containing the exact geometry and frequency blocks to evaluate for ADS. S-parameter values are ignored. Use this when you already know the ADS optimization grid. | <nobr><code>--template-mdif ads_sweep_template.mdif</code></nobr> |
| <nobr><code>--verification-mdif PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Optional. Separate MDIF containing verification blocks. When supplied, every block in `--mdif` is treated as training data and every block in this file is treated as verification data. | <nobr><code>--verification-mdif verify.mdif</code></nobr> |

#### Data selection and loss weighting

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--frequency-weights SPEC</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Optional per-frequency fitting and sweep-selection weights. Select exact frequencies or inclusive ranges; later rules override earlier rules and weights are normalized to mean 1. | <nobr><code>--frequency-weights 'default=1;1GHz=5;2GHz:4GHz=3'</code></nobr> |
| <nobr><code>--holdout-fraction FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Fraction of blocks to reserve for verification when no split values are found in a combined MDIF. Default: `0.2`. | <nobr><code>--holdout-fraction 0.25</code></nobr> |
| <nobr><code>--parameter-names LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Comma-separated geometry/process variable names to use as DNN inputs. If omitted, the trainer infers numeric `VAR`s common to all blocks, excluding the split variable. | <nobr><code>--parameter-names W,L,H</code></nobr> |
| <nobr><code>--sparam-weights SPEC</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Optional S-parameter fitting and sweep-selection weights. ADS ANN export records the stored or overridden weights in its manifest, but the generated ADS script cannot apply per-output weights because the documented ADS ANN API does not expose them. | <nobr><code>--sparam-weights 'diag=1;offdiag=0.2'</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | <code>inspect-mdif</code>, <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Name of the <code>VAR</code> used to split or summarize a combined MDIF. Default: <code>dataset</code>. | <nobr><code>--split-var dataset</code></nobr> |
| <nobr><code>--train-values LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Comma-separated values of `--split-var` that identify training blocks. Default: `train,training`. | <nobr><code>--train-values train,training</code></nobr> |
| <nobr><code>--verify-values LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Comma-separated values of `--split-var` that identify verification blocks. Default: `verify,verification,test,validation`. | <nobr><code>--verify-values verification,test</code></nobr> |

#### Model architecture and fitting

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--activation {tanh,relu}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Hidden-layer activation. `tanh` is smoother for small microwave datasets; `relu` can help larger datasets. Default: `tanh`. | <nobr><code>--activation tanh</code></nobr> |
| <nobr><code>--activations LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated activation functions to try. `--activation` accepts one train-compatible value; `--activation-options` remains an alias. Default: `tanh,relu`. | <nobr><code>--activations tanh,relu</code></nobr> |
| <nobr><code>--batch-size INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Number of frequency-sample rows per Adam update. Default: `256`. | <nobr><code>--batch-size 256</code></nobr> |
| <nobr><code>--debug</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Enable diagnostic output and command tracebacks. Sweeps also print the candidate list and retain failed-trial tracebacks; use `--jobs 1` for the cleanest trace. | <nobr><code>--debug</code></nobr> |
| <nobr><code>--epochs INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Maximum Adam training epochs. Early stopping may stop before this value. Default: `2000`. | <nobr><code>--epochs 2000</code></nobr> |
| <nobr><code>--freq-transform {log,linear,log-linear}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Frequency input transform. `log` uses `log10(freq_hz)`, `linear` uses raw Hz, and `log-linear` uses both. Default: `log`. | <nobr><code>--freq-transform log-linear</code></nobr> |
| <nobr><code>--freq-transforms LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated frequency transforms to try. `--freq-transform` accepts one train-compatible value; `--freq-transform-options` remains an alias. Default: `log,log-linear`. | <nobr><code>--freq-transforms log,log-linear</code></nobr> |
| <nobr><code>--hidden-layers LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Comma-separated hidden-layer sizes for one model. Sweeps also accept semicolon-separated candidate layouts. Train default: `128,128,64`; sweep default: `64,64;128,128,64;128,128,128;256,128,64`. `--hidden-layer-layouts` and `--hidden-layer-options` remain aliases. | <nobr><code>--hidden-layers 128,128,64</code></nobr> |
| <nobr><code>--learning-rate FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Adam optimizer step size. Lower values are safer; higher values may converge faster but can overshoot. Default: `0.002`. | <nobr><code>--learning-rate 0.002</code></nobr> |
| <nobr><code>--learning-rates LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated Adam learning rates to try. `--learning-rate` accepts one train-compatible value. Default: `0.001,0.002,0.005`. | <nobr><code>--learning-rates 0.001,0.002,0.005</code></nobr> |
| <nobr><code>--loss-interval INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Full train/verification loss check interval in epochs. Increasing this reduces full-dataset scoring overhead during long runs while early stopping still uses epoch-based patience. Default: `1`. | <nobr><code>--loss-interval 5</code></nobr> |
| <nobr><code>--output-domain {s,y}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Training target domain. `s` predicts S-parameters and is compatible with every export path. `y` converts the MDIF S-data to admittance targets using `--target-z0`; this is the fastest formulation for direct Verilog-A solve speed. Default: `s`. | <nobr><code>--output-domain y</code></nobr> |
| <nobr><code>--patience INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Early-stopping patience measured in epochs without validation-loss improvement. Use `0` to disable early stopping. Default: `200`. | <nobr><code>--patience 200</code></nobr> |
| <nobr><code>--progress-interval INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Console progress update interval in epochs. Updates redraw one terminal status line and include epoch count, elapsed time, and loss values when that epoch also matches `--loss-interval`. Use `0` to disable. Default: `25`. | <nobr><code>--progress-interval 10</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Random seed for data splitting, model initialization, minibatch order, ADS ANN data preparation, and sweep candidate selection where applicable. Default: `1234`. | <nobr><code>--seed 1234</code></nobr> |
| <nobr><code>--target-z0 FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Reference impedance used only when `--output-domain y` converts S-parameters into Y-parameter training targets. Use the same value as the MDIF option line reference impedance. Default: `50.0`. | <nobr><code>--target-z0 50</code></nobr> |
| <nobr><code>--worst-plots INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Number of worst verification S/Y plot pairs to generate. In a sweep it applies to a final `--retrain-best`; otherwise the promoted trial retains its `--trial-worst-plots` output. Default: `6`. | <nobr><code>--worst-plots 6</code></nobr> |

#### Sweep and model selection

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--best-model-dir PATH</code></nobr> | <code>rerank-sweep</code> | Destination for `--promote-best`. Default: `<sweep-dir>/best_model_reranked`. | <nobr><code>--best-model-dir dnn_sweep/best_model_passive</code></nobr> |
| <nobr><code>--jobs INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Number of sweep trials to train in parallel. Use up to the number of physical cores and lower it if memory use gets high. Default: `1`. | <nobr><code>--jobs 4</code></nobr> |
| <nobr><code>--keep-trial-models</code></nobr> | <code>sweep</code>, <code>optimize</code> | Keep full per-trial model directories under `trials/`. By default, each trial keeps lightweight summary and plot artifacts while large model files are removed. | <nobr><code>--keep-trial-models</code></nobr> |
| <nobr><code>--max-passivity-sigma FLOAT</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials whose worst predicted S-matrix singular value is at or below this value when selecting `best_model/`. | <nobr><code>--max-passivity-sigma 1.000001</code></nobr> |
| <nobr><code>--max-passivity-violations INT</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials with this many or fewer passivity-violating frequency points when selecting `best_model/`. | <nobr><code>--max-passivity-violations 0</code></nobr> |
| <nobr><code>--max-trials INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Maximum number of candidate configurations to evaluate. In `random` mode this limits the random sample; in `grid` mode it truncates the product list. Default: `24`. | <nobr><code>--max-trials 40</code></nobr> |
| <nobr><code>--overwrite</code></nobr> | <code>rerank-sweep</code> | Allow `--promote-best` to replace an existing `--best-model-dir`. | <nobr><code>--overwrite</code></nobr> |
| <nobr><code>--promote-best</code></nobr> | <code>rerank-sweep</code> | Copy the selected trial model to `--best-model-dir` if that trial still contains `model.npz` and `metadata.json`. Requires the original sweep to have used `--keep-trial-models`. | <nobr><code>--promote-best</code></nobr> |
| <nobr><code>--replace-current-best</code></nobr> | <code>rerank-sweep</code> | Overwrite `<sweep-dir>/best_model` with the selected trial model if the trial model files are available. | <nobr><code>--replace-current-best</code></nobr> |
| <nobr><code>--require-passive</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials with zero passivity-violating frequency points when selecting `best_model/`. Equivalent to `--max-passivity-violations 0` unless a stricter value is supplied. | <nobr><code>--require-passive</code></nobr> |
| <nobr><code>--retrain-best</code></nobr> | <code>sweep</code>, <code>optimize</code> | Retrain the selected best configuration at the end of the sweep instead of using the best completed trial model promoted during the sweep. Use this when you want `--worst-plots` to apply only to the final model. | <nobr><code>--retrain-best</code></nobr> |
| <nobr><code>--search-mode {grid,random}</code></nobr> | <code>sweep</code>, <code>optimize</code> | Search strategy. `grid` evaluates combinations in deterministic product order; `random` samples combinations from the full grid. Legacy `--mode` remains an alias. Default: `random`. | <nobr><code>--search-mode random</code></nobr> |
| <nobr><code>--selection-metric NAME</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Metric minimized when choosing the best trial. Options include `evm_pct`, `rmse_abs`, passivity metrics, and weighted metrics such as `weighted_evm_pct` and `weighted_rmse_abs`. Default: `rmse_abs`. | <nobr><code>--selection-metric weighted_evm_pct</code></nobr> |
| <nobr><code>--sweep-dir PATH</code></nobr> | <code>rerank-sweep</code> | Required. Existing DNN sweep or optimize output directory. | <nobr><code>--sweep-dir dnn_sweep</code></nobr> |
| <nobr><code>--trial-seed-mode {fixed,indexed}</code></nobr> | <code>sweep</code>, <code>optimize</code> | Controls the seed used inside each sweep trial. `fixed` uses `--seed` for every trial so repeated candidates compare directly across sweeps. `indexed` restores the older `--seed + trial_number` behavior. Default: `fixed`. | <nobr><code>--trial-seed-mode fixed</code></nobr> |
| <nobr><code>--trial-worst-plots INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Number of lightweight worst-case S/Y PDF pairs generated and linked for each sweep trial. Default: `1`. | <nobr><code>--trial-worst-plots 1</code></nobr> |

#### Export and ADS integration

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--ads-hidden-layers INT</code></nobr> | <code>export-ads-ann</code> | Override ADS `AnnSetup.num_hidden_layers`. If omitted, this is derived from `--hidden-layers`. | <nobr><code>--ads-hidden-layers 3</code></nobr> |
| <nobr><code>--ads-iterations INT</code></nobr> | <code>export-ads-ann</code> | ADS ANN maximum training iterations. Default: `500`. | <nobr><code>--ads-iterations 1000</code></nobr> |
| <nobr><code>--ads-network-training-type {standard,adjoint,classification}</code></nobr> | <code>export-ads-ann</code> | ADS ANN training type. Use `standard` for normal S-parameter regression. Default: `standard`. | <nobr><code>--ads-network-training-type standard</code></nobr> |
| <nobr><code>--ads-neurons-per-layer INT</code></nobr> | <code>export-ads-ann</code> | Override ADS `AnnSetup.num_neurons_per_layer`. If omitted, this is derived from the average of `--hidden-layers`. | <nobr><code>--ads-neurons-per-layer 128</code></nobr> |
| <nobr><code>--ads-optimizer {quasi-newton,bayesian-regularization}</code></nobr> | <code>export-ads-ann</code> | ADS ANN modeler optimizer. `bayesian-regularization` can improve generalization at additional training cost. Default: `quasi-newton`. | <nobr><code>--ads-optimizer bayesian-regularization</code></nobr> |
| <nobr><code>--ads-output-format {all,verilog-a,c-code,equation,struct-scale}</code></nobr> | <code>export-ads-ann</code> | ADS ANN native artifact format. `all` requests every documented output. Default: `all`. | <nobr><code>--ads-output-format all</code></nobr> |
| <nobr><code>--ads-training-stop-tolerance FLOAT</code></nobr> | <code>export-ads-ann</code> | ADS ANN RMSE stop tolerance. Use `0` to rely on the iteration limit. Default: `0.0`. | <nobr><code>--ads-training-stop-tolerance 0</code></nobr> |
| <nobr><code>--freqs SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Frequency grid used with `--parameter-grid`. `SPEC` can be a comma list or `start:stop:count`. | <nobr><code>--freqs 1GHz:20GHz:401</code></nobr> |
| <nobr><code>--frequency-expression EXPR</code></nobr> | <code>export-veriloga</code> | Verilog-A expression for simulator frequency in Hz. Default: `$freq`. Change this only if your ADS Verilog-A release requires a different frequency expression. | <nobr><code>--frequency-expression '$freq'</code></nobr> |
| <nobr><code>--module-name NAME</code></nobr> | <code>export-veriloga</code> | Optional Verilog-A module name. If omitted, the exporter derives one from the output directory. | <nobr><code>--module-name my_dnn_4port</code></nobr> |
| <nobr><code>--no-fold-scalers</code></nobr> | <code>export-veriloga</code> | Debug option. Keep input/output standardization as explicit Verilog-A arithmetic instead of folding it into the first and final neural layers. Leaving this unset is faster. | <nobr><code>--no-fold-scalers</code></nobr> |
| <nobr><code>--parameter-grid NAME=SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Optional repeatable grid definition. `SPEC` can be a comma list or `start:stop:count`. Repeat once for every model parameter when not using `--template-mdif`. | <nobr><code>--parameter-grid W=0.40mm:0.80mm:9</code></nobr> |
| <nobr><code>--parameter-input-scales SCALE</code></nobr> | <code>export-veriloga</code> | Optional positive ADS/base-unit scale applied to every geometry/process parameter before it is fed to the trained model. Default: `1.0`. | <nobr><code>--parameter-input-scales 1um</code></nobr> |
| <nobr><code>--z0 FLOAT</code></nobr> | <code>export-veriloga</code> | Reference impedance used when exporting an S-output model and converting predicted S-parameters to admittance. Direct-Y models use the saved training `--target-z0` metadata instead. Default: `50.0`. | <nobr><code>--z0 50</code></nobr> |

---

## KBNN

This is the KBNN companion to the Neuro-TF prototype. It trains a neural model
from a fine/target S-parameter MDIF and, for knowledge-based modes, the
predictions of a frozen S-domain DNN previously fitted to the coarse response.

Supported forms:

```text
plain        : NN(geometry, frequency) -> fine S
residual     : coarse S + NN(geometry, frequency[, coarse S]) -> fine S
prior-input  : NN(geometry, frequency, coarse S) -> fine S
```

The default is `residual`, which is the classic knowledge-based difference
method: the fitted coarse DNN carries most of the physics and the KBNN learns
the remaining correction. Using the fitted response here makes training match
the two-network model used for prediction and self-contained export.

The trainer automatically floors zero-variance output scaler columns to a
representative response scale, which prevents constant residual or isolation
terms from becoming oversized learned delta-S errors in exported models.

### Expected MDIF Shape

Fine and coarse MDIF files use the same generic block structure as the Neuro-TF
trainer. Supply them together with `--mdif` and `--coarse-mdif`. The integrated
workflow fits the coarse DNN first and enforces the same parameter names/order
and S-parameter labels when fitting the fine KBNN.

```text
VAR dataset=train
VAR W=0.40mm
VAR L=1.20mm
BEGIN ACDATA
% Freq S11 S12 S21 S22
# Hz S RI R 50
1.0e9  0.08 -0.12  0 0  0.92 -0.10  0.08 -0.12
2.0e9  0.03 -0.18  0 0  0.73 -0.24  0.03 -0.18
END
```

KBNN evaluates the fitted coarse DNN directly at every fine-data geometry and
frequency point. The original coarse grid therefore does not need to match the
fine grid, provided the fitted DNN is valid across the fine model's domain.

### Inspect MDIF

Use `inspect-mdif` first when you want to confirm block count, S-parameter
labels, inferred numeric variables, split values, and frequency span.

```bash
python3 kbnn.py inspect-mdif \
  --mdif fine_train_verify.mdif
```

### Usage

Train one KBNN model with `train`:

```bash
python3 kbnn.py train \
  --mdif fine_train_verify.mdif \
  --coarse-mdif coarse_train_verify.mdif \
  --out-dir kbnn_model \
  --parameter-names W,L \
  --mode residual
```

Outputs:

- `model.npz` and `metadata.json`: trained fine KBNN/correction network
- `coarse_model/model.npz` and `coarse_model/metadata.json`: fitted frozen
  coarse S-domain DNN
- `coarse_model/`: the coarse model's training history, verification metrics,
  predicted verification MDIF, summary, and plots
- `composite_model_manifest.json`: both required model paths, file hashes,
  fit order, and a copyable self-contained Verilog-A extraction command
- `predicted_verification.mdif`: model predictions at verification points
- `verification_metrics.csv`: per-block and per-S-parameter errors, including EVM
- `verification_summary.json`: global error, passivity summary, plot paths
- `training_summary.md`: settings, final loss values, verification metrics,
  passivity summary, links to S- and Y-parameter worst-case plots, and
  copyable self-contained Verilog-A and sampled ADS MDIF export commands
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
default. The S-parameter plots include Smith/complex response grids along with
magnitude, phase, and error-focus pages. The matching Y-parameter plots under
`worst_case_y_plots/` show the admittance response as modeled-vs-measured
real/imaginary frequency plots. Use `--worst-plots 0` to skip both plot sets
during large experiments.

For `plain` models, omit both coarse-source options. For `residual` and
`prior-input`, use `--coarse-mdif` to fit and package both models together.
`--coarse-model-dir` remains available when intentionally reusing a previously
fitted coarse DNN. Residual targets are computed as
`fine - fitted_coarse_dnn`; when coarse inputs are enabled, those same
predictions are appended to the input. Prior-input mode always uses the
predictions as inputs. The KBNN metadata records relative and absolute coarse
model paths plus file hashes for later prediction and export.

### Sweeping / Optimizing

Use `sweep` or its alias `optimize` to try multiple KBNN configurations. The
command writes `kbnn_sweep_results.csv` and `kbnn_sweep_summary.md`, chooses
the best trial using `--selection-metric`, and keeps the current best completed
trial in `best_model/` as the sweep runs. This avoids a final refit after all
trials finish. When the sweep completes, it prints a copyable standalone
`train` command for the winning configuration and records that command in
`kbnn_best_config.json` and `kbnn_sweep_summary.md`.

```bash
python3 kbnn.py optimize \
  --mdif fine_train_verify.mdif \
  --coarse-mdif coarse_train_verify.mdif \
  --out-dir kbnn_sweep \
  --parameter-names W,L \
  --modes residual,prior-input \
  --include-coarse-inputs false,true \
  --freq-transforms log,linear \
  --hidden-layers '32;64;64,64' \
  --activations tanh,relu \
  --learning-rates 0.001,0.002,0.005 \
  --sparam-weights 'diag=1;offdiag=0.2' \
  --max-trials 24 \
  --selection-metric weighted_evm_pct \
  --require-passive
```

In a sweep, `--include-coarse-inputs` is the list of boolean values to
try for the single-model `--include-coarse-input` switch. `false` trains
residual candidates from geometry and frequency only; `true` also feeds the
coarse real/imaginary S-parameters into the residual network. Impossible mode
combinations are skipped: `plain` forces this off and `prior-input` forces it
on. With `--coarse-mdif`, the coarse DNN is fitted exactly once under
`kbnn_sweep/coarse_model/` and the same frozen model is evaluated for every
trial. The winning copyable `train` command repeats the integrated coarse fit
with the same coarse MDIF and settings, so it regenerates both saved networks.
The coarse package is also copied into `best_model/coarse_model/`, making
`best_model/` independently movable. Its `composite_model_manifest.json`
records both selected networks and the Verilog-A extraction command.

All integrated coarse-DNN controls listed for `train` (`--coarse-hidden-layers`,
`--coarse-epochs`, activation, learning rate, batch size, patience, frequency
transform, seed, weights, and reporting intervals) also apply to `optimize`.
They configure the one shared coarse fit, not separate per-trial fits.

For fitting failures that do not produce an obvious Python error, rerun a small
or representative sweep with `--debug --jobs 1`. The shared sweep debug mode
prints the selected candidate list and failed-trial tracebacks. KBNN also prints
per-trial block/sample counts, feature and target scaling statistics, constant
columns, and initial-to-final scaled losses, and each trial writes
`kbnn_training_debug.json` in its trial directory.

Use `--sparam-weights` to make some S-parameters matter less during training
and sweep selection. In residual KBNN mode, the weights apply to the residual
target for each S-parameter. Rules are applied left to right, so later rules
override earlier broad rules. Weights are normalized internally so their average
across S-parameter labels is 1.0 before they are applied to the training loss
and scale-sensitive weighted metrics.

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
CSV/PDF; the CSV records how many samples were excluded for each setting. During
a KBNN sweep, parsed MDIF
blocks, aligned/interpolated coarse responses, and prepared feature/target
matrices are cached inside each process. Repeated trials with the same data,
mode, coarse-input setting, and frequency transform reuse those arrays instead
of rebuilding them.

### Post-Run Sweep Reranking

Passivity is computed and saved for every sweep trial, regardless of the
selection metric used during the original run. If you later decide that the
best model should be the lowest-error passive candidate, rerank the existing
sweep instead of rerunning the whole optimization:

```bash
python3 kbnn.py rerank-sweep \
  --sweep-dir kbnn_sweep \
  --selection-metric weighted_evm_pct \
  --require-passive
```

This writes `kbnn_reranked_sweep_results.csv`,
`kbnn_reranked_sweep_summary.md`, `kbnn_reranked_best_config.json`, and
refreshed diagnostic artifacts under `sweep_diagnostics/`. The reranker accepts
both current `kbnn_sweep_results.csv` folders and older `sweep_results.csv`
folders.

If the original sweep used `--keep-trial-models`, the selected model can be
copied without retraining:

```bash
python3 kbnn.py rerank-sweep \
  --sweep-dir kbnn_sweep \
  --selection-metric weighted_evm_pct \
  --require-passive \
  --promote-best
```

Without kept trial models, reranking still identifies the winning
configuration, but the script cannot copy deleted `model.npz` files. In that
case, retrain only the selected configuration rather than rerunning the full
sweep.

### Predict

Predict new parameter blocks after training:

```bash
python3 kbnn.py predict \
  --model-dir kbnn_model \
  --mdif new_fine_shape.mdif \
  --out-mdif predicted.mdif
```

For residual and prior-input models, prediction evaluates the same frozen
coarse DNN used during KBNN training. The packaged relative path is used first,
then the recorded absolute path. If the coarse model was moved separately, pass
its new path with
`--coarse-model-dir`; the saved model and metadata hashes must still match.

### ADS MDIF Export

After training, export a parameterized S-parameter table that ADS can use
directly through an MDIF-capable data-based n-port or data access component.
For residual and prior-input KBNNs, the frozen fitted coarse DNN is evaluated
during export; ADS only needs the final exported fine-response MDIF.

The safest export is template driven: provide an MDIF containing the exact
geometry `VAR`s and frequency grids you want available in ADS. Placeholder
fine S-parameter values are accepted and ignored.

```bash
python3 kbnn.py export-ads-mdif \
  --model-dir kbnn_model \
  --out-dir ads_export \
  --template-mdif ads_sweep_template.mdif
```

You can also generate a rectangular parameter/frequency grid directly. The
exporter evaluates the packaged coarse DNN at every generated point. If that
model was moved separately, provide its new path with `--coarse-model-dir`.

```bash
python3 kbnn.py export-ads-mdif \
  --model-dir kbnn_model \
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

### ADS ANN Export

Use `export-ads-ann` when you want ADS ANN to train/extract the neural model
natively and emit the ADS ANN artifacts, including Verilog-A-oriented `.inc`,
C `.c`, text equation `.equation`, `.struc`, and `.scale` files.

```bash
python3 kbnn.py export-ads-ann \
  --mdif fine_train_verify.mdif \
  --coarse-mdif coarse_train_verify.mdif \
  --model-dir kbnn_sweep/best_model \
  --out-dir kbnn_ads_ann \
  --ads-ann-target native \
  --ads-iterations 1000 \
  --ads-output-format all
```

The default `--ads-ann-target native` preserves the KBNN formulation. In
`residual` mode the ADS ANN output is `delta_S*`, so the final response is
`coarse_S* + delta_S*`. Use `--ads-ann-target fine` when you want ADS ANN to
emit final fine S-parameter outputs directly; that is simpler to consume in ADS
but does not preserve the residual target that usually reduces sample count.

The export writes `ads_ann_training.csv`, optional
`ads_ann_verification.csv`, `ads_ann_manifest.json`, `train_ads_ann.py`, and
`ADS_ANN_README.md`. Run `train_ads_ann.py` with the ADS Python interpreter on
a licensed ADS machine. This path retrains the network in ADS ANN; it does not
import the local NumPy `model.npz` weights. Consequently, this separate ADS ANN
retraining workflow still accepts raw coarse MDIF data; it is distinct from
local KBNN fitting, prediction, sampled-MDIF export, and direct Verilog-A export.

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
   If coarse S-parameter columns are listed as inputs, the wrapper must evaluate
   or instantiate the coarse circuit response at the same parameter/frequency
   point and pass those values into the ANN.
3. For `--ads-ann-target fine`, interpret `output_columns` as final fine
   S-parameters. For the default native residual export, interpret
   `output_columns` as `delta_S*` and reconstruct
   `fine_Sij = coarse_Sij + delta_Sij`.
4. Convert the final complex S-matrix to a circuit relation before driving the
   schematic pins. For reference impedance `Z0`, use
   `Y = (I - S) * inverse(I + S) / Z0`, then `Iport = Y * Vport`.
5. Validate the wrapper in an S-parameter or AC simulation before circuit
   optimization.

### Direct Verilog-A Export

Use `export-veriloga` when you want a self-contained Verilog-A n-port instead
of exporting a sampled MDIF table or retraining with ADS ANN. A residual or
prior-input KBNN needs two saved models: the optimized KBNN and an S-domain DNN
trained on the coarse MDIF. The KBNN itself must have been trained with that
frozen DNN, so its fitted response is represented in both optimization and
export.

Fit the coarse DNN once and optimize the fine KBNN in one command. Supply
`--parameter-names` explicitly so both model inputs have the same order:

```bash
python3 kbnn.py optimize \
  --mdif fine_train_verify.mdif \
  --coarse-mdif coarse_train_verify.mdif \
  --out-dir kbnn_sweep \
  --parameter-names W,L \
  --coarse-hidden-layers 64,64 \
  --modes residual,prior-input
```

This writes the coarse DNN and all of its verification outputs under
`kbnn_sweep/coarse_model/`, the selected fine KBNN under
`kbnn_sweep/best_model/`, a packaged copy under
`kbnn_sweep/best_model/coarse_model/`, and a composite manifest under
`best_model/`. Review both training summaries and verification reports because
coarse-model approximation error contributes to the final composite response.

Finally, export the composite model. The packaged relative coarse-model path is
used automatically, so `best_model/` can be moved as one unit:

```bash
python3 kbnn.py export-veriloga \
  --model-dir kbnn_sweep/best_model \
  --out-dir kbnn_veriloga \
  --module-name my_kbnn_4port
```

The export writes `<module>.va`, `veriloga_manifest.json`, and
`VERILOGA_README.md`. The generated module evaluates the neural network at the
simulator frequency, reconstructs the complex S-matrix, converts it to
admittance with `Y = (I - S) * inverse(I + S) / Z0`, and contributes the
corresponding port currents. The model must contain a complete square
S-parameter matrix, such as all 16 terms for a four-port.

Plain KBNN exports contain one network. Residual and prior-input exports contain
two networks. The embedded coarse DNN evaluates
`Scoarse(geometry, frequency)` internally. A residual export adds the KBNN
correction to that response; a prior-input export feeds that response into the
KBNN. The final S-matrix is then converted to Y and stamped at the electrical
ports. No coarse MDIF, coarse circuit, coarse S-parameter instance settings, or
extra pins are needed in ADS.

For safety, residual and prior-input exports verify the coarse DNN's saved-model
and metadata hashes against the identity recorded during KBNN training. If the
joint output directory moved, its packaged relative path is used automatically.
If the coarse model moved separately, pass its new location with
`--coarse-model-dir`; a different model is rejected. `--allow-coarse-hooks`
restores the legacy zero-default hook
package only for fixed-point diagnostics or hand-written coarse equations; that
package is explicitly marked as not self-contained.

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

If all MDIF training parameters were scaled dimensionless values but the ADS
schematic uses base units, export with one common scale. For example, if the
MDIF parameter values are expressed in microns and ADS passes them in meters,
use `--parameter-input-scales 1um`. The generated Verilog-A divides every
fitted parameter by its generated input-scale parameter before evaluating both
the fine and embedded coarse networks.

### ADS Note

The `export-ads-mdif` command is the lowest-risk direct ADS handoff. It exports
the trained KBNN response onto a dense parameter/frequency table, so ADS can use
normal MDIF interpolation during circuit optimization without embedding Python,
NumPy, or the coarse circuit evaluator in the simulator. The `export-veriloga`
command embeds the KBNN and, for residual or prior-input mode, the supplied
coarse S-domain DNN into one Verilog-A n-port. The `export-ads-ann` command is
the native ADS ANN handoff for generating ADS ANN Verilog-A/C/equation
artifacts on an ADS machine.

### Options Reference

Options are grouped by purpose below. Rows are alphabetical within each table;
the **Subcommands** column includes accepted command aliases.

#### Files, data, and outputs

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--coarse-mdif PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Recommended coarse source for `residual` and `prior-input`. Fits an S-domain DNN first and saves its complete outputs under `<out-dir>/coarse_model/`. Mutually exclusive with `--coarse-model-dir`. | <nobr><code>--coarse-mdif coarse_train_verify.mdif</code></nobr> |
| <nobr><code>--coarse-model-dir PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>predict</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-veriloga</code> | Directory containing the frozen coarse DNN used by the KBNN. Training may create it from coarse MDIF data; prediction and export can use the packaged model or a validated relocated copy. | <nobr><code>--coarse-model-dir coarse_dnn_model</code></nobr> |
| <nobr><code>--coarse-verification-mdif PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Optional separate coarse/prior verification MDIF. Use this with `--verification-mdif` when fine and coarse verification data are stored separately. | <nobr><code>--coarse-verification-mdif coarse_verify.mdif</code></nobr> |
| <nobr><code>--mdif PATH</code></nobr> | <code>inspect-mdif</code>, <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>predict</code>, <code>export-ads-ann</code> | Fine/target MDIF to inspect, fit, predict, or use as an ADS ANN retraining source, depending on the subcommand. | <nobr><code>--mdif fine_train_verify.mdif</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | <code>predict</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-ann</code>, <code>export-veriloga</code> | Directory containing the trained <code>model.npz</code> and <code>metadata.json</code> used for prediction or export. | <nobr><code>--model-dir kbnn_model</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-ann</code>, <code>export-veriloga</code> | Destination directory for the model, sweep, or export artifacts generated by the selected command. | <nobr><code>--out-dir kbnn_model</code></nobr> |
| <nobr><code>--out-mdif PATH</code></nobr> | <code>predict</code> | Required. Output MDIF containing predicted S-parameters. | <nobr><code>--out-mdif predicted.mdif</code></nobr> |
| <nobr><code>--output-name NAME</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Output MDIF file name. Default: `surrogate_ads.mdif`. | <nobr><code>--output-name kbnn_ads.mdif</code></nobr> |
| <nobr><code>--output-prefix NAME</code></nobr> | <code>export-ads-ann</code> | Prefix for native ADS ANN outputs such as `.inc`, `.c`, `.equation`, `.scale`, and `.struc`. Default: `kbnn_ads_ann`. | <nobr><code>--output-prefix kbnn_filter_ann</code></nobr> |
| <nobr><code>--template-mdif PATH</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Optional. MDIF containing the exact geometry and frequency blocks to evaluate for ADS. Fine S-parameter values are ignored. | <nobr><code>--template-mdif ads_sweep_template.mdif</code></nobr> |
| <nobr><code>--verification-mdif PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Optional separate fine/target verification MDIF. When supplied, all blocks in `--mdif` are training blocks. | <nobr><code>--verification-mdif fine_verify.mdif</code></nobr> |

#### Data selection and loss weighting

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--coarse-frequency-weights SPEC</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Optional coarse-DNN frequency loss weights. Defaults to the fine `--frequency-weights`. | <nobr><code>--coarse-frequency-weights 'default=1;1GHz=4'</code></nobr> |
| <nobr><code>--coarse-sparam-weights SPEC</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Optional coarse-DNN loss weights. Defaults to the fine `--sparam-weights`. | <nobr><code>--coarse-sparam-weights 'diag=1;offdiag=0.2'</code></nobr> |
| <nobr><code>--frequency-weights SPEC</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Optional per-frequency fitting and sweep-selection weights. Exact frequencies and inclusive ranges are supported. | <nobr><code>--frequency-weights 'default=1;2GHz:4GHz=3'</code></nobr> |
| <nobr><code>--holdout-fraction FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Fraction of blocks reserved for verification when no split values are found. Default: `0.2`. | <nobr><code>--holdout-fraction 0.2</code></nobr> |
| <nobr><code>--parameter-names LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Comma-separated geometry/process variables used as KBNN and ADS ANN inputs. If omitted, common numeric `VAR`s are inferred. | <nobr><code>--parameter-names W,L,H</code></nobr> |
| <nobr><code>--sparam-weights SPEC</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Optional S-parameter fitting and sweep-selection weights. ADS ANN export records the stored or overridden weights in its manifest, but the generated ADS script cannot apply per-output weights because the documented ADS ANN API does not expose them. | <nobr><code>--sparam-weights 'diag=1;offdiag=0.2'</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | <code>inspect-mdif</code>, <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Name of the <code>VAR</code> used to split or summarize a combined MDIF. Default: <code>dataset</code>. | <nobr><code>--split-var dataset</code></nobr> |
| <nobr><code>--train-values LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Comma-separated split values that identify training blocks. Default: `train,training`. | <nobr><code>--train-values train,training</code></nobr> |
| <nobr><code>--verify-values LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Comma-separated split values that identify verification blocks. Default: `verify,verification,test,validation`. | <nobr><code>--verify-values verification,test</code></nobr> |

#### Model architecture and fitting

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--activation {tanh,relu}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Hidden-layer activation for one KBNN or ADS ANN configuration. In ADS, `tanh` maps to `HYPERBOLIC_TANGENT` and `relu` maps to `RELU`. Train default: `tanh`. | <nobr><code>--activation tanh</code></nobr> |
| <nobr><code>--activations LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated activations to try. `--activation` accepts one train-compatible value; `--activation-options` remains an alias. Default: `tanh,relu`. | <nobr><code>--activations tanh,relu</code></nobr> |
| <nobr><code>--batch-size INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Frequency-sample rows per Adam update in each candidate. Default: `256`. | <nobr><code>--batch-size 256</code></nobr> |
| <nobr><code>--coarse-activation {tanh,relu}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN hidden activation. Default: `tanh`. | <nobr><code>--coarse-activation tanh</code></nobr> |
| <nobr><code>--coarse-batch-size INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN batch size. Defaults to `--batch-size`. | <nobr><code>--coarse-batch-size 256</code></nobr> |
| <nobr><code>--coarse-epochs INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN maximum epochs. Defaults to `--epochs`. | <nobr><code>--coarse-epochs 2000</code></nobr> |
| <nobr><code>--coarse-freq-transform {log,linear,log-linear}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN frequency transform. Defaults to the fine KBNN `--freq-transform`. | <nobr><code>--coarse-freq-transform log-linear</code></nobr> |
| <nobr><code>--coarse-hidden-layers LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN hidden layout. Default: `64,64`. | <nobr><code>--coarse-hidden-layers 64,64</code></nobr> |
| <nobr><code>--coarse-learning-rate FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN Adam step size. Default: `0.002`. | <nobr><code>--coarse-learning-rate 0.002</code></nobr> |
| <nobr><code>--coarse-loss-interval INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN full-loss check interval. Defaults to `--loss-interval`. | <nobr><code>--coarse-loss-interval 5</code></nobr> |
| <nobr><code>--coarse-patience INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN early-stopping patience. Defaults to `--patience`. | <nobr><code>--coarse-patience 200</code></nobr> |
| <nobr><code>--coarse-progress-interval INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN console progress interval. Uses the same terminal-width-aware stderr redraw as the fine KBNN and other fits, so updates do not wrap into retained lines; completed fit metrics replace the status line. Defaults to `--progress-interval`. | <nobr><code>--coarse-progress-interval 25</code></nobr> |
| <nobr><code>--coarse-seed INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN random seed. Defaults to `--seed`. | <nobr><code>--coarse-seed 1234</code></nobr> |
| <nobr><code>--coarse-worst-plots INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Coarse-DNN worst verification plots. Defaults to `--worst-plots`. | <nobr><code>--coarse-worst-plots 6</code></nobr> |
| <nobr><code>--debug</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Enable KBNN data/loss diagnostics and tracebacks. Sweeps also print the candidate list and retain per-trial debug output; use `--jobs 1` for the cleanest trace. | <nobr><code>--debug</code></nobr> |
| <nobr><code>--epochs INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Maximum Adam training epochs for one fit or each sweep candidate. Early stopping may finish sooner. Default: `2000`. | <nobr><code>--epochs 2000</code></nobr> |
| <nobr><code>--freq-transform {log,linear}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Frequency input transform. `log` uses `log10(freq_hz)` and is usually better for wideband data. Default: `log`. | <nobr><code>--freq-transform log</code></nobr> |
| <nobr><code>--freq-transforms LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated frequency transforms to try. `--freq-transform` accepts one train-compatible value; `--freq-transform-options` remains an alias. | <nobr><code>--freq-transforms log,linear</code></nobr> |
| <nobr><code>--hidden-layers LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Comma-separated hidden-layer sizes for one model. Sweeps also accept semicolon-separated candidate layouts. Train default: `64,64`; sweep default: `32;64;64,64`. `--hidden-layer-layouts` and `--hidden-layer-options` remain aliases. | <nobr><code>--hidden-layers 64,64</code></nobr> |
| <nobr><code>--include-coarse-input</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | In `residual` mode, append coarse real/imaginary S-parameters to the NN input vector. This can improve accuracy if the correction depends strongly on the coarse response. Forced on for `prior-input` and off for `plain`. | <nobr><code>--include-coarse-input</code></nobr> |
| <nobr><code>--include-coarse-inputs LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated boolean candidates for `--include-coarse-input`. Supplying the singular flag selects only `true`; `--include-coarse-input-options` remains an alias. Default: `false,true`. | <nobr><code>--include-coarse-inputs false,true</code></nobr> |
| <nobr><code>--learning-rate FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Adam step size. Default: `0.002`. | <nobr><code>--learning-rate 0.002</code></nobr> |
| <nobr><code>--learning-rates LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated Adam learning rates. `--learning-rate` accepts one train-compatible value. Default: `0.001,0.002,0.005`. | <nobr><code>--learning-rates 0.001,0.002,0.005</code></nobr> |
| <nobr><code>--loss-interval INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Full train/verification loss check interval in epochs. Increasing this reduces full-dataset scoring overhead during long runs while early stopping still uses epoch-based patience. Default: `1`. | <nobr><code>--loss-interval 5</code></nobr> |
| <nobr><code>--mode {plain,residual,prior-input}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | KBNN formulation. `residual` learns `fine - fitted_coarse_dnn`; `prior-input` predicts fine S using fitted coarse-DNN predictions as inputs; `plain` uses no coarse model. Default: `residual`. | <nobr><code>--mode residual</code></nobr> |
| <nobr><code>--modes LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated KBNN model modes. The singular `--mode` accepts one train-compatible value; `--mode-options` remains an alias. Default: `plain,residual,prior-input`. | <nobr><code>--modes residual,prior-input</code></nobr> |
| <nobr><code>--patience INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Early-stopping patience in epochs for each candidate. Use `0` to disable. Default: `200`. | <nobr><code>--patience 200</code></nobr> |
| <nobr><code>--progress-interval INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Console progress update interval in epochs. Updates redraw one terminal status line and include epoch count, elapsed time, and loss values when that epoch also matches `--loss-interval`. Use `0` to disable. Default: `25`. | <nobr><code>--progress-interval 10</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Random seed for data splitting, model initialization, minibatch order, ADS ANN data preparation, and sweep candidate selection where applicable. Default: `1234`. | <nobr><code>--seed 1234</code></nobr> |
| <nobr><code>--worst-plots INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Number of worst verification S/Y plot pairs to generate. In a sweep it applies to a final `--retrain-best`; otherwise the promoted trial retains its `--trial-worst-plots` output. Default: `6`. | <nobr><code>--worst-plots 6</code></nobr> |

#### Sweep and model selection

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--best-model-dir PATH</code></nobr> | <code>rerank-sweep</code> | Destination for `--promote-best`. Default: `<sweep-dir>/best_model_reranked`. | <nobr><code>--best-model-dir kbnn_sweep/best_model_passive</code></nobr> |
| <nobr><code>--jobs INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Number of sweep trials to train in parallel. Use up to the number of physical cores and lower it if memory use gets high. Default: `1`. | <nobr><code>--jobs 4</code></nobr> |
| <nobr><code>--keep-trial-models</code></nobr> | <code>sweep</code>, <code>optimize</code> | Keep full per-trial model directories under `trials/`. By default, each trial keeps lightweight summary and plot artifacts while large model files are removed. | <nobr><code>--keep-trial-models</code></nobr> |
| <nobr><code>--max-passivity-sigma FLOAT</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials whose worst predicted S-matrix singular value is at or below this value when selecting `best_model/`. | <nobr><code>--max-passivity-sigma 1.000001</code></nobr> |
| <nobr><code>--max-passivity-violations INT</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials with this many or fewer passivity-violating frequency points when selecting `best_model/`. | <nobr><code>--max-passivity-violations 0</code></nobr> |
| <nobr><code>--max-trials INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Maximum candidate configurations to evaluate. Default: `24`. | <nobr><code>--max-trials 24</code></nobr> |
| <nobr><code>--overwrite</code></nobr> | <code>rerank-sweep</code> | Allow `--promote-best` to replace an existing `--best-model-dir`. | <nobr><code>--overwrite</code></nobr> |
| <nobr><code>--promote-best</code></nobr> | <code>rerank-sweep</code> | Copy the selected trial model to `--best-model-dir` if that trial still contains `model.npz` and `metadata.json`. Requires the original sweep to have used `--keep-trial-models`. | <nobr><code>--promote-best</code></nobr> |
| <nobr><code>--replace-current-best</code></nobr> | <code>rerank-sweep</code> | Overwrite `<sweep-dir>/best_model` with the selected trial model if the trial model files are available. | <nobr><code>--replace-current-best</code></nobr> |
| <nobr><code>--require-passive</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials with zero passivity-violating frequency points when selecting `best_model/`. Equivalent to `--max-passivity-violations 0` unless a stricter value is supplied. | <nobr><code>--require-passive</code></nobr> |
| <nobr><code>--retrain-best</code></nobr> | <code>sweep</code>, <code>optimize</code> | Retrain the selected best configuration at the end of the sweep instead of using the best completed trial model promoted during the sweep. Use this when you want `--worst-plots` to apply only to the final model. | <nobr><code>--retrain-best</code></nobr> |
| <nobr><code>--search-mode {grid,random}</code></nobr> | <code>sweep</code>, <code>optimize</code> | Search strategy. Legacy `--mode grid` and `--mode random` remain valid. Default: `random`. | <nobr><code>--search-mode random</code></nobr> |
| <nobr><code>--selection-metric NAME</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Metric minimized to choose the best model. Options include `evm_pct`, `rmse_abs`, passivity metrics, and weighted metrics such as `weighted_evm_pct` and `weighted_rmse_abs`. Default: `rmse_abs`. | <nobr><code>--selection-metric weighted_evm_pct</code></nobr> |
| <nobr><code>--sweep-dir PATH</code></nobr> | <code>rerank-sweep</code> | Required. Existing KBNN sweep or optimize output directory. | <nobr><code>--sweep-dir kbnn_sweep</code></nobr> |
| <nobr><code>--trial-seed-mode {fixed,indexed}</code></nobr> | <code>sweep</code>, <code>optimize</code> | Controls the seed used inside each sweep trial. `fixed` uses `--seed` for every trial so repeated candidates compare directly across sweeps. `indexed` restores the older `--seed + trial_number` behavior. Default: `fixed`. | <nobr><code>--trial-seed-mode fixed</code></nobr> |
| <nobr><code>--trial-worst-plots INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Number of lightweight worst-case S/Y PDF pairs generated and linked for each sweep trial. Default: `1`. | <nobr><code>--trial-worst-plots 1</code></nobr> |

#### Export and ADS integration

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--ads-ann-target {native,fine}</code></nobr> | <code>export-ads-ann</code> | ADS ANN target definition. `native` preserves the KBNN target, so residual mode outputs `delta_S*`; `fine` trains ADS ANN to output final fine S-parameters directly. Default: `native`. | <nobr><code>--ads-ann-target native</code></nobr> |
| <nobr><code>--ads-hidden-layers INT</code></nobr> | <code>export-ads-ann</code> | Override ADS `AnnSetup.num_hidden_layers`. If omitted, this is derived from `--hidden-layers`. | <nobr><code>--ads-hidden-layers 2</code></nobr> |
| <nobr><code>--ads-iterations INT</code></nobr> | <code>export-ads-ann</code> | ADS ANN maximum training iterations. Default: `500`. | <nobr><code>--ads-iterations 1000</code></nobr> |
| <nobr><code>--ads-network-training-type {standard,adjoint,classification}</code></nobr> | <code>export-ads-ann</code> | ADS ANN training type. Use `standard` for normal S-parameter regression. Default: `standard`. | <nobr><code>--ads-network-training-type standard</code></nobr> |
| <nobr><code>--ads-neurons-per-layer INT</code></nobr> | <code>export-ads-ann</code> | Override ADS `AnnSetup.num_neurons_per_layer`. If omitted, this is derived from the average of `--hidden-layers`. | <nobr><code>--ads-neurons-per-layer 64</code></nobr> |
| <nobr><code>--ads-optimizer {quasi-newton,bayesian-regularization}</code></nobr> | <code>export-ads-ann</code> | ADS ANN modeler optimizer. `bayesian-regularization` can improve generalization at additional training cost. Default: `quasi-newton`. | <nobr><code>--ads-optimizer bayesian-regularization</code></nobr> |
| <nobr><code>--ads-output-format {all,verilog-a,c-code,equation,struct-scale}</code></nobr> | <code>export-ads-ann</code> | ADS ANN native artifact format. `all` requests every documented output. Default: `all`. | <nobr><code>--ads-output-format all</code></nobr> |
| <nobr><code>--ads-training-stop-tolerance FLOAT</code></nobr> | <code>export-ads-ann</code> | ADS ANN RMSE stop tolerance. Use `0` to rely on the iteration limit. Default: `0.0`. | <nobr><code>--ads-training-stop-tolerance 0</code></nobr> |
| <nobr><code>--allow-coarse-hooks</code></nobr> | <code>export-veriloga</code> | Explicitly allow the legacy non-self-contained residual/prior-input export when `--coarse-model-dir` is omitted. The generated coarse values default to zero and are intended only for fixed-point diagnostics or hand-written equations. | <nobr><code>--allow-coarse-hooks</code></nobr> |
| <nobr><code>--freqs SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Frequency grid used with `--parameter-grid`. `SPEC` can be a comma list or `start:stop:count`. | <nobr><code>--freqs 1GHz:20GHz:401</code></nobr> |
| <nobr><code>--frequency-expression EXPR</code></nobr> | <code>export-veriloga</code> | Verilog-A expression for simulator frequency in Hz. Default: `$freq`. Change this only if your ADS Verilog-A release requires a different frequency expression. | <nobr><code>--frequency-expression '$freq'</code></nobr> |
| <nobr><code>--module-name NAME</code></nobr> | <code>export-veriloga</code> | Optional Verilog-A module name. If omitted, the exporter derives one from the output directory. | <nobr><code>--module-name my_kbnn_4port</code></nobr> |
| <nobr><code>--parameter-grid NAME=SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Optional repeatable grid definition. `SPEC` can be a comma list or `start:stop:count`. Repeat once for every model parameter when not using `--template-mdif`. | <nobr><code>--parameter-grid W=0.40mm:0.80mm:9</code></nobr> |
| <nobr><code>--parameter-input-scales SCALE</code></nobr> | <code>export-veriloga</code> | Optional positive ADS/base-unit scale applied to every geometry/process parameter before it is fed to the trained fine and coarse networks. Default: `1.0`. | <nobr><code>--parameter-input-scales 1um</code></nobr> |
| <nobr><code>--z0 FLOAT</code></nobr> | <code>export-veriloga</code> | Reference impedance used when converting predicted S-parameters to admittance. Default: `50.0`. | <nobr><code>--z0 50</code></nobr> |

---

## Neuro-TF

This is a self-contained prototype for training a Neuro-transfer-function
surrogate from parameterized S-parameter MDIF data.

Model structure:

```text
geometry/process VARs -> small neural network -> rational TF coefficients -> S-parameters
```

The rational transfer functions use fixed stable poles, so coefficient
extraction for each geometry is linear least squares. The neural network then
learns the geometry-to-coefficients map.

### Expected MDIF Shape

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

### Inspect MDIF

Use `inspect-mdif` first when you want to confirm block count, S-parameter
labels, inferred numeric variables, split values, and frequency span.

```bash
python3 neuro_tf.py inspect-mdif \
  --mdif train_verify.mdif
```

### Usage

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
- `training_summary.md`: settings, metrics, plot links, and copyable
  self-contained Verilog-A and sampled ADS MDIF export commands
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
links to the generated S- and Y-parameter worst-case plots, and copyable
self-contained Verilog-A and sampled ADS MDIF export commands.

### Sweeping / Optimizing

Use `sweep` or its alias `optimize` to try multiple rational orders and neural
network settings. The command writes `neurotf_sweep_results.csv` and
`neurotf_sweep_summary.md`, chooses the best trial using `--selection-metric`,
and keeps the current best completed trial in `best_model/` as the sweep runs.
This avoids a final refit after all trials finish. When the sweep completes,
it prints a copyable standalone `train` command for the winning configuration
and records that command in `neurotf_best_config.json` and
`neurotf_sweep_summary.md`.

```bash
python3 neuro_tf.py optimize \
  --mdif train_verify.mdif \
  --out-dir neuro_tf_sweep \
  --parameter-names W,L,H \
  --orders 8,10,12,16 \
  --hidden-layers '32;64;64,64;128,64' \
  --activations tanh,relu \
  --learning-rates 0.001,0.002,0.005 \
  --search-mode random \
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
- `weighted_rmse_abs`, `weighted_evm_pct`, and the other `weighted_*`
  variants: the same metrics using `--frequency-weights`

Set `--search-mode grid` to exhaustively test all combinations, or keep the default
`--search-mode random --max-trials N` for direct hyperparameter optimization over a
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

### Predict

Predict new parameter blocks after training:

```bash
python3 neuro_tf.py predict \
  --model-dir neuro_tf_model \
  --mdif new_parameter_blocks.mdif \
  --out-mdif predicted.mdif
```

### Direct Verilog-A Export

Export the saved geometry-to-coefficient network and its fixed rational poles
as one self-contained Verilog-A n-port:

```bash
python3 neuro_tf.py export-veriloga \
  --model-dir neuro_tf_model \
  --out-dir neuro_tf_veriloga \
  --module-name my_neuro_tf_4port \
  --parameter-input-scales 1.0
```

The generated module evaluates the neural coefficient map, constructs each
S-parameter from `c0 + sum(c_k / (j*f/f_scale - pole_k))`, converts the complete
S-matrix to Y, and stamps the small-signal port currents. It requires no Python
runtime or MDIF table in ADS. The package contains `<module>.va`,
`veriloga_manifest.json`, and `VERILOGA_README.md`.

This export is intended for S-parameter and small-signal AC analysis. Validate
it against `predicted_verification.mdif` with the target ADS Verilog-A compiler
before using it in optimization.

### Export Sampled ADS MDIF

Export the fitted fixed-pole Neuro-TF response on either the exact geometry and
frequency blocks of a template MDIF or an explicit parameter/frequency grid:

```bash
python3 neuro_tf.py export-ads-mdif \
  --model-dir neuro_tf_model \
  --out-dir neuro_tf_ads_export \
  --template-mdif ads_sweep_template.mdif
```

The command writes `surrogate_ads.mdif`, `ads_model_manifest.json`, and
`ADS_README.md`. The template's S-parameter values are ignored; only its
parameter blocks and frequency grids are used. Instead of `--template-mdif`,
repeat `--parameter-grid` once for each model parameter and supply `--freqs`.

### ADS Note

Use `export-ads-mdif` for the lowest-risk interpolation-based handoff, or
`export-veriloga` when ADS should evaluate the trained coefficient network and
fixed-pole rational response directly. The Verilog-A package is self-contained;
the sampled MDIF remains useful as a simulator-independent reference for
cross-checking the exported component.

### Options Reference

Options are grouped by purpose below. Rows are alphabetical within each table;
the **Subcommands** column includes accepted command aliases.

#### Files, data, and outputs

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--mdif PATH</code></nobr> | <code>inspect-mdif</code>, <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>predict</code> | Input MDIF to inspect, fit, or predict, depending on the subcommand. | <nobr><code>--mdif train_verify.mdif</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | <code>predict</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-veriloga</code> | Directory containing the trained <code>model.npz</code> and <code>metadata.json</code> used for prediction or export. | <nobr><code>--model-dir neuro_tf_model</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-veriloga</code> | Destination directory for the model, sweep, or export artifacts generated by the selected command. | <nobr><code>--out-dir neuro_tf_model</code></nobr> |
| <nobr><code>--out-mdif PATH</code></nobr> | <code>predict</code> | Required. Output MDIF containing predicted S-parameters. | <nobr><code>--out-mdif predicted.mdif</code></nobr> |
| <nobr><code>--output-name NAME</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Exported MDIF file name. Default: `surrogate_ads.mdif`. | <nobr><code>--output-name neuro_tf_ads.mdif</code></nobr> |
| <nobr><code>--template-mdif PATH</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | MDIF whose parameter/frequency blocks define the export sampling grid. Mutually exclusive in practice with the explicit-grid form. | <nobr><code>--template-mdif ads_sweep_template.mdif</code></nobr> |
| <nobr><code>--verification-mdif PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Optional. Separate MDIF containing verification blocks. When supplied, every block in `--mdif` is treated as training data and every block in this file is treated as verification data. | <nobr><code>--verification-mdif verify.mdif</code></nobr> |

#### Data selection and loss weighting

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--frequency-weights SPEC</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Optional weights for the rational least-squares coefficient fit and weighted sweep-selection metrics. Exact frequencies and inclusive ranges are supported. | <nobr><code>--frequency-weights 'default=1;2GHz:4GHz=3'</code></nobr> |
| <nobr><code>--holdout-fraction FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Fraction of blocks to reserve for verification when no split values are found in a combined MDIF. Default: `0.2`. | <nobr><code>--holdout-fraction 0.25</code></nobr> |
| <nobr><code>--parameter-names LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Comma-separated geometry/process variable names to use as neural-network inputs. If omitted, the trainer infers numeric `VAR`s common to all blocks, excluding the split variable. | <nobr><code>--parameter-names W,L,H</code></nobr> |
| <nobr><code>--split-var NAME</code></nobr> | <code>inspect-mdif</code>, <code>train</code>, <code>sweep</code>, <code>optimize</code> | Name of the <code>VAR</code> used to split or summarize a combined MDIF. Default: <code>dataset</code>. | <nobr><code>--split-var dataset</code></nobr> |
| <nobr><code>--train-values LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Comma-separated values of `--split-var` that identify training blocks. Default: `train,training`. | <nobr><code>--train-values train,training</code></nobr> |
| <nobr><code>--verify-values LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Comma-separated values of `--split-var` that identify verification blocks. Default: `verify,verification,test,validation`. | <nobr><code>--verify-values verification,test</code></nobr> |

#### Model architecture and fitting

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--activation {tanh,relu}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Hidden-layer activation. `tanh` is usually smoother for microwave response fitting; `relu` can be useful for larger datasets. Default: `tanh`. | <nobr><code>--activation tanh</code></nobr> |
| <nobr><code>--activations LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated activation functions. `--activation` accepts one train-compatible value; `--activation-options` remains an alias. Default: `tanh,relu`. | <nobr><code>--activations tanh,relu</code></nobr> |
| <nobr><code>--batch-size INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Number of training geometries per Adam update. The implementation clamps this to the number of available training blocks. Default: `64`. | <nobr><code>--batch-size 64</code></nobr> |
| <nobr><code>--debug</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Enable diagnostic output and command tracebacks. Sweeps also print the candidate list and retain failed-trial tracebacks; use `--jobs 1` for the cleanest trace. | <nobr><code>--debug</code></nobr> |
| <nobr><code>--epochs INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Maximum Adam training epochs. Early stopping may stop before this value. Default: `2000`. | <nobr><code>--epochs 2000</code></nobr> |
| <nobr><code>--hidden-layers LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Comma-separated hidden-layer sizes for one model. Sweeps also accept semicolon-separated candidate layouts. Train default: `64,64`; sweep default: `32;64;64,64`. `--hidden-layer-layouts` and `--hidden-layer-options` remain aliases. | <nobr><code>--hidden-layers 64,64</code></nobr> |
| <nobr><code>--learning-rate FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Adam optimizer step size. Lower values are safer; higher values may converge faster but can overshoot. Default: `0.002`. | <nobr><code>--learning-rate 0.002</code></nobr> |
| <nobr><code>--learning-rates LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated Adam learning rates. `--learning-rate` accepts one train-compatible value. Default: `0.001,0.002,0.005`. | <nobr><code>--learning-rates 0.001,0.002,0.005</code></nobr> |
| <nobr><code>--loss-interval INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Full train/verification loss check interval in epochs. Increasing this reduces full-dataset scoring overhead during long runs while early stopping still uses epoch-based patience. Default: `1`. | <nobr><code>--loss-interval 5</code></nobr> |
| <nobr><code>--order INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Number of fixed stable rational poles used for each S-parameter transfer function. Higher values can fit sharper frequency behavior but increase coefficient count and NN output dimension. Default: `10`. | <nobr><code>--order 12</code></nobr> |
| <nobr><code>--orders LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated rational pole counts. `--order` accepts one train-compatible value. Default: `6,10,14`. | <nobr><code>--orders 8,10,12,16</code></nobr> |
| <nobr><code>--patience INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Early-stopping patience measured in epochs without validation-loss improvement. Use `0` to disable early stopping. Default: `200`. | <nobr><code>--patience 200</code></nobr> |
| <nobr><code>--pole-damping FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Real-part damping factor for the fixed pole grid. Larger values make poles more damped and smoother; smaller values can follow sharper resonances but may be more sensitive. Default: `0.18`. | <nobr><code>--pole-damping 0.18</code></nobr> |
| <nobr><code>--pole-dampings LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated pole damping values. `--pole-damping` accepts one train-compatible value. Default: `0.12,0.18,0.28`. | <nobr><code>--pole-dampings 0.12,0.18,0.28</code></nobr> |
| <nobr><code>--progress-interval INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Console progress update interval in epochs. Updates redraw one terminal status line and include epoch count, elapsed time, and loss values when that epoch also matches `--loss-interval`. Use `0` to disable. Default: `25`. | <nobr><code>--progress-interval 10</code></nobr> |
| <nobr><code>--ridge FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Ridge regularization used during linear least-squares TF coefficient fitting. Increase this if coefficient fits become noisy or ill-conditioned. Default: `1e-8`. | <nobr><code>--ridge 1e-8</code></nobr> |
| <nobr><code>--ridges LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated coefficient-fit ridge values. `--ridge` accepts one train-compatible value; `--ridge-values` remains an alias. Default: `1e-10,1e-8,1e-6`. | <nobr><code>--ridges 1e-10,1e-8,1e-6</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Random seed for data splitting, model initialization, minibatch order, and sweep candidate selection where applicable. Default: `1234`. | <nobr><code>--seed 1234</code></nobr> |
| <nobr><code>--worst-plots INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Number of worst verification fits to render as PDFs. Each selected case gets an S-parameter plot and a Y-parameter implementation-view plot. Ranking uses max absolute complex response error, with RMSE also reported in the title and plot index CSV. Use `0` to skip plot generation. Default: `6`. | <nobr><code>--worst-plots 6</code></nobr> |

#### Sweep and model selection

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--jobs INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Number of sweep trials to train in parallel. Use up to the number of physical cores and lower it if memory use gets high. Default: `1`. | <nobr><code>--jobs 4</code></nobr> |
| <nobr><code>--keep-trial-models</code></nobr> | <code>sweep</code>, <code>optimize</code> | Keep full per-trial model directories under `trials/`. By default, each trial keeps lightweight summary and plot artifacts while large model files are removed. | <nobr><code>--keep-trial-models</code></nobr> |
| <nobr><code>--max-passivity-sigma FLOAT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Only consider trials whose worst predicted S-matrix singular value is at or below this value when selecting `best_model/`. | <nobr><code>--max-passivity-sigma 1.000001</code></nobr> |
| <nobr><code>--max-passivity-violations INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Only consider trials with this many or fewer passivity-violating frequency points when selecting `best_model/`. | <nobr><code>--max-passivity-violations 0</code></nobr> |
| <nobr><code>--max-trials INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Maximum number of candidate configurations to evaluate. In `random` mode this limits the random sample; in `grid` mode it truncates the product list. Default: `24`. | <nobr><code>--max-trials 40</code></nobr> |
| <nobr><code>--require-passive</code></nobr> | <code>sweep</code>, <code>optimize</code> | Only consider trials with zero passivity-violating frequency points when selecting `best_model/`. Equivalent to `--max-passivity-violations 0` unless a stricter value is supplied. | <nobr><code>--require-passive</code></nobr> |
| <nobr><code>--retrain-best</code></nobr> | <code>sweep</code>, <code>optimize</code> | Retrain the selected best configuration at the end of the sweep instead of using the best completed trial model promoted during the sweep. Use this when you want `--worst-plots` to apply only to the final model. | <nobr><code>--retrain-best</code></nobr> |
| <nobr><code>--search-mode {grid,random}</code></nobr> | <code>sweep</code>, <code>optimize</code> | Search strategy. Legacy `--mode` remains an alias. Default: `random`. | <nobr><code>--search-mode random</code></nobr> |
| <nobr><code>--selection-metric NAME</code></nobr> | <code>sweep</code>, <code>optimize</code> | Metric minimized when choosing the best trial. Includes unweighted error, passivity, and `weighted_*` metrics that apply `--frequency-weights`. Default: `rmse_abs`. | <nobr><code>--selection-metric weighted_rmse_abs</code></nobr> |
| <nobr><code>--trial-seed-mode {fixed,indexed}</code></nobr> | <code>sweep</code>, <code>optimize</code> | Controls the seed used inside each sweep trial. `fixed` uses `--seed` for every trial so repeated candidates compare directly across sweeps. `indexed` restores the older `--seed + trial_number` behavior. Default: `fixed`. | <nobr><code>--trial-seed-mode fixed</code></nobr> |
| <nobr><code>--trial-worst-plots INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Number of lightweight worst-case S/Y PDF pairs generated and linked for each sweep trial. Default: `1`. | <nobr><code>--trial-worst-plots 1</code></nobr> |

#### Export and ADS integration

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--freqs SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Frequency grid used with `--parameter-grid`. | <nobr><code>--freqs 1GHz:20GHz:401</code></nobr> |
| <nobr><code>--frequency-expression EXPR</code></nobr> | <code>export-veriloga</code> | Verilog-A expression for simulator frequency in Hz. Default: `$freq`. | <nobr><code>--frequency-expression '$freq'</code></nobr> |
| <nobr><code>--module-name NAME</code></nobr> | <code>export-veriloga</code> | Optional Verilog-A module name. If omitted, the exporter derives one from the model directory. | <nobr><code>--module-name my_neuro_tf_4port</code></nobr> |
| <nobr><code>--parameter-grid SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Explicit grid for one model parameter. Repeat once per parameter; requires `--freqs`. | <nobr><code>--parameter-grid W=0.4mm:0.8mm:9</code></nobr> |
| <nobr><code>--parameter-input-scales SCALE</code></nobr> | <code>export-veriloga</code> | Optional positive ADS/base-unit scale applied to every geometry/process parameter before it is fed to the trained coefficient network. Default: `1.0`. | <nobr><code>--parameter-input-scales 1um</code></nobr> |
| <nobr><code>--z0 FLOAT</code></nobr> | <code>export-veriloga</code> | Reference impedance used when converting predicted S-parameters to admittance. Default: `50.0`. | <nobr><code>--z0 50</code></nobr> |
