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
self-contained Verilog-A n-ports. All three also export self-contained, linear
ADS SDD subnetworks for harmonic balance. DNN and KBNN additionally support
native ADS ANN package generation.

## Repository Layout

```text
.
|-- dnn.py                                Direct DNN CLI and implementation
|-- kbnn.py                               KBNN CLI and implementation
|-- neuro_tf.py                           Neuro-TF CLI and implementation
|-- surrogate_common.py                   Shared training, reporting, and ADS export utilities
|-- generate_points.py                    Geometry/process point-set generator
|-- de_generated_scripts/
|   `-- parse_ads_hb_solver_log.py        ADS Newton/Krylov status-log comparison utility
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
Verilog-A/SDD file generation. An ADS installation is required to compile and
simulate the generated circuit models. The generated ADS ANN training package must be run on
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
simulation queues independently. Use `--decimal-places N` to round generated
parameter values to at most `N` decimal places in each parameter's declared
unit; normalized coordinates are recalculated from those rounded values. During
a range extension, this applies to the newly appended points while the original
CSV rows remain unchanged.

Every geometry CSV also gets an automatic same-stem JSON file. For example,
`geometries.csv` produces `geometries.json`. The JSON records the generation
method, point and dataset counts, and each parameter's lower bound, upper bound,
unit, base-unit bounds, and linear/log scale. When `--write-split-files` is
used, the one JSON describes the complete combined geometry; the separate
train/verification CSVs do not receive duplicate JSON files. Targeted
additional-point CSVs receive their own JSON files.

### Using GP to Determine Additional Points

This is the direct workflow for using the Gaussian-process selector to choose
new EM geometries. It is separate from GP hyperparameter optimization: this
command models the current surrogate's **geometry-level verification error**
and writes the next physical data points to simulate.

The GP workflow requires an existing DNN, KBNN, or Neuro-TF fit containing
`verification_metrics.csv`; it cannot choose an informed first design before
any verification errors exist.

#### First GP Addition Round

Assume:

- `geometries.csv` contains every geometry already simulated;
- `geometries.json` is the companion metadata written when that CSV was
  generated and contains the parameter names, ranges, units, and linear/log
  scaling;
- `outputs/dnn_adaptive/best_model/` is the winning optimize result; and
- the fit's verification metrics use the same geometry parameters.

Request eight additional training geometries with this complete command:

```bash
python3 generate_points.py suggest-additional \
  --count 8 \
  --fit-dir outputs/dnn_adaptive/best_model \
  --existing-points geometries.csv \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --candidate-count 1600 \
  --lhs-candidates 32 \
  --metric evm_pct \
  --exploration-weight 2.0 \
  --gp-noise-variance 1e-6 \
  --novelty-power 1.0 \
  --min-distance 0.05 \
  --seed 1234 \
  --decimal-places 4 \
  --include-normalized \
  --target-dataset train \
  --analysis-out outputs/gp_round_1_error_regions.csv \
  --out outputs/gp_round_1_points.csv
```

The important inputs are:

| Input | Purpose |
| --- | --- |
| `--fit-dir` | Directory containing the current fit's `verification_metrics.csv`. For a successful optimize run, use its `best_model/` directory. |
| `--existing-points` | CSV of already simulated geometries that must not be suggested again. Its same-stem JSON is loaded automatically as the parameter domain. Repeat for multiple CSVs. `--existing-mdif` can be used together with it. |
| `--parameter-json` | Explicit geometry metadata source when no original point CSV is supplied, such as an MDIF-only workflow. It is normally unnecessary. |
| `--parameter` | Optional backward-compatible domain override. Repeat it for every parameter only when intentionally bypassing the generated JSON. |
| `--count` | Number of new expensive geometries to return, not candidate-pool size. Start with roughly one or two points per dimension. |
| `--candidate-count` | Number of inexpensive candidate locations scored by the GP. Only the best `--count` locations are written for simulation. |
| `--metric` | Per-row error column from `verification_metrics.csv`. Typical choices are `evm_pct`, `rmse_abs`, and `max_abs`; `auto` chooses an available metric. |
| `--exploration-weight` | GP-UCB uncertainty weight. `2.0` is balanced; smaller values exploit known high-error regions and larger values explore uncertain regions. |
| `--novelty-power` and `--min-distance` | Encourage separation from simulated points and from other points selected in the same batch. Distances use normalized geometry coordinates. |

The command writes three analysis artifacts:

- `outputs/gp_round_1_points.csv`: the eight new geometries to simulate;
- `outputs/gp_round_1_points.json`: parameter ranges and GP/acquisition
  metadata; and
- `outputs/gp_round_1_error_regions.csv`: current verification geometries
  ranked by the error used to fit the GP.

The new-point CSV includes `predicted_error`, `gp_log_uncertainty`,
`gp_upper_confidence_error`, `distance_to_existing`, and
`acquisition_score`. It does not modify `geometries.csv` or an MDIF.

The selector first looks for `geometries.json` beside
`--existing-points geometries.csv`. If a split file such as
`geometries_train.csv` is supplied, it also finds the single combined
`geometries.json`; split generation intentionally does not create duplicate
JSON files. Use `--parameter-json PATH` only when the CSV/JSON names no longer
match or when occupied points are supplied only through `--existing-mdif`.

#### After Simulating the First GP Batch

Run the following sequence:

1. Simulate every row in `outputs/gp_round_1_points.csv`.
2. Add the resulting blocks to the training MDIF.
3. Refit the same provisional model architecture and fitting objective.
4. Produce a new `verification_metrics.csv` from verification data not used as
   training data.
5. Run a second GP command using the new fit and mark both the original and
   first-round CSVs as occupied.

For example:

```bash
python3 generate_points.py suggest-additional \
  --count 6 \
  --fit-dir outputs/dnn_gp_round_1_refit \
  --existing-points geometries.csv \
  --existing-points outputs/gp_round_1_points.csv \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --candidate-count 1400 \
  --metric evm_pct \
  --exploration-weight 1.5 \
  --gp-noise-variance 1e-6 \
  --novelty-power 1.0 \
  --min-distance 0.05 \
  --seed 1235 \
  --target-dataset train \
  --analysis-out outputs/gp_round_2_error_regions.csv \
  --out outputs/gp_round_2_points.csv
```

Repeat the simulate, append, refit, and acquire cycle until the verification
target stops improving or the EM budget is exhausted. Keep a separate final
audit set that never supplies GP error observations.

#### Which Fit Path to Use

| Current model result | GP error input |
| --- | --- |
| Direct `train` run | `--fit-dir outputs/dnn_model` (or the corresponding KBNN/Neuro-TF model directory) |
| Successful DNN optimize run | `--fit-dir outputs/dnn_adaptive/best_model` |
| Successful KBNN optimize run | `--fit-dir outputs/kbnn_adaptive/best_model` |
| Successful Neuro-TF optimize run | `--fit-dir outputs/neuro_tf_adaptive/best_model` |
| All optimize trials failed the passivity criteria | Run optimize with `--keep-trial-models`, take `best_available_trial` from the best-config JSON, and use `--verification-metrics <sweep>/trials/trial_NNNN/verification_metrics.csv`. |

For example, if an all-ineligible report identifies one-based trial 7:

```bash
python3 generate_points.py suggest-additional \
  --count 8 \
  --verification-metrics outputs/dnn_adaptive/trials/trial_0007/verification_metrics.csv \
  --existing-points geometries.csv \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --metric evm_pct \
  --exploration-weight 2.5 \
  --gp-noise-variance 1e-5 \
  --novelty-power 1.25 \
  --min-distance 0.05 \
  --target-dataset train \
  --out outputs/gp_from_trial_0007.csv
```

More GP examples—including six-parameter, range-extension, KBNN, Neuro-TF,
and exploration-versus-exploitation commands—are provided under
[Copyable Adaptive Point-Generation Examples](#copyable-adaptive-point-generation-examples).

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
recommendation. Let $r$ be the added design-space volume divided by the old
volume. For a one-variable linear extension this is the added width divided by
the old width; log variables use log-width. Let $d$ be the number of geometry
parameters.

| New point group | Recommended count |
| --- | --- |
| Training | $\max(\lceil n_{\mathrm{train,old}}r\rceil, 4d)$ |
| Verification, when the original set contains verification points | $\max(\lceil n_{\mathrm{verify,old}}r\rceil, 2d)$ |

For example, extending one range by 50% from an 80-point, two-parameter design containing 64 training and
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
with a full frequency sweep. For a one-shot design that is expected to work
without adaptive refinement, a practical initial design size is:

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
$15d$ training points, with a minimum of about 30, and $4d$ to $6d$
verification points, with a minimum of about 12. Keep the verification set
fixed across model comparisons, then grow the training set in targeted batches
of about $3d$ to $5d$ points using the current worst-fit regions.

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

### Legacy Error-Distance Additional Points

The older `error-distance` method remains available for comparison. It ranks
verification points by error and scores candidates by proximity to high-error
regions and distance from existing points. The following command intentionally
omits `--acquisition` and therefore uses the legacy default:

```bash
python3 generate_points.py suggest-additional \
  --count 12 \
  --fit-dir outputs/dnn_model \
  --existing-points geometries.csv \
  --out targeted_additional_points.csv
```

The suggested-point CSV uses `dataset=targeted` by default and includes the
nearest high-error verification source, distance from existing points, and
acquisition score. A companion `*_fit_error_regions.csv` file ranks the current
verification points by the selected metric, which defaults to `evm_pct`.

### Gaussian-Process Adaptive Loop

`suggest-additional` also provides a true Gaussian-process upper-confidence
bound acquisition mode. It fits a Matérn-5/2 GP to the natural logarithm of the
current geometry-level error and scores candidate point $\mathbf p$ using

$$
A(\mathbf p)=
\exp\!\left(\mu_{\log e}(\mathbf p)+
\kappa\sigma_{\log e}(\mathbf p)\right)
D(\mathbf p)^{\nu},
$$

where $D$ is normalized distance from already simulated or selected points,
$\kappa$ is `--exploration-weight`, and $\nu$ is `--novelty-power`. The GP
length scale is selected from a normalized candidate grid by log marginal
likelihood unless `--gp-length-scale` is supplied.
[Appendix C](#appendix-c-gaussian-process-adaptive-point-selection) documents
the exact target construction, covariance, posterior, batch acquisition,
diagnostics, limitations, and references.

There is no enforced $8d$ or power-of-two initial-point minimum. For six
geometry parameters, a cost-conscious first trial can start with 32 points: 24
training and 8 acquisition-validation points. This is intentionally lean and
should be grown adaptively; it is not a guarantee that 32 points can resolve
every six-dimensional response.

```bash
python3 generate_points.py generate \
  --parameter P1=0:1 --parameter P2=0:1 \
  --parameter P3=0:1 --parameter P4=0:1 \
  --parameter P5=0:1 --parameter P6=0:1 \
  --count 32 \
  --verification-count 8 \
  --out adaptive_round_0.csv
```

Both initial generation and adaptive candidate generation use `maximin-lhs` by
default. `minimax-lhs` is accepted as an alias for the same standard maximin
Latin-hypercube method. After simulating the initial points and fitting the
chosen provisional network, request a small next batch:

```bash
python3 generate_points.py suggest-additional \
  --count 6 \
  --fit-dir outputs/dnn_compact \
  --existing-points adaptive_round_0.csv \
  --acquisition gp-ucb \
  --target-dataset train \
  --out adaptive_round_1.csv
```

For each round:

1. Simulate the suggested geometry batch.
2. Evaluate the pre-refit network at those new geometries when practical; a
   geometry-level cross-validation or accumulated pre-refit error CSV is a
   better GP target than training residuals.
3. Append the simulated blocks to the training MDIF and retrain the same
   provisional architecture and fitting objective.
4. Regenerate `verification_metrics.csv`, run `suggest-additional` again, and
   repeat until the error target stops improving or the simulation budget is
   reached.
5. Evaluate a separate final audit set that was never used by GP acquisition.

The acquisition is model-aware: keep the model family, S/Y output domain,
frequency transform, weighting, and provisional network architecture fixed
during the loop. The physical samples remain reusable, so after acquisition
you can rescreen architectures and run one cleanup GP round with the final
configuration. Persistent failure of only the compact network, while a larger
reference succeeds at the same points, indicates capacity limitation rather
than missing data.

The suggested CSV records `predicted_error`, `gp_log_uncertainty`, and
`gp_upper_confidence_error`. Its companion JSON records the fitted length
scale, observation count, likelihood, noise variance, and exploration weight.
The original `error-distance` selector remains the default acquisition mode
until this alternative has been validated against the target EM/model flow.

### Copyable Adaptive Point-Generation Examples

Most examples below use the same two geometry variables so the differences
between workflows are explicit. `geometries.csv` is assumed to list every
geometry already simulated, `geometries.json` is its automatically generated
parameter metadata, and each `--fit-dir` is assumed to contain the
`verification_metrics.csv` produced by the current fit. Replace the DNN output
directory with a KBNN or Neuro-TF output directory without changing the point
generation options.

#### GP Additions From a Successful DNN Optimize Run

For an optimize or sweep result, point `--fit-dir` at `best_model/`, not at the
sweep root. The promoted model directory contains the winning trial's
`verification_metrics.csv`. `--existing-mdif` prevents the selector from
requesting a geometry already present in the simulated MDIF:

```bash
python3 generate_points.py suggest-additional \
  --count 10 \
  --fit-dir outputs/dnn_adaptive/best_model \
  --existing-points geometries.csv \
  --existing-mdif train_verify.mdif \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --candidate-count 2000 \
  --lhs-candidates 32 \
  --metric evm_pct \
  --exploration-weight 2.0 \
  --gp-noise-variance 1e-6 \
  --novelty-power 1.0 \
  --min-distance 0.05 \
  --seed 1234 \
  --decimal-places 4 \
  --include-normalized \
  --target-dataset train \
  --analysis-out outputs/gp_round_1_error_regions.csv \
  --out outputs/gp_round_1_points.csv
```

Use a per-row column present in `verification_metrics.csv`, such as `evm_pct`,
`rmse_abs`, or `max_abs`, or specify `--metric auto`. The geometry aggregation
automatically uses `normalized_sparam_weight` when it is present, so an
`evm_pct` target still respects the S-parameter priorities stored by the fit.
Global summary fields such as `weighted_evm_pct` are not per-geometry input
columns. The requested CSV contains only the ten new geometries; it does not
append or rewrite the original MDIF.

#### GP Additions From KBNN and Neuro-TF Optimize Runs

The point selector is model-family independent. Use the fine-model verification
metrics under the selected KBNN model:

```bash
python3 generate_points.py suggest-additional \
  --count 8 \
  --fit-dir outputs/kbnn_adaptive/best_model \
  --parameter-json fine_geometries.json \
  --existing-mdif fine_train_verify.mdif \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --metric evm_pct \
  --exploration-weight 2.0 \
  --novelty-power 1.0 \
  --min-distance 0.05 \
  --target-dataset train \
  --out outputs/kbnn_gp_round_1.csv
```

Use the same workflow with the selected Neuro-TF model:

```bash
python3 generate_points.py suggest-additional \
  --count 8 \
  --fit-dir outputs/neuro_tf_adaptive/best_model \
  --parameter-json geometries.json \
  --existing-mdif train_verify.mdif \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --metric rmse_abs \
  --exploration-weight 2.0 \
  --novelty-power 1.0 \
  --min-distance 0.05 \
  --target-dataset train \
  --out outputs/neuro_tf_gp_round_1.csv
```

#### GP Additions When Every Optimize Trial Failed Passivity

An all-ineligible sweep has no `best_model/`. Run the optimization with
`--keep-trial-models`, open its Markdown report, and use the closest available
trial identified there. For example, if the report identifies trial 7, point
directly to that retained trial's metrics:

```bash
python3 generate_points.py suggest-additional \
  --count 10 \
  --verification-metrics outputs/dnn_adaptive/trials/trial_0007/verification_metrics.csv \
  --parameter-json geometries.json \
  --existing-mdif train_verify.mdif \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --candidate-count 2000 \
  --metric evm_pct \
  --exploration-weight 2.5 \
  --gp-noise-variance 1e-5 \
  --novelty-power 1.25 \
  --min-distance 0.05 \
  --target-dataset train \
  --out outputs/gp_from_ineligible_trial.csv
```

The `best_available_trial` field in the sweep's best-config JSON gives the same
one-based trial number. Without `--keep-trial-models`, nonwinning trial
`verification_metrics.csv` files are intentionally removed after the sweep,
so they cannot be used for a later GP round.

#### A Second GP Acquisition Round

After simulating `gp_round_1_points.csv`, add those blocks to the training
MDIF, refit the same provisional model, and generate fresh verification
metrics. Pass every earlier point CSV with a repeatable `--existing-points` so
round 2 cannot suggest them again:

```bash
python3 generate_points.py suggest-additional \
  --count 6 \
  --fit-dir outputs/dnn_gp_round_1_refit \
  --existing-points geometries.csv \
  --existing-points outputs/gp_round_1_points.csv \
  --existing-mdif train_verify_plus_round_1.mdif \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --candidate-count 1600 \
  --metric evm_pct \
  --exploration-weight 1.5 \
  --gp-noise-variance 1e-6 \
  --novelty-power 1.0 \
  --min-distance 0.05 \
  --seed 1235 \
  --target-dataset train \
  --analysis-out outputs/gp_round_2_error_regions.csv \
  --out outputs/gp_round_2_points.csv
```

Using both CSV and MDIF sources is allowed; duplicate occupied geometries are
collapsed internally. Incrementing `--seed` changes the finite candidate pool.
Keep the seed fixed instead when comparing GP tuning settings on exactly the
same candidate population.

#### Six-Parameter GP Batch With Linear and Logarithmic Ranges

This higher-dimensional example assumes `six_parameter_geometries.json` was
written with the original six-parameter geometry CSV and therefore already
contains the complete mix of linear and logarithmic ranges. It uses an explicit
candidate budget and includes the normalized coordinates in the result for
auditing:

```bash
python3 generate_points.py suggest-additional \
  --count 8 \
  --fit-dir outputs/dnn_six_parameter/best_model \
  --parameter-json six_parameter_geometries.json \
  --existing-mdif six_parameter_train_verify.mdif \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --candidate-count 2000 \
  --lhs-candidates 24 \
  --metric auto \
  --exploration-weight 2.5 \
  --gp-noise-variance 1e-5 \
  --novelty-power 1.25 \
  --min-distance 0.04 \
  --seed 2026 \
  --decimal-places 5 \
  --include-normalized \
  --target-dataset train \
  --out outputs/six_parameter_gp_additional.csv
```

For a first adaptive round, request roughly one or two new points per geometry
dimension; the example requests eight for six dimensions. Increase the next
batch only when the first acquisition remains broadly uncertain. Small batches
allow the GP error surface to be rebuilt after each set of expensive
simulations.

#### Exploitation-Focused and Exploration-Focused Variants

To concentrate near currently predicted high-error regions, start from the
successful-DNN command and change these controls:

```bash
  --exploration-weight 0.5 \
  --novelty-power 0.75 \
  --min-distance 0.03
```

To search uncertain or sparsely sampled portions of the range more strongly,
use:

```bash
  --exploration-weight 3.0 \
  --novelty-power 1.5 \
  --min-distance 0.07
```

These are starting points, not universal optima. For a controlled comparison,
keep `--seed`, `--candidate-count`, `--candidate-method`, `--metric`, the fit,
and all occupied-point inputs identical; change only the acquisition controls.

Before sending a suggested batch to EM simulation, inspect the generated
outputs:

- `<out>.csv` is ordered by `additional_sequence` and contains the selected
  geometries, `predicted_error`, `gp_log_uncertainty`,
  `gp_upper_confidence_error`, `distance_to_existing`, and the final
  `acquisition_score`.
- The same-stem `*_fit_error_regions.csv`, or the explicit `--analysis-out`,
  ranks the measured verification geometries used to fit the GP. Use it to
  confirm that the expected bad regions and parameter values were parsed.
- The same-stem JSON records the normalized parameter ranges, selected Matérn
  length scale, GP observation count, likelihood, noise variance, and
  acquisition settings needed to reproduce the batch.

If the command warns that there are fewer distinct error observations than
dimensions plus one, treat the batch as exploration-heavy. Simulate a small
batch, refit, and rebuild the GP rather than requesting one large batch from a
sparsely observed error surface.

For a one-sided range change, first create guaranteed coverage of only the new
slab. This shared seed step extends the upper `W` bound from `0.80mm` to
`1.00mm`, retains the original rows, and appends 20 maximin-LHS points:

```bash
python3 generate_points.py generate \
  --parameter W=0.40mm:0.80mm \
  --parameter L=1.00mm:1.60mm \
  --extend-range W=0.40mm:1.00mm \
  --existing-points geometries.csv \
  --count 20 \
  --verification-count 4 \
  --method maximin-lhs \
  --include-normalized \
  --out geometries_extended_seed.csv
```

Simulate the appended rows, include their results in the training and
verification MDIF, and refit over the expanded domain. Then use one of the two
following alternatives for another range-extension refinement batch.

#### Range Extension With the Legacy Error-Distance Selector

```bash
python3 generate_points.py suggest-additional \
  --count 8 \
  --fit-dir outputs/dnn_extended_seed \
  --existing-points geometries_extended_seed.csv \
  --acquisition error-distance \
  --candidate-method maximin-lhs \
  --metric evm_pct \
  --focus-radius 0.25 \
  --novelty-power 1.0 \
  --min-distance 0.05 \
  --target-dataset train \
  --out range_extension_legacy_additional.csv
```

#### Range Extension With the GP-UCB Selector

```bash
python3 generate_points.py suggest-additional \
  --count 8 \
  --fit-dir outputs/dnn_extended_seed \
  --existing-points geometries_extended_seed.csv \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --metric evm_pct \
  --exploration-weight 2.0 \
  --gp-noise-variance 1e-6 \
  --novelty-power 1.0 \
  --min-distance 0.05 \
  --target-dataset train \
  --out range_extension_gp_additional.csv
```

These refinement commands score candidates over the entire expanded rectangle,
so they may also repair an error remaining in the original range. The initial
`generate --extend-range` command is what guarantees coverage specifically in
the new slab.

When the parameter ranges are unchanged, omit the slab-generation step.

#### Standard Addition With the Legacy Error-Distance Selector

Use the legacy selector to concentrate additions around observed high-error
verification geometries:

```bash
python3 generate_points.py suggest-additional \
  --count 8 \
  --fit-dir outputs/dnn_current \
  --existing-points geometries.csv \
  --acquisition error-distance \
  --candidate-method maximin-lhs \
  --metric evm_pct \
  --focus-radius 0.25 \
  --focus-power 1.0 \
  --novelty-power 1.0 \
  --min-distance 0.05 \
  --target-dataset train \
  --out standard_legacy_additional.csv
```

#### Standard Addition With the GP-UCB Selector

Use GP-UCB over the same unchanged range to balance predicted error against
uncertainty and point separation:

```bash
python3 generate_points.py suggest-additional \
  --count 8 \
  --fit-dir outputs/dnn_current \
  --existing-points geometries.csv \
  --acquisition gp-ucb \
  --candidate-method maximin-lhs \
  --metric evm_pct \
  --exploration-weight 2.0 \
  --gp-noise-variance 1e-6 \
  --novelty-power 1.0 \
  --min-distance 0.05 \
  --target-dataset train \
  --out standard_gp_additional.csv
```

For a controlled comparison, run the legacy and GP commands from the same fit,
with the same existing-point files, candidate count, random seed, metric, and
batch size. Simulate the two output batches separately; do not combine one
method's new results into the fit used to judge the other method.

The explicit `generate` subcommand is optional; invoking `generate_points.py`
without a subcommand uses it automatically.

### Parameter Ranges

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--extend-range SPEC</code></nobr> | <code>generate</code> | Optional one-sided append workflow. Supplies the new overall bounds as <code>NAME=NEW_LOW:NEW_HIGH</code>; exactly one bound must match the original <code>--parameter</code> range and the other must move outward. Requires <code>--existing-points</code>. | <nobr><code>--extend-range W=0.40mm:1.00mm</code></nobr> |
| <nobr><code>--parameter SPEC</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Repeatable definition in the form <code>NAME=LOW:HIGH[:linear\|log]</code>. Required for <code>generate</code>. For <code>suggest-additional</code>, it is an optional complete override when intentionally bypassing generated metadata. | <nobr><code>--parameter W=0.40mm:0.80mm</code></nobr> |
| <nobr><code>--parameter-json PATH</code></nobr> | <code>suggest-additional</code> | Explicit generated geometry JSON from which to load every parameter name, bound, unit, and linear/log scale. Omit it when a companion JSON can be inferred from <code>--existing-points</code>. Cannot be combined with <code>--parameter</code>. | <nobr><code>--parameter-json geometries.json</code></nobr> |
| <nobr><code>--range-factor SPEC</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Optional and repeatable. Expands the named parameter's total span around its existing center, whether the domain came from <code>--parameter</code> or JSON. The finite factor must be greater than 1. | <nobr><code>--range-factor W=1.5</code></nobr> |

### Sampling

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--candidate-method METHOD</code></nobr> | <code>suggest-additional</code> | Candidate-pool method: <code>halton</code>, <code>latin-hypercube</code>, <code>maximin-lhs</code>, or <code>sobol</code>. <code>minimax-lhs</code> is accepted as an alias for <code>maximin-lhs</code>. Default: <code>maximin-lhs</code>. | <nobr><code>--candidate-method sobol</code></nobr> |
| <nobr><code>--lhs-candidates INT</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Candidate Latin hypercubes tried when using <code>maximin-lhs</code>. Default: <code>64</code>. | <nobr><code>--lhs-candidates 128</code></nobr> |
| <nobr><code>--method METHOD</code></nobr> | <code>generate</code> | Repeat or comma-separate point-set methods. Choices: <code>halton</code>, <code>latin-hypercube</code>, <code>maximin-lhs</code>, and <code>sobol</code>. <code>minimax-lhs</code> is accepted as an alias for <code>maximin-lhs</code>. Default: <code>maximin-lhs</code>. | <nobr><code>--method sobol,maximin-lhs</code></nobr> |
| <nobr><code>--no-scramble</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Disables Sobol scrambling, which is enabled by default. | <nobr><code>--no-scramble</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Random seed used by randomized sampling methods. Default: <code>1234</code>. | <nobr><code>--seed 42</code></nobr> |
| <nobr><code>--skip INT</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Non-negative number of leading Sobol or Halton points to skip. Default: <code>0</code>. | <nobr><code>--skip 64</code></nobr> |

### Gaussian-Process Acquisition

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--acquisition MODE</code></nobr> | <code>suggest-additional</code> | Candidate acquisition: original <code>error-distance</code> or Matérn-5/2 <code>gp-ucb</code>. Default: <code>error-distance</code>. | <nobr><code>--acquisition gp-ucb</code></nobr> |
| <nobr><code>--exploration-weight FLOAT</code></nobr> | <code>suggest-additional</code> | Non-negative GP-UCB multiplier $\kappa$ on posterior log-error uncertainty. Default: <code>2.0</code>. | <nobr><code>--exploration-weight 2.5</code></nobr> |
| <nobr><code>--gp-error-floor FLOAT</code></nobr> | <code>suggest-additional</code> | Positive floor applied before taking the natural logarithm of geometry error. Default: <code>1e-12</code>. | <nobr><code>--gp-error-floor 1e-9</code></nobr> |
| <nobr><code>--gp-length-scale FLOAT</code></nobr> | <code>suggest-additional</code> | Optional positive Matérn-5/2 length scale in normalized geometry coordinates. When omitted, it is selected by log marginal likelihood. | <nobr><code>--gp-length-scale 0.4</code></nobr> |
| <nobr><code>--gp-noise-variance FLOAT</code></nobr> | <code>suggest-additional</code> | Non-negative normalized covariance nugget for GP stability and noisy error observations. Default: <code>1e-6</code>. | <nobr><code>--gp-noise-variance 1e-5</code></nobr> |

### Output and Dataset Splits

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--analysis-out PATH</code></nobr> | <code>suggest-additional</code> | Ranked fit-error-region CSV. Default: <code>&lt;out&gt;_fit_error_regions.csv</code>. | <nobr><code>--analysis-out error_regions.csv</code></nobr> |
| <nobr><code>--count INT</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Positive number of new points. Required except for <code>generate --extend-range</code>, which calculates and uses a recommendation when omitted. | <nobr><code>--count 80</code></nobr> |
| <nobr><code>--decimal-places INT</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | Rounds generated parameter values to this many decimal places in their declared units. Accepts <code>0</code> through <code>15</code>; omitted values retain the existing full-precision behavior. | <nobr><code>--decimal-places 3</code></nobr> |
| <nobr><code>--existing-points PATH</code></nobr> | <code>generate</code>, <code>suggest-additional</code> | With <code>generate --extend-range</code>, the original CSV retained at the start of the combined output. With <code>suggest-additional</code>, a repeatable CSV of simulated points to avoid; its same-stem geometry JSON supplies the parameter domain automatically. A <code>*_train.csv</code> or <code>*_verification.csv</code> split also resolves the combined JSON. | <nobr><code>--existing-points geometries.csv</code></nobr> |
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
| <nobr><code>--focus-power FLOAT</code></nobr> | <code>suggest-additional</code> | With <code>--acquisition error-distance</code>, non-negative exponent applied to verification-error scores. Default: <code>1.0</code>. | <nobr><code>--focus-power 1.5</code></nobr> |
| <nobr><code>--focus-radius FLOAT</code></nobr> | <code>suggest-additional</code> | With <code>--acquisition error-distance</code>, positive unit-cube radius around high-error verification points. Default: <code>0.25</code>. | <nobr><code>--focus-radius 0.2</code></nobr> |
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

Use `--search-mode adaptive|grid|random` for the optimize search strategy.
Legacy `--mode grid|random` commands remain valid; on KBNN optimize commands,
`--mode plain|residual|prior-input` now has the same model meaning as it does
for `train`.

Run a discrete random sweep and keep the best completed model:

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

### Running Adaptive Hyperparameter Optimization

Use `--search-mode adaptive` when a grid is too large and random trials are not
learning from previous results. Specify only the settings that should vary with
repeatable `--optimize-parameter` options. Unspecified model settings stay at
the first value supplied through their normal optimize option, including the
documented default when that option is omitted.

The practical workflow is:

1. Start from a working `train` command and change `train` to `optimize`.
2. Add `--search-mode adaptive`.
3. Add one repeatable `--optimize-parameter NAME=DOMAIN` option for every
   setting that should vary. Leave fixed settings on their normal train-style
   options.
4. Set the actual fitting budget with `--max-trials`. A useful first run is
   24–40 trials, with 6–8 initial space-filling trials.
5. Choose the performance objective with `--selection-metric` and add
   `--require-passive` when passivity is mandatory.

`--adaptive-candidate-pool` is the number of unevaluated configurations made
available to the optimizer; it is not the number of models fitted.
`--max-trials` is the fitting budget. `--adaptive-initial-trials` controls how
many maximin-separated trials run before GP guidance, and
`--adaptive-exploration` controls how strongly the later lower-confidence-bound
selection favors uncertain regions. Adaptive fitting is sequential and forces
one job because every new selection uses the preceding trial results.

Supported adaptive domains by model are:

| Model | `--optimize-parameter` names |
| --- | --- |
| DNN | `activation`, `batch_size`, `epochs`, `freq_transform`, `hidden_layers`, `learning_rate`, `output_domain`, `patience`, `target_z0` |
| KBNN | `activation`, `batch_size`, `epochs`, `freq_transform`, `hidden_layers`, `include_coarse_input`, `learning_rate`, `mode`, `patience` |
| Neuro-TF | `activation`, `batch_size`, `epochs`, `hidden_layers`, `learning_rate`, `order`, `patience`, `pole_damping`, `ridge` |

This example searches learning rate, activation, and neural architecture while
requiring a passive result:

```bash
python3 dnn.py optimize \
  --mdif train_verify.mdif \
  --out-dir outputs/dnn_adaptive \
  --parameter-names W,L,H \
  --search-mode adaptive \
  --optimize-parameter learning_rate=1e-4:1e-2:log \
  --optimize-parameter activation=tanh,relu \
  --optimize-parameter 'hidden_layers=1:4x32:256:log' \
  --adaptive-initial-trials 8 \
  --adaptive-candidate-pool 768 \
  --adaptive-exploration 1.5 \
  --max-trials 32 \
  --selection-metric weighted_evm_pct \
  --require-passive
```

For a one-parameter study, provide only one range and fix the architecture with
the normal train-compatible option:

```bash
python3 dnn.py optimize \
  --mdif train_verify.mdif \
  --out-dir outputs/dnn_learning_rate \
  --parameter-names W,L,H \
  --hidden-layers 128,128,64 \
  --activation tanh \
  --search-mode adaptive \
  --optimize-parameter learning_rate=2e-4:8e-3:log \
  --max-trials 20 \
  --selection-metric rmse_abs \
  --require-passive
```

Domain syntax:

| Domain type | Syntax | Example |
| --- | --- | --- |
| Linear numeric range | `NAME=LOW:HIGH` or `NAME=LOW:HIGH:linear` | `batch_size=64:512` |
| Logarithmic numeric range | `NAME=LOW:HIGH:log` | `learning_rate=1e-4:1e-2:log` |
| Categorical choices | `NAME=VALUE1,VALUE2,...` | `activation=tanh,relu` |
| Explicit hidden layouts | `hidden_layers=LAYOUT1;LAYOUT2;...` | `hidden_layers=64,64;128,128,64;256,128,64` |
| Hidden depth/width range | `hidden_layers=MIN_DEPTH:MAX_DEPTHxMIN_WIDTH:MAX_WIDTH[:SCALE]`, where scale is `linear` or `log` | `hidden_layers=1:4x32:256:log` |

Integer parameters such as Neuro-TF `order`, `batch_size`, `epochs`, and
`patience` are sampled as integers. A structured hidden-layer range creates
uniform-width networks; widths are rounded to `--adaptive-hidden-width-step`,
which defaults to 8. Use explicit layouts when tapered networks are important.
Quote hidden-layer domains in the shell because semicolons are command
separators.

The search begins with maximin-separated configurations, then fits a small
Matérn-5/2 GP to a feasibility-aware objective and selects each later trial by
a lower confidence bound. With `--require-passive` or a passivity limit,
eligible trials are ranked by `--selection-metric`; until one exists,
passivity-violation count and singular-value excess guide the search. Adaptive
search is sequential, so it uses one job even if a larger `--jobs` value is
given.

After the command finishes, open the model-specific Markdown report in the
chosen output directory:

| Model | Trial results | Markdown report | Selection JSON |
| --- | --- | --- | --- |
| DNN | `dnn_sweep_results.csv` | `dnn_sweep_summary.md` | `dnn_best_config.json` |
| KBNN | `kbnn_sweep_results.csv` | `kbnn_sweep_summary.md` | `kbnn_best_config.json` |
| Neuro-TF | `neurotf_sweep_results.csv` | `neurotf_sweep_summary.md` | `neurotf_best_config.json` |

The report contains the ranked trial table, adaptive search stage and
uncertainty, inline trend plots, and links to the detailed diagnostics. When an
eligible winner exists, `best_model/` contains the promoted model and the
report includes a copyable command for reproducing it by itself.

If every completed trial fails training or the passivity constraints, the
command returns a nonzero status and does not promote `best_model/`, but it
still writes the normal results CSV, best-config JSON, Markdown sweep summary,
diagnostic PDF, inline SVG trend plots, and diagnostic CSV. The Markdown
identifies the closest available ineligible trial and embeds the trend plots
directly in the report. Diagnostic plots retain all trial points, and the CSV
contains both passive-only statistics and `all_*` statistics so trends are
visible even when the passive subset is empty.

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

For a passive, power-independent component that ADS can use directly in
harmonic balance, export the HB-native SDD subnetwork instead:

```bash
python3 dnn.py export-ads-hb \
  --model-dir outputs/dnn_model \
  --out-dir outputs/dnn_ads_hb \
  --module-name my_dnn_4port_hb \
  --parameter-input-scales 1.0
```

The SDD implements

$$
\mathbf V-Z_0\mathbf I=\mathbf S(f)(\mathbf V+Z_0\mathbf I)
$$

and ADS evaluates the embedded
surrogate independently at every HB spectral frequency. It has no input-power
parameter and creates no compression; compression and harmonic generation come
only from nonlinear devices elsewhere in the circuit. Negative-frequency
weights are the complex conjugates of the corresponding positive-frequency
predictions, preserving the real-waveform symmetry expected from an
S-parameter file.

Cover every HB frequency that can carry meaningful energy in the training and
verification range. The exporter preserves linearity and power independence,
but it does not project an unconstrained fit onto a passive matrix; use
passivity-aware sweep selection and validate the exported component across the
full parameter/frequency range. Frequencies outside the fitted range remain
model extrapolation, just as an under-ranged S-parameter dataset would require
an extrapolation policy.

Residual and prior-input KBNNs use a frozen coarse DNN during fitting and at
runtime. The integrated KBNN command fits the coarse model once, saves it under
`coarse_model/`, fits the fine network from its predictions, and retains both
models for one self-contained Verilog-A component or ADS HB subnetwork:

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

python3 kbnn.py export-ads-hb \
  --model-dir outputs/kbnn_model \
  --out-dir outputs/kbnn_ads_hb \
  --module-name my_kbnn_4port_hb
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

- `model.npz` and `metadata.json` with the trained RF model state and assumptions.
- `dc_model.npz` and `dc_model.json` with the separate geometry-to-DC model
  and its extraction diagnostics.
- `predicted_verification.mdif` for held-out verification blocks.
- `verification_metrics.csv` with per-block and per-S-parameter errors,
  including EVM.
- `verification_summary.json` with global errors and passivity summary data.
- `training_history.csv` and `training_history.pdf` with train/verification
  loss history and convergence plots.
- `dc_training_history.csv` and `dc_training_history.pdf` with convergence of
  the independent geometry-only DC model.
- `training_summary.md` with a human-readable run summary and copyable export
  commands using paths relative to the repository root.
- `worst_case_plots/*.pdf` with S-parameter Smith/complex, magnitude, phase,
  and error views.
- `worst_case_y_plots/*.pdf` with real/imaginary Y-parameter diagnostics.

An integrated residual or prior-input KBNN run also writes a complete coarse
DNN package under `coarse_model/` and a `composite_model_manifest.json` that
identifies and hashes both saved networks for later Verilog-A or ADS HB extraction.
The reported circuit-export commands include an explicit default module name and a
single `--parameter-input-scales 1.0` value applied to every fitted parameter.
A new model's saved DC network is self-contained, so its report does not add
`--dc-mdif` or export-time open-threshold overrides. Legacy-model commands add
`--dc-open-threshold 1e12` and `--dc-open-resistance 1e19`. When the fit saved
an explicit path selection, commands also include `--dc-port-paths`. These
values are easy to edit before export.

### Distinct DC Point

DC is a separate, geometry-dependent model derived only from the actual
exact-zero-frequency S-matrix in each fitted MDIF block.
It is not a target of the RF DNN, KBNN, or Neuro-TF. Zero-Hz rows are removed
before RF neural training and rational fitting, so changing DC data cannot
change RF weights, poles, sweep ranking, or the positive-frequency response.

Declare the only viable DC connections with `--dc-port-paths`. For example,
`--dc-port-paths 1-2,3-4` extracts and stamps independent paths between ports
1–2 and 3–4. Every undeclared path remains open at DC. Use `1-ground` for a
port-to-simulator-reference path. Omitting the option instead fits both the real
and imaginary component of every ordered DC S-parameter: `S11.real`,
`S11.imag`, …, `SNN.imag`. The complete complex S matrix is then converted to Y
for electrical stamping. This does not discard information in an intermediate
real-Y projection and imposes no resistor-graph or reciprocity constraint.
Reports therefore show `dc_port_paths: []` for this unrestricted mode and list
all modeled entries separately under `dc_sparameter_entries` and
`dc_matrix_entries`; the empty path list does not mean S-parameters were omitted.
When `--dc-mdif` is supplied without `--dc-port-paths`, this full-complex-S mode is
also used to upgrade an older saved path-only DC model. Repeat an explicit
`--dc-port-paths` value when a restricted resistor topology is intentional.
Export stops with upgrade guidance if an older model saved the former automatic
path graph but no source MDIF is available; it never silently presents that
subset as a full-complex-S result.

1. Each finite exact-zero-Hz S-matrix is checked for passivity using its largest
   singular value. Rows above $1+10^{-6}$ are ignored.
2. With no path option, both components of every ordered $S_{ij}$ value are fitted
   directly. No intermediate real-Y projection is used.
3. With an explicit path option, the S matrix is converted to Y and all declared
   branches are solved together with a non-negative least-squares projection
   onto that resistor graph.
4. In explicit-path mode, conductances below the reciprocal of
   `--dc-open-threshold` are represented by the reciprocal of
   `--dc-open-resistance`. The natural logarithm of each positive branch
   conductance is then fitted by a small geometry-only MLP.
5. The saved diagnostics include the measured-S → extracted-Y → model-Y →
   reconstructed-S round-trip error, filtered-row counts, matrix/path errors,
   and train/verification errors.

The DC MLP reuses the command's hidden-layer layout, activation, epoch, batch,
learning-rate, patience, seed, and progress settings, but it has its own scaler,
weights, history, and loss. Its inputs are geometry/process parameters only;
frequency, RF samples, KBNN coarse responses, and S-parameter/frequency loss
weights are not inputs to the DC fit.

Every training and verification geometry must contain at least one exact-zero-
Hz row. Non-passive or non-finite zero-Hz rows are discarded, and a geometry
with no usable passive DC row is excluded from the separate DC fit. Fitting
stops only when no usable passive DC geometries remain. A missing zero-Hz row
in a training block is still an error. The lowest positive frequency is never substituted,
and the RF response is never extrapolated to DC. RF verification metrics,
worst-case plots, sweep ranking, and passivity selection use only positive-
frequency rows. `predicted_verification.mdif` still contains both the separately
predicted DC point and all RF points, while `verification_summary.json` reports
the DC network's own train/verification round-trip errors separately.

At exactly zero Hz, prediction and sampled-MDIF export evaluate only the saved
geometry-to-DC model. Sampled ADS exports
prepend this zero-Hz point automatically. Direct Verilog-A exports embed the DC
MLP and electrically enable its branch matrix only at zero frequency. Positive
frequencies use the RF model. The exporter selects DC or fitted Y
coefficients before an unconditional current contribution, so `ddt()` is never
placed in a conditional and the generated source remains legal for ADS
Verilog-A. Export also verifies that DC came from exact-zero-frequency data.
ADS HB exports make the same DC/RF separation in the SDD frequency weights:
the geometry-dependent exact-DC matrix or explicit-path network is used only at
$f=0$, and the fitted RF surrogate is used only at non-zero spectral frequencies.

To export an older fitted model without retraining it, pass the original DC data
directly to `export-veriloga`, `export-ads-hb`, or `export-ads-mdif`:

```bash
python3 dnn.py export-veriloga \
  --model-dir dnn_model \
  --out-dir dnn_model/veriloga_export \
  --dc-mdif training_with_dc.mdif \
  --dc-port-paths 1-2,3-4 \
  --dc-open-threshold 1e12 \
  --dc-open-resistance 1e19
```

`--dc-mdif` validates a saved geometry-dependent DC network directly against
the supplied exact-DC rows. If it differs by more than $10^{-4}$ in maximum
absolute S-parameter error, or if the saved model is legacy, export fits a new
DC-only full-complex-S or explicit-path network from that MDIF. This never changes
or refits the RF model. The export manifest records `dc_mdif_action` plus the topology and final
DC-model S-parameter errors. An explicit resistor-path export is rejected if its
maximum absolute S-parameter error remains above $10^{-3}$, because that means the
declared topology cannot reproduce the data. The unrestricted complex-S mode
has no topology/projection mismatch: it receives a larger export-only network
and convergence budget, and any remaining interpolation error is reported as
`dc_mdif_warning` without preventing the requested export.
For a combined training/verification MDIF, export reuses the fitted model's
`--split-var`, `--train-values`, and `--verify-values` metadata and fits or
validates DC from the training split only. Verification blocks are never added
to the DC optimizer; any verification blocks that do contain usable DC may be
reported as validation during the original model fit, while verification blocks
without DC are skipped. Missing-DC errors identify one-based `ACDATA` block
positions in the original MDIF.

For an integrated residual or prior-input KBNN, the composite Verilog-A and ADS
HB components use the DC conductance surrogate fitted only from the fine-data
MDIF and bypass both fine and coarse RF networks at zero Hz. The coarse model
and coarse MDIF are not used to calculate composite DC. Native ADS ANN
retraining excludes zero-Hz rows,
but its generated ANN alone does not implement the distinct resistor branch;
use the direct Verilog-A or sampled-MDIF handoff when DC behavior is required.

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
| Native ADS HB passive network | `export-ads-hb` | You need one integrated, linear, power-independent component whose fitted matrix is evaluated at every HB spectral frequency and explicitly stamped as admittance. |
| Native ADS ANN package | `export-ads-ann` | You want ADS to retrain/extract the neural network and emit native ANN artifacts on a licensed ADS machine. |
| Direct Verilog-A | `export-veriloga` | You want to embed local trained weights for S-parameter or small-signal AC analysis and validate them with the target ADS Verilog-A compiler. |

Use `export-ads-hb` for harmonic balance. The generated SDD is a linear
frequency-dependent multiport, just like an S-parameter file, but its response
is calculated from the geometry-dependent surrogate instead of a fixed table.
Every model type uses the explicit circuit stamp
$\mathbf I=\mathbf Y(f)\mathbf V$. DNN models trained
with `--output-domain y` supply Y directly. S-output DNNs, KBNNs, and Neuro-TFs
are converted to Y by generated frequency-only equations before the explicit
stamp. The separately extracted DC conductance is a different parallel branch:
the RF branch is open at DC and the DC branch is open at every RF frequency.
This avoids the additional modified-nodal branch unknowns created by the former
implicit S-wave implementation.

DNN `export-ads-hb` can also write a direct-Y comparison trial while the
S-domain baseline remains unchanged. Train the comparison DNN separately with
`--output-domain y`, then pass its directory through
`--direct-y-trial-model-dir`. The exporter requires matching parameters,
S-parameter order, frequency transform, activation, layer sizes, and reference
impedance. Both packages reuse the baseline's exact-DC model and two-branch SDD
topology; only the RF model formulation changes. The default manifest lists the
trial files and compatibility checks under `trial_exports`.

### Parsing ADS Gain Compression solver logs

The standalone
`de_generated_scripts/parse_ads_hb_solver_log.py` utility converts the plain
Status/Summary text from an ADS Gain Compression or HB run into comparable
tables. It requires only standard Python and does not need to run inside ADS.

Set the Gain Compression controller's **Freq > Levels > Status level** to `4`
and copy each model's Status/Summary text into a plain log file. Level 4 is
sufficient because every Newton summary row contains the number of Krylov
iterations used for that step. Level 5 also works; the parser deliberately
ignores its additional inner-Krylov residual table to avoid double-counting.

Compare any number of model logs in one command:

```bash
python3 de_generated_scripts/parse_ads_hb_solver_log.py \
  baseline_status.log direct_y_status.log \
  --labels baseline direct_y \
  --out-dir hb_solver_comparison
```

For a single copied log, pass `-` instead of a filename, paste the text into the
terminal, and end standard input with Ctrl-D on Linux/macOS or Ctrl-Z followed
by Enter on Windows.

The report directory contains:

- `ads_hb_solver_points.csv`: one row per detected HB solve, including
  frequency and input power when ADS printed them, Newton count, total Krylov
  count, final residuals, line numbers, and convergence/retry messages;
- `ads_hb_solver_summary.csv`: totals plus mean, median, 95th-percentile, and
  worst-case solver work, ADS total and simulation stopwatch times, total
  stopwatch time per detected solve, and CPU time for every model;
- `ads_hb_solver_summary.json`: the same aggregate data and messages in a
  machine-readable form, plus the exact content-versioned SVG filenames used
  by the Markdown report;
- `ads_hb_solver_report.md`: an easy-to-scan comparison report containing the
  runtime and solver-work summary tables, changes relative to the first model,
  per-frequency results, highest-work solves, source coverage, and inline
  plots;
- `runtime_comparison.svg`: ADS total stopwatch time, simulation stopwatch time,
  and derived total stopwatch time per detected HB solve;
- `solver_work_totals.svg`: total Newton and Krylov work by model;
- `krylov_per_solve_statistics.svg`: mean, median, 95th-percentile, and maximum
  Krylov work per detected HB solve;
- `krylov_by_solve.svg`: solve-sequence comparison for finding localized
  convergence-cost differences.

Each stable SVG also has a content-versioned copy such as
`runtime_comparison.a1b2c3d4e5f6.svg`. The Markdown report references these
physical copies directly, and `embedded_plot_artifacts` in the summary JSON
lists all four filenames.

The SVG plots are generated with the Python standard library and are referenced
with relative paths inside `ads_hb_solver_report.md`, so the report directory is
portable and the plots render inline in normal Markdown viewers. The image links
point to real SVG files whose filenames include a content fingerprint. This
keeps the links portable across Markdown renderers and prevents a rerun in the
same directory from displaying a stale cached plot. The stable SVG filenames
are also retained for direct access and automation. The first log is treated as
the baseline for percentage-change tables; put the standard model first in the
command.

#### ADS Resource usage timing

The parser reads these exact fields from the `Resource usage` block at the end
of an ADS log:

- `Total stopwatch time` is the primary end-to-end wall-clock comparison;
- `Simulation stopwatch time` is reported separately as time spent in the
  simulation portion;
- `Total CPU time` is retained as supporting processor-time context.

The exact stopwatch labels take priority over generic timing lines elsewhere in
the log. The parser also accepts fallback labels such as `Total elapsed time`,
`Total simulation time`, `Wall clock time`, `CPU time`, and paired
`CPU/Elapsed time` values for older or differently configured logs. Clock-form
durations such as `0:24:52` are converted to seconds. The selected source line
for every metric is included in the Markdown report and summary outputs so a
timing match is visible rather than silent.

If ADS does not print timing, enable its diagnostic event recording before
starting ADS, then display that recording in the simulation log:

1. set `ADSSIM_ENABLE_DEBUG_EVENTS=Y` in the environment or the applicable
   `hpeesofsim.cfg` before ADS starts;
2. place an Options controller, expose **Display > Other**, and set
   `Other=ResourceUsage=2`;
3. use the same event settings for every model being timed.

Keysight documents that the event log includes elapsed and CPU time for license
acquisition, netlist parsing, simulation, and matrix-solver steps, and that its
recording system is disabled by default in releases using the newer event
subsystem. See the [Keysight circuit-simulator documentation](https://edadownload.software.keysight.com/eedl/ads/2011/pdf/cktsim.pdf)
and [ADS diagnostic-event release note](https://docs.keysight.com/download/attachments/4403386/Advanced%20Design%20System%202017%20Release%20Notes_RC1.pdf?api=v2).

For logs without an embedded `Total stopwatch time`, supply independently
measured times in log order. These values override parsed total timing:

```bash
python3 de_generated_scripts/parse_ads_hb_solver_log.py \
  baseline_status.log trial_status.log \
  --labels baseline trial \
  --wall-clock-seconds 123.4 118.9 \
  --cpu-time-seconds 211.2 205.7 \
  --out-dir hb_solver_comparison
```

The report never estimates runtime from iteration counts. `Total/solve` is only
`Total stopwatch time` divided by the number of detected HB solves. Use total
stopwatch time as the primary end-to-end comparison, use simulation stopwatch
time to isolate the simulator portion, and treat CPU time as supporting context,
especially for multi-core runs. Diagnostic event recording can add overhead to
very short sweeps, so benchmark every model with identical logging settings.

The parser starts a new solve when a printed frequency or input-power label
changes, or when the Newton iteration counter resets. The reset fallback still
produces correct aggregate solver-work comparisons when ADS does not print the
adaptive Gain Compression power values. In that case, the frequency and power
columns are intentionally blank. UTF-8, UTF-8-with-BOM, and UTF-16 Message
Window exports are detected automatically, and wrapped multi-line table headers
are accepted.

For model $m$, the most useful normalized comparison in the summary is

$$
\overline{K}_m=
\frac{\sum_{q=1}^{N_m}\sum_{n=1}^{M_q}K_{q,n}}{N_m},
$$

where $N_m$ is the number of detected HB solves, $M_q$ is the number of Newton
steps in solve $q$, and $K_{q,n}$ is the Krylov iteration count printed on a
Newton summary row. Also compare `solve_count`: Gain Compression searches
adaptively, so models can require different power-point sequences. Use the same
status level and initial-guess policy for every timed run.

If a particular ADS release uses different frequency or power wording, provide
a release-specific regular expression with a named `value` group and optional
`unit` group through `--frequency-regex` or `--power-regex`. Run `--help` for
the exact syntax. A log that contains no recognized Newton/Krylov summary rows
fails with an explicit request to enable status level 4 or 5 and prints the
relevant candidate lines it found for release-specific diagnosis.

### Reusing an ADS HB model at multiple parameter values

The exported `.net` file contains one native ADS subnetwork definition. Load
that definition once and call it any number of times:

```text
top-level HB testbench
|- one NetlistInclude -> loads my_model_hb.net
|- my_model_hb:X1     -> parameter set A
|- my_model_hb:X2     -> parameter set B
`- HB controller
```

Copy the complete export directory below the ADS workspace, for example
`./hb_models/my_model_hb/`. On the top-level schematic containing the HB
controller, place one `NetlistInclude` from the **Data Items** palette and set:

```text
IncludePath="./hb_models/my_model_hb"
IncludeFiles[1]="my_model_hb.net"
UsePreprocessor=yes
```

Use the generated filename and directory in place of the example. The include
must be at the top simulation level because the file contains a `define`
subnetwork declaration. Do not put the include in every model instance.
`NetlistInclude` does not have electrical pins and receives no geometry or
process parameters; it only makes the definition available to the simulator.

Geometry/process parameters belong on each subnetwork call after its ordered
electrical nodes. A two-port with parameters `W` and `L` can be called twice as:

```text
my_model_hb:X1 x1_p1 x1_p2 W=W_A L=L_A
my_model_hb:X2 x2_p1 x2_p2 W=W_B L=L_B
```

`W_A`, `L_A`, `W_B`, and `L_B` may be top-level ADS `VAR` expressions or the
values can be written directly. The call must use the sanitized names from
`parameter_identifiers` in `ads_hb_manifest.json`; its node order is `p1`,
`p2`, and so on from the opening `define` line of the generated `.net` file.
Parameters omitted from a call use their generated defaults.

The package includes `ADS_HB_INSTANCE_TEMPLATE.txt` with calls matching the
actual module, port count, and parameter names. One direct integration method
is to copy those calls into a second top-level `.net` fragment, replace its node
labels with matching top-level schematic net labels, and list that fragment in
the same `NetlistInclude` after the model definition, for example
`IncludeFiles[2]="my_model_instances.net"`. These instances work but are
visually hidden. For normal schematic reuse, create one custom ADS adapter
component/symbol whose generated native ADS line has the same call form, expose
the geometry parameters on it, and then place that symbol repeatedly. The
export is already native ADS syntax, so do not import its `.net` file through a
SPICE parser.

The parameter scaling equation is

$$
p_{\mathrm{model}}=
\frac{p_{\mathrm{ADS\ instance}}}{p_{\mathrm{input\ scale}}}.
$$

Pass the physical ADS-side value; do not pass a manually pre-scaled model
value. For a model trained with dimensionless micron counts such as `W=10`,
export with `--parameter-input-scales 1um` and pass `W=10um` on the instance.
The generated `W_input_scale=1um` makes the embedded network receive `10`. If
the MDIF value already parsed into SI, for example `W=0.40mm` became `0.0004`,
export with scale `1.0` and pass `W=0.40mm`. Normally leave all generated
`*_input_scale` parameters unchanged on every instance.

For KBNN, the external call is identical: pass one set of geometry parameters
per instance. The embedded fine and frozen coarse networks both receive the
same converted model values internally; the coarse model is not instantiated
or parameterized separately.

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

$$
[\mathbf p,\boldsymbol\phi(f)]
\xrightarrow{\mathrm{DNN}}
\widehat{\mathbf S}(\mathbf p,f)
\quad\text{or}\quad
\widehat{\mathbf Y}(\mathbf p,f)
$$

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

#### Adaptive range optimization

This is the recommended command when the useful values are not already known
as a short discrete list. It searches continuous learning-rate and integer
batch-size ranges, categorical activations, and the hidden-layer depth/width
space:

```bash
python3 dnn.py optimize \
  --mdif train_verify.mdif \
  --out-dir dnn_adaptive \
  --parameter-names W,L,H \
  --search-mode adaptive \
  --optimize-parameter learning_rate=1e-4:1e-2:log \
  --optimize-parameter batch_size=64:512:log \
  --optimize-parameter activation=tanh,relu \
  --optimize-parameter freq_transform=log,log-linear \
  --optimize-parameter 'hidden_layers=1:4x32:256:log' \
  --adaptive-initial-trials 8 \
  --adaptive-candidate-pool 768 \
  --adaptive-exploration 1.5 \
  --max-trials 32 \
  --sparam-weights 'diag=1;offdiag=0.2' \
  --selection-metric weighted_evm_pct \
  --require-passive
```

Here, only the five named domains vary. Options such as `--epochs`,
`--patience`, `--output-domain`, and `--target-z0` retain their normal values
unless they are also supplied through `--optimize-parameter`. To test specific
architectures instead of uniform-width depth/width combinations, use an
explicit domain such as
`--optimize-parameter 'hidden_layers=64,64;128,128,64;256,128,64'`.

#### Discrete grid or random optimization

Use the plural list options with `grid` or `random` when the complete candidate
set is already known. This is the original discrete optimization method:

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
- `diag`, `diagonal`, `return`, or `reflection`: $S_{ii}$
- `offdiag`, `off-diagonal`, or `transmission`: $S_{ij}$ where $i \ne j$
- `upper`: all $S_{ij}$ where $i < j$
- `lower`: all $S_{ij}$ where $i > j$
- `rowN`, `outN`, or `outputN`: all $S_{Nj}$
- `colN`, `columnN`, `inN`, or `inputN`: all $S_{iN}$
- Wildcards such as `S1*` or `S*1`
- Explicit groups such as `S11,S22,S33,S44`

#### Frequency weighting

Use `--frequency-weights` with DNN, KBNN, or Neuro-TF training and sweep
commands to prioritize particular frequencies or bands. Rules are separated by
semicolons, applied left to right, and normalized over the positive-frequency
training samples so their mean is 1.0. Zero Hz remains the separate
data-derived DC model and is never part of the RF fitted loss.

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

Set `--search-mode grid` to exhaustively test all combinations, keep the
default `--search-mode random --max-trials N` to sample a discrete product, or
use `--search-mode adaptive` with `--optimize-parameter` ranges so later trials
target the best results observed so far.

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
embeds SVG trend plots and links a diagnostics PDF and CSV under
`sweep_diagnostics/`, comparing error metrics against each swept option.
Passivity-failing trials are shown in red on those plots. Passive-only grouped
statistics remain available, while dashed all-trial means and `all_*` CSV
columns preserve trends when every trial fails passivity. If the goal is
fastest direct
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
   For example, `freq_log10_hz` means the wrapper must pass
   $\log_{10}(f_{\mathrm{Hz}})$.
3. Interpret `output_columns` as the final fine S-parameters, with all real
   columns followed by the matching imaginary columns.
4. Convert the final complex S-matrix to a circuit relation before driving the
   schematic pins. For reference impedance $Z_0$, use
   $\mathbf Y=(\mathbf I-\mathbf S)(\mathbf I+\mathbf S)^{-1}/Z_0$, then
   $\mathbf I_{\mathrm{port}}=\mathbf Y\mathbf V_{\mathrm{port}}$.
5. Validate the wrapper in an S-parameter or AC simulation before circuit
   optimization.

### ADS Harmonic-Balance Passive Network Export

Use `export-ads-hb` when the fitted structure must behave like a linear
S-parameter network inside harmonic balance:

```bash
python3 dnn.py export-ads-hb \
  --model-dir dnn_model \
  --out-dir dnn_ads_hb \
  --module-name my_dnn_4port_hb \
  --parameter-input-scales 1.0 \
  --z0 50
```

The default package contains `<module>.net`, `ads_hb_manifest.json`,
`ADS_HB_INSTANCE_TEMPLATE.txt`, and `ADS_HB_README.md`. Include the netlist with
an ADS `NetlistInclude`, then
instantiate `<module>:X1` with the electrical nodes and geometry parameters.
ADS applies the embedded matrix independently to the fundamental, harmonics,
and mixing products requested by the HB controller. The model is linear and
power independent. Direct-Y DNNs use
$\mathbf I=\mathbf Y(f)\mathbf V$ immediately; S-output DNNs are
converted to Y in frequency-only equations and use the same explicit current
stamp. DC is stamped separately from the fitted RF response.

The next timing trial requires a separate direct-Y fit. Start from the exact
training command used for the S-domain baseline. Keep its MDIFs, split options,
parameter and S-parameter order, frequency transform, activation, hidden layer
sizes, weights, and seed. Change only the output directory and response domain,
and set the reference impedance used by the MDIF:

```bash
python3 dnn.py train \
  --mdif train_verify.mdif \
  --out-dir dnn_model_direct_y_trial \
  --parameter-names W,L \
  --freq-transform log \
  --hidden-layers 128,128,64 \
  --activation tanh \
  --output-domain y \
  --target-z0 50 \
  --seed 1234
```

Then export the unchanged S-domain baseline and the separately trained trial
together:

```bash
python3 dnn.py export-ads-hb \
  --model-dir dnn_model \
  --direct-y-trial-model-dir dnn_model_direct_y_trial \
  --out-dir dnn_ads_hb \
  --module-name my_dnn_4port_hb \
  --parameter-input-scales 1.0 \
  --z0 50
```

In addition to the default package, this writes:

- `<module>_direct_y_trial.net`
- `ads_hb_direct_y_trial_manifest.json`
- `ADS_HB_DIRECT_Y_TRIAL_INSTANCE_TEMPLATE.txt`
- `ADS_HB_DIRECT_Y_TRIAL_README.md`

The baseline learns $\widehat{\mathbf S}(\mathbf p,f)$ and evaluates

$$
\mathbf Y_{\mathrm{RF}}=
\frac{1}{Z_0}
(\mathbf I-\widehat{\mathbf S})
(\mathbf I+\widehat{\mathbf S})^{-1}
$$

inside ADS at every RF spectral frequency. The trial instead learns
$\widehat{\mathbf Y}(\mathbf p,f)$ and stamps that matrix directly. Consequently,
the trial netlist has `response_domain: "y"` and `rf_source_conversion: "none"`.
It deliberately retains the baseline two-SDD topology and explicit neural
scalers, and it uses the baseline's exact-DC model rather than the DC model from
the direct-Y directory. Earlier combined-SDD, scaler-folding, and
constant-output trials are not stacked into it.

This is a refitted formulation, not an algebraically identical weight
transformation. Validate S-parameters, passivity, DC, HB fundamental power,
convergence, and both cold- and warm-run timing. The export stops on architecture
or impedance mismatch. Differences found in comparable training metadata are
recorded under `direct_y_comparison.training_metadata_differences` so the timing
trial does not silently combine unrelated fitting changes.

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
admittance with
$\mathbf Y=(\mathbf I-\mathbf S)(\mathbf I+\mathbf S)^{-1}/Z_0$. For Y-output models it
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
or NumPy in the simulator. The `export-ads-hb` command embeds the same local
weights in a linear SDD whose frequency weights are evaluated per HB spectral
component. The `export-veriloga` command targets SP/AC use and should be
validated in the target ADS Verilog-A compiler. The `export-ads-ann` command is
the native ADS ANN handoff for generating ADS ANN
Verilog-A/C/equation artifacts on an ADS machine.

### Options Reference

Options are grouped by purpose below. Rows are alphabetical within each table;
the **Subcommands** column includes accepted command aliases.

#### Files, data, and outputs

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--dc-mdif PATH</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Exact-DC validation/override source. A mismatch or legacy model triggers a DC-only conductance fit for the export; RF is never refitted. | <nobr><code>--dc-mdif train_with_dc.mdif</code></nobr> |
| <nobr><code>--direct-y-trial-model-dir PATH</code></nobr> | <code>export-ads-hb</code> | Optional separately trained direct-Y DNN used to emit the next ADS HB timing trial beside an unchanged S-domain baseline. Parameters, response order, frequency transform, activation, layer sizes, and reference impedance must match. | <nobr><code>--direct-y-trial-model-dir dnn_model_direct_y_trial</code></nobr> |
| <nobr><code>--mdif PATH</code></nobr> | <code>inspect-mdif</code>, <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>predict</code>, <code>export-ads-ann</code> | Input MDIF to inspect, fit, predict, or use as an ADS ANN retraining source, depending on the subcommand. | <nobr><code>--mdif train_verify.mdif</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | <code>predict</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-ann</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Directory containing the trained <code>model.npz</code> and <code>metadata.json</code> used for prediction or export. | <nobr><code>--model-dir dnn_model</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-ann</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Destination directory for the model, sweep, or export artifacts generated by the selected command. | <nobr><code>--out-dir dnn_model</code></nobr> |
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
| <nobr><code>--freq-transform {log,linear,log-linear}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Frequency input transform. `log` uses $\log_{10}(f_{\mathrm{Hz}})$, `linear` uses raw Hz, and `log-linear` uses both. Default: `log`. | <nobr><code>--freq-transform log-linear</code></nobr> |
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
| <nobr><code>--adaptive-candidate-pool INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Raw candidate configurations requested for adaptive search. The generator requests at least `--max-trials`, removes duplicates, and warns if the requested ranges contain fewer unique configurations. Must be positive. Default: `512`. | <nobr><code>--adaptive-candidate-pool 768</code></nobr> |
| <nobr><code>--adaptive-exploration FLOAT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Non-negative GP lower-confidence-bound uncertainty multiplier. Larger values explore uncertain configurations more strongly. Default: `1.5`. | <nobr><code>--adaptive-exploration 2</code></nobr> |
| <nobr><code>--adaptive-hidden-width-step INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Positive neuron-width increment used by structured `hidden_layers` ranges. Default: `8`. | <nobr><code>--adaptive-hidden-width-step 16</code></nobr> |
| <nobr><code>--adaptive-initial-trials INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Maximin-separated trials evaluated before GP guidance. Default: `6`. | <nobr><code>--adaptive-initial-trials 8</code></nobr> |
| <nobr><code>--best-model-dir PATH</code></nobr> | <code>rerank-sweep</code> | Destination for `--promote-best`. Default: `<sweep-dir>/best_model_reranked`. | <nobr><code>--best-model-dir dnn_sweep/best_model_passive</code></nobr> |
| <nobr><code>--jobs INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Number of independent grid/random trials to train in parallel. Adaptive search is sequential and forces one job. Default: `1`. | <nobr><code>--jobs 4</code></nobr> |
| <nobr><code>--keep-trial-models</code></nobr> | <code>sweep</code>, <code>optimize</code> | Keep full per-trial model directories under `trials/`. By default, each trial keeps lightweight summary and plot artifacts while large model files are removed. | <nobr><code>--keep-trial-models</code></nobr> |
| <nobr><code>--max-passivity-sigma FLOAT</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials whose worst predicted S-matrix singular value is at or below this value when selecting `best_model/`. | <nobr><code>--max-passivity-sigma 1.000001</code></nobr> |
| <nobr><code>--max-passivity-violations INT</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials with this many or fewer passivity-violating frequency points when selecting `best_model/`. | <nobr><code>--max-passivity-violations 0</code></nobr> |
| <nobr><code>--max-trials INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Maximum configurations evaluated. In `adaptive` mode this is the sequential trial budget; in `random` mode it limits the sample; in `grid` mode it truncates the product list. Default: `24`. | <nobr><code>--max-trials 40</code></nobr> |
| <nobr><code>--optimize-parameter SPEC</code></nobr> | <code>sweep</code>, <code>optimize</code> | Repeatable adaptive domain. DNN supports `freq_transform`, `hidden_layers`, `activation`, `learning_rate`, `output_domain`, `target_z0`, `batch_size`, `epochs`, and `patience`. | <nobr><code>--optimize-parameter learning_rate=1e-4:1e-2:log</code></nobr> |
| <nobr><code>--overwrite</code></nobr> | <code>rerank-sweep</code> | Allow `--promote-best` to replace an existing `--best-model-dir`. | <nobr><code>--overwrite</code></nobr> |
| <nobr><code>--promote-best</code></nobr> | <code>rerank-sweep</code> | Copy the selected trial model to `--best-model-dir` if that trial still contains `model.npz` and `metadata.json`. Requires the original sweep to have used `--keep-trial-models`. | <nobr><code>--promote-best</code></nobr> |
| <nobr><code>--replace-current-best</code></nobr> | <code>rerank-sweep</code> | Overwrite `<sweep-dir>/best_model` with the selected trial model if the trial model files are available. | <nobr><code>--replace-current-best</code></nobr> |
| <nobr><code>--require-passive</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials with zero passivity-violating frequency points when selecting `best_model/`. Equivalent to `--max-passivity-violations 0` unless a stricter value is supplied. | <nobr><code>--require-passive</code></nobr> |
| <nobr><code>--retrain-best</code></nobr> | <code>sweep</code>, <code>optimize</code> | Retrain the selected best configuration at the end of the sweep instead of using the best completed trial model promoted during the sweep. Use this when you want `--worst-plots` to apply only to the final model. | <nobr><code>--retrain-best</code></nobr> |
| <nobr><code>--search-mode {adaptive,grid,random}</code></nobr> | <code>sweep</code>, <code>optimize</code> | Search strategy. `adaptive` learns sequentially from completed trials, `grid` follows product order, and `random` samples the discrete product. Legacy `--mode` remains an alias. Default: `random`. | <nobr><code>--search-mode adaptive</code></nobr> |
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
| <nobr><code>--dc-open-resistance FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Finite resistance used to represent an open DC branch. Default: `1e19` ohm. | <nobr><code>--dc-open-resistance 1e19</code></nobr> |
| <nobr><code>--dc-open-threshold FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | A selected branch conductance below the reciprocal of this resistance is treated as open. Default: `1e12` ohm. | <nobr><code>--dc-open-threshold 1e12</code></nobr> |
| <nobr><code>--dc-port-paths SPEC</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Optional comma-separated restricted DC resistor paths. If omitted, both components of every ordered complex DC $S_{ij}$ value are fitted directly. | <nobr><code>--dc-port-paths 1-2,3-4</code></nobr> |
| <nobr><code>--freqs SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Frequency grid used with `--parameter-grid`. `SPEC` can be a comma list or `start:stop:count`. | <nobr><code>--freqs 1GHz:20GHz:401</code></nobr> |
| <nobr><code>--frequency-expression EXPR</code></nobr> | <code>export-veriloga</code> | Verilog-A expression for simulator frequency in Hz. Default: `$freq`. Change this only if your ADS Verilog-A release requires a different frequency expression. | <nobr><code>--frequency-expression '$freq'</code></nobr> |
| <nobr><code>--module-name NAME</code></nobr> | <code>export-ads-hb</code>, <code>export-veriloga</code> | Optional ADS subnetwork or Verilog-A module name. If omitted, the exporter derives one from the model directory. | <nobr><code>--module-name my_dnn_4port</code></nobr> |
| <nobr><code>--no-fold-scalers</code></nobr> | <code>export-veriloga</code> | Debug option. Keep input/output standardization as explicit Verilog-A arithmetic instead of folding it into the first and final neural layers. Leaving this unset is faster. | <nobr><code>--no-fold-scalers</code></nobr> |
| <nobr><code>--parameter-grid NAME=SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Optional repeatable grid definition. `SPEC` can be a comma list or `start:stop:count`. Repeat once for every model parameter when not using `--template-mdif`. | <nobr><code>--parameter-grid W=0.40mm:0.80mm:9</code></nobr> |
| <nobr><code>--parameter-input-scales SCALE</code></nobr> | <code>export-ads-hb</code>, <code>export-veriloga</code> | Common positive ADS-side unit scale used for every geometry/process parameter: $p_{\mathrm{model}}=p_{\mathrm{instance}}/s_{\mathrm{input}}$. Default: `1.0`. | <nobr><code>--parameter-input-scales 1um</code></nobr> |
| <nobr><code>--z0 FLOAT</code></nobr> | <code>export-ads-hb</code>, <code>export-veriloga</code> | S-parameter reference impedance. Direct-Y DNNs use the saved training `--target-z0` metadata instead. Default: `50.0`. | <nobr><code>--z0 50</code></nobr> |

---

## KBNN

This is the KBNN companion to the Neuro-TF prototype. It trains a neural model
from a fine/target S-parameter MDIF and, for knowledge-based modes, the
predictions of a frozen S-domain DNN previously fitted to the coarse response.

Supported forms:

$$
\begin{aligned}
\text{plain:}\quad
\widehat{\mathbf s}_{\mathrm f}
  &=n(\mathbf p,\boldsymbol\phi(f)),\\
\text{residual:}\quad
\widehat{\mathbf s}_{\mathrm f}
  &=\mathbf c(\mathbf p,f)
    +n(\mathbf p,\boldsymbol\phi(f)[,\mathbf c]),\\
\text{prior-input:}\quad
\widehat{\mathbf s}_{\mathrm f}
  &=n(\mathbf p,\boldsymbol\phi(f),\mathbf c(\mathbf p,f)).
\end{aligned}
$$

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

#### Adaptive range optimization

For an integrated KBNN, pass `--coarse-mdif` as usual. The coarse DNN is fitted
once, frozen, and reused while the adaptive optimizer trials different fine
model configurations:

```bash
python3 kbnn.py optimize \
  --mdif fine_train_verify.mdif \
  --coarse-mdif coarse_train_verify.mdif \
  --out-dir kbnn_adaptive \
  --parameter-names W,L \
  --search-mode adaptive \
  --optimize-parameter mode=residual,prior-input \
  --optimize-parameter include_coarse_input=false,true \
  --optimize-parameter freq_transform=log,linear \
  --optimize-parameter learning_rate=1e-4:8e-3:log \
  --optimize-parameter batch_size=64:512:log \
  --optimize-parameter activation=tanh,relu \
  --optimize-parameter 'hidden_layers=1:4x32:192:log' \
  --adaptive-initial-trials 8 \
  --adaptive-candidate-pool 768 \
  --adaptive-exploration 1.5 \
  --max-trials 32 \
  --sparam-weights 'diag=1;offdiag=0.2' \
  --selection-metric weighted_evm_pct \
  --require-passive
```

Invalid KBNN combinations are removed automatically: `plain` cannot use a
coarse input, and `prior-input` always requires it. Residual candidates may be
tested with either coarse-input setting. The coarse-model architecture is
fixed by the normal `--coarse-*` options; the adaptive hidden-layer and fitting
domains above apply to the fine network. Use `--coarse-model-dir` in place of
`--coarse-mdif` only when intentionally reusing a previously fitted coarse
network.

#### Discrete grid or random optimization

Use the plural list options when the KBNN modes and network configurations are
already known as a finite candidate set:

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

In addition to the domains shown in the adaptive example, `epochs` and
`patience` can be ranged with integer `--optimize-parameter` domains. Any
setting omitted from those domains remains fixed at its normal option value.
The fitted coarse model is still prepared once and shared by every fine-model
trial.

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
- `diag`, `diagonal`, `return`, or `reflection`: $S_{ii}$
- `offdiag`, `off-diagonal`, or `transmission`: $S_{ij}$ where $i \ne j$
- `upper`: all $S_{ij}$ where $i < j$
- `lower`: all $S_{ij}$ where $i > j$
- `rowN`, `outN`, or `outputN`: all $S_{Nj}$
- `colN`, `columnN`, `inN`, or `inputN`: all $S_{iN}$
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
embeds SVG trend plots and links a diagnostics PDF and CSV under
`sweep_diagnostics/`, comparing error metrics against each swept option.
Passivity-failing trials are shown in red. Passive-only grouped statistics
remain available, while dashed all-trial means and `all_*` CSV columns preserve
trends when every trial fails passivity. During a KBNN sweep, parsed MDIF
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
$\mathbf S_{\mathrm{fine}}=\mathbf S_{\mathrm{coarse}}+\Delta\mathbf S$.
Use `--ads-ann-target fine` when you want ADS ANN to
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
   For example, `freq_log10_hz` means the wrapper must pass
   $\log_{10}(f_{\mathrm{Hz}})$.
   If coarse S-parameter columns are listed as inputs, the wrapper must evaluate
   or instantiate the coarse circuit response at the same parameter/frequency
   point and pass those values into the ANN.
3. For `--ads-ann-target fine`, interpret `output_columns` as final fine
   S-parameters. For the default native residual export, interpret
   `output_columns` as `delta_S*` and reconstruct
   $S_{ij,\mathrm{fine}}=S_{ij,\mathrm{coarse}}+\Delta S_{ij}$.
4. Convert the final complex S-matrix to a circuit relation before driving the
   schematic pins. For reference impedance $Z_0$, use
   $\mathbf Y=(\mathbf I-\mathbf S)(\mathbf I+\mathbf S)^{-1}/Z_0$, then
   $\mathbf I_{\mathrm{port}}=\mathbf Y\mathbf V_{\mathrm{port}}$.
5. Validate the wrapper in an S-parameter or AC simulation before circuit
   optimization.

### ADS Harmonic-Balance Passive Network Export

Use `export-ads-hb` to package the fitted fine KBNN and its exact frozen coarse
DNN as one linear ADS subnetwork:

```bash
python3 kbnn.py export-ads-hb \
  --model-dir kbnn_model \
  --out-dir kbnn_ads_hb \
  --module-name my_kbnn_4port_hb \
  --parameter-input-scales 1.0 \
  --z0 50
```

For residual and prior-input modes, export is refused unless the matching
coarse model can be loaded and its saved hashes match. Both networks are then
embedded in `<module>.net`: the coarse prediction is evaluated at the same HB
spectral frequency as the fine network, and the final fine S-matrix drives the
generated S-to-Y conversion followed by an explicit current stamp. The DC
conductance is stamped by a separate branch, and no external coarse hooks,
implicit port-current unknowns, or power parameter remain.

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
admittance with
$\mathbf Y=(\mathbf I-\mathbf S)(\mathbf I+\mathbf S)^{-1}/Z_0$, and contributes the
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
coarse S-domain DNN into one Verilog-A n-port for SP/AC use. The
`export-ads-hb` command embeds both networks in one linear SDD subnetwork for
harmonic balance. The `export-ads-ann` command is
the native ADS ANN handoff for generating ADS ANN Verilog-A/C/equation
artifacts on an ADS machine.

### Options Reference

Options are grouped by purpose below. Rows are alphabetical within each table;
the **Subcommands** column includes accepted command aliases.

#### Files, data, and outputs

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--coarse-mdif PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Recommended coarse source for `residual` and `prior-input`. Fits an S-domain DNN first and saves its complete outputs under `<out-dir>/coarse_model/`. Mutually exclusive with `--coarse-model-dir`. | <nobr><code>--coarse-mdif coarse_train_verify.mdif</code></nobr> |
| <nobr><code>--coarse-model-dir PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>predict</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Directory containing the frozen coarse DNN used by the KBNN. Training may create it from coarse MDIF data; prediction and export can use the packaged model or a validated relocated copy. | <nobr><code>--coarse-model-dir coarse_dnn_model</code></nobr> |
| <nobr><code>--coarse-verification-mdif PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Optional separate coarse/prior verification MDIF. Use this with `--verification-mdif` when fine and coarse verification data are stored separately. | <nobr><code>--coarse-verification-mdif coarse_verify.mdif</code></nobr> |
| <nobr><code>--dc-mdif PATH</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Exact fine-data DC validation/override source. A mismatch or legacy model triggers a fine-data DC-only fit; neither RF network is refitted. | <nobr><code>--dc-mdif fine_with_dc.mdif</code></nobr> |
| <nobr><code>--mdif PATH</code></nobr> | <code>inspect-mdif</code>, <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>predict</code>, <code>export-ads-ann</code> | Fine/target MDIF to inspect, fit, predict, or use as an ADS ANN retraining source, depending on the subcommand. | <nobr><code>--mdif fine_train_verify.mdif</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | <code>predict</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-ann</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Directory containing the trained <code>model.npz</code> and <code>metadata.json</code> used for prediction or export. | <nobr><code>--model-dir kbnn_model</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-ann</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Destination directory for the model, sweep, or export artifacts generated by the selected command. | <nobr><code>--out-dir kbnn_model</code></nobr> |
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
| <nobr><code>--freq-transform {log,linear}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Frequency input transform. `log` uses $\log_{10}(f_{\mathrm{Hz}})$ and is usually better for wideband data. Default: `log`. | <nobr><code>--freq-transform log</code></nobr> |
| <nobr><code>--freq-transforms LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated frequency transforms to try. `--freq-transform` accepts one train-compatible value; `--freq-transform-options` remains an alias. | <nobr><code>--freq-transforms log,linear</code></nobr> |
| <nobr><code>--hidden-layers LIST</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Comma-separated hidden-layer sizes for one model. Sweeps also accept semicolon-separated candidate layouts. Train default: `64,64`; sweep default: `32;64;64,64`. `--hidden-layer-layouts` and `--hidden-layer-options` remain aliases. | <nobr><code>--hidden-layers 64,64</code></nobr> |
| <nobr><code>--include-coarse-input</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | In `residual` mode, append coarse real/imaginary S-parameters to the NN input vector. This can improve accuracy if the correction depends strongly on the coarse response. Forced on for `prior-input` and off for `plain`. | <nobr><code>--include-coarse-input</code></nobr> |
| <nobr><code>--include-coarse-inputs LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated boolean candidates for `--include-coarse-input`. Supplying the singular flag selects only `true`; `--include-coarse-input-options` remains an alias. Default: `false,true`. | <nobr><code>--include-coarse-inputs false,true</code></nobr> |
| <nobr><code>--learning-rate FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Adam step size. Default: `0.002`. | <nobr><code>--learning-rate 0.002</code></nobr> |
| <nobr><code>--learning-rates LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated Adam learning rates. `--learning-rate` accepts one train-compatible value. Default: `0.001,0.002,0.005`. | <nobr><code>--learning-rates 0.001,0.002,0.005</code></nobr> |
| <nobr><code>--loss-interval INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Full train/verification loss check interval in epochs. Increasing this reduces full-dataset scoring overhead during long runs while early stopping still uses epoch-based patience. Default: `1`. | <nobr><code>--loss-interval 5</code></nobr> |
| <nobr><code>--mode {plain,residual,prior-input}</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | KBNN formulation. `residual` learns $\mathbf S_{\mathrm{fine}}-\widehat{\mathbf S}_{\mathrm{coarse}}$; `prior-input` predicts fine S using fitted coarse-DNN predictions as inputs; `plain` uses no coarse model. Default: `residual`. | <nobr><code>--mode residual</code></nobr> |
| <nobr><code>--modes LIST</code></nobr> | <code>sweep</code>, <code>optimize</code> | Comma-separated KBNN model modes. The singular `--mode` accepts one train-compatible value; `--mode-options` remains an alias. Default: `plain,residual,prior-input`. | <nobr><code>--modes residual,prior-input</code></nobr> |
| <nobr><code>--patience INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Early-stopping patience in epochs for each candidate. Use `0` to disable. Default: `200`. | <nobr><code>--patience 200</code></nobr> |
| <nobr><code>--progress-interval INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Console progress update interval in epochs. Updates redraw one terminal status line and include epoch count, elapsed time, and loss values when that epoch also matches `--loss-interval`. Use `0` to disable. Default: `25`. | <nobr><code>--progress-interval 10</code></nobr> |
| <nobr><code>--seed INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-ann</code> | Random seed for data splitting, model initialization, minibatch order, ADS ANN data preparation, and sweep candidate selection where applicable. Default: `1234`. | <nobr><code>--seed 1234</code></nobr> |
| <nobr><code>--worst-plots INT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code> | Number of worst verification S/Y plot pairs to generate. In a sweep it applies to a final `--retrain-best`; otherwise the promoted trial retains its `--trial-worst-plots` output. Default: `6`. | <nobr><code>--worst-plots 6</code></nobr> |

#### Sweep and model selection

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--adaptive-candidate-pool INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Raw candidate configurations requested for adaptive search. The generator requests at least `--max-trials`, removes duplicates, and warns if the requested ranges contain fewer unique configurations. Must be positive. Default: `512`. | <nobr><code>--adaptive-candidate-pool 768</code></nobr> |
| <nobr><code>--adaptive-exploration FLOAT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Non-negative GP lower-confidence-bound uncertainty multiplier. Larger values explore uncertain configurations more strongly. Default: `1.5`. | <nobr><code>--adaptive-exploration 2</code></nobr> |
| <nobr><code>--adaptive-hidden-width-step INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Positive neuron-width increment used by structured `hidden_layers` ranges. Default: `8`. | <nobr><code>--adaptive-hidden-width-step 16</code></nobr> |
| <nobr><code>--adaptive-initial-trials INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Maximin-separated trials evaluated before GP guidance. Default: `6`. | <nobr><code>--adaptive-initial-trials 8</code></nobr> |
| <nobr><code>--best-model-dir PATH</code></nobr> | <code>rerank-sweep</code> | Destination for `--promote-best`. Default: `<sweep-dir>/best_model_reranked`. | <nobr><code>--best-model-dir kbnn_sweep/best_model_passive</code></nobr> |
| <nobr><code>--jobs INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Number of independent grid/random trials to train in parallel. Adaptive search is sequential and forces one job. Default: `1`. | <nobr><code>--jobs 4</code></nobr> |
| <nobr><code>--keep-trial-models</code></nobr> | <code>sweep</code>, <code>optimize</code> | Keep full per-trial model directories under `trials/`. By default, each trial keeps lightweight summary and plot artifacts while large model files are removed. | <nobr><code>--keep-trial-models</code></nobr> |
| <nobr><code>--max-passivity-sigma FLOAT</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials whose worst predicted S-matrix singular value is at or below this value when selecting `best_model/`. | <nobr><code>--max-passivity-sigma 1.000001</code></nobr> |
| <nobr><code>--max-passivity-violations INT</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials with this many or fewer passivity-violating frequency points when selecting `best_model/`. | <nobr><code>--max-passivity-violations 0</code></nobr> |
| <nobr><code>--max-trials INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Maximum configurations evaluated. In `adaptive` mode this is the sequential trial budget. Default: `24`. | <nobr><code>--max-trials 24</code></nobr> |
| <nobr><code>--optimize-parameter SPEC</code></nobr> | <code>sweep</code>, <code>optimize</code> | Repeatable adaptive domain. KBNN supports `mode`, `include_coarse_input`, `freq_transform`, `hidden_layers`, `activation`, `learning_rate`, `batch_size`, `epochs`, and `patience`. | <nobr><code>--optimize-parameter mode=residual,prior-input</code></nobr> |
| <nobr><code>--overwrite</code></nobr> | <code>rerank-sweep</code> | Allow `--promote-best` to replace an existing `--best-model-dir`. | <nobr><code>--overwrite</code></nobr> |
| <nobr><code>--promote-best</code></nobr> | <code>rerank-sweep</code> | Copy the selected trial model to `--best-model-dir` if that trial still contains `model.npz` and `metadata.json`. Requires the original sweep to have used `--keep-trial-models`. | <nobr><code>--promote-best</code></nobr> |
| <nobr><code>--replace-current-best</code></nobr> | <code>rerank-sweep</code> | Overwrite `<sweep-dir>/best_model` with the selected trial model if the trial model files are available. | <nobr><code>--replace-current-best</code></nobr> |
| <nobr><code>--require-passive</code></nobr> | <code>sweep</code>, <code>optimize</code>, <code>rerank-sweep</code> | Only consider trials with zero passivity-violating frequency points when selecting `best_model/`. Equivalent to `--max-passivity-violations 0` unless a stricter value is supplied. | <nobr><code>--require-passive</code></nobr> |
| <nobr><code>--retrain-best</code></nobr> | <code>sweep</code>, <code>optimize</code> | Retrain the selected best configuration at the end of the sweep instead of using the best completed trial model promoted during the sweep. Use this when you want `--worst-plots` to apply only to the final model. | <nobr><code>--retrain-best</code></nobr> |
| <nobr><code>--search-mode {adaptive,grid,random}</code></nobr> | <code>sweep</code>, <code>optimize</code> | Search strategy. `adaptive` learns sequentially from completed trials. Legacy `--mode adaptive`, `--mode grid`, and `--mode random` remain valid. Default: `random`. | <nobr><code>--search-mode adaptive</code></nobr> |
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
| <nobr><code>--dc-open-resistance FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Finite resistance used to represent an open fine-data DC branch. Default: `1e19` ohm. | <nobr><code>--dc-open-resistance 1e19</code></nobr> |
| <nobr><code>--dc-open-threshold FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | A selected fine-data branch conductance below the reciprocal of this resistance is treated as open. Default: `1e12` ohm. | <nobr><code>--dc-open-threshold 1e12</code></nobr> |
| <nobr><code>--dc-port-paths SPEC</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Optional comma-separated restricted fine-data DC resistor paths. If omitted, both components of every ordered complex fine-data DC $S_{ij}$ value are fitted directly. | <nobr><code>--dc-port-paths 1-2,3-4</code></nobr> |
| <nobr><code>--freqs SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Frequency grid used with `--parameter-grid`. `SPEC` can be a comma list or `start:stop:count`. | <nobr><code>--freqs 1GHz:20GHz:401</code></nobr> |
| <nobr><code>--frequency-expression EXPR</code></nobr> | <code>export-veriloga</code> | Verilog-A expression for simulator frequency in Hz. Default: `$freq`. Change this only if your ADS Verilog-A release requires a different frequency expression. | <nobr><code>--frequency-expression '$freq'</code></nobr> |
| <nobr><code>--module-name NAME</code></nobr> | <code>export-ads-hb</code>, <code>export-veriloga</code> | Optional ADS subnetwork or Verilog-A module name. If omitted, the exporter derives one from the model directory. | <nobr><code>--module-name my_kbnn_4port</code></nobr> |
| <nobr><code>--parameter-grid NAME=SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Optional repeatable grid definition. `SPEC` can be a comma list or `start:stop:count`. Repeat once for every model parameter when not using `--template-mdif`. | <nobr><code>--parameter-grid W=0.40mm:0.80mm:9</code></nobr> |
| <nobr><code>--parameter-input-scales SCALE</code></nobr> | <code>export-ads-hb</code>, <code>export-veriloga</code> | Common positive ADS-side unit scale used before both fine and coarse networks: $p_{\mathrm{model}}=p_{\mathrm{instance}}/s_{\mathrm{input}}$. Default: `1.0`. | <nobr><code>--parameter-input-scales 1um</code></nobr> |
| <nobr><code>--z0 FLOAT</code></nobr> | <code>export-ads-hb</code>, <code>export-veriloga</code> | S-parameter reference impedance used by the exported wave or admittance relation. Default: `50.0`. | <nobr><code>--z0 50</code></nobr> |

---

## Neuro-TF

This is a self-contained prototype for training a Neuro-transfer-function
surrogate from parameterized S-parameter MDIF data.

Model structure:

$$
\mathbf p
\xrightarrow{\mathrm{MLP}}
\widehat{\mathbf C}(\mathbf p)
\xrightarrow{\text{fixed-pole rational basis}}
\widehat{\mathbf S}(\mathbf p,f)
$$

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

#### Adaptive range optimization

Neuro-TF can adapt both the rational transfer-function fit and the neural
coefficient model. This example searches pole count, damping, ridge
regularization, learning rate, activation, and hidden-layer structure:

```bash
python3 neuro_tf.py optimize \
  --mdif train_verify.mdif \
  --out-dir neuro_tf_adaptive \
  --parameter-names W,L,H \
  --search-mode adaptive \
  --optimize-parameter order=6:20 \
  --optimize-parameter pole_damping=0.08:0.35:log \
  --optimize-parameter ridge=1e-10:1e-5:log \
  --optimize-parameter learning_rate=1e-4:8e-3:log \
  --optimize-parameter activation=tanh,relu \
  --optimize-parameter 'hidden_layers=1:4x32:192:log' \
  --adaptive-initial-trials 8 \
  --adaptive-candidate-pool 768 \
  --adaptive-exploration 1.5 \
  --max-trials 32 \
  --selection-metric rmse_abs \
  --require-passive
```

`order` is sampled as an integer. The positive `pole_damping`, `ridge`, and
`learning_rate` ranges use logarithmic sampling so each decade is represented.
To weight specific frequency bands during coefficient fitting and model
selection, add a normal option such as
`--frequency-weights 'default=1;2GHz:4GHz=5'`.

#### Discrete grid or random optimization

Use the plural list options when the rational and network candidates are
already known as a finite set:

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

Set `--search-mode grid` to exhaustively test all combinations, keep the
default `--search-mode random --max-trials N` to sample a discrete product, or
use `--search-mode adaptive` with ranges for `order`, `pole_damping`, `ridge`,
`hidden_layers`, `activation`, `learning_rate`, `batch_size`, `epochs`, or
`patience`.

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
embeds SVG trend plots and links a diagnostics PDF and CSV under
`sweep_diagnostics/`, comparing error metrics against each swept option.
Passivity-failing trials are shown in red on those plots. Passive-only grouped
statistics remain available, while dashed all-trial means and `all_*` CSV
columns preserve trends when every trial fails passivity.

### Predict

Predict new parameter blocks after training:

```bash
python3 neuro_tf.py predict \
  --model-dir neuro_tf_model \
  --mdif new_parameter_blocks.mdif \
  --out-mdif predicted.mdif
```

### ADS Harmonic-Balance Passive Network Export

Export the trained coefficient network and fixed-pole response as one linear
ADS HB subnetwork:

```bash
python3 neuro_tf.py export-ads-hb \
  --model-dir neuro_tf_model \
  --out-dir neuro_tf_ads_hb \
  --module-name my_neuro_tf_4port_hb \
  --parameter-input-scales 1.0 \
  --z0 50
```

The generated equations evaluate the rational S-matrix at every HB spectral
frequency, convert it to Y, and apply it through an explicit current stamp. A
separate explicit branch stamps DC conductance only at zero frequency. The
model remains linear and power independent; the fixed poles provide the
frequency dependence, not signal-amplitude dependence.

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

The generated module evaluates the neural coefficient map and constructs each
S-parameter as

$$
S_{ij}(\mathbf p,f)
=c_{ij,0}(\mathbf p)
+\sum_k\frac{c_{ij,k}(\mathbf p)}{j f/f_{\mathrm{scale}}-p_k},
$$

then converts the complete S-matrix to Y and stamps the small-signal port
currents. It requires no Python
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

Use `export-ads-mdif` for the lowest-risk interpolation-based handoff,
`export-ads-hb` for an integrated harmonic-balance component, or
`export-veriloga` for direct SP/AC evaluation of the trained coefficient
network and fixed-pole response. Both direct packages are self-contained; the
sampled MDIF remains useful as a simulator-independent cross-check.

### Options Reference

Options are grouped by purpose below. Rows are alphabetical within each table;
the **Subcommands** column includes accepted command aliases.

#### Files, data, and outputs

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--dc-mdif PATH</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Exact-DC validation/override source. A mismatch or legacy model triggers a DC-only conductance fit; the RF coefficient network is never refitted. | <nobr><code>--dc-mdif train_with_dc.mdif</code></nobr> |
| <nobr><code>--mdif PATH</code></nobr> | <code>inspect-mdif</code>, <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>predict</code> | Input MDIF to inspect, fit, or predict, depending on the subcommand. | <nobr><code>--mdif train_verify.mdif</code></nobr> |
| <nobr><code>--model-dir PATH</code></nobr> | <code>predict</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Directory containing the trained <code>model.npz</code> and <code>metadata.json</code> used for prediction or export. | <nobr><code>--model-dir neuro_tf_model</code></nobr> |
| <nobr><code>--out-dir PATH</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Destination directory for the model, sweep, or export artifacts generated by the selected command. | <nobr><code>--out-dir neuro_tf_model</code></nobr> |
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
| <nobr><code>--adaptive-candidate-pool INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Raw candidate configurations requested for adaptive search. The generator requests at least `--max-trials`, removes duplicates, and warns if the requested ranges contain fewer unique configurations. Must be positive. Default: `512`. | <nobr><code>--adaptive-candidate-pool 768</code></nobr> |
| <nobr><code>--adaptive-exploration FLOAT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Non-negative GP lower-confidence-bound uncertainty multiplier. Larger values explore uncertain configurations more strongly. Default: `1.5`. | <nobr><code>--adaptive-exploration 2</code></nobr> |
| <nobr><code>--adaptive-hidden-width-step INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Positive neuron-width increment used by structured `hidden_layers` ranges. Default: `8`. | <nobr><code>--adaptive-hidden-width-step 16</code></nobr> |
| <nobr><code>--adaptive-initial-trials INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Maximin-separated trials evaluated before GP guidance. Default: `6`. | <nobr><code>--adaptive-initial-trials 8</code></nobr> |
| <nobr><code>--jobs INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Number of independent grid/random trials to train in parallel. Adaptive search is sequential and forces one job. Default: `1`. | <nobr><code>--jobs 4</code></nobr> |
| <nobr><code>--keep-trial-models</code></nobr> | <code>sweep</code>, <code>optimize</code> | Keep full per-trial model directories under `trials/`. By default, each trial keeps lightweight summary and plot artifacts while large model files are removed. | <nobr><code>--keep-trial-models</code></nobr> |
| <nobr><code>--max-passivity-sigma FLOAT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Only consider trials whose worst predicted S-matrix singular value is at or below this value when selecting `best_model/`. | <nobr><code>--max-passivity-sigma 1.000001</code></nobr> |
| <nobr><code>--max-passivity-violations INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Only consider trials with this many or fewer passivity-violating frequency points when selecting `best_model/`. | <nobr><code>--max-passivity-violations 0</code></nobr> |
| <nobr><code>--max-trials INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Maximum configurations evaluated. In `adaptive` mode this is the sequential trial budget; in `random` mode it limits the sample; in `grid` mode it truncates the product list. Default: `24`. | <nobr><code>--max-trials 40</code></nobr> |
| <nobr><code>--optimize-parameter SPEC</code></nobr> | <code>sweep</code>, <code>optimize</code> | Repeatable adaptive domain. Neuro-TF supports `order`, `pole_damping`, `ridge`, `hidden_layers`, `activation`, `learning_rate`, `batch_size`, `epochs`, and `patience`. | <nobr><code>--optimize-parameter order=6:20</code></nobr> |
| <nobr><code>--require-passive</code></nobr> | <code>sweep</code>, <code>optimize</code> | Only consider trials with zero passivity-violating frequency points when selecting `best_model/`. Equivalent to `--max-passivity-violations 0` unless a stricter value is supplied. | <nobr><code>--require-passive</code></nobr> |
| <nobr><code>--retrain-best</code></nobr> | <code>sweep</code>, <code>optimize</code> | Retrain the selected best configuration at the end of the sweep instead of using the best completed trial model promoted during the sweep. Use this when you want `--worst-plots` to apply only to the final model. | <nobr><code>--retrain-best</code></nobr> |
| <nobr><code>--search-mode {adaptive,grid,random}</code></nobr> | <code>sweep</code>, <code>optimize</code> | Search strategy. `adaptive` learns sequentially from completed trials. Legacy `--mode` remains an alias. Default: `random`. | <nobr><code>--search-mode adaptive</code></nobr> |
| <nobr><code>--selection-metric NAME</code></nobr> | <code>sweep</code>, <code>optimize</code> | Metric minimized when choosing the best trial. Includes unweighted error, passivity, and `weighted_*` metrics that apply `--frequency-weights`. Default: `rmse_abs`. | <nobr><code>--selection-metric weighted_rmse_abs</code></nobr> |
| <nobr><code>--trial-seed-mode {fixed,indexed}</code></nobr> | <code>sweep</code>, <code>optimize</code> | Controls the seed used inside each sweep trial. `fixed` uses `--seed` for every trial so repeated candidates compare directly across sweeps. `indexed` restores the older `--seed + trial_number` behavior. Default: `fixed`. | <nobr><code>--trial-seed-mode fixed</code></nobr> |
| <nobr><code>--trial-worst-plots INT</code></nobr> | <code>sweep</code>, <code>optimize</code> | Number of lightweight worst-case S/Y PDF pairs generated and linked for each sweep trial. Default: `1`. | <nobr><code>--trial-worst-plots 1</code></nobr> |

#### Export and ADS integration

| Option | Subcommands | Description | Example |
| --- | --- | --- | --- |
| <nobr><code>--dc-open-resistance FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Finite resistance used to represent an open DC branch. Default: `1e19` ohm. | <nobr><code>--dc-open-resistance 1e19</code></nobr> |
| <nobr><code>--dc-open-threshold FLOAT</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | A selected branch conductance below the reciprocal of this resistance is treated as open. Default: `1e12` ohm. | <nobr><code>--dc-open-threshold 1e12</code></nobr> |
| <nobr><code>--dc-port-paths SPEC</code></nobr> | <code>train</code>, <code>sweep</code>, <code>optimize</code>, <code>export-ads-mdif</code>, <code>export-ads</code>, <code>export-ads-hb</code>, <code>export-veriloga</code> | Optional comma-separated restricted DC resistor paths. If omitted, both components of every ordered complex DC $S_{ij}$ value are fitted directly. | <nobr><code>--dc-port-paths 1-2,3-4</code></nobr> |
| <nobr><code>--freqs SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Frequency grid used with `--parameter-grid`. | <nobr><code>--freqs 1GHz:20GHz:401</code></nobr> |
| <nobr><code>--frequency-expression EXPR</code></nobr> | <code>export-veriloga</code> | Verilog-A expression for simulator frequency in Hz. Default: `$freq`. | <nobr><code>--frequency-expression '$freq'</code></nobr> |
| <nobr><code>--module-name NAME</code></nobr> | <code>export-ads-hb</code>, <code>export-veriloga</code> | Optional ADS subnetwork or Verilog-A module name. If omitted, the exporter derives one from the model directory. | <nobr><code>--module-name my_neuro_tf_4port</code></nobr> |
| <nobr><code>--parameter-grid SPEC</code></nobr> | <code>export-ads-mdif</code>, <code>export-ads</code> | Explicit grid for one model parameter. Repeat once per parameter; requires `--freqs`. | <nobr><code>--parameter-grid W=0.4mm:0.8mm:9</code></nobr> |
| <nobr><code>--parameter-input-scales SCALE</code></nobr> | <code>export-ads-hb</code>, <code>export-veriloga</code> | Common positive ADS-side unit scale used for every geometry/process parameter: $p_{\mathrm{model}}=p_{\mathrm{instance}}/s_{\mathrm{input}}$. Default: `1.0`. | <nobr><code>--parameter-input-scales 1um</code></nobr> |
| <nobr><code>--z0 FLOAT</code></nobr> | <code>export-ads-hb</code>, <code>export-veriloga</code> | S-parameter reference impedance used by the exported wave or admittance relation. Default: `50.0`. | <nobr><code>--z0 50</code></nobr> |

## Appendix A: Exact-DC Extraction and Export Method

This appendix describes the complete exact-DC data path shared by DNN, KBNN,
and Neuro-TF models. The DC model is intentionally independent of the RF model:
it has its own samples, scaler, neural-network weights, training history,
validation metrics, and saved files. An exact-zero-Hz MDIF row never contributes
to the positive-frequency RF loss, while a positive-frequency row is never used
as a substitute for DC.

### A.1 End-to-end data flow

For each geometry, the implementation follows this sequence:

1. Split the MDIF blocks into training and verification geometries.
2. Copy only rows with $f>0$ into the RF fitting data.
3. Inspect rows with $f=0$ separately for DC extraction.
4. Reject non-finite and non-passive zero-Hz matrices.
5. Form one exact-DC target per usable geometry.
6. Fit a geometry-only DC MLP independently from the RF network.
7. Save the DC model as `dc_model.npz` and `dc_model.json`, with its optimizer
   history in `dc_training_history.csv`.
8. At prediction or export, select the DC model only at exactly zero Hz and the
   RF model only at nonzero frequency.

The same separation applies to all three model families. In particular, a KBNN
extracts its DC target exclusively from the **fine-data MDIF**. Its fitted or
integrated coarse model participates in the positive-frequency KBNN response,
but does not supply, modify, or regularize the DC target.

Frequency weights and S-parameter weights apply to RF fitting only. They do not
change zero-Hz row selection, DC passivity filtering, or DC-model loss.

### A.2 Selecting usable exact-DC samples

The extractor requires a complete ordered N-port S-matrix ($S_{11}$ through
$S_{NN}$) and the reference impedance $Z_0$ used by the command. It applies the
following rules independently to each ACDATA block:

- A DC row must have a numeric frequency exactly equal to `0.0` Hz. The lowest
  positive frequency is not treated as DC and the RF model is not extrapolated.
- Every DC-training block must contain at least one exact-zero-Hz row. Missing
  rows are reported using one-based ACDATA block positions.
- A verification block may omit DC. Such a block remains available for RF
  verification but is excluded from DC verification.
- Every element of the zero-Hz S-matrix must be finite.
- The matrix must pass the singular-value test
  $\max_k\sigma_k(\mathbf S)\leq 1+10^{-6}$. Rows that fail this test are considered
  non-passive and are ignored.
- If a block contains multiple usable zero-Hz rows, their complex S-matrices are
  averaged component by component to create one DC target for that geometry.
- A block with zero-Hz rows but no usable row is excluded from the DC data. The
  fit fails only if no usable passive DC geometries remain. A training block
  with no zero-Hz row at all remains an error because that usually indicates an
  incomplete dataset rather than a deliberately filtered measurement.

When `--dc-mdif` points to a combined training/verification MDIF, the exporter
reuses the saved `--split-var`, `--train-values`, and `--verify-values` settings.
Only the training split is eligible for export-time DC fitting; verification
data is not silently added to the optimizer.

### A.3 Default unrestricted full-complex-S extraction

When `--dc-port-paths` is omitted, the extractor does not infer a resistor
network. Instead, it preserves every ordered complex S-parameter directly. For
an N-port model, the DC MLP has $2N^2$ outputs:

```text
S11.real, S12.real, ..., SNN.real,
S11.imag, S12.imag, ..., SNN.imag
```

The matrices are flattened in row-major order. The real components are stored
first and the imaginary components second. For example, a four-port DC model
contains all 16 ordered S-parameters and therefore has 32 scalar outputs.

Let $\mathbf p$ be the vector of geometry/process parameters. The model
evaluates

$$
\begin{aligned}
\widetilde{\mathbf p}
  &=\frac{\mathbf p-\boldsymbol\mu_p}{\boldsymbol\sigma_p},\\
\widetilde{\mathbf s}
  &=\operatorname{MLP}(\widetilde{\mathbf p}),\\
\mathbf s_{\mathrm{components}}
  &=\widetilde{\mathbf s}\odot\boldsymbol\sigma_s+\boldsymbol\mu_s,\\
\mathbf S_{\mathrm{DC}}
  &=\operatorname{reshape}
    (\mathbf s_{\mathrm{real}}+j\mathbf s_{\mathrm{imag}},N,N).
\end{aligned}
$$

Input and output standardizers are fitted only from DC-training geometries. If
an output component is constant across those geometries, its stored output
scale is set to zero so the constant is reproduced exactly instead of depending
on neural-network convergence.

This default representation deliberately imposes none of the following:

- reciprocity ($S_{ij}$ need not equal $S_{ji}$);
- a real-only admittance approximation;
- a resistor-graph topology;
- a projection onto a subset of port paths.

Consequently, `dc_port_paths: []` means **unrestricted full-S extraction**. It
does not mean that no DC entries were created. The complete ordered entries are
listed in `dc_sparameter_entries`, and their real/imaginary components are
listed in `dc_matrix_entries`.

When an electrical admittance representation is required, the predicted matrix
is converted using

$$
\mathbf Y_{\mathrm{DC}}
=\frac{1}{Z_0}
(\mathbf I-\mathbf S_{\mathrm{DC}})
(\mathbf I+\mathbf S_{\mathrm{DC}})^{-1}.
$$

The implementation solves the equivalent complex linear system rather than
forming the inverse explicitly. A singular or nearly singular relation uses
the exporter's guarded matrix-solve behavior.

Passivity filtering is applied to the supplied zero-Hz rows, but the MLP itself
is not a constrained passive-network fit. Interpolated geometries should
therefore still be checked using the saved validation metrics or a sampled-MDIF
comparison.

### A.4 Explicit resistor-path extraction

Supplying `--dc-port-paths` selects a different, intentionally restricted
model. For example:

```text
--dc-port-paths 1-2,3-ground
```

declares one resistor between ports 1 and 2 and another from port 3 to the
simulator reference. Undeclared DC paths remain open.

For each usable exact-DC S-matrix, the extractor first computes

$$
\mathbf Y_{\mathrm{measured}}
=\frac{1}{Z_0}
(\mathbf I-\mathbf S_{\mathrm{DC}})
(\mathbf I+\mathbf S_{\mathrm{DC}})^{-1}.
$$

It then takes the real symmetric target

$$
\mathbf Y_{\mathrm{target}}
=\operatorname{Re}\!\left(
\frac{\mathbf Y_{\mathrm{measured}}
      +\mathbf Y_{\mathrm{measured}}^{\mathsf T}}{2}
\right).
$$

and solves a non-negative least-squares problem for the declared branch
conductances. A branch of conductance $g$ between ports $i$ and $j$ contributes
$+g$ to $Y_{ii}$ and $Y_{jj}$, and $-g$ to $Y_{ij}$ and $Y_{ji}$. A
port-to-ground branch contributes $+g$ only to its diagonal entry. This construction guarantees a
real, reciprocal, non-negative resistor graph, but it cannot represent a
general complex or nonreciprocal DC S-matrix.

After projection:

- conductance below the reciprocal of `--dc-open-threshold` is classified as open;
- an open branch is represented by the reciprocal of `--dc-open-resistance` so the exported
  model remains finite;
- repeated usable zero-Hz rows in one block are averaged in conductance space;
- the MLP fits the natural logarithm of each branch conductance, which preserves
  positive conductance after interpolation;
- constant path outputs use their stored mean exactly.

The projected network is converted back to S and compared with the supplied
matrix. `dc_topology_s_rmse` and `dc_topology_s_max_abs_error` quantify error
introduced by the selected topology before considering neural interpolation.
An explicit-path export stops when its final maximum absolute S error exceeds
$10^{-3}$, because continuing would claim that the declared physical graph matches
data it cannot reproduce.

For a two-port `--dc-port-paths 1-2` model, the only available admittance is

$$
\mathbf Y_{\mathrm{DC}}
=\begin{bmatrix}
g & -g\\
-g & g
\end{bmatrix}.
$$

so it cannot independently reproduce all four ordered S-parameters. Use the
default unrestricted mode unless this physical restriction is intentional.

### A.5 DC fitting and saved artifacts

Normal `train`, `sweep`, and `optimize` operations fit the DC MLP alongside the
RF model, but the two optimizations remain numerically separate. The DC network
uses the command's geometry parameter list, hidden-layer layout, activation,
epoch count, batch size, learning rate, patience, seed, loss interval, and
progress interval. It does not receive frequency, RF response samples, KBNN
coarse responses, or RF loss weights as inputs.

The model directory contains:

| File | Purpose |
| --- | --- |
| `dc_model.npz` | DC input/output scalers and MLP weights and biases. |
| `dc_model.json` | Representation, matrix/path ordering, reference impedance, extraction counts, fit metrics, and topology metadata. |
| `dc_training_history.csv` | Independent DC training and verification loss history. |

The RF model does not need to be refitted when only the DC representation is
updated. This is why `--dc-mdif` can upgrade an older model during export.

### A.6 Export-time validation and legacy upgrades

Without `--dc-mdif`, a current saved `full_s_matrix` DC model is embedded or
sampled directly. If a saved model uses the former lossy full-Y or automatically
inferred path representation, the exporter requires the original MDIF so it can
reconstruct information that is not present in the old saved model:

```bash
python3 dnn.py export-veriloga \
  --model-dir dnn_model \
  --out-dir dnn_model/veriloga_export \
  --dc-mdif training_with_dc.mdif
```

When `--dc-mdif` is supplied, the exporter first validates a compatible saved
DC model directly against every usable exact-zero-Hz training row. A maximum
absolute S error at or below $10^{-4}$ reuses the saved model. Otherwise, only the
DC network is fitted again; RF weights, poles, KBNN coarse models, and
positive-frequency predictions remain unchanged.

For an unrestricted export-only refit, the implementation uses two hidden
layers whose width is at least $\max(64,4P,2L)$, where $P$ is the parameter
count and $L$ is the S-parameter count, up to 8000 epochs, and early-stopping
patience of 800.
This larger budget is used because a full N-port model has $2N^2$ outputs.
The preferred final maximum absolute S error is $10^{-3}$. Exceeding it sets
`dc_mdif_match_within_tolerance: false` and records `dc_mdif_warning`, but does
not stop an unrestricted export because every ordered complex component is
still represented and no topology projection has discarded entries.

For an explicit-path export-only refit, the implementation uses up to 4000
epochs and patience of 400. Error above $10^{-3}$ stops the export because it
indicates that the declared resistor graph cannot reproduce the data.

The `dc_mdif_action` manifest value records whether the exporter used
`validated_saved_dc_model` or `fitted_dc_only_model`.

### A.7 Behavior of each export target

#### Sampled ADS MDIF

Each exported geometry receives an exact-zero-Hz row if its sampling grid does
not already contain one. The DC MLP supplies every S-parameter on that row.
Every positive-frequency row comes only from the RF surrogate. This format is
the most direct way to compare the extracted DC S-matrix numerically with the
source MDIF.

#### ADS harmonic-balance network

The generated SDD equations evaluate the geometry-only DC MLP at the
zero-frequency spectral component. In unrestricted mode, the complete complex
S matrix is converted to Y for the port-current equations. Positive spectral
frequencies use the RF model, and negative-frequency weights use the conjugate
of the corresponding positive-frequency response. The DC and RF models are
therefore part of one instantiated ADS component without sharing fitted data.

#### Verilog-A

The generated module evaluates the DC MLP when the simulator frequency
expression is exactly zero. In unrestricted mode it reconstructs
$\mathbf S_{\mathrm{DC}}$, solves
the complex S-to-Y relation, and selects that Y matrix before stamping port
currents. At positive frequency it selects the RF-derived Y matrix instead.

The current contribution is structurally unconditional:

$$
\mathbf I_{\mathrm{port}}
\mathrel{+}=
\operatorname{Re}(\mathbf Y)\mathbf V_{\mathrm{port}}
+\frac{\operatorname{Im}(\mathbf Y)}{\omega}
\operatorname{ddt}(\mathbf V_{\mathrm{port}}).
$$

Only the coefficient selection is conditional. Keeping `ddt()` outside the
frequency `if` statement avoids the ADS compiler error "Analog operators are
forbidden in this context."

The imaginary-admittance term is an AC/S-parameter representation, not a
static DC conductance. At a true DC operating point `ddt(V)` is zero. If an
exact-zero-Hz dataset contains a materially imaginary DC response, use the
sampled MDIF or ADS HB export for a direct complex-S comparison and treat the
Verilog-A DC-bias result as the real static network behavior.

### A.8 Diagnostics recorded in reports and manifests

The following fields are the primary audit trail for DC extraction:

| Field | Meaning |
| --- | --- |
| `dc_model_representation` | `full_s_matrix` for unrestricted extraction or `path_conductance` for an explicit resistor graph. |
| `dc_sparameter_entries` | Every ordered S-parameter represented by the unrestricted model. |
| `dc_matrix_entries` | Ordered scalar outputs, including `.real` and `.imag`. |
| `dc_port_paths` | Canonical explicit paths. An empty list is expected for unrestricted full-S extraction. |
| `dc_usable_block_positions` | One-based source ACDATA positions used to form DC samples. |
| `dc_missing_block_positions` | Training blocks with no exact-zero-Hz row. |
| `dc_unusable_block_positions` | Blocks with zero-Hz rows but no finite passive row. |
| `dc_ignored_nonpassive_count` | Exact-zero-Hz rows rejected by the singular-value passivity test. |
| `dc_ignored_nonfinite_count` | Exact-zero-Hz rows rejected for non-finite data or an invalid matrix decomposition. |
| `dc_model_train_s_max_abs_error` | Worst S-parameter error on fitted DC-training geometries. |
| `dc_model_verify_s_max_abs_error` | Worst S-parameter error on usable DC-verification geometries. |
| `dc_topology_s_max_abs_error` | Error caused by explicit resistor-graph projection before neural interpolation; zero for unrestricted full-S extraction. |
| `dc_mdif_model_s_max_abs_error` | Direct error against usable exact-DC rows supplied through `--dc-mdif`. |
| `dc_mdif_match_within_tolerance` | Whether export validation met the preferred final tolerance. |
| `dc_mdif_warning` | Non-fatal unrestricted interpolation warning retained in the export manifest. |

For unrestricted extraction, inspect `dc_sparameter_entries` and
`dc_matrix_entries` first. For an explicit resistor model, inspect
`dc_port_paths` and `dc_topology_s_max_abs_error` first. These distinguish
neural interpolation error from a topology that is incapable of representing
the supplied S-matrix.

### A.9 Failure conditions and recommended response

| Failure | Meaning | Recommended response |
| --- | --- | --- |
| Missing exact-zero-Hz training block | A training geometry has RF rows but no true DC row. | Add a `0 Hz` row for that geometry; do not substitute the lowest RF point. |
| No usable passive exact-zero-Hz rows | Every available DC row was non-finite or failed passivity. | Correct the source data or provide a dataset with at least one passive DC geometry. |
| Legacy lossy representation | The saved model predates full-complex-S extraction. | Re-export once with `--dc-mdif` pointing to the original training data. |
| Explicit topology exceeds $10^{-3}$ | The selected resistor graph cannot reproduce the supplied matrix closely enough. | Correct `--dc-port-paths` or omit it to use unrestricted full-S extraction. |
| Unrestricted fit exceeds $10^{-3}$ | The full-S MLP interpolation missed the preferred tolerance, but no matrix entries were discarded. | Review `dc_mdif_warning`, add DC geometries, or retrain with a larger/smoother parameter sampling plan; export is still produced. |
| Incomplete `dc_model.npz` / `dc_model.json` pair | Only part of the saved DC model is present. | Restore both files or regenerate the model/export from the source MDIF. |

## Appendix B: DNN, KBNN, and Neuro-TF Implementation

This appendix documents the implementation in this repository, including the
model equations, sample construction, optimization, persistence, and simulator
translation. The names DNN, KBNN, and Neuro-TF describe broad families of
methods in the literature; the equations below define the precise variants
implemented here. Appendix A separately documents the exact-DC network that is
attached to every family.

### B.1 Shared data model and notation

An MDIF ACDATA block represents one geometry or process point. Let

- $\mathbf p=[p_1,\ldots,p_P]$ be the selected numeric geometry/process `VAR` values;
- $f$ be frequency in Hz;
- $\mathbf S(\mathbf p,f)$ be the complex N-port scattering matrix;
- $L$ be the number of common ordered S-parameter labels ($L=N^2$ for the
  complete matrices required by direct N-port export);
- $\mathbf s(\mathbf p,f)$ be the row-major vector
  $[S_{11},S_{12},\ldots,S_{NN}]$;
- $\mathcal R(\mathbf v)=[\operatorname{Re}(\mathbf v),\operatorname{Im}(\mathbf v)]$
  be the real-column representation of a complex vector, with all real entries
  followed by all imaginary entries.

The parser finds S-parameter labels common to all selected blocks, orders them
by port indices, and infers numeric parameters common to the blocks unless
`--parameter-names` fixes the names and order. Direct N-port exports require a
complete matrix with contiguous port numbers.

The split variable is evaluated per block, not per frequency row. Values named
by `--train-values` and `--verify-values` select the two sets. If no explicit
training values are found, a seeded random block holdout is used. Keeping an
entire geometry in one split prevents different frequencies of the same
geometry from leaking between training and verification.

All three RF implementations retain only $f>0$ when constructing their RF
training arrays. Exact DC is handled only by Appendix A's independent
geometry-only model.

### B.2 Shared neural-network engine

Each learned map uses the same dense multilayer perceptron (MLP). For standardized
input $\mathbf a_0$, hidden layer $\ell$ computes

$$
\begin{aligned}
\mathbf z_\ell
  &=\mathbf a_{\ell-1}\mathbf W_\ell+\mathbf b_\ell,\\
\mathbf a_\ell
  &=\varphi(\mathbf z_\ell).
\end{aligned}
$$

and the final layer is linear. Supported hidden activations are `tanh` and
ReLU. For layer fan-in $n_{\mathrm{in}}$ and fan-out $n_{\mathrm{out}}$,
weights are initialized uniformly over

$$
\left[
-\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}},
+\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}
\right]
$$

and biases start at zero. This is the Glorot/Xavier uniform initialization.

Each input or target column is standardized as

$$
\widetilde x=\frac{x-\mu_{x,\mathrm{train}}}{\sigma_{x,\mathrm{train}}}
$$

using training data only. Predictions are inverse-transformed after the final
linear layer. Near-constant direct-response output columns use the median
standard deviation of the varying columns as a numerical floor; the original
mean is retained.

Training uses shuffled mini-batches and Adam with $\beta_1=0.9$,
$\beta_2=0.999$, and $\epsilon=10^{-8}$. The implementation stores the parameters with the
lowest checked verification loss and restores them after training. When no
verification data exists, checked training loss is used. `--loss-interval`
controls how often the complete datasets are evaluated, and `--patience` is
the number of epochs since the best checked epoch before early stopping.

For DNN and KBNN, the scaled-domain training objective is a weighted mean
squared error. If sample $k$ is at frequency $f_k$, output column $q$ belongs to
S-parameter $\ell(q)$, and $e_{kq}$ is the scaled prediction error, the reported
objective is

$$
\mathcal L
=\frac{1}{KQ}
\sum_{k=1}^{K}\sum_{q=1}^{Q}
w_f(f_k)\,w_s(\ell(q))\,e_{kq}^2.
$$

The same S-parameter weight is applied to the real and imaginary columns.
Raw S-parameter weights are normalized to mean one over S-parameters. Raw
frequency weights are normalized to mean one over RF training rows. This keeps
the average gradient scale roughly unchanged when relative priorities change.
Zero weights are permitted as long as at least one weight remains positive.

Neuro-TF uses frequency weights in the per-geometry rational least-squares
stage described below. Its neural coefficient map then uses unweighted scaled
coefficient MSE; the coefficient targets have already changed in response to
the frequency weighting.

### B.3 DNN: direct response surrogate

#### Architecture

The DNN learns the RF response directly as a function of parameters and
frequency:

$$
\begin{aligned}
\mathbf x(\mathbf p,f)&=[\mathbf p,\boldsymbol\phi(f)],\\
\mathcal R(\widehat{\mathbf r}(\mathbf p,f))
&=\operatorname{inverse\_scale}\!\left(
  \operatorname{MLP}(\operatorname{scale}(\mathbf x(\mathbf p,f)))
  \right).
\end{aligned}
$$

The frequency feature $\boldsymbol\phi(f)$ is selected by `--freq-transform`:

| Transform | Feature columns |
| --- | --- |
| `log` | $[\log_{10}(\max(f,1\,\mathrm{Hz}))]$ |
| `linear` | $[f]$ |
| `log-linear` | $[\log_{10}(\max(f,1\,\mathrm{Hz})),f]$ |

Because RF construction has already removed zero-Hz rows, the 1-Hz clamp is a
numerical guard and does not create a fitted DC point.

Each frequency row is one neural-training sample. A block containing $F$
positive frequencies produces $F$ samples with the same parameter values and
different frequency features. For an N-port S-domain model, the output layer
has $2N^2$ values.

#### S-domain and Y-domain targets

With the default `--output-domain s`, the target is
$\mathcal R(\mathbf s(\mathbf p,f))$. With
`--output-domain y`, each complete S-matrix is converted before training:

$$
\mathbf Y
=\frac{1}{Z_0}(\mathbf I-\mathbf S)(\mathbf I+\mathbf S)^{-1}
$$

and the target is the ordered real/imaginary Y vector. Prediction converts Y
back to S with

$$
\mathbf S
=(\mathbf I-Z_0\mathbf Y)(\mathbf I+Z_0\mathbf Y)^{-1}.
$$

The implementation solves the transposed matrix relation with `numpy.linalg.solve`
and falls back to a pseudoinverse when the system is singular. A Y-domain
model fixes its reference impedance through `--target-z0`; export must use the
same value.

#### Consequences

The DNN has no rational frequency structure and no coarse prior. It can model
arbitrary smooth response shapes represented by the data, but frequency and
geometry interpolation are learned simultaneously. It therefore commonly
needs more full-wave geometries than KBNN when a useful coarse model exists,
and it offers less structural frequency regularization than Neuro-TF. No RF
passivity or reciprocity constraint is embedded in the loss; these properties
are measured on predictions and may be required during sweep selection.

#### Persistence and inference

`model.npz` stores input/output means and standard deviations plus every
$\mathbf W_\ell$ and $\mathbf b_\ell$. `metadata.json` stores layer sizes, activation, parameter and
S-parameter ordering, frequency transform, output domain, reference impedance,
training configuration, and metrics. Prediction reconstructs the exact saved
MLP and scalers; it does not retrain.

### B.4 KBNN: fitted-coarse knowledge-based surrogate

#### Coarse model contract

This repository's KBNN uses a **frozen S-domain DNN** as prior knowledge. It
does not feed raw coarse MDIF samples directly into fine fitting. When
`--coarse-mdif` is provided, the command first trains a standalone coarse DNN,
saves it under the KBNN output, reloads it, and evaluates it on the fine-model
geometry/frequency grids. `--coarse-model-dir` reuses an already fitted DNN.

The frozen coarse model must have the same parameter names and order, complete
S-parameter labels and order, and S-domain output as the KBNN. Its model and
metadata SHA-256 hashes are recorded. Prediction and self-contained export
verify those hashes, preventing an accidentally different coarse model from
being substituted after KBNN fitting.

This design makes the KBNN optimize against the same fitted coarse
response that will be embedded in the final Verilog-A or ADS HB component. It
avoids training against exact coarse samples and later deploying a separately
fitted approximation of those samples.

#### Alignment

During normal KBNN fitting, the frozen coarse DNN is evaluated directly at each
fine geometry and each fine positive frequency. No raw coarse-MDIF frequency
interpolation occurs in the fine-model optimizer. This is the
deployment-matched path used by self-contained Verilog-A and ADS HB export.

The separate native ADS ANN handoff can instead accept coarse MDIF response
blocks. In that path, fine and coarse blocks are matched using the ordered
parameter tuple rounded to 15 decimal places, and coarse responses are linearly
interpolated onto the fine positive-frequency grid. If the lists already match
one-to-one in parameter order, that fast path is used. Missing geometry matches
are errors.

#### Implemented modes

Let $\mathbf c(\mathbf p,f)$ be the complex response predicted by the frozen
coarse DNN, and let $n(\cdot)$ denote the KBNN MLP after scaling and inverse
scaling.

| Mode | MLP input | MLP target | Final prediction |
| --- | --- | --- | --- |
| `plain` | $[\mathbf p,\boldsymbol\phi(f)]$ | $\mathcal R(\mathbf s_{\mathrm{fine}})$ | $n(\mathbf p,f)$ |
| `residual`, coarse input off | $[\mathbf p,\boldsymbol\phi(f)]$ | $\mathcal R(\mathbf s_{\mathrm{fine}}-\mathbf c)$ | $\mathbf c+n(\mathbf p,f)$ |
| `residual`, coarse input on | $[\mathbf p,\boldsymbol\phi(f),\mathcal R(\mathbf c)]$ | $\mathcal R(\mathbf s_{\mathrm{fine}}-\mathbf c)$ | $\mathbf c+n(\mathbf p,f,\mathbf c)$ |
| `prior-input` | $[\mathbf p,\boldsymbol\phi(f),\mathcal R(\mathbf c)]$ | $\mathcal R(\mathbf s_{\mathrm{fine}})$ | $n(\mathbf p,f,\mathbf c)$ |

`prior-input` always enables coarse-response input. `plain` always disables
it. Residual mode can optionally include the coarse response as input in
addition to adding it at the output.

The residual target is computed from the **fitted coarse DNN prediction**:

$$
\boldsymbol\Delta_{\mathrm{target}}(\mathbf p,f)
=\mathbf s_{\mathrm{fine}}(\mathbf p,f)
-\mathbf c_{\mathrm{fitted}}(\mathbf p,f).
$$

not from the original coarse MDIF value. Thus deployment evaluates

$$
\widehat{\mathbf s}(\mathbf p,f)
=\mathbf c_{\mathrm{fitted}}(\mathbf p,f)
+\widehat{\boldsymbol\Delta}(\mathbf p,f).
$$

using exactly the same coarse-model definition used to create the training
target.

The KBNN output scaler is fitted to either fine S or delta S according to the
mode. Real and imaginary values remain separate scalar outputs. Frequency and
S-parameter weights are applied to the KBNN objective. When a coarse DNN is
trained by the same command, it inherits those weights unless corresponding
`--coarse-*` options override them.

#### Coarse versus fine DC

The frozen coarse DNN is evaluated only on positive-frequency fine grids during
KBNN RF fitting. The KBNN's distinct DC model is extracted solely from exact
zero-Hz rows in the fine-data MDIF. At zero Hz, the KBNN and coarse RF DNN are
bypassed; Appendix A's fine-data DC model supplies the response.

#### Persistence and self-contained export

The fine KBNN MLP is saved in the standard `model.npz` and `metadata.json`
files. A KBNN created from `--coarse-mdif` also packages the fitted coarse DNN
and records its relative location and identity hashes. Direct Verilog-A and ADS
HB export embed both evaluators when the mode requires coarse knowledge:

1. evaluate the frozen coarse DNN from instance parameters and frequency;
2. feed its complex response into the KBNN when configured;
3. evaluate the fine or residual KBNN;
4. add the coarse response for residual mode;
5. convert the final complete S-matrix to Y and stamp the ports.

Legacy editable coarse hooks are available only when explicitly requested and
are not equivalent to the normal self-contained path.

#### Relation to published KBNN methods

Published KBNN and neuro-space-mapping work incorporates empirical,
semi-analytical, equivalent-circuit, or other coarse knowledge into a neural
model. This repository implements a pragmatic response-domain specialization:
a fitted coarse S-parameter DNN is frozen and used as a residual baseline
and/or prior input. It does not implement every internal knowledge neuron or
input/output space-mapping topology described in the literature.

### B.5 Neuro-TF: fixed-pole rational coefficient surrogate

#### Two-stage construction

Neuro-TF separates frequency representation from geometry interpolation:

1. fit one rational response at every training geometry using a common fixed
   pole set;
2. train an MLP from geometry parameters to the real and imaginary rational
   coefficients.

This differs from the DNN, which treats every $(\mathbf p,f)$ row as a neural sample.
A Neuro-TF geometry contributes one neural sample regardless of its frequency
count.

#### Frequency normalization and fixed poles

From all positive training frequencies, the implementation computes

$$
\begin{aligned}
f_{\mathrm{scale}}&=\sqrt{f_{\min}f_{\max}},\\
x_{\min}&=\frac{f_{\min}}{f_{\mathrm{scale}}},\\
x_{\max}&=\frac{f_{\max}}{f_{\mathrm{scale}}},\\
s&=j\frac{f}{f_{\mathrm{scale}}}.
\end{aligned}
$$

For $K$ requested poles, $\lfloor K/2\rfloor$ logarithmically spaced center
frequencies cover $0.75x_{\min}$ through $1.25x_{\max}$. Each center $\omega_k$ produces a
conjugate pair

$$
p_{k,\pm}=-d\,\omega_k\pm j\omega_k,
$$

where $d$ is the configured pole damping. If $K$ is odd, the last pole is the
real negative pole

$$
p_{\mathrm{last}}=-\sqrt{x_{\min}x_{\max}}.
$$

Every pole has a negative real part for a positive damping value, giving a
stable fixed basis. Poles are shared by all geometries and all S-parameters;
the MLP predicts coefficients, not pole locations.

#### Per-geometry coefficient extraction

The rational basis is

$$
\mathbf B(f)=
\left[1,\frac{1}{s-p_1},\ldots,\frac{1}{s-p_K}\right]
$$

and each ordered S-parameter is represented as

$$
S_{ij}(\mathbf p,f)
=c_{ij,0}(\mathbf p)
+\sum_{k=1}^{K}
\frac{c_{ij,k}(\mathbf p)}{j f/f_{\mathrm{scale}}-p_k}.
$$

For each geometry and S-parameter, coefficients solve the complex weighted
ridge least-squares problem

$$
\widehat{\mathbf c}
=\underset{\mathbf c}{\operatorname{arg\,min}}
\left\|
\mathbf W_f^{1/2}(\mathbf B\mathbf c-\mathbf s_{\mathrm{sample}})
\right\|_2^2
+\lambda\|\mathbf c\|_2^2.
$$

using `numpy.linalg.lstsq`. The regularization is implemented by appending
$\sqrt{\lambda}\mathbf I$ and a zero right-hand side. `--frequency-weights` therefore
changes the extracted coefficient targets themselves. Coefficient extraction
uses positive frequencies only.

Each geometry's coefficient matrices are flattened by S-parameter, with the
constant coefficient followed by its pole coefficients. All real coefficients
are concatenated before all imaginary coefficients. For $L$ ordered
S-parameters and $K$ poles, the neural target dimension is $2L(K+1)$.

#### Geometry-to-coefficient MLP and evaluation

The neural map is

$$
\mathcal R(\mathbf C(\mathbf p))
=\operatorname{inverse\_scale}\!\left(
\operatorname{MLP}(\operatorname{scale}(\mathbf p))
\right).
$$

Frequency is not a neural input. At prediction, the coefficient row is
unflattened and multiplied by the rational basis evaluated at each requested
frequency. This makes evaluation on a new frequency grid inexpensive and keeps
the geometry-to-network map lower dimensional than direct row-wise regression
when many frequency samples are present.

#### Important implementation boundaries

- The pole set is generated once from the training band; this implementation
  does not relocate poles with vector fitting.
- Pole tracking across geometries is unnecessary because poles are fixed.
- The common pole basis provides stable denominators, but arbitrary learned
  complex coefficients do not by themselves guarantee passivity, reciprocity,
  real-time conjugate symmetry, or a minimal realization.
- Accuracy is determined jointly by rational order, pole damping, frequency
  weighting, ridge regularization, geometry coverage, and MLP capacity.
- Direct export evaluates the fixed-pole equation; it does not sample and
  interpolate a hidden lookup table.

#### Persistence and export

In addition to the normal MLP arrays, `model.npz` stores the real and imaginary
parts of every pole and $f_{\mathrm{scale}}$. `metadata.json` stores the pole count,
coefficient count per S-parameter, damping, ridge value, and fit configuration.
Verilog-A and ADS HB exports evaluate the coefficient MLP, reconstruct the
rational S-matrix at simulator frequency, convert it to Y, and stamp the same
N-port relation used by the other families.

This repository's method is best described as a **fixed-pole pole-residue
Neuro-TF variant**. It is inspired by combined neural-network/transfer-function
modeling, but it is not the pole-relocation, pole-tracking, hybrid
pole-residue/rational, or sensitivity-assisted algorithm from any one cited
paper.

### B.6 Prediction, validation, and model selection

All family predictions are converted to complex S-parameters before common
verification metrics are calculated. Per-block and aggregate outputs include
complex-magnitude RMSE, mean and maximum absolute error, EVM, and magnitude-dB
error where both magnitudes exceed the numerical floor. Weighted metrics
combine normalized S-parameter and frequency weights.

Passivity diagnostics reconstruct the complete S-matrix at each evaluated
point and compute its largest singular value. A point is counted as violating
when that value exceeds $1+10^{-6}$. This is a diagnostic and sweep-selection
constraint, not a projection or enforcement step. `--require-passive`,
`--max-passivity-violations`, and `--max-passivity-sigma` control whether a
sweep trial is eligible for promotion.

Sweep/optimize candidates vary the family-specific and shared hyperparameters,
train each candidate, and rank completed trials using the selected verification
metric. The chosen model is a saved trained trial unless `--retrain-best`
requests another fit with the winning configuration.

### B.7 Direct circuit implementation

The direct Verilog-A and ADS HB generators do not invoke Python during ADS
simulation. They serialize numeric scalers, weights, biases, and—where
applicable—coarse-model data or rational poles into simulator equations.

For an S-domain RF prediction, all model families ultimately use

$$
\begin{aligned}
\mathbf Y(f)
  &=\frac{1}{Z_0}
    (\mathbf I-\mathbf S(f))(\mathbf I+\mathbf S(f))^{-1},\\
\mathbf I_{\mathrm{port}}(f)
  &=\mathbf Y(f)\mathbf V_{\mathrm{port}}(f).
\end{aligned}
$$

The generated Verilog-A performs a complex Gauss-Jordan solve with partial
pivot selection and a small pivot floor. The ADS HB package generates explicit
SDD frequency-domain current equations. Negative-frequency HB weights are the
complex conjugate of their matching positive-frequency response, as required
for real-valued time-domain port voltages and currents.

These are linear parameterized N-port models: parameters change the network,
but RF response is not a function of incident power. They can participate in a
harmonic-balance circuit because the simulator evaluates the linear network at
each spectral frequency; the models do not create compression or new
harmonics.

Parameter input scaling is applied before the saved training standardization:

$$
\begin{aligned}
p_{\mathrm{model}}
  &=\frac{p_{\mathrm{ADS\ instance}}}{p_{\mathrm{input\ scale}}},\\
\widetilde p
  &=\frac{p_{\mathrm{model}}-\mu_{p,\mathrm{saved}}}
          {\sigma_{p,\mathrm{saved}}}.
\end{aligned}
$$

Thus scaling changes the ADS unit convention without changing the fitted model.

### B.8 Model-family comparison

| Property | DNN | KBNN | Neuro-TF |
| --- | --- | --- | --- |
| Neural input | Parameters plus frequency features | Parameters, frequency features, and optionally fitted coarse S | Parameters only |
| Neural target | Complex S or Y at each RF row | Fine complex S or fine-minus-fitted-coarse complex S | Complex rational coefficients per geometry |
| RF neural samples per geometry | Number of positive-frequency rows | Number of positive-frequency rows | One |
| Frequency structure | Learned directly | Learned directly around coarse knowledge | Fixed stable pole basis |
| Prior model | None | Frozen fitted S-domain DNN | Fixed rational basis |
| Typical strength | Maximum response flexibility | Efficient correction when useful coarse data exists | Compact broadband frequency representation |
| Main risk | Data demand and unconstrained frequency interpolation | Bias or error inherited from coarse model and mode choice | Insufficient pole order/basis placement or nonsmooth coefficient map |
| RF passivity enforcement | None | None | None |
| Exact DC | Separate Appendix A model | Separate fine-data Appendix A model | Separate Appendix A model |

### B.9 References and implementation provenance

The references below motivate the model families or specific numerical
building blocks. They are not claims that this repository exactly reproduces
every algorithm in those publications.

1. F. Wang and Q.-J. Zhang, “Knowledge-based neural models for microwave
   design,” *IEEE Transactions on Microwave Theory and Techniques*, vol. 45,
   no. 12, pp. 2333–2343, 1997.
   [doi:10.1109/22.643839](https://doi.org/10.1109/22.643839). Foundational
   microwave KBNN motivation: incorporate empirical or analytical prior
   knowledge into a neural model.
2. H. Kabir, Q.-J. Zhang, M. Yu, L. Zhang, P. H. Aaen, and J. Wood, “Smart
   modeling of microwave devices,” *IEEE Microwave Magazine*, vol. 11, no. 3,
   pp. 105–118, 2010.
   [doi:10.1109/MMM.2010.936079](https://doi.org/10.1109/MMM.2010.936079).
   Overview of neural, knowledge-based, and neuro-space-mapping approaches for
   microwave modeling.
3. Y. Cao, G. Wang, and Q.-J. Zhang, “A new training approach for parametric
   modeling of microwave passive components using combined neural networks and
   transfer functions,” *IEEE Transactions on Microwave Theory and Techniques*,
   vol. 57, no. 11, pp. 2727–2742, 2009.
   [doi:10.1109/TMTT.2009.2032476](https://doi.org/10.1109/TMTT.2009.2032476).
   Establishes the geometry-to-transfer-function-coefficient modeling concept.
4. F. Feng, C. Zhang, J. Ma, and Q.-J. Zhang, “Parametric modeling of EM
   behavior of microwave components using combined neural networks and
   pole-residue-based transfer functions,” *IEEE Transactions on Microwave
   Theory and Techniques*, vol. 64, no. 1, pp. 60–77, 2016.
   [doi:10.1109/TMTT.2015.2504099](https://doi.org/10.1109/TMTT.2015.2504099).
   Pole-residue Neuro-TF background and the motivation for smooth parametric
   coefficient representations.
5. B. Gustavsen and A. Semlyen, “Rational approximation of frequency domain
   responses by vector fitting,” *IEEE Transactions on Power Delivery*, vol.
   14, no. 3, pp. 1052–1061, 1999.
   [doi:10.1109/61.772353](https://doi.org/10.1109/61.772353). General rational
   frequency-response fitting background. This repository uses fixed poles and
   linear ridge least squares rather than vector-fitting pole relocation.
6. X. Glorot and Y. Bengio, “Understanding the difficulty of training deep
   feedforward neural networks,” *Proceedings of AISTATS*, PMLR 9, pp. 249–256,
   2010. [PMLR paper](https://proceedings.mlr.press/v9/glorot10a.html).
   Source for the uniform fan-in/fan-out initialization used by the MLP.
7. D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
   *International Conference on Learning Representations*, 2015.
   [arXiv:1412.6980](https://arxiv.org/abs/1412.6980). Optimizer implemented by
   the shared MLP engine.
8. Keysight Technologies, *Using Circuit Simulators*, ADS 2011 documentation,
   sections on MDIF/S2PMDIF `VAR` declarations and ACDATA blocks.
   [Keysight PDF](https://edadownload.software.keysight.com/eedl/ads/2011_01/pdf/cktsim.pdf).
   Format reference for multidimensional MDIF small-signal data.
9. Keysight Technologies, *User-Defined Models*, ADS documentation, sections
   on SDD equations and harmonic-balance frequency-domain evaluation.
   [Keysight PDF](https://edadownload.software.keysight.com/eedl/ads/2009u1/pdf/modbuild.pdf).
   Background for the generated linear ADS HB SDD packages.
10. Keysight Technologies, *Guide to Harmonic Balance Simulation in ADS*.
    [Keysight PDF](https://edadownload.software.keysight.com/eedl/ads/2011/pdf/adshbapp.pdf).
    Harmonic-balance frequency-domain simulation background.

The native ADS ANN export additionally follows the installed ADS 2026 Update
2.1 examples and API pages listed in each generated `ADS_ANN_README.md`, most
notably `doc/ann/examples/inmemory_extraction.py`. Those installed product files
are the version-specific reference for `keysight.ads.ann`; the local exporter
records their paths in `ads_ann_manifest.json`.

### B.10 Normative source-code map

The documentation above is explanatory; the following repository files are the
normative implementation:

| Area | Source |
| --- | --- |
| DNN features, S/Y targets, training, persistence, and commands | [`dnn.py`](dnn.py) |
| KBNN coarse fitting, identity checks, modes, targets, and composite export | [`kbnn.py`](kbnn.py) |
| Fixed-pole construction, rational coefficient extraction, and Neuro-TF evaluation | [`neuro_tf.py`](neuro_tf.py) |
| MDIF parsing, splitting, weighting, MLP/Adam, metrics, exact DC, and simulator generators | [`surrogate_common.py`](surrogate_common.py) |
| DC and export regression coverage | [`tests/test_dc_conductance_model.py`](tests/test_dc_conductance_model.py) and [`tests/test_ads_hb_export.py`](tests/test_ads_hb_export.py) |

## Appendix C: Gaussian-Process Adaptive Point Selection

This appendix documents the exact GP-assisted point-selection implementation
in `generate_points.py`. The GP is not the exported RF surrogate and does not
predict S-parameters. It is a small auxiliary model of **geometry-level
surrogate error** used only to choose the next expensive EM geometries.

### C.1 End-to-end data flow

One adaptive round follows this sequence:

1. Train or load a DNN, KBNN, or Neuro-TF using the currently simulated MDIF.
2. Evaluate that surrogate on verification geometries and produce
   `verification_metrics.csv`.
3. Aggregate the selected metric into one non-negative error value per
   geometry.
4. Normalize every geometry variable to the unit hypercube.
5. Fit the GP to the logarithm of geometry error.
6. Generate a finite maximin-LHS candidate pool.
7. Score candidates using an upper confidence bound and a diversity factor.
8. Select a batch, write its physical parameter values to CSV, and simulate
   that batch externally.
9. Append the resulting MDIF blocks, refit the RF surrogate, and repeat.

The GP is rebuilt on every invocation. It is not serialized as part of the DNN,
KBNN, Neuro-TF, Verilog-A, ADS ANN, or ADS HB model.

### C.2 Geometry normalization

Let geometry parameter $p_j$ have lower and upper bounds $a_j$ and $b_j$. A
linear parameter is mapped to

$$
u_j=\frac{p_j-a_j}{b_j-a_j}.
$$

A log-scaled parameter is mapped to

$$
u_j=
\frac{\log p_j-\log a_j}{\log b_j-\log a_j}.
$$

Thus every observation and candidate is represented by
$\mathbf u\in[0,1]^d$. Distances, the GP length scale, `--focus-radius`, and
`--min-distance` all operate in this normalized geometry space. The resolved
parameter bounds—from the companion geometry JSON, an explicit
`--parameter-json`, or explicit `--parameter` overrides—must therefore describe
the complete domain being scored.

### C.3 Geometry-level error target

The selected `--metric` is read from `verification_metrics.csv`. For each
geometry, the implementation combines its S-parameter rows using

$$
e(\mathbf u)=
\sqrt{
\frac{\sum_q w_q e_q^2}
     {\sum_q w_q}
},
$$

where $q$ indexes the available metric rows and $w_q$ is
`normalized_sparam_weight`, then `sparam_weight`, or $1$ when neither usable
weight is present. Most metrics are converted to absolute values. An `evm_db`
metric is first converted to a linear amplitude ratio.

The GP target is

$$
y(\mathbf u)=\log\!\left(\max(e(\mathbf u),\epsilon)\right),
$$

where $\epsilon$ is `--gp-error-floor`. The log transform makes multiplicative
changes in error easier to represent and prevents a few large errors from
dominating the covariance calculation. Duplicate normalized geometries are
collapsed, retaining the largest error. At least two distinct geometries are
required.

The target is standardized before GP fitting:

$$
z_i=\frac{y_i-\overline y}{s_y},
\qquad
s_y=\max\!\left(\operatorname{std}(y),0.25\right).
$$

The $0.25$ floor retains useful posterior uncertainty when the first observed
errors are nearly equal; it corresponds to approximately a 28% multiplicative
one-standard-deviation interval in the original error domain.

### C.4 Matérn-5/2 covariance and fitting

The implementation uses one isotropic length scale $\ell$ for all normalized
geometry dimensions. Define

$$
r(\mathbf u,\mathbf u')=
\sqrt{\sum_{j=1}^{d}
\left(\frac{u_j-u'_j}{\ell}\right)^2}.
$$

The unit-amplitude Matérn-5/2 covariance is

$$
k(\mathbf u,\mathbf u')=
\left(1+\sqrt{5}r+\frac{5}{3}r^2\right)
\exp(-\sqrt{5}r).
$$

For $n$ observed geometries, the covariance matrix is

$$
K_{ij}=k(\mathbf u_i,\mathbf u_j)+\eta\,\delta_{ij},
$$

where $\eta$ is `--gp-noise-variance`. It acts as a normalized covariance
nugget for noisy error observations and numerical stability. The system is
factored using a dense Cholesky decomposition; progressively larger diagonal
jitter is added only if needed to stabilize that factorization.

When `--gp-length-scale` is omitted, the code tests

$$
\ell\in
\{0.08,0.12,0.18,0.27,0.40,0.60,0.90,1.35\}
$$

and chooses the value with the largest log marginal likelihood

$$
\log p(\mathbf z\mid\ell)=
-\frac{1}{2}\mathbf z^{\mathsf T}K^{-1}\mathbf z
-\sum_{i=1}^{n}\log L_{ii}
-\frac{n}{2}\log(2\pi),
$$

where $K=LL^{\mathsf T}$. This is a small deterministic grid search, not a
continuous optimizer and not automatic relevance determination. Supplying
`--gp-length-scale` bypasses this search.

### C.5 Posterior prediction

For a candidate $\mathbf u_*$, let $\mathbf k_*$ contain its covariance with
each observed geometry and let
$\boldsymbol\alpha=K^{-1}\mathbf z$. The normalized posterior is

$$
\begin{aligned}
m_* &= \mathbf k_*^{\mathsf T}\boldsymbol\alpha,\\
v_* &= \max\!\left(
0,
1-\left\|L^{-1}\mathbf k_*\right\|_2^2
\right).
\end{aligned}
$$

The posterior log-error mean and standard deviation are

$$
\mu_{\log e}=\overline y+s_y m_*,
\qquad
\sigma_{\log e}=s_y\sqrt{v_*}.
$$

The reported `predicted_error` is
$\exp(\mu_{\log e})$. The reported `gp_log_uncertainty` is
$\sigma_{\log e}$, so it is dimensionless and multiplicative in the original
error domain rather than an additive error bar.

### C.6 Acquisition and batch diversity

The GP upper-confidence error is

$$
U(\mathbf u)=
\exp\!\left(
\mu_{\log e}(\mathbf u)+
\kappa\sigma_{\log e}(\mathbf u)
\right),
$$

where $\kappa$ is `--exploration-weight`. Small $\kappa$ emphasizes predicted
high-error regions; large $\kappa$ emphasizes uncertain regions.

Let $\mathcal O$ contain all observed points and points already chosen for the
current batch. The normalized diversity factor is

$$
D(\mathbf u)=
\min\!\left(
1,
\frac{\min_{\mathbf v\in\mathcal O}
\|\mathbf u-\mathbf v\|_2}{\sqrt d}
\right).
$$

The implemented acquisition score is

$$
A(\mathbf u)=U(\mathbf u)D(\mathbf u)^{\nu},
$$

where $\nu$ is `--novelty-power`. Candidates closer than `--min-distance` to
any occupied point are rejected before scoring. After selecting the best
candidate, that point is added to $\mathcal O$ and all remaining candidates
are rescored. This sequential diversity penalty keeps one batch from
collapsing around a single peak.

The GP posterior itself is not fantasy-updated after each point in the batch;
only the diversity term changes. Consequently, this is a practical batch
GP-UCB-inspired acquisition rule, not an implementation of a joint batch
Bayesian-optimization proof or its convergence guarantees.

### C.7 Candidate generation

Both initial point generation and adaptive candidate generation default to
`maximin-lhs`. The accepted `minimax-lhs` spelling is an alias for the same
maximin criterion. In each Latin-hypercube candidate design, every dimension is
divided into equally populated strata. The script creates `--lhs-candidates`
such designs and retains the design with the largest minimum pairwise distance:

$$
X^*=\underset{X}{\operatorname{arg\,max}}
\;\min_{\mathbf u_i\ne\mathbf u_j\in X}
\|\mathbf u_i-\mathbf u_j\|_2.
$$

For adaptive selection, the pool size is `--candidate-count`, or

$$
\max(1000,\;200\,n_{\mathrm{requested}})
$$

by default. GP-UCB scores this finite pool; it does not continuously optimize
the acquisition function. Increasing the pool improves search resolution but
also increases point-generation and posterior-evaluation time.

### C.8 Range extension behavior

`generate --extend-range` and `suggest-additional --acquisition gp-ucb` serve
different purposes:

- `generate --extend-range` samples only the new one-sided slab and appends
  those points to the original geometry CSV. It provides guaranteed initial
  boundary coverage without using model error.
- `suggest-additional` scores the full domain loaded from the expanded
  geometry's JSON (or explicitly overridden). When those bounds include an
  extension, GP uncertainty and the diversity term can favor the new region,
  but the selector may legitimately choose an old-region point with a larger
  acquisition score.

For that reason, the recommended range workflow is to seed the new slab first,
simulate it, refit the surrogate, and then compare legacy and GP refinement
batches from that identical expanded-domain fit. This also gives the GP actual
error observations inside the new region instead of asking it to rely entirely
on extrapolated uncertainty.

### C.9 Diagnostics and tuning

The suggested-point CSV contains:

| Column | Meaning |
| --- | --- |
| `acquisition_score` | Final GP upper-confidence error multiplied by the diversity penalty. |
| `distance_to_existing` | Raw Euclidean distance to the nearest occupied normalized point at selection time. |
| `fit_error_score` | Error at the nearest observed verification geometry; this is contextual and is not the GP prediction. |
| `gp_log_uncertainty` | Posterior standard deviation in natural-log-error space. |
| `gp_upper_confidence_error` | $\exp(\mu_{\log e}+\kappa\sigma_{\log e})$ before the diversity penalty. |
| `predicted_error` | Posterior median-scale error, $\exp(\mu_{\log e})$. |

The companion JSON records the kernel, target transform, observation count,
chosen length scale, selection mode, nugget, exploration weight, and log
marginal likelihood. The companion `*_fit_error_regions.csv` ranks the source
error geometries used by both acquisition modes.

Practical tuning guidance:

| Observation | Adjustment |
| --- | --- |
| Most points cluster near one measured error peak | Increase `--novelty-power`, `--min-distance`, or `--exploration-weight`. |
| Points are too exploratory and ignore known bad regions | Decrease `--exploration-weight`; optionally decrease `--novelty-power`. |
| Selected points change excessively between fits with similar errors | Increase `--gp-noise-variance`. |
| The error surface is known to change over short parameter distances | Supply a smaller `--gp-length-scale`. |
| The error surface should be broad and smooth | Supply a larger `--gp-length-scale`. |
| Too few requested points survive distance filtering | Lower `--min-distance` or increase `--candidate-count`. |

Use error measured before retraining on a newly acquired point when practical.
A training residual after the point has already been absorbed by the network
systematically understates how valuable that point was. Keep the provisional
model family, architecture, frequency transform, output domain, loss weights,
and random-seed policy fixed while comparing acquisition rounds. Reserve a
final audit set that never enters GP acquisition.

### C.10 Limitations and computational cost

- The auxiliary GP models one scalar aggregate error, not individual
  S-parameter errors or the underlying complex response.
- One isotropic length scale assumes comparable smoothness across normalized
  dimensions. Strongly anisotropic problems may need more observations or a
  future per-dimension length-scale implementation.
- GP quality depends on the provisional RF surrogate. A persistent error caused
  by insufficient neural capacity may attract additional EM points without
  resolving the architecture limitation.
- Sparse initial observations can make uncertainty dominate. The command warns
  when there are fewer distinct error observations than $d+1$.
- Dense GP fitting costs $O(n^3)$ in the number of observed error geometries,
  and posterior evaluation uses dense triangular solves for every candidate.
  The implementation is intended for expensive-EM campaigns with modest
  geometry counts, not millions of observations.
- The method does not automatically run ADS/EM simulation, merge MDIF blocks,
  retrain a surrogate, or decide convergence. Those remain explicit steps so
  each new expensive simulation batch can be inspected.

### C.11 References and normative source map

1. C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine
   Learning*, MIT Press, 2006, Chapters 2 and 4.
   [Official open-access text](https://gaussianprocess.org/gpml/chapters/).
   Source for GP regression, marginal likelihood, and Matérn covariance
   background.
2. N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger, “Gaussian process
   optimization in the bandit setting: No regret and experimental design,”
   *Proceedings of ICML*, 2010.
   [arXiv:0912.3995](https://arxiv.org/abs/0912.3995). Source for the GP-UCB
   exploration/exploitation principle. The repository's finite-candidate,
   diversity-penalized batch formulation is an engineering adaptation.
3. M. D. Morris and T. J. Mitchell, “Exploratory designs for computational
   experiments,” *Journal of Statistical Planning and Inference*, vol. 43,
   no. 3, pp. 381–402, 1995.
   [doi:10.1016/0378-3758(94)00035-T](https://doi.org/10.1016/0378-3758(94)00035-T).
   Background for maximin distance designs within Latin hypercubes.

| Area | Source |
| --- | --- |
| Geometry parsing, normalization, Latin hypercubes, GP fitting, acquisition, CSV/JSON output, and CLI | [`generate_points.py`](generate_points.py) |
| GP, alias, default-method, and legacy-compatibility regression tests | [`tests/test_generate_points_gp.py`](tests/test_generate_points_gp.py) |
