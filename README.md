# RC2 Refactor

This directory contains the refactored release-candidate layout.

Shared infrastructure lives in `common/`:

- `common/surrogate_common.py`: MDIF input/output, ADS and Verilog-A export
  helpers, metrics, summaries, plotting, passivity analysis, and shared sweep
  utilities.
- [MODEL_PLUGIN_API.md](MODEL_PLUGIN_API.md): developer API for adding another
  model extraction plugin.

Model-specific calculations and command wiring live in separate modules:

- `dnn/model.py`
- `kbnn/model.py`
- `neuro_tf/model.py`

The user-facing entry points remain thin scripts with the same command style:

```bash
python3 dnn/dnn.py train --mdif train_verify.mdif --out-dir dnn_model
python3 kbnn/kbnn.py train --mdif fine.mdif --coarse-mdif coarse.mdif --out-dir kbnn_model
python3 neuro_tf/neuro_tf.py train --mdif train_verify.mdif --out-dir neuro_tf_model
```

The top-level `outputs/dnn`, `outputs/kbnn`, and `outputs/neuro_tf` folders are
left in place so existing scripts keep working while rc2 is evaluated.
