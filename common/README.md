# Common Support Layer

`surrogate_common.py` is the shared infrastructure layer for rc2. It contains
the pieces that are not specific to one fitting formulation:

- MDIF parsing/writing and block splitting
- S-parameter label handling, weighting, S-to-Y conversion, and passivity checks
- Verification metrics, EVM, training/sweep Markdown summaries, and CSV helpers
- Worst-case S/Y plotting and sweep diagnostic plotting
- ADS MDIF, ADS ANN, and Verilog-A export package helpers
- Shared MLP/standardizer utilities, sweep orchestration, and rerank helpers

The model directories import this module and keep their own fitting
calculations in `model.py`.
