"""
One module per separation baseline (baseline1.py .. baseline5.py) --
split out of what used to be one large baselines.py. See
code_description.md for each baseline's full writeup. Baseline 0
(raw mixture, no separation) and the SSA-paper-style synthetic-set /
report-generation glue that ties all of them together still live in
src/baselines.py; Baseline 6 (Conv-TasNet-lite) lives in src/convtasnet.py
-- a different kind of module (trains a model) rather than a fixed
separate_fn/fit_fn pair.

No re-exports here on purpose: import each baseline directly from its own
module (`from baseline.baseline1 import fit_bandpass_baseline`), not
through this package's __init__.
"""
