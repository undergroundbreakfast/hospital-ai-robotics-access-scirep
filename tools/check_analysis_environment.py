#!/usr/bin/env python3
"""Import the analysis stack and run a synthetic cross-fit smoke check. No database."""
from pathlib import Path
import importlib
import logging
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
for name in ["replicate_scirep_outcomes", "geospatial_access_lorenz", "generate_moderation_plots", "econml.dml", "pygam", "plotly"]:
    importlib.import_module(name)
    print(f"Import OK: {name}", flush=True)
import numpy as np
import pandas as pd
from replicate_scirep_outcomes import crossfit_aipw_point_estimate
rng=np.random.default_rng(20260905)
n=200
x=rng.normal(size=n)
t=np.tile([0,1],n//2)
y=2*t+x+rng.normal(scale=0.2,size=n)
result=crossfit_aipw_point_estimate(pd.DataFrame({"t":t,"y":y,"x":x}), "t", "y", ["x"], logging.getLogger("synthetic"), n_splits=2, rf_n_jobs=1)
assert not result["error"], result
assert result["n_treated"]+result["n_control"]==n
assert np.isfinite(result["ate"])
assert result["ci_lower"] < result["ate"] < result["ci_upper"]
assert abs(result["ate"]-2)<0.5, result
print("Synthetic cross-fit smoke check passed (not a study-data rerun).")
