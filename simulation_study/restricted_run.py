"""
restricted_run.py - order-restricted part: ORMLE and OR-BE, on the SAME datasets as unrestricted_run.py.
Merges both parts and writes the final four-estimator tables (MLE, ORMLE, BE, OR-BE).
Run:
    python restricted_run.py        # full run
    python restricted_run.py 10     # quick test with 10 replications

Results go to results/restricted/ :
    summary_all.csv       bias, AB, var, MSE, AL, CP for MLE, ORMLE, BE, OR-BE (reps where every D_i >= 1)
    summary_allreps.csv   BE and OR-BE over all reps with at least one failure (supplementary)
    diagnostics.csv       case mix, avg failures, exclusions, how often ORMLE pools, acceptance, ESS
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
from sim import Setting, run_setting, summarise, diagnostics
from unrestricted_run import CASES, THETAS, LAM, B, N, BURN, JOBS, OUT as OUT_U

OUT = os.path.join(BASE_DIR, "results", "restricted")
PRIORS = ("vague", "oracle")
NEW = ("ormle", "orbe", "pooled", "acc_or", "ess_or")       # columns from the restricted part


def run():
    all_rows, allreps_rows, diag_rows = [], [], []
    for size, n, m, plan, level, T1, T2 in CASES:
        for config, theta in THETAS.items():
            st = Setting(n=(n, n), m=(m, m), plan=plan, T1=T1, T2=T2, theta=theta, lam=LAM)
            lab = dict(size=size, plan=plan, censoring=level, config=config)

            du, _ = run_setting(st, B=B, priors=PRIORS, N=N, burn=BURN,
                                n_jobs=JOBS, outdir=os.path.join(OUT_U, "raw"))       # unrestricted part
            dr, _ = run_setting(st, B=B, priors=PRIORS, N=N, burn=BURN, n_jobs=JOBS,
                                outdir=os.path.join(OUT, "raw"), part="restricted")   # restricted part
            keep = ["rep"] + [c for c in dr.columns if c.startswith(NEW)]
            df = du.merge(dr[keep], on="rep")                                          # same rep = same dataset

            all_rows.append(summarise(df, st).assign(**lab))
            allreps_rows.append(summarise(df, st, all_reps=True).assign(**lab))
            diag_rows.append({**lab, **diagnostics(df, st)})
            print("done:", st.tag(), flush=True)
    return all_rows, allreps_rows, diag_rows


def save(tables, name):
    df = pd.concat(tables) if isinstance(tables[0], pd.DataFrame) else pd.DataFrame(tables)
    front = ["size", "plan", "censoring", "config"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    a, b, c = run()
    save(a, "summary_all")
    save(b, "summary_allreps")
    save(c, "diagnostics")
