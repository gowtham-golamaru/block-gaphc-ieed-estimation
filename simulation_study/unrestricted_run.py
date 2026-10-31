"""
unrestricted_run.py - runs the unrestricted simulation (MLE + Bayes) and saves csv files.
Run:
    python unrestricted_run.py        # full run (B=1000)
    python unrestricted_run.py 50     # quick test with 50 replications

Results go to results/unrestricted/ :
    summary_main.csv      bias, AB, var, MSE, AL, CP
    summary_allreps.csv   Bayes over all replications (supplementary)
    diagnostics.csv       case mix, avg failures, exclusions, MH acceptance, ESS
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"          # one numpy thread per worker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import pandas as pd
from sim import Setting, run_setting, summarise, diagnostics

# ---- settings
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000   # replications per setting
N, BURN = 10000, 1000                                 # Gibbs draws kept / burn-in
JOBS = 8                                              # cores
OUT = os.path.join(BASE_DIR, "results", "unrestricted")
LAM = 2.0
RUN_SENSITIVITY = False                               # a = 1, 5 priors, not used in the report

# true shapes: order right / equal / order wrong
THETAS = {"ordered": (1.0, 2.0), "equal": (1.5, 1.5), "reversed": (2.0, 1.0)}

# size, n, m, removal plan, censoring level, T1, T2
CASES = [
    ("n30m20", 30, 20, "last", "moderate", 2.5, 4.5),
    ("n30m20", 30, 20, "last", "heavy",    1.5, 2.5),
    ("n30m20", 30, 20, "even", "moderate", 8.0, 25.0),
    ("n30m20", 30, 20, "even", "heavy",    5.0, 12.0),
    ("n50m30", 50, 30, "last", "moderate", 2.0, 3.5),
    ("n50m30", 50, 30, "last", "heavy",    1.5, 2.5),
    ("n50m30", 50, 30, "even", "moderate", 10.0, 30.0),
    ("n50m30", 50, 30, "even", "heavy",    6.0, 14.0),
    ("n30m20", 30, 20, "first", "moderate", 8.0, 25.0),
    ("n30m20", 30, 20, "first", "heavy",    5.0, 12.0),
    ("n50m30", 50, 30, "first", "moderate", 10.0, 30.0),
    ("n50m30", 50, 30, "first", "heavy",    6.0, 14.0),
]
PRIORS = ("vague", "oracle", "oracle1", "oracle5") if RUN_SENSITIVITY else ("vague", "oracle")


def run():
    all_rows, allreps_rows, diag_rows = [], [], []
    for size, n, m, plan, level, T1, T2 in CASES:
        for config, theta in THETAS.items():
            st = Setting(n=(n, n), m=(m, m), plan=plan, T1=T1, T2=T2, theta=theta, lam=LAM)
            lab = dict(size=size, plan=plan, censoring=level, config=config)

            df, _ = run_setting(st, B=B, priors=PRIORS, N=N, burn=BURN, n_jobs=JOBS,
                                outdir=os.path.join(OUT, "raw"))
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
    save(a, "summary_main")
    save(b, "summary_allreps")
    save(c, "diagnostics")
    if RUN_SENSITIVITY:
        df = pd.concat(a)
        sens = df[df.method.isin(["bayes_oracle1", "bayes_oracle5"])]
        sens.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
