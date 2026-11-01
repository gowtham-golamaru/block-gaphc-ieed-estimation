"""
data_analysis.py - real-data analysis (report Section 5.2). Put it next to sim.py.

Nothing about the data is hard-coded. It reads a csv with one row per unit: a group column and the
failure time (complete data, all units failed). Only the groups listed in ORDER are used.

    python data_analysis.py            # full analysis with the settings below
    python data_analysis.py explore    # just show which GAPHC case each group lands in (to choose T1, T2)

Steps: 1 state the order, 2 fit each group + goodness of fit (bootstrap KS p-value),
3 test a common lambda (LRT), 4 apply GAPHC censoring to the complete data and estimate with
MLE, ORMLE, BE, OR-BE (vague prior). Results are saved in results_real/.
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import sim

# ----------------------------------------------------------------------------- settings (edit here)
DATA_FILE = os.path.join(BASE_DIR, "insulating_fluid_nelson.csv")
GROUP_COL, TIME_COL = "group", "time"
# Step 1: the order you argue for BEFORE looking at the estimates: theta_1 <= theta_2 <= ...
# (the group with the longest lives first)
ORDER = ["32kV", "36kV"]
OUT = os.path.join(BASE_DIR, "results_real")
N_BOOT = 1000                    # bootstrap size for the KS p-value
N_DRAWS, BURN = 20000, 2000      # Gibbs draws kept / burn-in
SEED = 1
# Censoring scenarios applied to the complete data (this censoring is artificial).
#   plan : last / first / even / split  (see sim.make_R)
#   m    : target failures per group (list), or frac: share of each group's units, m_i = round(frac * n_i)
#   T1,T2: the two time limits (same for every group), or q1,q2: quantiles of the pooled complete data
# Run `python data_analysis.py explore` first and adjust until cases I, II and III all appear.
SCENARIOS = [
    dict(name="A", plan="last",  m=[11, 11], T1=10.0, T2=60.0),
    dict(name="B", plan="even",  m=[11, 11], T1=10.0, T2=60.0),
    dict(name="C", plan="first", m=[11, 11], T1=10.0, T2=60.0),
    dict(name="D", plan="last",  m=[11, 11], T1=3.0, T2=12.0),
    dict(name="E", plan="even",  m=[11, 11], T1=3.0, T2=12.0),
    dict(name="F", plan="first", m=[11, 11], T1=3.0, T2=12.0),
]
# A scenario can give T1, T2 directly (as above) or as pooled-data quantiles q1, q2; and m as a list (one per group)
# or as a share frac of each group's units. Order of the groups is the order in ORDER.
# -----------------------------------------------------------------------------

PARS = [f"theta[{g}]" for g in ORDER] + ["lambda"]
ROMAN = {1: "I", 2: "II", 3: "III"}


def load():
    df = pd.read_csv(DATA_FILE)
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    missing = [g for g in ORDER if g not in set(df[GROUP_COL])]
    if missing:
        sys.exit(f"groups {missing} are not in {DATA_FILE}")
    data = {g: np.sort(df.loc[df[GROUP_COL] == g, TIME_COL].to_numpy(float)) for g in ORDER}
    if any((x <= 0).any() for x in data.values()):
        sys.exit("all failure times must be positive")
    return data


def complete(x):                     # one facility, nothing censored, in the format sim.Data expects
    return (x, np.zeros(len(x), int), 0, 1)


# ----------------------------------------------------------------------------- steps 2 and 3
def cdf(x, th, lam):
    return 1 - (-np.expm1(-lam / x)) ** th


def draw(n, th, lam, rng):           # inverse cdf
    p = rng.random(n)
    return -lam / np.log(1 - (1 - p) ** (1 / th))


def fit1(x):                         # one group, own lambda
    P = sim.Data([complete(x)], 2 * x.max())
    th, lam, _ = sim.mle(P)
    return th[0], lam, sim.loglik(th, lam, P)


def ks_stat(x, th, lam):
    return kstest(x, lambda v: cdf(v, th, lam)).statistic


def qq_plot(x, th, lam, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    p = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    q = -lam / np.log(1 - (1 - p) ** (1 / th))
    lim = [min(q.min(), x.min()), max(q.max(), x.max())]
    plt.figure(figsize=(4, 4))
    plt.loglog(q, x, "o")
    plt.plot(lim, lim, "k--")
    plt.xlabel("fitted quantile"), plt.ylabel("observed"), plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{OUT}/qq_{name}.png", dpi=150)
    plt.close()


def fit_and_test(data):
    rng = np.random.default_rng(SEED)
    rows, fits = [], []
    for g, x in data.items():
        th, lam, ll = fit1(x)
        obs = ks_stat(x, th, lam)
        boot = []                    # the usual KS p-value is not valid (parameters are estimated), so simulate it
        for _ in range(N_BOOT):
            y = np.sort(draw(len(x), th, lam, rng))
            t, l, _ = fit1(y)
            boot.append(ks_stat(y, t, l))
        rows.append(dict(group=g, n=len(x), theta=th, lam=lam, KS=obs, p_boot=np.mean(np.array(boot) >= obs)))
        fits.append(ll)
        qq_plot(x, th, lam, g)
    gof = pd.DataFrame(rows).round(4)
    print("\nStep 2: IEED fit to each group (own lambda) and bootstrap KS p-value\n", gof.to_string(index=False))

    P = sim.Data([complete(x) for x in data.values()], 2 * max(x.max() for x in data.values()))
    th_c, lam_c, _ = sim.mle(P)      # common lambda, theta_i free
    lrt = 2 * (sum(fits) - sim.loglik(th_c, lam_c, P))
    df = len(data) - 1
    pv = chi2.sf(lrt, df)
    print(f"\nStep 3: common lambda. LRT = {lrt:.3f}, df = {df}, p = {pv:.4f}")
    print("  common lambda =", round(lam_c, 4), " theta_i =", np.round(th_c, 4))
    print("  A large p means no evidence against a common lambda; it does not prove it.")
    return gof, dict(LRT=lrt, df=df, p=pv, lam_common=lam_c)


# ----------------------------------------------------------------------------- step 4
def censor(life, R, T1, T2, rng):
    """GAPHC test run on a complete sample (same rules as sim.censor_one, but the lifetimes are given)"""
    life = np.sort(life)
    n, m = len(life), len(R)
    alive = np.ones(n, bool)
    x, Rp, nalive = [], [], n
    for j in range(m):
        i = int(np.argmax(alive))                    # smallest lifetime still on test
        t = life[i]
        if t >= T2:                                  # test stops at T2, this failure is not recorded
            break
        alive[i] = False
        nalive -= 1
        x.append(t)
        if j == m - 1:                               # target reached: withdraw everyone left
            r = nalive
            alive[:] = False
        else:
            r = int(R[j]) if t < T1 else 0           # no planned removals after T1
            if r > 0:
                drop = rng.choice(np.flatnonzero(alive), r, replace=False)
                alive[drop] = False
        nalive -= r
        Rp.append(r)
    D = len(x)
    case = 3 if D < m else (1 if x[-1] < T1 else 2)
    return np.array(x), np.array(Rp, int), (nalive if D < m else 0), case


def scenario_data(data, sc, seed):
    if "T1" in sc:
        T1, T2 = sc["T1"], sc["T2"]
    else:
        T1, T2 = np.quantile(np.concatenate(list(data.values())), [sc["q1"], sc["q2"]])
    rng = np.random.default_rng(seed)
    per, ms = [], []
    for gi, x in enumerate(data.values()):
        m = int(sc["m"][gi]) if "m" in sc else max(2, round(sc["frac"] * len(x)))
        per.append(censor(x, sim.make_R(len(x), m, sc["plan"]), T1, T2, rng))
        ms.append(m)
    return per, ms, T1, T2


def estimate(P, seed):
    """MLE (+ACI), ORMLE, BE, OR-BE (vague prior) on one dataset"""
    k = P.k
    th, lam, _ = sim.mle(P)
    ot, olam, _ = sim.mle(P, restricted=True)
    lo, hi = sim.aci(th, lam, P)
    est = {"MLE": (np.append(th, lam), lo, hi), "ORMLE": (np.append(ot, olam), None, None)}
    diag = {}
    prior = sim.make_prior(sim.Setting(n=(1,) * k), "vague")
    runs = (("BE", False, th, lam), ("OR-BE", True, np.maximum.accumulate(np.maximum(ot, 1e-3)), olam))
    for name, ordered, start, l0 in runs:
        dr, acc = sim.gibbs(P.s, P.c, P.fac, P.D, P.rstar, P.s2, *prior,
                            np.asarray(start, float), float(l0), N_DRAWS, BURN, seed, ordered)
        ci = np.array([sim.hpd(dr[:, j]) for j in range(k + 1)])
        est[name] = (dr.mean(0), ci[:, 0], ci[:, 1])
        diag[name] = (round(acc, 3), round(min(sim.ess(dr[:, j]) for j in range(k + 1))))
    flags = dict(unrestricted_in_order=bool(np.all(np.diff(th) >= 0)), ormle_pooled=bool(np.any(np.diff(ot) == 0)))
    return est, diag, flags


def show(title, P, est, diag, flags, extra=""):
    print(f"\n--- {title} {extra}")
    if P.cases:
        print("   groups: " + ", ".join(f"{g}: D={int(d)}, case {ROMAN[c]}" for g, d, c in zip(ORDER, P.D, P.cases)))
    rows = []
    for j, par in enumerate(PARS):
        row = {"par": par}
        for meth, (e, lo, hi) in est.items():
            row[meth] = f"{e[j]:.3f}" + ("" if lo is None else f" ({lo[j]:.3f}, {hi[j]:.3f})")
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("   unrestricted estimates follow the assumed order:", flags["unrestricted_in_order"],
          "| ORMLE merged groups:", flags["ormle_pooled"], "| (acceptance, ESS):", diag)


def long_rows(label, est):
    out = []
    for meth, (e, lo, hi) in est.items():
        for j, par in enumerate(PARS):
            out.append(dict(scenario=label, method=meth, par=par, estimate=e[j],
                            lower=np.nan if lo is None else lo[j], upper=np.nan if hi is None else hi[j]))
    return out


def run_estimates(data):
    rows, seen = [], set()
    ref = sim.Data([complete(x) for x in data.values()], 2 * max(x.max() for x in data.values()))
    est, diag, flags = estimate(ref, SEED)
    show("complete data (reference, no censoring)", ref, est, diag, flags)
    rows += long_rows("complete", est)
    for i, sc in enumerate(SCENARIOS):
        per, ms, T1, T2 = scenario_data(data, sc, SEED + 1 + i)
        if any(len(p[0]) == 0 for p in per):
            print(f"\n--- scenario {sc['name']}: a group has no observed failure, skipped (change T1/T2/frac)")
            continue
        P = sim.Data(per, T2)
        est, diag, flags = estimate(P, SEED + 100 + i)
        show(f"scenario {sc['name']}: plan={sc['plan']}, m={ms}, T1={T1:.3f}, T2={T2:.3f}", P, est, diag, flags)
        seen |= set(P.cases)
        rows += long_rows(sc["name"], est)
    missing = {1, 2, 3} - seen
    if missing:
        print("\nNote: case(s)", sorted(ROMAN[c] for c in missing), "did not appear in any scenario; adjust SCENARIOS.")
    return pd.DataFrame(rows)


def explore(data):
    print("case of each group (I, II or III) for several choices; groups in order:", ORDER)
    for plan in ("last", "even", "first"):
        for q1 in (0.2, 0.4, 0.6):
            for q2 in (0.6, 0.8, 0.95):
                if q2 <= q1:
                    continue
                sc = dict(plan=plan, frac=0.6, q1=q1, q2=q2)
                per, ms, T1, T2 = scenario_data(data, sc, SEED)
                print(f"plan={plan:5s} q1={q1:.1f} q2={q2:.2f} (T1={T1:8.3f}, T2={T2:8.3f}): "
                      + "  ".join(ROMAN[p[3]] for p in per))


if __name__ == "__main__":
    data = load()
    print("Step 1: assumed order theta_1 <= ... <= theta_k:", " <= ".join(ORDER))
    for g, x in data.items():
        print(f"   {g}: n={len(x)}, median={np.median(x):.3f}, range=({x.min():.3f}, {x.max():.3f})")
    if len(sys.argv) > 1 and sys.argv[1] == "explore":
        explore(data)
        sys.exit()
    os.makedirs(OUT, exist_ok=True)
    gof, lrt = fit_and_test(data)
    gof.to_csv(f"{OUT}/gof.csv", index=False)
    pd.DataFrame([lrt]).to_csv(f"{OUT}/common_lambda_lrt.csv", index=False)
    res = run_estimates(data)
    res.to_csv(f"{OUT}/estimates.csv", index=False)
    print(f"\nsaved in {OUT}/: gof.csv, common_lambda_lrt.csv, estimates.csv, qq_*.png")
