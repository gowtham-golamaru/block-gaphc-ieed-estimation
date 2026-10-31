"""
sim.py - Stage A (unrestricted) simulation for block GAPHC, IEED (Q(t) = t), k facilities.

Everything lives in this one file:
  1. Setting / priors       -> what to simulate
  2. censor_one             -> generate one facility's GAPHC sample
  3. likelihood + MLE + ACI -> classical part
  4. gibbs + HPD            -> Bayes part
  5. run_setting / sweep    -> repeat B times, summarise, save csv

Quick use:
    python sim.py test      # sanity checks
    python sim.py pilot     # small run, prints tables + timing
In code:
    from sim import Setting, run_setting, sweep
    st = Setting(n=(50, 50), m=(30, 30), T1=3, T2=8)
    df, tab = run_setting(st, B=200)
"""
import os, sys, time, math
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

try:                                   # numba is optional, just slower without it
    from numba import njit
except ImportError:
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f

# knobs live here, not buried inside functions
ALPHA = 0.05                                       # 95% intervals (ACI and HPD)
Z = norm.ppf(1 - ALPHA / 2)
STEP0, WIN, ACC_LO, ACC_HI = 0.5, 50, 0.20, 0.40   # MH step tuning during burn-in


# ----------------------------------------------------------------------------
# 1. settings
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Setting:
    n: tuple = (30, 30)          # units on test, per facility
    m: tuple = (20, 20)          # target failures, per facility
    plan: str = "last"           # removal plan: last / first / split / even
    T1: float = 3.0
    T2: float = 8.0
    theta: tuple = (1.0, 2.0)    # true shapes (one per facility)
    lam: float = 2.0             # true common scale

    def tag(self):
        f = lambda t: "-".join(str(x) for x in t)
        return (f"n{f(self.n)}_m{f(self.m)}_{self.plan}_T{self.T1:g}-{self.T2:g}"
                f"_th{f(self.theta)}_lam{self.lam:g}")


def make_R(n, m, plan):
    """planned removals R_1..R_m, they sum to n - m"""
    R, tot = np.zeros(m, np.int64), n - m
    if plan == "last":
        R[-1] = tot
    elif plan == "first":
        R[0] = tot
    elif plan == "split":
        R[0] = tot // 2
        R[-1] = tot - tot // 2
    elif plan == "even":
        R[:] = tot // m
        R[:tot % m] += 1
    else:
        raise ValueError(plan)
    return R


def make_prior(st, kind="vague"):
    """
    returns (a_i, b_i, c, d).
      'vague'  : every hyperparameter 0.001
      'oracle' : Gamma(a, a/truth), prior mean = true value (simulation only), a = 2
      'oracle1' / 'oracle5' : same with a = 1 or 5 (sensitivity check)
    """
    k = len(st.n)
    if kind == "vague":
        return np.full(k, 1e-3), np.full(k, 1e-3), 1e-3, 1e-3
    a = float(kind[6:] or 2)
    ai = np.full(k, a)
    bi = a / np.array(st.theta)
    return ai, bi, a, a / st.lam


# ----------------------------------------------------------------------------
# 2. data generation (one facility)
# ----------------------------------------------------------------------------
@njit(cache=True)
def censor_one(n, R, T1, T2, theta, lam, seed):
    """
    Latent lifetimes -> run the GAPHC test in time order.
    Returns failure times, actual removals, number withdrawn at T2, case (1/2/3).
    """
    np.random.seed(seed)
    m = len(R)
    life = np.empty(n)
    for i in range(n):
        p = np.random.random()
        life[i] = -lam / np.log(1.0 - (1.0 - p) ** (1.0 / theta))   # IEED inverse cdf
    life.sort()

    alive = np.ones(n, np.bool_)
    x = np.empty(m)
    Rp = np.zeros(m, np.int64)
    nalive, D, ptr = n, 0, 0

    for j in range(m):
        while not alive[ptr]:          # next failure = smallest lifetime still alive
            ptr += 1
        t = life[ptr]
        if t >= T2:                    # test ends at T2, this failure is not recorded
            break
        alive[ptr] = False
        nalive -= 1
        x[D] = t
        ptr += 1
        if j == m - 1:                 # last failure: everyone left is withdrawn
            r = nalive
            alive[:] = False
        else:
            r = R[j] if t < T1 else 0  # no planned removals once we pass T1
            c = 0
            while c < r:               # pick r survivors at random
                kk = np.random.randint(ptr, n)
                if alive[kk]:
                    alive[kk] = False
                    c += 1
        nalive -= r
        Rp[D] = r
        D += 1

    rstar = nalive if D < m else 0     # survivors withdrawn at T2 (case III)
    if D < m:
        case = 3
    elif x[m - 1] < T1:
        case = 1
    else:
        case = 2
    return x[:D], Rp[:D], rstar, case


class Data:
    """all facilities flattened into arrays so likelihood code stays simple"""
    def __init__(self, per_fac, T2):
        self.k = len(per_fac)
        self.s = np.concatenate([1.0 / f[0] for f in per_fac])           # 1/x
        self.c = np.concatenate([1.0 + f[1] for f in per_fac])           # 1 + R'
        self.fac = np.concatenate([np.full(len(f[0]), i) for i, f in enumerate(per_fac)])
        self.D = np.array([len(f[0]) for f in per_fac], float)
        self.rstar = np.array([f[2] for f in per_fac], float)
        self.cases = [f[3] for f in per_fac]
        self.s2 = 1.0 / T2
        self.M = int(self.D.sum())


# ----------------------------------------------------------------------------
# 3. classical: likelihood, MLE, ACI
# ----------------------------------------------------------------------------
def _l1m(z):                           # log(1 - exp(-z)), stable
    return np.log(-np.expm1(-z))


def expo(lam, P):
    """A_i(lam): total exposure per facility"""
    w = np.bincount(P.fac, weights=-P.c * _l1m(lam * P.s), minlength=P.k)
    return w - P.rstar * _l1m(lam * P.s2)


def loglik(th, lam, P):
    A = expo(lam, P)
    return (np.sum(P.D * np.log(th) - th * A)
            + np.sum(np.log(lam) - lam * P.s - _l1m(lam * P.s)))


def profile(loglam, P):
    lam = np.exp(loglam)
    A = expo(lam, P)
    ok = P.D > 0
    if not np.all(A[ok] > 1e-300):     # exposure underflowed at huge lam: not a real maximum
        return -np.inf
    return (np.sum(P.D[ok] * np.log(P.D[ok] / A[ok]))
            + P.M * np.log(lam) - lam * P.s.sum() - np.sum(_l1m(lam * P.s)))


def pool(D, A):
    """
    order-restricted theta for fixed lam (report eq. 7): isotonic fit of D_i/A_i with weights A_i.
    Pool adjacent violators: whenever a block's rate beats the next block's, merge them.
    """
    bD, bA, bn = [], [], []                  # per block: sum of D, sum of A, number of facilities
    for d, a in zip(D, A):
        bD.append(d); bA.append(a); bn.append(1)
        while len(bD) > 1 and bD[-2] * bA[-1] > bD[-1] * bA[-2]:    # previous rate > last rate
            d2, a2, n2 = bD.pop(), bA.pop(), bn.pop()
            bD[-1] += d2; bA[-1] += a2; bn[-1] += n2
    out = np.empty(len(D))
    i = 0
    for d, a, n in zip(bD, bA, bn):
        out[i:i + n] = d / a
        i += n
    return out


def profile_or(loglam, P):
    """same as profile(), but theta is pooled (order-restricted) at every lam"""
    lam = np.exp(loglam)
    A = expo(lam, P)
    if not np.all(A > 1e-300):
        return -np.inf
    th = pool(P.D, A)
    ok = P.D > 0
    return (np.sum(P.D[ok] * np.log(th[ok])) - np.sum(th * A)
            + P.M * np.log(lam) - lam * P.s.sum() - np.sum(_l1m(lam * P.s)))


def mle(P, restricted=False):
    """
    theta_i(lam) = D_i / A_i(lam) in closed form (pooled if restricted), so only lam needs a 1-D search.
    restricted=True gives the order-restricted MLE (ORMLE).
    """
    f = profile_or if restricted else profile
    g = np.log(np.median(1.0 / P.s)) + np.linspace(-4, 4, 81)     # coarse grid first
    v = np.array([f(z, P) for z in g])
    i = int(np.argmax(v))
    lo, hi = g[max(i - 1, 0)], g[min(i + 1, len(g) - 1)]
    obj = lambda z: -v if np.isfinite(v := f(z, P)) else 1e300
    r = minimize_scalar(obj, bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-9})
    lam = float(np.exp(r.x))
    A = expo(lam, P)
    th = pool(P.D, A) if restricted else P.D / A
    return th, lam, (i == 0 or i == len(g) - 1)   # last flag: hit grid edge


def aci(th, lam, P):
    """log-scale asymptotic CI, analytic observed information"""
    k = P.k
    w = np.expm1(lam * P.s)
    ws = np.expm1(lam * P.s2)
    A1 = -np.bincount(P.fac, weights=P.c * P.s / w, minlength=k) - P.rstar * P.s2 / ws
    A2 = (np.bincount(P.fac, weights=P.c * P.s**2 * (w + 1) / w**2, minlength=k)
          + P.rstar * P.s2**2 * (ws + 1) / ws**2)
    I = np.zeros((k + 1, k + 1))
    I[range(k), range(k)] = P.D / th**2
    I[:k, k] = I[k, :k] = A1
    I[k, k] = (th @ A2 + P.M / lam**2 - np.sum(P.s**2 * (w + 1) / w**2))
    try:
        se = np.sqrt(np.diag(np.linalg.inv(I)))
    except np.linalg.LinAlgError:
        return np.full(k + 1, np.nan), np.full(k + 1, np.nan)
    est = np.append(th, lam)
    f = np.exp(Z * se / est)
    return est / f, est * f


# ----------------------------------------------------------------------------
# 4. Bayes: Gibbs for theta, MH on log(lam). ordered=True gives the order-restricted version
# ----------------------------------------------------------------------------
# truncated gamma draws (needed for the order restriction). numba has no incomplete gamma
# function, so a small one is written here (series / continued fraction, Numerical Recipes style)
@njit(cache=True)
def _gser(a, x):                           # P(a, x) by series, used for x < a + 1
    ap, s = a, 1.0 / a
    d = s
    for _ in range(1000):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-15:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


@njit(cache=True)
def _gcf(a, x):                            # Q(a, x) by continued fraction, used for x >= a + 1
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


@njit(cache=True)
def _pq(a, x):                             # (cdf, survival) of a standard Gamma(a)
    if x <= 0.0:
        return 0.0, 1.0
    if x < a + 1.0:
        p = _gser(a, x)
        return p, 1.0 - p
    q = _gcf(a, x)
    return 1.0 - q, q


@njit(cache=True)
def trunc_gamma(a, rate, L, U):
    """one draw from Gamma(shape a, rate) cut to [L, U]  (U >= 1e300 means no upper limit)"""
    for _ in range(10):                    # cheap route: plain draw, keep it if it lands inside
        x = np.random.gamma(a, 1.0) / rate
        if x >= L and x <= U:
            return x
    lo, hi = L, U                          # exact route: invert the cut cdf by bisection
    if hi >= 1e300:
        hi = max(lo, a / rate) * 2.0 + 1.0
        while _pq(a, rate * hi)[1] > 1e-17:
            hi *= 2.0
    pl, ql = _pq(a, rate * lo)
    ph, qh = _pq(a, rate * hi)
    u = np.random.random()
    up = rate * lo > a                     # both ends above the mean: use the survival function (keeps precision)
    if up:
        if ql - qh <= 0.0:
            return lo
        target = qh + (1.0 - u) * (ql - qh)
    else:
        if ph - pl <= 0.0:
            return lo
        target = pl + u * (ph - pl)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p, q = _pq(a, rate * mid)
        if up:
            if q > target:
                lo = mid
            else:
                hi = mid
        else:
            if p < target:
                lo = mid
            else:
                hi = mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def _logt(lam, rs_th, s, c, fac, th, s2, M, cc, dd, S):
    v = (M + cc - 1.0) * np.log(lam) - lam * (dd + S)
    for o in range(len(s)):
        v += (th[fac[o]] * c[o] - 1.0) * np.log(-np.expm1(-lam * s[o]))
    return v + rs_th * np.log(-np.expm1(-lam * s2))


@njit(cache=True)
def gibbs(s, c, fac, D, rstar, s2, a, b, cc, dd, th0, lam0, N, burn, seed, ordered=False):
    np.random.seed(seed)
    k, M, S = len(D), len(s), s.sum()
    th, lam = th0.copy(), lam0
    out = np.empty((N, k + 1))
    A = np.empty(k)
    step, win, tot = STEP0, 0, 0
    for it in range(N + burn):
        for i in range(k):
            A[i] = -rstar[i] * np.log(-np.expm1(-lam * s2))
        for o in range(M):
            A[fac[o]] -= c[o] * np.log(-np.expm1(-lam * s[o]))
        for i in range(k):                                  # theta_i | lam ~ Gamma(D+a, A+b)
            if ordered:                                     # cut to [theta_{i-1} (new), theta_{i+1} (old)]
                lo_i = th[i - 1] if i > 0 else 0.0
                hi_i = th[i + 1] if i < k - 1 else 1e300
                th[i] = trunc_gamma(D[i] + a[i], A[i] + b[i], lo_i, hi_i)
            else:
                th[i] = np.random.gamma(D[i] + a[i], 1.0) / (A[i] + b[i])
        rs = 0.0
        for i in range(k):
            rs += th[i] * rstar[i]
        prop = lam * np.exp(step * np.random.randn())       # MH on log(lam)
        la = (_logt(prop, rs, s, c, fac, th, s2, M, cc, dd, S) + np.log(prop)
              - _logt(lam, rs, s, c, fac, th, s2, M, cc, dd, S) - np.log(lam))
        h = 0
        if np.log(np.random.random()) < la:
            lam, h = prop, 1
        if it < burn:                                       # tune step during burn-in only
            win += h
            if (it + 1) % WIN == 0:
                r = win / WIN
                if r > ACC_HI:
                    step *= 1.3
                elif r < ACC_LO:
                    step *= 0.7
                win = 0
        else:
            out[it - burn, :k] = th
            out[it - burn, k] = lam
            tot += h
    return out, tot / N


def hpd(x, alpha=ALPHA):
    """shortest interval holding 100(1-alpha)% of the draws"""
    x = np.sort(x)
    w = int((1 - alpha) * len(x))
    j = int(np.argmin(x[w:] - x[:len(x) - w]))
    return x[j], x[j + w]


def ess(x):
    """effective sample size from the autocorrelations (stops at the first negative pair)"""
    n = len(x)
    x = x - x.mean()
    if not x.any():
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    s = 0.0
    for t in range(0, n - 1, 2):
        pair = ac[t] + ac[t + 1]
        if pair < 0:
            break
        s += pair
    return n / max(2 * s - 1, 1.0)


def bayes(P, th0, lam0, prior, N, burn, seed, ordered=False):
    a, b, cc, dd = prior
    dr, acc = gibbs(P.s, P.c, P.fac, P.D, P.rstar, P.s2, a, b, cc, dd,
                    np.asarray(th0, float), float(lam0), N, burn, seed, ordered)
    est = dr.mean(0)
    ci = np.array([hpd(dr[:, j]) for j in range(dr.shape[1])])
    return est, ci[:, 0], ci[:, 1], acc, min(ess(dr[:, j]) for j in range(dr.shape[1]))


# ----------------------------------------------------------------------------
# 5. one replication -> many replications -> summary
# ----------------------------------------------------------------------------
def names(k):
    return [f"t{i + 1}" for i in range(k)] + ["lam"]


def one_rep(st, rep, base_seed=0, priors=("vague",), N=4000, burn=1000, do_bayes=True,
           part="unrestricted"):
    """
    one dataset, same seeds for both parts so they see identical data.
    part="unrestricted": MLE/ACI once, then Bayes under every prior in `priors`
    part="restricted"  : ORMLE once (no interval), then OR-BE under every prior in `priors`
    """
    k = len(st.n)
    ss = np.random.SeedSequence([base_seed, rep]).generate_state(k + 2 * len(priors))
    per = [censor_one(st.n[i], make_R(st.n[i], st.m[i], st.plan), st.T1, st.T2,
                      st.theta[i], st.lam, int(ss[i])) for i in range(k)]
    P = Data(per, st.T2)
    row = {"rep": rep, "ok": bool(np.all(P.D > 0))}
    for i in range(k):
        row[f"D{i + 1}"] = int(P.D[i])
        row[f"case{i + 1}"] = P.cases[i]
    if P.M == 0:                           # no failure anywhere: nothing can be estimated
        return row
    if part == "restricted":
        ot, olam, _ = mle(P, restricted=True)
        if row["ok"]:                      # ORMLE has no interval (report, Section 3.2): bias and MSE only
            row["pooled"] = int(np.any(np.diff(ot) == 0))      # 1 if the order restriction merged facilities
            for j, nm in enumerate(names(k)):
                row[f"ormle_{nm}"] = np.append(ot, olam)[j]
        if do_bayes:                       # OR-BE exists even when D_i = 0, so run it on every rep
            start = np.maximum.accumulate(np.maximum(ot, 1e-3))  # ordered and strictly positive
            for q, kind in enumerate(priors):
                est, blo, bhi, acc, e = bayes(P, start, olam, make_prior(st, kind), N, burn,
                                              int(ss[k + len(priors) + q]), ordered=True)
                row[f"acc_or_{kind}"], row[f"ess_or_{kind}"] = acc, e
                for j, nm in enumerate(names(k)):
                    row[f"orbe_{kind}_{nm}"] = est[j]
                    row[f"orbe_{kind}_{nm}_lo"], row[f"orbe_{kind}_{nm}_hi"] = blo[j], bhi[j]
        return row
    th, lam, edge = mle(P)                 # a facility with D_i = 0 gets theta_i = 0 (boundary, not a real MLE)
    row["edge"] = edge
    if row["ok"]:                          # MLE and ACI only count when every D_i >= 1
        lo, hi = aci(th, lam, P)
        for j, nm in enumerate(names(k)):
            row[f"mle_{nm}"] = np.append(th, lam)[j]
            row[f"mle_{nm}_lo"], row[f"mle_{nm}_hi"] = lo[j], hi[j]
    if do_bayes:                           # Bayes exists even when D_i = 0, so run it on every rep
        for q, kind in enumerate(priors):
            est, blo, bhi, acc, e = bayes(P, th, lam, make_prior(st, kind), N, burn, int(ss[k + q]))
            row[f"acc_{kind}"], row[f"ess_{kind}"] = acc, e
            for j, nm in enumerate(names(k)):
                row[f"bayes_{kind}_{nm}"] = est[j]
                row[f"bayes_{kind}_{nm}_lo"], row[f"bayes_{kind}_{nm}_hi"] = blo[j], bhi[j]
    return row


def _work(args):
    return one_rep(*args)


def summarise(df, st, all_reps=False):
    """
    bias, AB (|bias|), Var, MSE, average length (AL), coverage (CP) for every method found in df.
    Default: only reps where every D_i >= 1, so all methods are judged on the same reps.
    all_reps=True: Bayes and OR-BE only, every rep with at least one failure (the supplementary table).
    ORMLE has no interval, so its AL and CP are left empty.
    """
    k = len(st.n)
    truth = dict(zip(names(k), list(st.theta) + [st.lam]))
    d = df if all_reps else df[df.ok]
    rows = []
    for meth in [c[:-3] for c in d.columns if c.endswith("_t1")]:
        if all_reps and not meth.startswith(("bayes", "orbe")):
            continue
        for nm in names(k):
            e, t = d[f"{meth}_{nm}"], truth[nm]
            keep = e.notna()
            e = e[keep]
            al = cp = np.nan
            if f"{meth}_{nm}_lo" in d:
                lo, hi = d[f"{meth}_{nm}_lo"][keep], d[f"{meth}_{nm}_hi"][keep]
                ok = lo.notna()
                al, cp = (hi - lo)[ok].mean(), ((lo <= t) & (t <= hi))[ok].mean()
            rows.append(dict(method=meth, par=nm, true=t, reps=len(e), bias=(e - t).mean(),
                             AB=abs((e - t).mean()), var=e.var(ddof=0), mse=((e - t) ** 2).mean(),
                             AL=al, CP=cp))
    return pd.DataFrame(rows).round(4)


def diagnostics(df, st):
    """case mix, failures, exclusions, missing intervals, sampler health"""
    k = len(st.n)
    out = {"reps": len(df),
           "some D_i=0 (excluded from MLE tables)": int((~df.ok).sum()),
           "D_i=0 in every facility": int(df["edge"].isna().sum()) if "edge" in df else 0}
    lo_cols = [c for c in df if c.startswith("mle_") and c.endswith("_lo")]
    if lo_cols:
        out["ACI not computable"] = int(df[df.ok][lo_cols].isna().any(axis=1).sum())
    for i in range(k):
        for cs in (1, 2, 3):
            out[f"fac{i + 1}_case{cs}%"] = round(100 * (df[f"case{i + 1}"] == cs).mean(), 1)
        out[f"fac{i + 1}_avgD"] = round(df[f"D{i + 1}"].mean(), 2)
    if "pooled" in df:                    # how often the order restriction merged facilities (ORMLE)
        out["ORMLE pooled %"] = round(100 * df[df.ok]["pooled"].mean(), 1)
    for c in df.columns:                  # per prior: MH acceptance, effective sample size
        if c.startswith("acc_") or c.startswith("ess_"):
            out[c] = round(df[c].mean(), 3 if c.startswith("acc_") else 0)
    return out


def run_setting(st, B=200, base_seed=0, priors=("vague",), N=4000, burn=1000, do_bayes=True,
                n_jobs=1, outdir="results", force=False, part="unrestricted"):
    """
    one setting, B replications. Saved per setting, so a crash never loses finished work.
    part="restricted" runs ORMLE + OR-BE on the same datasets (same seeds) and saves a separate file.
    """
    os.makedirs(outdir, exist_ok=True)
    suffix = "" if part == "unrestricted" else "_" + part       # unrestricted file names stay as before
    f = os.path.join(outdir, f"{st.tag()}_{'+'.join(priors)}_B{B}{suffix}.csv")
    if os.path.exists(f) and not force:
        df = pd.read_csv(f)
    else:
        jobs = [(st, r, base_seed, tuple(priors), N, burn, do_bayes, part) for r in range(B)]
        if n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(n_jobs) as ex:
                rows = list(ex.map(_work, jobs, chunksize=max(1, B // (4 * n_jobs))))
        else:
            rows = [_work(j) for j in jobs]
        df = pd.DataFrame(rows)
        df.to_csv(f, index=False)
    return df, summarise(df, st)


def sweep(base, B=200, vary=None, **kw):
    """
    Try many settings fast. vary = {field: [values]}, e.g.
        sweep(Setting(), vary={"n": [(30,30),(50,50)], "m": [(20,20),(30,30)]})
    """
    keys = list(vary or {})
    combos = [[]]
    for kk in keys:
        combos = [c + [v] for c in combos for v in vary[kk]]
    res = []
    for c in combos:
        st = replace(base, **dict(zip(keys, c)))
        df, tab = run_setting(st, B=B, **kw)
        tab.insert(0, "setting", st.tag())
        res.append(tab)
    return pd.concat(res, ignore_index=True)


# ----------------------------------------------------------------------------
# sanity checks + pilot
# ----------------------------------------------------------------------------
def selftest():
    # 1) every sample accounts for all n units, and all three cases show up
    st = Setting()
    seen = set()
    for plan in ("last", "first", "split", "even"):
        for seed in range(300):
            R = make_R(30, 20, plan)
            x, Rp, rs, case = censor_one(30, R, st.T1, st.T2, 1.5, 2.0, seed)
            assert len(x) + Rp.sum() + rs == 30 and np.all(np.diff(x) > 0)
            assert (len(x) == 20) == (case != 3) and (x < st.T2).all()
            seen.add(case)
    print("invariants ok, cases seen:", sorted(seen))

    # 2) analytic information matrix vs numerical hessian of the loglik
    per = [censor_one(40, make_R(40, 25, "split"), 3.0, 8.0, t, 2.0, 7 + i)
           for i, t in enumerate((1.0, 2.0))]
    P = Data(per, 8.0)
    th, lam, _ = mle(P)
    lo, hi = aci(th, lam, P)
    est = np.append(th, lam)
    def f(v): return loglik(v[:-1], v[-1], P)
    H = np.zeros((3, 3)); h = 1e-4 * est
    for i in range(3):
        for j in range(3):
            def sh(a, b):
                v = est.copy(); v[i] += a * h[i]; v[j] += b * h[j]; return f(v)
            H[i, j] = (sh(1, 1) - sh(1, -1) - sh(-1, 1) + sh(-1, -1)) / (4 * h[i] * h[j])
    se_num = np.sqrt(np.diag(np.linalg.inv(-H)))
    se_ana = np.log(hi / lo) / (2 * Z) * est / est      # back out se from the interval
    se_ana = est * np.log(hi / est) / Z
    print("se numeric :", se_num.round(4)); print("se analytic:", se_ana.round(4))

    # 3) big complete-ish sample should recover the truth
    per = [censor_one(4000, make_R(4000, 3990, "last"), 1e9, 2e9, t, 2.0, 11 + i)
           for i, t in enumerate((1.0, 2.0))]
    th, lam, _ = mle(Data(per, 2e9))
    print("large-n MLE (truth 1, 2, 2):", th.round(3), round(lam, 3))


@njit(cache=True)
def _tg_many(a, rate, L, U, n, seed):      # helper for the test below only
    np.random.seed(seed)
    out = np.empty(n)
    for i in range(n):
        out[i] = trunc_gamma(a, rate, L, U)
    return out


def selftest_or():
    """checks for the order-restricted pieces: python sim.py test_or"""
    from scipy.stats import gamma as G, kstest
    rng = np.random.default_rng(1)

    # 1) pooling must equal the min-max formula (report eq. 7), computed the slow way
    for _ in range(300):
        k = 4
        D = rng.integers(0, 8, k).astype(float)
        A = rng.uniform(0.5, 6.0, k)
        mm = np.array([max(min(D[s:t + 1].sum() / A[s:t + 1].sum() for t in range(i, k))
                           for s in range(i + 1)) for i in range(k)])
        assert np.allclose(pool(D, A), mm) and np.all(np.diff(pool(D, A)) >= -1e-12)
    print("pool matches the min-max formula")

    # 2) truncated gamma draws vs the exact truncated cdf (plain route, tail route, narrow far tail)
    cases = [("inside      ", 3.5, 2.0, 2.5, 4.0), ("upper tail  ", 3.0, 1.0, 15.0, 1e300),
             ("lower end   ", 5.0, 1.0, 0.0, 0.5), ("far narrow  ", 2.0, 1.0, 30.0, 31.0)]
    for name, a, rate, L, U in cases:
        x = _tg_many(a, rate, L, U, 20000, 3)
        sf = lambda v: G.sf(v, a, scale=1 / rate)
        top = min(U, 1e6)
        cdf = lambda v: (sf(L) - sf(v)) / (sf(L) - sf(top))     # exact cut cdf, written with sf for tail accuracy
        p = kstest(x, cdf).pvalue
        assert x.min() >= L and x.max() <= U and p > 1e-3, (name, p)
        print(f"truncated gamma {name} ks p-value {p:.3f}")

    # 3) ORMLE: ordered, never above the unrestricted maximum, equal to the MLE when the MLE is already ordered
    for thetas in ((2.0, 1.0), (1.0, 2.0)):
        per = [censor_one(40, make_R(40, 25, "split"), 3.0, 8.0, t, 2.0, 21 + i)
               for i, t in enumerate(thetas)]
        P = Data(per, 8.0)
        th, lam, _ = mle(P)
        ot, olam, _ = mle(P, restricted=True)
        assert np.all(np.diff(ot) >= 0)
        assert loglik(ot, olam, P) <= loglik(th, lam, P) + 1e-6
        if np.all(np.diff(th) >= 0):
            assert np.allclose(ot, th, rtol=1e-3)
        print("truth", thetas, "MLE", th.round(3), "ORMLE", ot.round(3))

    # 4) every draw of the ordered sampler must satisfy theta1 <= theta2
    a, b, cc, dd = make_prior(Setting(), "vague")
    start = np.maximum.accumulate(np.maximum(ot, 1e-3))
    dr, acc = gibbs(P.s, P.c, P.fac, P.D, P.rstar, P.s2, a, b, cc, dd, start, olam, 3000, 500, 5, True)
    assert np.all(np.diff(dr[:, :2], axis=1) >= 0)
    print("ordered Gibbs ok, acceptance", round(acc, 3), "posterior means", dr.mean(0).round(3))


def pilot():
    st = Setting()
    t = time.time()
    df, tab = run_setting(st, B=50, priors=("vague", "oracle"), force=True)
    print(f"B=50 took {time.time() - t:.1f}s (includes numba compile on first run)")
    print(diagnostics(df, st)); print(tab.to_string(index=False))


if __name__ == "__main__":
    {"test": selftest, "test_or": selftest_or, "pilot": pilot}.get(sys.argv[1] if len(sys.argv) > 1 else "pilot")()