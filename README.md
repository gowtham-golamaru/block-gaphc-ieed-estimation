# Classical and Bayesian Estimation for Block Generalized Adaptive Progressive Hybrid Censoring (IEED)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the simulation study implementation for the research paper:

> **"Classical and Bayesian Estimation for Block Generalized Adaptive Progressive Hybrid Censoring for a General Class of Inverted Exponentiated Distributions"**  
> *Mehta Family School of Data Science and Artificial Intelligence, Indian Institute of Technology Guwahati (IITG), India.*

---

## 📌 Overview

This project studies inference procedures for the **Inverted Exponentiated Exponential Distribution (IEED)** under the **Block Generalized Adaptive Progressive Hybrid Censoring (GAPHC)** scheme with two time thresholds $T_1 < T_2$. Multiple testing facilities share a common baseline scale parameter $\lambda$, while each facility has its own shape parameter $\theta_i$ that may satisfy a physical order restriction:

$$\theta_1 \le \theta_2 \le \cdots \le \theta_k$$

### Key Methodologies

1. **Unconstrained MLE & ACIs**:
   - Closed-form conditional estimator $\hat{\theta}_i(\lambda) = D_i / A_i(\lambda)$, reducing the $(k+1)$-dimensional likelihood maximization to a 1D profile search over $\lambda > 0$.
   - Asymptotic confidence intervals (ACIs) constructed via the exact observed Fisher information matrix and log-delta method.
2. **Order-Restricted MLE (ORMLE)**:
   - Constrained maximization via the Pool-Adjacent-Violators Algorithm (PAVA) / min-max isotonic regression formula on the restricted parameter cone.
3. **Bayesian Estimation (BE)**:
   - Conjugate Gamma conditional updates for $\theta_i$ and random-walk Metropolis–Hastings updates on $\ln \lambda$ (Gibbs-within-MH).
   - Highest Posterior Density (HPD) credible intervals computed via the Chen–Shao algorithm.
4. **Order-Restricted Bayes (OR-BE)**:
   - Direct sampling on the ordered cone by drawing sequentially from conjugate Gamma conditionals truncated to $[\theta_{i-1}, \theta_{i+1}]$.

---

## 📁 Repository Structure

```text
.
├── .gitignore                    # Git ignore file (excludes caches and large raw replications)
├── LICENSE                       # MIT License
├── README.md                     # Project documentation
├── requirements.txt              # Python package dependencies
├── sim.py                        # Core statistical & MCMC engine
│
└── simulation_study/
    ├── unrestricted_run.py       # Stage A: Runs unrestricted simulation grid (MLE + BE)
    ├── restricted_run.py         # Stage B: Runs order-restricted grid (ORMLE + OR-BE)
    ├── sim_data_analysis.py      # Generates paper figures (Figures 1, 2, and 3)
    │
    ├── figures/                  # Publication figures
    │   ├── fig_mse.png           # Figure 1: MSE comparison across 12 settings
    │   ├── fig_cp.png            # Figure 2: Empirical coverage probabilities (ACI vs HPD)
    │   └── fig_ratio.png         # Figure 3: MSE ratios (Restricted / Unrestricted)
    │
    └── results/                  # Summarized simulation outputs
        ├── unrestricted/         # Tables 2–7 (Unrestricted MLE & BE)
        │   ├── summary_main.csv  # Bias, MSE, AL, CP under vague and informative priors
        │   ├── diagnostics.csv   # Failure counts, Case % (I, II, III), ESS, acceptance
        │   └── summary_allreps.csv
        │
        └── restricted/           # Tables 8–15 (Order-Restricted ORMLE & OR-BE)
            ├── summary_all.csv   # Four-estimator merged tables (MLE, ORMLE, BE, OR-BE)
            ├── diagnostics.csv   # Pooling percentages and MCMC diagnostics
            └── summary_allreps.csv
```

---

## 🚀 Quickstart & Installation

### 1. Requirements
Ensure you have Python 3.9+ installed, then install dependencies:

```bash
pip install -r requirements.txt
```

*Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `numba`.*

### 2. Fast Sanity Checks
Verify the core statistical calculations and data generation:

```bash
python sim.py test
```

Run a quick pilot simulation ($B = 200$ replications):

```bash
python sim.py pilot
```

---

## 🔬 Reproducing Simulation Experiments

### Run Unrestricted Estimators (Stage A)
To run the full Monte Carlo simulation grid ($B = 1000$ replications across all 12 settings):

```bash
python simulation_study/unrestricted_run.py
```
*(To run a fast test with fewer replications, e.g. $B=20$, run `python simulation_study/unrestricted_run.py 20`)*

### Run Order-Restricted Estimators (Stage B)
Evaluates the ORMLE and OR-BE on the identical simulated datasets and produces the final merged 4-estimator summaries:

```bash
python simulation_study/restricted_run.py
```

### Generate Publication Figures
To regenerate Figures 1, 2, and 3 from the summary tables into `simulation_study/figures/`:

```bash
python simulation_study/sim_data_analysis.py
```

---

## 📊 Mapping Results to the Paper

| Table / Figure in Paper | Output File | Description |
| :--- | :--- | :--- |
| **Tables 2–3** | `results/unrestricted/summary_main.csv` | MLE with 95% ACI for $n=30$ and $n=50$ |
| **Tables 4–5** | `results/unrestricted/summary_main.csv` | BE with vague prior and 95% HPD intervals |
| **Tables 6–7** | `results/unrestricted/summary_main.csv` | BE with informative prior |
| **Tables 8–9** | `results/restricted/summary_all.csv` | ORMLE performance |
| **Tables 10–11** | `results/restricted/summary_all.csv` | OR-BE with vague prior and 95% HPD intervals |
| **Tables 12–13** | `results/restricted/summary_all.csv` | OR-BE with informative prior |
| **Tables 14–15** | `results/restricted/diagnostics.csv` | Empirical % of Cases I, II, III and ORMLE pooling rates |
| **Figure 1** | `figures/fig_mse.png` | MSE of $\theta_1, \theta_2, \lambda$ across plans and sample sizes |
| **Figure 2** | `figures/fig_cp.png` | Empirical 95% coverage probabilities (with Monte Carlo error bands) |
| **Figure 3** | `figures/fig_ratio.png` | MSE ratios ($\text{ORMLE}/\text{MLE}$ and $\text{OR-BE}/\text{BE}$) |

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
