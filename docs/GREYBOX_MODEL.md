# Grey-box model for the bioreactor forecast

**Version 2.** A mechanistic fed-batch model whose rates are set per batch from a data-driven forecast. Every forecast is a trajectory of the ODE; nothing is added on top of it.

Status: candidate. Scored on held-out training batches only. It has not been run on the 12 hidden batches and has not been submitted.

## Changes from version 1

| | Version 1 | Version 2 |
|---|---|---|
| Maintenance uptake | $m_S\,S/(S + 0.05)$, half-saturation fixed | $m_S\,S/(S + K_q)$, $K_q$ fitted (0.40–0.61 g/L across folds) |
| Shared parameters | 17 | 18 |
| Registry name of the ODE | `ode-S v5` | `ode-S v25` |
| Registry name of the grey box | `greybox v3` | `greybox v4` |
| Oracle substrate error on day 11 | 12.6% | 2.8% |
| Grey-box Score | 7.84 ± 0.15 | 7.91 ± 0.22 |

The change removes a structural error: with the batch's own rates fitted, version 1 still missed substrate on day 11 by 12.6%. The forecast Score does not change beyond the fold spread. Version 2 is kept because it is the better description of the culture at no cost in accuracy. See [Structural diagnosis](#structural-diagnosis).

## Contents

1. [Summary](#summary)
2. [Task and inputs](#task-and-inputs)
3. [Mechanistic model](#mechanistic-model)
4. [Data-driven forecast](#data-driven-forecast)
5. [Grey-box step](#grey-box-step)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Oracle](#oracle)
9. [Structural diagnosis](#structural-diagnosis)
10. [History matching](#history-matching)
11. [Limitations](#limitations)
12. [Reproducibility](#reproducibility)

## Summary

| Model | VCD | TCD | Substrate | Byproduct | Score ± fold SD |
|---|---|---|---|---|---|
| ODE, one set of rates for all batches | 7.4 | 7.2 | 10.3 | 12.2 | 9.28 ± 0.27 |
| **Grey box: ODE with per-batch rates** | **6.9** | **6.7** | **8.7** | **9.4** | **7.91 ± 0.22** |
| Grey box + substrate day correction | 6.9 | 6.7 | 8.2 | 9.4 | 7.79 ± 0.19 |
| Data-driven forecast the grey box is matched to | 6.6 | 6.5 | 7.7 | 8.3 | 7.30 ± 0.33 |
| Geometric mean of grey box and data-driven forecast | 6.6 | 6.5 | 7.8 | 8.5 | 7.32 ± 0.24 |
| Oracle: per-batch rates fitted to the batch's own measurements, robust loss | 5.5 | 5.2 | 3.9 | 5.9 | 5.12–5.16 |
| Oracle, squared loss | 5.6 | 5.4 | 4.0 | 7.1 | 5.54 ± 0.91 |
| Grey box, version 1 (for comparison) | 6.8 | 6.7 | 8.4 | 9.4 | 7.84 ± 0.15 |

Values are MAPE in % on held-out batches; Score is the mean of the four. Lower is better. Differences below about 0.3 are not distinguishable (see [Evaluation](#evaluation)).

The method in three steps, for one batch:

1. A data-driven model forecasts the batch's trajectory from its day-0 values, feed record and volumes.
2. Eleven rates of the ODE are adjusted for this batch until the ODE trajectory matches that forecast. How far each rate may move is learned from the training batches.
3. The adjusted ODE trajectory is the forecast.

## Task and inputs

Forecast four variables of a fed-batch mammalian cell culture on days 1 to 13.

| Variable | Meaning | Inferred unit |
|---|---|---|
| VCD | viable cell density | 10⁶ cells/mL |
| TCD | total cell density | 10⁶ cells/mL |
| Substrate | nutrient concentration | g/L |
| Byproduct | metabolic byproduct concentration | g/L |

Units are not given with the data; those above are inferred.

Data: 61 batches, 49 with full trajectories (training) and 12 hidden (test).

| Input | Covers | Available for hidden batches |
|---|---|---|
| VCD, TCD, substrate, byproduct | day 0 only | yes |
| Feed doses: substrate feed, Feed B, Feed C, Feed D (time and volume of each bolus) | days 0 to 13 | yes |
| Reactor volume at each sample | days 0 to 13 | yes |

The forecast is one shot: no measurement of the culture after day 0 is used. The feed and volume records for the whole run are inputs.

Score: MAPE per variable over the 156 hidden points (12 batches × 13 samples), then the mean of the four:

$$
\text{MAPE}_j = \frac{100}{|\mathcal P|}\sum_{(b,k)\in\mathcal P}\frac{|\hat y_{bj}(t_k) - y_{bj}(t_k)|}{y_{bj}(t_k)}, \qquad \text{Score} = \frac14\sum_{j=1}^{4}\text{MAPE}_j
$$

## Mechanistic model

The structure is called `ode-S v25` in the model registry.

### States

| Symbol | Meaning | Measured |
|---|---|---|
| $X_v$ | viable cells | yes (VCD) |
| $X_d$ | dead cells | yes (TCD − VCD) |
| $S$ | substrate | yes |
| $B$ | byproduct | yes |
| $N$ | growth nutrient supplied by Feed B | no |
| $G$ | overflow driver: a component of the starting medium that the cells use up | no |

### Balances between feed events

$$
\begin{aligned}
\dot X_v &= (\mu - k_d)\,X_v \\
\dot X_d &= k_d\,X_v - k_l\,X_d \\
\dot S &= -\Big(m_S\,\frac{S}{S + K_q} + q_{ov}\Big)\,X_v \\
\dot B &= \Big(y_{bg}\,\mu + Y_{bo}\,q_{ov} - k_{bc}\,B\,\frac{K_i}{K_i + S}\Big)\,X_v \\
\dot N &= -\frac{\mu}{Y_{XN}}\,X_v \\
\dot G &= -k_g\,X_v\,G
\end{aligned}
$$

### Rates

$$
\begin{aligned}
\mu &= r_N\,N\,\frac{S}{S + 0.05}\,\big(1 - e^{-t/\tau}\big) && \text{specific growth rate} \\
k_d &= k_{dx}\,X_v && \text{death rate, rising with density} \\
q_{ov} &= q_o\,G && \text{overflow uptake}
\end{aligned}
$$

| Term | Meaning |
|---|---|
| $r_N N$ | Growth follows a nutrient delivered by Feed B and consumed by growth. |
| $S/(S+0.05)$ in $\mu$ | Growth stops when substrate runs out. The constant 0.05 is fixed. |
| $m_S\,S/(S+K_q)$ | Maintenance uptake per cell, saturable in substrate (Michaelis–Menten). It slows once substrate falls below about $K_q$. New in version 2. |
| $1 - e^{-t/\tau}$ | Short lag before growth reaches full speed. |
| $k_{dx} X_v$ | Death rate proportional to viable density. |
| $q_o G$ | Early overflow: the cells take up excess substrate while $G$ lasts and release part of it as byproduct. $G$ decays with accumulated viable cell-days, which gives a sigmoid-shaped decline in time over the first three days. |
| $y_{bg}\mu$ | Byproduct formed with growth. |
| $k_{bc} B\,K_i/(K_i+S)$ | Byproduct consumption, held back while substrate is plentiful. |
| $k_l X_d$ | Lysis of dead cells, so that they leave the total count. |

**Why $K_q$ is fitted.** Uptake of a sugar substrate by mammalian cells is carrier-mediated and saturates. The fitted $K_q$ of 0.40–0.61 g/L is about 2–3 mM if the substrate is glucose, the order of the affinity of glucose transport in mammalian cells, and close to the substrate threshold of 0.58 g/L below which growth stopped in a recombinant CHO line [1]. The term matters only below about 1 g/L, which 7% of training samples reach, nearly all on day 11 (median 0.87 g/L). Version 1 fixed $K_q = 0.05$, so uptake continued at full rate until substrate was almost exhausted.

### Feed events and volume

Feeds are boluses. At a bolus with volumes $v_{sub}$ (substrate feed), $v_B$, $v_C$, $v_D$:

$$
\begin{aligned}
V^+ &= V + v_{sub} + v_B + v_C + v_D \\
C^+ &= C\,\frac{V}{V^+} \quad \text{for every state } C \\
S^+ &\mathrel{+}= \frac{c_{sub}\,v_{sub} + c_B\,v_B}{V^+} \\
N^+ &\mathrel{+}= \frac{v_B}{V^+}
\end{aligned}
$$

At each sampling time the volume is reset to the recorded value.

### Initial conditions and the day-0 count correction

$$
\begin{aligned}
X_v(0) &= \bar x\,\Big(\frac{x_0}{\bar x}\Big)^{\lambda}, \qquad X_d(0) = \text{TCD}_0 - \text{VCD}_0 \\
S(0) &= S_0, \qquad B(0) = B_0, \qquad N(0) = n_0, \qquad G(0) = 1
\end{aligned}
$$

$x_0$ is the batch's day-0 VCD and $\bar x$ the geometric mean of day-0 VCD over the fitting batches. Reported densities are scaled back:

$$
\widehat{\text{VCD}} = X_v\,\Big(\frac{x_0}{\bar x}\Big)^{\gamma}, \qquad \widehat{\text{TCD}} = (X_v + X_d)\,\Big(\frac{x_0}{\bar x}\Big)^{\gamma}
$$

With $\lambda$ about 0.4 and $\gamma$ about 0.44, the day-0 count is treated as a noisy reading: only part of a batch's deviation from the typical day-0 count is carried into the simulation.

### Shared parameters

Eighteen parameters, the same for all batches. They are fitted separately in each of the five training folds; the table gives the median and range over folds, with version 1 for comparison.

| Parameter | Meaning | Version 2: median (range over folds) | Version 1 median |
|---|---|---|---|
| $r_N$ | growth per unit nutrient | 1.19 (0.86–1.49) | 1.50 |
| $Y_{XN}$ | cells formed per unit nutrient | 66.6 (51.6–84.5) | 89.7 |
| $n_0$ | nutrient in the starting medium | 0.69 (0.55–0.95) | 0.55 |
| $\tau$ | growth lag, days | 0.088 (0.080–0.119) | 0.078 |
| $k_{dx}$ | death rate per unit density | 0.0052 (0.0046–0.0064) | 0.0068 |
| $k_l$ | lysis rate of dead cells, per day | 2.69 (2.32–3.43) | 3.94 |
| $m_S$ | maintenance uptake per cell, per day | 0.143 (0.129–0.151) | 0.101 |
| $K_q$ | substrate level that halves maintenance uptake, g/L | 0.52 (0.40–0.61) | 0.05 (fixed) |
| $q_o$ | overflow uptake per cell at $G = 1$, per day | 0.554 (0.538–0.559) | 0.587 |
| $k_g$ | decay of the overflow driver per cell-day | 0.236 (0.230–0.243) | 0.255 |
| $Y_{bo}$ | byproduct per unit overflow uptake | 0.80 (0.75–0.85) | 0.67 |
| $y_{bg}$ | byproduct per unit growth | 0.106 (0.085–0.140) | 0.175 |
| $k_{bc}$ | byproduct consumption | 0.036 (0.026–0.053) | 0.068 |
| $K_i$ | substrate level that halves byproduct consumption, g/L | 4.93 (4.04–6.28) | 4.09 |
| $c_{sub}$ | substrate content of the substrate feed | 215 (211–218) | 223 |
| $c_B$ | substrate content of Feed B | 58.8 (55.1–61.4) | 47.7 |
| $\lambda$ | share of the day-0 count deviation used as the starting density | 0.41 (0.40–0.43) | 0.51 |
| $\gamma$ | share of the day-0 count deviation applied to reported densities | 0.44 (0.36–0.44) | 0.40 |

With uptake now slowing at low substrate, the fit raises maintenance uptake per cell ($m_S$ 0.10 → 0.14) and the substrate content of Feed B, and lowers byproduct consumption and growth-linked byproduct formation. The parameters are correlated, so these shifts are compensation within the model, not independent findings.

### Numerical method and fit

| Item | Choice |
|---|---|
| Integration | Runge–Kutta 4, step 0.025 d, bolus events applied on the time grid |
| Simulation | open loop from day 0; no measured value is fed back |
| Objective | least squares on $\log(\hat y / y)$ over all four variables, soft-L1 loss with scale 0.1 |
| Parameter space | logarithms of the parameters, with bounds |
| Optimiser | trust-region reflective (`scipy.optimize.least_squares`), finite-difference Jacobian |

Checks on the fit, run on version 1 (no parameter on a bound in version 2 either):

| Check | Result |
|---|---|
| 12 random starts with SD 1.0 in log units | same optimum |
| Basin hopping, 20 hops | same optimum |
| Loss closer to absolute error (scale 0.02) | 9.48 against 9.45 |
| Gaussian prior of SD 0.3 in log space (MAP) | 9.43 against 9.45 |
| Condition number of the Fisher information | 7.4·10² |
| Two other fold splits | 9.47 and 9.51 against 9.45 |

## Data-driven forecast

The grey box needs a forecast of each batch to match (`dd-best v1` in the registry). It is chosen from a benchmark of our own data-driven models. All are fitted inside the training folds, work on $\log y$ and use only the inputs listed above. Unchanged from version 1.

### Inputs

| Group | Features |
|---|---|
| Day 0 | log VCD, log TCD, log substrate, log byproduct, volume, log viability |
| Feed totals | total volume of each feed, number of Feed C doses, number of small Feed C doses |
| Substrate shots | early-shot flag and time, volume of the shots after the day-11 and day-12 samples |
| Local feeds (per sampling day) | Feed B and substrate feed in the two intervals before the sample, the one starting at it and the next; small Feed C doses and Feed D in the interval |
| Volume (per sampling day) | volume relative to day 0, cumulative volume of each feed |
| Unlogged volume | $u_b(t_k) = V_b(t_k) - V_b(0) - \sum_{\tau \le t_k} v$: level and daily change per sampling day, and the level on days 2, 3, 4, 5 and 13 |

Unlogged volume is volume that entered or left the reactor without being logged as a feed. One reading is base added for pH control, which would track byproduct formation; the data do not confirm that reading.

### Benchmark

Held-out MAPE in % on the same five folds.

| Model | VCD | TCD | Substrate | Byproduct | Score ± fold SD |
|---|---|---|---|---|---|
| LightGBM, absolute-error loss | 7.0 | 7.0 | 8.9 | 9.1 | 7.98 ± 0.34 |
| TabPFN (pretrained tabular transformer, v2.0) | 7.4 | 7.3 | 8.3 | 9.2 | 8.04 ± 0.26 |
| LightGBM, squared-error loss | 7.2 | 7.1 | 8.6 | 9.4 | 8.08 ± 0.28 |
| Extra-trees, one model per sampling day | 7.5 | 7.3 | 8.3 | 9.6 | 8.15 ± 0.36 |
| Elastic net per day (L1 + L2) | 7.6 | 7.6 | 8.7 | 8.8 | 8.15 ± 0.41 |
| Varying-coefficient spline | 6.9 | 6.8 | 10.7 | 8.3 | 8.16 ± 0.59 |
| Lasso per day (L1) | 7.6 | 7.6 | 8.7 | 8.8 | 8.17 ± 0.40 |
| GAM (splines on time, feeds and unlogged volume; linear in the rest) | 7.0 | 6.9 | 8.8 | 10.1 | 8.20 ± 0.35 |
| Ridge per day (L2) | 7.7 | 7.7 | 8.8 | 8.9 | 8.29 ± 0.48 |
| Recursive ridge rollout | 7.2 | 7.1 | 8.8 | 11.8 | 8.71 ± 0.32 |
| Principal curves + ridge | 6.9 | 6.9 | 10.4 | 10.7 | 8.71 ± 0.60 |
| Partial least squares | 7.1 | 7.0 | 11.3 | 10.5 | 8.96 ± 0.28 |
| Median of the 9 nearest batches | 6.7 | 6.5 | 12.0 | 11.4 | 9.14 ± 0.30 |
| Gaussian process | 7.6 | 7.5 | 9.4 | 14.9 | 9.84 ± 1.00 |

The eight best single models are within 0.3 of each other, which is less than one fold SD. Nearest batches by dynamic time warping of the input series scored 9.30, no better than plain distance (9.33): the batches are aligned in time (best per-batch time shift varies by 0.1 day).

Model notes:

- **Per-day linear models** (ridge, lasso, elastic net): one regression per sampling day and variable on batch-level features, the 13 Feed B doses and the 13 unlogged-volume levels. Penalties by cross-validation inside the training folds.
- **Varying-coefficient spline:** $\log y(b,t) = s(t) + \sum_i z_{b,i}\,g_i(t)$, a smooth curve in elapsed time plus batch features whose effect is itself a smooth function of time (quadratic spline, 4 knots), fitted by ridge.
- **Tree models** (LightGBM, extra-trees): one row per batch and sampling day; the target is $\log y$ minus the typical trajectory of the fitting batches.
- **Recursive ridge:** predicts each day from its own prediction for the previous day plus the feeds around that interval; rolled out from day 0.

### LightGBM

LightGBM is the best single data-driven model (7.98 ± 0.34 with the absolute-error loss).

| Item | Choice |
|---|---|
| Data layout | one row per batch and sampling day: 637 rows in all, about 510 per training fold |
| Target | $\log y$ minus the typical log trajectory of the fitting batches on that day, one model per variable |
| Features | all groups in the Inputs table, plus the sampling day and elapsed time |
| Loss | absolute error (`objective="l1"`), which is closer to MAPE than squared error |
| Trees | 600 trees, learning rate 0.03, at most 8 leaves, at least 10 rows per leaf |
| Randomisation | 80% of rows and 70% of features per tree |
| Squared-error variant | 500 trees, L2 leaf penalty 2.0; Score 8.08 ± 0.28 |

Settings were fixed in advance and not tuned on the folds. In the combined forecast LightGBM is one of the models chosen for substrate.

### Combination

For each variable $j$ and fold, the models are ranked on the other folds, and the forecast is the geometric mean of the top $m_j$, with $m_j$ from 1 to 6 chosen on those same folds:

$$
\tilde y_{bj}(t_k) = \exp\Big(\frac{1}{m_j}\sum_{i \in \mathcal T_j}\log \hat y^{(i)}_{bj}(t_k)\Big)
$$

| Variable | Models used in at least 3 of 5 folds | MAPE |
|---|---|---|
| VCD, TCD | nearest batches, principal curves + ridge, varying-coefficient spline | 6.6, 6.5 |
| Substrate | extra-trees, LightGBM, elastic net, lasso, recursive ridge, GAM | 7.7 |
| Byproduct | varying-coefficient spline, elastic net | 8.3 |

TCD is raised to VCD where the forecast is lower. Score: 7.30 ± 0.33. TabPFN is left out of the pool: it takes about ten minutes on a CPU, and with it the combined Score is 7.33 ± 0.33.

## Grey-box step

Unchanged from version 1, applied to the version-2 ODE.

### Per-batch rates

Eleven of the ODE's parameters receive a per-batch multiplier. For batch $b$ and parameter $p_i$:

$$
p_{i,b} = p_i\,e^{\theta_{i,b}}
$$

$\theta_b = 0$ gives the shared-rate ODE. The eleven parameters are $r_N$, $k_{dx}$, $k_l$, $m_S$, $q_o$, $k_g$, $Y_{bo}$, $y_{bg}$, $k_{bc}$, $c_{sub}$, $c_B$.

### Estimate for one batch

Let $\tilde y_b(k, j)$ be the data-driven forecast for sample $k$ and variable $j$, and $f_j(k;\theta)$ the ODE output for the batch's own day-0 values and feed record. The per-batch values are the maximum a posteriori estimate

$$
\hat\theta_b = \arg\min_{\theta}\; \sum_{k=1}^{13}\sum_{j=1}^{4} \frac{\big(\log \tilde y_b(k,j) - \log f_j(k;\theta)\big)^2}{R_{kj}} \;+\; \theta^{\top}\Sigma^{-1}\theta
$$

and the forecast is the ODE trajectory $f(\cdot\,;\hat\theta_b)$.

### Learned weights

Both weights come from the training batches of the fold, not from choices made by hand.

| Weight | Meaning | How it is learned |
|---|---|---|
| $\Sigma$ | how much the rates vary between batches, and how they vary together | The eleven multipliers are fitted to each training batch's own measurements. $\Sigma$ is half the covariance of those fits; the halving allows for measurement noise absorbed by the fits and was not tuned. |
| $R_{kj}$ | how wrong the data-driven forecast is at sample $k$ for variable $j$ | Mean squared log error of the data-driven forecast on the training batches, each forecast made without that batch. |

A rate that hardly varies between batches gets a tight prior and is effectively not used. A sample where the data-driven forecast is unreliable pulls the ODE less.

### Algorithm

| Step | Detail |
|---|---|
| Solver | Gauss–Newton, 10 iterations, all batches of a fold simulated together |
| Jacobian | forward differences, step 0.05 in $\theta$ |
| Safeguards | step limited to ±0.3 per iteration, $\theta$ limited to ±1.2 |
| Start | $\theta = 0$ |

### Substrate day correction (optional)

$\hat S_b(t_k)\,e^{c_k}$, with $c_k$ the median of $\log(S/\hat S)$ on sampling day $k$ over the training batches of the fold. Applied to substrate only. It lowers substrate MAPE from 8.7 to 8.2 (Score 7.91 → 7.79), inside the fold spread.

## Evaluation

| Item | Choice |
|---|---|
| Protocol | batch-level 5-fold cross-validation on the 49 training batches |
| Folds | `np.array_split(np.random.default_rng(42).permutation(train_ids), 5)`, batches sorted by number |
| Held-out points | 637 (49 batches × 13 samples; day 0 excluded) |
| Metric | pooled MAPE per variable; Score = mean of the four |
| Spread | SD of the per-fold Scores |

For a held-out batch only its day-0 values, feed record and volumes are used. Everything else is fitted on the other four folds: the shared ODE parameters, the data-driven models with their penalties, the choice of models per variable, $\Sigma$, $R$ and the day correction.

Five other fold shuffles and leave-one-batch-out move a model's Score by about ±0.2, so differences below about 0.3 are treated as ties.

One point is not fully nested. When $\Sigma$ and $R$ are learned, the ODE and data-driven predictions for the training batches of a fold are themselves held-out predictions from the other folds (stacking), so models that saw the held-out batch contributed to them. The effect on the Score has not been measured.

## Results

### Error by variable

See the table in [Summary](#summary).

### Where the error falls

A batch's error has two parts: the batch sitting above or below the model for the whole run (offset), and scatter around that.

| Variable | Offset, shared rates → grey box | Scatter, shared rates → grey box |
|---|---|---|
| Byproduct | 11.4% → 7.7% | 12.8% → 11.6% |
| Substrate | 8.8% → 6.9% | 11.8% → 10.9% |
| VCD | 5.6% → 4.7% | 7.9% → 8.0% |

Per-batch rates remove offsets, which is what a batch-specific rate can do.

| MAPE by day, % | Day 8 | Day 11 | Day 12 | Day 13 |
|---|---|---|---|---|
| Substrate, shared rates | 9 | 30 | 15 | 14 |
| Substrate, grey box | 8 | 24 | 14 | 11 |
| Byproduct, shared rates | 13 | 18 | 20 | 24 |
| Byproduct, grey box | 9 | 10 | 12 | 18 |

The grey box stays close to its target: on average 2.5% away on cell densities and about 5% on substrate and byproduct. 40 of the 49 batches improve on the shared-rate ODE; 9 get worse.

### What differs between batches

Learned prior SD of each per-batch rate, averaged over folds:

| Rate | Prior SD |
|---|---|
| Byproduct per unit growth $y_{bg}$ | 33% |
| Byproduct consumption $k_{bc}$ | 30% |
| Lysis $k_l$ | 16% |
| Maintenance uptake $m_S$ | 14% |
| Substrate content of Feed B $c_B$ | 14% |
| Substrate content of the substrate feed $c_{sub}$ | 13% |
| Overflow decay $k_g$ | 13% |
| Overflow uptake $q_o$ | 10% |
| Death $k_{dx}$ | 10% |
| Byproduct per unit overflow $Y_{bo}$ | 9% |
| Growth $r_N$ | 2% |

Batches differ mainly in byproduct metabolism, and hardly at all in growth rate.

### Example

Batch 39, held out. The grey box set byproduct consumption 19% lower, byproduct per unit growth 17% lower, byproduct per overflow 14% higher and overflow uptake 7% lower than the shared values.

| | Byproduct day 9 | Byproduct day 13 | Substrate day 9 | Substrate day 13 |
|---|---|---|---|---|
| Measured | 1.29 | 0.66 | 2.34 | 1.25 |
| ODE, shared rates | 1.12 | 0.45 | 1.86 | 0.98 |
| Grey box | 1.26 | 0.54 | 2.20 | 1.31 |

The batch's mean error falls from 10.2% to 5.4%.

A counter-example is Batch 42: the data-driven forecast pointed the wrong way on substrate, the grey box lowered maintenance uptake and the feed substrate contents to follow it, and the error rose from 8.8% to 10.0%.

## Oracle

The oracle is the same ODE with the eleven per-batch rates fitted to a held-out batch's own measurements on days 1 to 13, with an almost flat prior:

$$
\theta^{\text{oracle}}_b = \arg\min_{\theta}\ \sum_{k,j}\frac{\big(\log y_{bj}(t_k) - \log f_j(k;\theta)\big)^2}{0.05^2} + \|\theta\|^2
$$

It cannot be used for hidden batches; it is the ceiling of what per-batch rates can achieve with this structure.

| Model | VCD | TCD | Substrate | Byproduct | Score ± fold SD |
|---|---|---|---|---|---|
| ODE, shared rates | 7.4 | 7.2 | 10.3 | 12.2 | 9.28 ± 0.27 |
| Grey box (rates inferred from the inputs) | 6.9 | 6.7 | 8.7 | 9.4 | 7.91 ± 0.22 |
| Oracle, version 2 | 5.6 | 5.4 | 4.0 | 7.1 | 5.54 ± 0.91 |
| Oracle, version 1 | 5.7 | 5.5 | 4.7 | 6.7 | 5.65 ± 0.72 |

| Oracle MAPE by day, % | Days 1–5 | Days 6–10 | Day 11 | Day 12 | Day 13 |
|---|---|---|---|---|---|
| Substrate, version 2 | 2–3 | 3–7 | 3 | 4 | 4 |
| Substrate, version 1 | 2–3 | 3–7 | 13 | 4 | 5 |
| Byproduct, version 2 | 2–5 | 7–10 | 12 | 9 | 11 |

- The ODE structure can describe a batch to about 5–6% on cell densities and 4% on substrate. The cell-density error of 5.5% is close to the count noise estimated from the data (about 6%).
- The grey box closes about 40% of the gap between the shared-rate ODE and the oracle. The rest is information about the batch that its feed and volume record does not carry.
- With squared loss, byproduct looked like the variable the equations describe worst (7.1%, rising to 11–12% on days 11–13).

### Robust oracle

A squared loss lets a single aberrant sample drag the whole fitted curve; batches 29 and 34, whose byproduct triples on day 13, are the clearest case. Refitting the oracle with a soft-L1 loss (scale 0.1 in log units, iteratively reweighted):

| Oracle | VCD | TCD | Substrate | Byproduct | Score |
|---|---|---|---|---|---|
| Squared loss | 5.6 | 5.4 | 4.0 | 7.1 | 5.54 |
| Robust loss | 5.5 | 5.2 | 3.9 | 5.9 | 5.12–5.16 |

Byproduct falls from 7.1 to 5.9% (from 6.2 to 5.7% without batches 29 and 34), about the per-sample noise implied by consecutive differences (about 6%). With the batch known, the version-2 equations describe all four variables to roughly measurement noise. The forecast's remaining error (Score 7.91) is lack of information about each batch.

## Structural diagnosis

### Method

If the per-batch rates are fitted to the batch's own data (the oracle), any error left that is the same for all batches comes from the equations. Two checks:

1. Compare the median signed error by day of the shared-rate ODE and of its oracle. With version 1 they correlate at 0.89–0.91 across days for all four variables: the day pattern is structural.
2. Test each candidate change both with shared rates and with the oracle, and look at the by-day error before and after.

### Changes tested

| Change to the equations | Shared rates | Oracle | Effect |
|---|---|---|---|
| Version 1 ($K_q = 0.05$ fixed) | 9.45 | 5.65 | Oracle substrate error on day 11: 12.6% |
| **Version 2: maintenance uptake saturable, $K_q$ fitted** | **9.28** | **5.54** | **Oracle day-11 substrate error 2.8%. Grey box 7.91: a tie** |
| Extra overflow uptake above a substrate threshold | 9.41 | 5.39 | Best oracle; threshold not stable across folds |
| Late substrate shots dose to a setpoint (event rule) | 9.17 | 6.00 | Helps when the batch's level is unknown, hurts when known |
| Decaying pulse after each Feed D dose on byproduct production, substrate uptake or byproduct consumption | 9.44–9.48 | 5.68 | Byproduct day pattern unchanged |
| Feed D or Feed C carrying substrate or byproduct; pulse after each Feed B dose; sharper substrate switch on consumption; consumption held back during growth | 9.40–9.50 | – | Within noise |
| Death rising at low substrate; byproduct released with death or formed again at low substrate | 9.40–9.44 | 5.46–5.65 | Within noise |
| Unlogged volume as evaporation (water leaves, all states concentrate) | 9.84 | 5.85 | Worse, mainly on cell densities |
| Overflow switch as a sigmoid or Weibull curve in time, shape fixed from the measured rates | 9.32 | – | Within noise |

### The byproduct pattern

Byproduct sits above the model on days 6–7 and below it on days 8 and 11, in all five folds. Per batch:

- Consecutive changes of the byproduct residual within a batch correlate negatively (about −0.3 to −0.55), the signature of independent error on each sample rather than of a process.
- Feed D dose, Feed B dose, Feed C and unlogged volume show no relation to the residual change on the same day (|ρ| < 0.1).
- The byproduct and substrate residual changes move together (ρ = 0.34).
- The common day pattern is about 20% of the oracle's byproduct error; the implied per-sample noise is about 6% on days 1–10.

Reading: a day-level offset common to all batches plus sample-level noise. No feed-driven mechanism was found. Correcting the pattern per day helps substrate but not byproduct, whose batch-to-batch spread is wider.

Further byproduct structures, all tested with shared rates and with the oracle:

| Change | Shared rates | Oracle | Effect |
|---|---|---|---|
| Consumption switched on as the overflow driver G runs out | 9.28 | – | fit turns the switch off |
| Byproduct production with its own driver \(G^r\), decoupled from substrate overflow | 9.28 | 5.44 (squared loss) | fitted \(r\) 1.2–1.5; gain almost entirely from batches 29 and 34 |
| First-order delay between the overflow burst and byproduct production, per batch | 9.28 | 5.21 (robust) against 5.16 | delay shrinks to its minimum in every batch |

No lagged dependence was found: on days 6–13 the byproduct residual is unrelated to cell density, substrate, byproduct, other residuals or Feed B 1 to 7 days earlier (mean |ρ| ≤ 0.2).

### Coupling of the per-batch rates

From the information matrix of the eleven per-batch log-rates (version 2, per batch, data only):

| Coupled pair | Correlation of estimates |
|---|---|
| Byproduct consumption ↔ byproduct per unit growth | +0.99 |
| Maintenance uptake ↔ substrate content of Feed B | +0.95 |
| Overflow uptake ↔ overflow decay | +0.94 |
| Byproduct per growth ↔ byproduct per overflow | −0.84 |
| Death ↔ lysis | −0.65 |

Eight of the eleven directions carry less than 1/100 of the information of the best-informed one. Individual per-batch rates are therefore not identifiable; they should be read as combinations (net byproduct turnover, net substrate balance). Restricting the per-batch adjustment to the best-informed directions loses accuracy:

| Directions fitted per batch | 3 | 4 | 6 | 11 (learned prior) |
|---|---|---|---|---|
| Robust oracle | 6.79 | 6.67 | 5.97 | 5.12 |
| Grey-box forecast | 8.29 | 8.22 | 8.07 | 7.93 |

The learned prior already regularises the weakly informed directions.

### Dosing events as measurements

Late substrate shots top up to a target (implied level after the dose 2.02 g/L on day 11, 2.10 g/L on day 12, CV 9–11%), so the dose size carries information on the substrate level before it. Each shot was added to the grey box as a pseudo-observation of the model's substrate just before the dose:

$$
\log S_b(t_{\text{shot}}^-) \sim \mathcal N\big(a_{\kappa} + b_{\kappa}\,v_{\text{shot}},\ \sigma_{\kappa}^2\big), \qquad \kappa \in \{\text{early}, \text{day 11}, \text{day 12}\},
$$

with \(a, b, \sigma\) learned on the training batches of each fold from their robust-oracle trajectories at their shot times (\(\sigma\) 0.09–0.12 in log units).

| Grey box | Substrate | Substrate, early-shot batches | Score |
|---|---|---|---|
| Without event observations | 8.71 | 11.87 | 7.92 ± 0.22 |
| With event observations (weight ×1) | 8.65 | 11.63 | 7.90 ± 0.26 |
| With event observations (weight ×2) | 8.67 | 11.52 | 7.91 ± 0.29 |
| Event observations + substrate day correction | 8.19 | – | 7.79 ± 0.24 |

No gain. The events move the substrate forecast by about 1% at most. The data-driven target already uses the shot sizes (its day-11 substrate error is 17.1%, against 20.8% for the dose rule alone), so the events add little new information.

## History matching

History matching asks which rate sets are not ruled out by the evidence, instead of looking for one best set [2, 3]. Because the ODE is cheap to simulate, no emulator is needed. It was run with the version-1 ODE and an earlier data-driven target (Score 7.49), and has not been rerun for version 2.

For each held-out batch:

1. Draw 300 candidate rate sets $\theta$ from the learned prior $\mathcal N(0, \Sigma)$.
2. Simulate the ODE for each.
3. Compute the implausibility

$$
I(\theta) = \max_{k,j}\; \frac{\big|\log \tilde y_b(k,j) - \log f_j(k;\theta)\big|}{\sqrt{R_{kj} + D_{kj}}}
$$

where $R_{kj}$ is the error variance of the data-driven forecast and $D_{kj}$ the model discrepancy: what the ODE misses on training batches even with rates fitted to the batch itself.

4. Keep the rate sets with $I < 3$. Refocus the draws on the least implausible tenth and repeat, for three waves.

| Estimate of the per-batch rates | VCD | TCD | Substrate | Byproduct | Score ± fold SD |
|---|---|---|---|---|---|
| MAP estimate (grey box, same target) | 6.8 | 6.7 | 8.6 | 10.0 | 8.01 ± 0.33 |
| History matching, median of the kept set | 7.1 | 6.9 | 8.8 | 10.5 | 8.36 ± 0.39 |
| History matching, likelihood-weighted mean of all draws | 7.0 | 6.8 | 8.5 | 10.1 | 8.09 ± 0.37 |

- About 92% of the prior draws are not ruled out: the data-driven forecast is too uncertain to exclude much of the prior.
- As a point forecast it is therefore slightly worse than the MAP estimate.
- Its use is the range: the 90% band covers 85–86% of held-out measurements, with a half-width of 13% for cell densities and 17–18% for substrate and byproduct.

## Limitations

- **Not better than its target.** The grey box scores 7.91 against 7.30 for the data-driven forecast it is matched to. Its value is that every forecast obeys the mass balances and comes with per-batch rates.
- **Version 2 is a structural, not a predictive, improvement.** It removes the day-11 substrate error when the batch is known; the forecast Score is a tie with version 1.
- **Per-batch rates are inferences.** They are estimated from the inputs through a data-driven forecast. For about one batch in five the forecast gets worse.
- **Byproduct carries a day pattern no tested structure removes.** With a robust fit the oracle reaches 5.9% on byproduct, about measurement noise; the forecast's 9.4% is mostly lack of information about the batch.
- **Early-shot batches are the weakest regime.** Substrate error 11.8% against 7.1% (7 training batches); the hidden set has 3 of 12 such batches (6, 35, 54).
- **Per-batch rates are strongly coupled** (see [Coupling](#coupling-of-the-per-batch-rates)); only combinations of them are identifiable.
- **Two hidden states.** $N$ and $G$ are not measured. They are identified only through their effect on growth, substrate and byproduct.
- **Feed contents are fitted.** The substrate content of the feeds is not given; $c_{sub}$ and $c_B$ are estimates. Feed C and Feed D enter only through dilution.
- **Units are inferred,** as are the identities of the substrate and the byproduct. The interpretation of $K_q$ in mM assumes the substrate is glucose.
- **Late byproduct jumps are not predictable.** Two training batches (29 and 34) show byproduct tripling on day 13; nothing in the inputs anticipates it.
- **Selection bias.** The ODE structure and the data-driven feature set were chosen after looking at all 49 training batches, so the held-out Scores are somewhat optimistic. The choice of models per variable is nested in the folds.

## Reproducibility

The code is not in this repository yet. The scripts used for the numbers above:

| Script | Purpose |
|---|---|
| `model.py` | data loading, time grid, feed events |
| `cand.py` | ODE structures (version 2 is `S5` with the flag `qsS`, registered as `M5`), simulator, least-squares fit, per-batch multipliers |
| `cand2.py` | event-driven byproduct terms tested in the structural diagnosis |
| `run_eval.py` | cross-validation of an ODE structure with shared rates |
| `dd_bench.py` | benchmark of the data-driven models |
| `dd_combine.py` | nested choice of the best models per variable (`dd-best`) |
| `greybox_eb2.py` | grey box with learned weights for any ODE structure (the model described here) |
| `sub_day_correct.py` | substrate day correction, nested in the folds |
| `oracle_struct.py` | oracle per structure |
| `history_match.py` | history matching for the per-batch rates |

Requirements: Python 3, NumPy, pandas, SciPy, scikit-learn, LightGBM (needs `libomp` on macOS), pyGAM.

To produce a forecast for the hidden batches, refit the shared ODE parameters and the data-driven models on all 49 training batches, learn $\Sigma$ and $R$ from them, then run the grey-box step for each hidden batch. This has not been done yet.

## References

1. López-Meza J. et al. (2015). Using simple models to describe the kinetics of growth, glucose consumption, and monoclonal antibody formation in naive and infliximab producer CHO cells. *Cytotechnology*. Abstract read only.
2. Craig P. S., Goldstein M., Seheult A. H., Smith J. A. (1997). Pressure matching for hydrocarbon reservoirs: a case study in the use of Bayes linear strategies for large computer experiments. *Case Studies in Bayesian Statistics III*.
3. Vernon I., Goldstein M., Bower R. G. (2010). Galaxy formation: a Bayesian uncertainty analysis. *Bayesian Analysis* 5(4).
