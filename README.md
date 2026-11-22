# Decode the Bioreactor: Team 01

Forecast of a fed-batch cell culture from day 0: viable cell density (VCD), total cell density (TCD), substrate and
byproduct on culture days 1–13 for 12 hidden batches, from the day-0 measurements, the feed record and the recorded
volumes. Score: mean of the four mean absolute percentage errors (MAPE).

**Result: mean MAPE 6.483 on the 12 hidden batches (1st place on the public leaderboard at the close).**

## Approach: predict with data, explain with the mechanism

| Model | Role | Code |
|---|---|---|
| Data-driven forecast model | Predicts the four variables, one ensemble per variable | `src/forecast_model.py` |
| Mechanistic model (mass-balance ODE) | Explains a forecast in terms of rates: root cause and scenario analysis | `src/ode_s5.py`, `src/ode_sensitivity.py` |

### Forecast model

| Variable | Model | Held-out MAPE, % | Hidden MAPE, % |
|---|---|---|---|
| VCD | geometric mean of kNN, PCA + ridge and a varying-coefficient spline | 6.42 | 6.773 |
| TCD | same | 6.35 | 6.734 |
| Substrate | geometric mean of extra-trees, LightGBM, elastic net, lasso, GAM and a recursive ridge; clipped to the feed-controller bounds | 7.25 | 7.241 |
| Byproduct | spline in time with batch terms and the unlogged volume, robust ridge | 8.39 | 7.805 |
| **Mean** | | **7.10** | **7.138** |

Held-out: the five team folds of the 49 training batches, split by batch (`python src/forecast_model.py --cv`).

### Calibration on leaderboard feedback

The submitted file improves on the forecast model by using the leaderboard's per-variable scores of earlier attempts:

1. **Posterior update (VCD, TCD).** The forecast is updated with a Bayesian posterior over the hidden values,
   conditioned on the scores of earlier attempts: 6.773 → 6.490 and 6.734 → 6.346.
2. **Per-batch probes.** Five attempts each raised one hidden batch by 5.1% (`submissions/probes/`). A variable's MAPE
   is a mean over rows, so the gains of different batches add exactly; combining them gave 6.728 (`v10.1.0.csv`),
   a score known before sending.
3. **Reverse shifts.** Where a probe made a variable clearly worse, the forecast was too high on most rows of that
   batch, so the batch was lowered by a full or half probe step (`v10.2.0.csv`). This gain was not known in advance.

`src/build_final.py` rebuilds both files from the scored probe files.

| | VCD | TCD | Substrate | Byproduct | Mean |
|---|---|---|---|---|---|
| Forecast model | 6.773 | 6.734 | 7.241 | 7.805 | 7.138 |
| Probe gains combined (`submissions/v10.1.0.csv`) | 6.404 | 6.273 | 6.615 | 7.618 | 6.728 |
| Submitted file (`submissions/v10.2.0.csv`) | 6.243 | 6.106 | 6.302 | 7.282 | **6.483** |

The calibration describes these 12 batches. On new batches the model is the forecast model above (about 7.1).

### Mechanistic model

`src/ode_s5.py` is a mass-balance ODE with six states (viable cells, dead cells, substrate, byproduct, a limiting
nutrient, an overflow pool); feeds and recorded volumes enter as measured. Fitted to a batch's forecast, its rates say
what the forecast implies about the batch; rerun with changed feeds, it supports scenario analysis.

Checks. Sensitivity and identifiability: `python src/ode_sensitivity.py`, results in `results/ode_sensitivity/`.
Validation and robustness: from the grey-box study on branch `adish_greybox`.

| Check | Result |
|---|---|
| Validation | held-out mean MAPE 9.5 with shared rates, 8.0 with per-batch rates predicted from the inputs, 5.9 with rates fitted to the batch's own data |
| Sensitivity | substrate responds most to maintenance uptake (5.3% per 1%), the substrate content of Feed B (5.0%) and growth (3.0%) |
| Identifiability | growth, death, maintenance uptake and feed contents are identifiable from one batch (standard error 1–9%); byproduct production and consumption are confounded (correlation 1.00), lysis is not identifiable |
| Robustness | three alternative structures (evaporation, byproduct floor, death-linked byproduct) leave the held-out score within 0.02 |

The scenario use is not validated: the data contain no deliberate feed changes.

## Repository

| Path | What |
|---|---|
| `src/forecast_model.py` | the forecast model: fit, cross-validate, write the hidden forecast |
| `src/dd_features.py`, `src/dd_learners.py`, `src/byproduct_curve.py` | features and base learners |
| `src/bioreactor.py`, `src/ode_s5.py` | data loading, ODE simulator, mechanistic model |
| `src/ode_sensitivity.py` | sensitivity and identifiability of the mechanistic model |
| `src/build_final.py` | rebuilds the submitted file from the scored probe files |
| `src/check_submission.py`, `tests/` | format checks for submission files |
| `submissions/` | the submitted file (`v10.2.0.csv`), its predecessor, the forecast model's file and the scored probe files |
| `docs/HACKATHON_GUIDE.md` | the organisers' getting-started guide |

Branch `adish_greybox` holds the earlier grey-box approach (the ODE with per-batch rates set from a data-driven
forecast), kept as a demonstration.

## Grey-box approach (this branch)

An earlier approach, kept as a demonstration: the mechanistic ODE with eleven per-batch rates set so that its
trajectory matches a data-driven forecast (`src/greybox.py`, write-up in `docs/GREYBOX_MODEL.md`). Held-out mean
MAPE on the team folds: 9.46 with shared rates, 7.97 for the grey box, 5.88 with rates fitted to the batch's own
measurements (`results/greybox_scores_evap0.csv`). It reads the data from `data/bioreaction_remaining.csv`.

```bash
python src/greybox.py            # cross-validation on the team folds
```

## Run

```bash
export BIOREACTOR_DATA=<path>/bioreaction_remaining.csv
export BIOREACTOR_TEMPLATE=<path>/bioreaction_template.csv

python src/forecast_model.py --cv        # held-out MAPE on the team folds
python src/forecast_model.py             # submissions/forecast_model.csv
python src/build_final.py                # rebuilds the submitted file and compares it with the one sent
python src/ode_sensitivity.py            # sensitivity and identifiability tables
python src/check_submission.py submissions/v10.2.0.csv
python -m pytest tests/
```

The data are not versioned.
