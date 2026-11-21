# Bioreactor trajectory prediction (SDSC STEM4Health, Project 1)

This README records what we did on 2026-11-21, in order, and why: each step's hypothesis, the metric that decided it, and the
decision. It also describes the repository as it stands, so a new teammate or agent can continue from here.

The hackathon platform guide (Renku sessions, git workflow) is in [RENKU_GETTING_STARTED.md](RENKU_GETTING_STARTED.md).

---

## TL;DR

| | |
|---|---|
| **Task** | For 12 test bioreactor batches, predict `vcd`, `tcd`, `substrate`, `byproduct` at days 1-13, given only the day-0 sample, the feed events and the recorded volume. |
| **Metric** | MAPE per variable over all 156 predicted points, then the plain mean of the 4 (lower is better). Submissions are not limited. |
| **Current best model** | Ridge + gradient-boosting ensemble on log ratios to day 0, with **volume-shortfall features** for substrate/byproduct and **substrate caps** from the `feed_substrate` controller. |
| **Cross-validated score** | **8.11 % mean MAPE** (team folds), 8.10 % (stratified folds), **7.95-8.01 % test-weighted** (folds reweighted to the test set's feeding mix). |
| **Starting point** | Median-trajectory baseline 10.33 %; mechanistic ODE 14.09 %. |
| **Submission files** | [`testing/submission_ensemble_caps_volume_shortfall.csv`](testing/submission_ensemble_caps_volume_shortfall.csv) (ens-rb v4 alone, CV 8.11); [`testing/submission_split_team_cells_ensrb_chem.csv`](testing/submission_split_team_cells_ensrb_chem.csv) (CV 7.71) and [`testing/submission_blend50_ensrb_team.csv`](testing/submission_blend50_ensrb_team.csv) (CV 7.82), both combined with the team's median ensemble (step 17). All generated and checked, **not yet submitted**. |
| **Reference** | Another team's leaderboard score: 7.96 % (vcd 7.64, tcd 7.60, substrate 8.18, byproduct 8.44). |

---

## 1. Repository map

```
.
├── README.md                    ← this file: chronology, decisions, current state
├── RENKU_GETTING_STARTED.md     ← original hackathon/Renku platform instructions
├── environment.yml              ← conda environment (pandas, scikit-learn, scipy, seaborn, …)
├── data/                        ← NOT in git (.gitignore); input data, see §2
│   ├── bioreaction_remaining.csv    event log of all 61 batches
│   └── bioreaction_template.csv     submission template (12 test batches × 13 time points)
├── testing/
│   ├── data_overview.ipynb      ← exploratory analysis (sections 1-8)
│   ├── model_playground.ipynb   ← validation harness + every model + submission writer
│   ├── submission_ensemble_caps_volume_shortfall.csv   ← ens-rb v4 alone
│   ├── submission_split_team_cells_ensrb_chem.csv      ← team ensemble for VCD/TCD, ens-rb v4 for substrate/byproduct (step 17)
│   └── submission_blend50_ensrb_team.csv               ← 50/50 geometric blend with the team ensemble (step 17)
├── src/main.py                  ← empty placeholder (all code currently lives in the notebooks)
└── init.d/                      ← Renku session init script (AI credentials)
```

**Both notebooks are self-contained.** All code is in their cells, they read data from `../data/`, and they're saved with outputs.
Run them from `testing/` with a Python 3 kernel that has the packages in `environment.yml`.

| Notebook | Runtime | What it contains |
|---|---|---|
| `data_overview.ipynb` | ~1 min | Data structure, train/test split, trajectories, feeds, the feeding-identifier analysis (§5b), clustering (§8), volume balance (§9) |
| `model_playground.ipynb` | ~7 min (the ODE takes most of it) | Validation harness (§1), baseline (§2), ODE (§3-3b), data-driven models (§3c-3f), comparisons (§4, §4b), submission (§5) |

---

## 2. Data and rules

- **`bioreaction_remaining.csv`** (2,858 rows) is a long *event log* for 61 fed-batch runs of ~13 days. Each row is one of:
  - a **sample**: `vcd`, `tcd`, `substrate`, `byproduct`, `volume`;
  - a **feed event**: exactly one of `feed_substrate`, `feed_b`, `feed_c`, `feed_d`;
  - a **hidden sample** (`to_predict = True`): only `volume` is given.
- **49 training batches** have ~14 daily samples. **12 test batches** (Batch5, 6, 12, 24, 32, 35, 40, 48, 54, 56, 57, 60) have only
  their day-0 sample measured. Their 13 hidden samples match the 156 template rows exactly.
- **Feed columns are volumes.** Measured volume = day-0 volume + cumulative feed volume (r = 0.9999), minus a small loss.
- **Allowed inputs for a prediction:** the day-0 sample, all feed events, and the recorded volume at each timestamp.
  No measurement after day 0 may enter a prediction.
- **Scoring:** see the TL;DR. MAPE is asymmetric: under-prediction costs at most 100 %, over-prediction is unbounded. So small true
  values (early VCD, late substrate, late byproduct) are expensive to get wrong.

---

## 3. How we evaluate (fixed early, used for every number below)

| Choice | Setting | Why |
|---|---|---|
| Unit of splitting | **Whole batches**, never rows | Rows of one batch are strongly correlated; row-level splits leak. |
| Folds | **Team folds**: 5 folds of 10/10/10/10/9 batches (49 batches permuted with seed 42) | Agreed with the team, so scores are comparable across members. |
| What a held-out batch shows the model | Day 0 + feeds + volume | Mirrors the test set exactly. |
| Score | **Pooled MAPE** over all 637 held-out points (49 × 13), per variable, then mean | Same formula as the leaderboard. |
| Secondary scores | Stratified folds (by feeding identifier); **test-weighted** MAPE (batches reweighted so feeding groups match the test set) | The team folds are unevenly mixed; the test set is over-represented in a rare group (§4, step 2). |

**How noisy is the leaderboard?** A score on 12 batches is noisy. Resampling 12-batch sets from our held-out predictions gave a
90 % range of about **±0.7 points** around the expected score (for the ensemble at the time: 7.8-9.3, median 8.5).
Differences below ~0.1 in cross-validation are within fold noise and were not treated as real.

---

## 4. Chronology: what we did, why, and what the metric said

Scores are mean MAPE on the team folds unless stated. ✅ = adopted, ❌ = rejected, 🔎 = finding used later.

### Step 1: Understand the data 🔎
*Notebook: `data_overview.ipynb` §1-7*

- The file is a sparse event log, not a table. Most cells are empty by design.
- All batches follow almost the same trajectory shape:
  - cells grow exponentially to a peak around day 7 (~23-39), then decline;
  - viability stays > 98 % until day 6, then falls to 84-92 %;
  - substrate is consumed early, and byproduct peaks around days 4-5, then is consumed.
- Each feed has its own fixed pattern:
  - `feed_b`: daily, dose ramps up and drops at day 10;
  - `feed_c`: 8 fixed large doses plus 3-27 irregular small ones;
  - `feed_d`: days 5/8/10, identical in every batch;
  - `feed_substrate`: mostly at days 11.5 and 12.5.
- Test batches look like training batches in their initial conditions, feeds and volume.

**Consequence:** the hard part isn't the average shape. It's each batch's small deviation from it.

### Step 2: "Are they testing different feeding regimens?" ❌ as designed regimens, 🔎 as information
*Notebook: `data_overview.ipynb` §5b*

- **Hypothesis (team):** batches test different feeding regimens, which explain different VCD/TCD.
- **Tests:**
  - `feed_b` has one continuous ±5 % spread (k-means silhouette ≤ 0.26, i.e. no clusters).
  - `feed_d` is identical in every batch.
  - `feed_substrate` takes discrete values, **but it's a feedback controller**: a dose follows exactly when the substrate measured just
    before is below a threshold. On days 7-9: dose ⇔ substrate ≤ 1.91 (else ≥ 1.99). On days 11-12: high dose ⇔ ≤ 0.96 (else ≥ 0.97).
    No exceptions in 49 batches.
  - The resulting 6 "feeding identifiers" don't differ in VCD/TCD (group means within ~1.5 units, within-group spread 2-4).
- **Decision:** there are no designed regimens in the data. The identifier is a *footprint of the hidden substrate*, which made it
  useful later, both for stratifying folds and for constraining predictions (step 10).

| Feeding identifier | Train | Test |
|---|---|---|
| no early dose · high/low (HL) | 26 | 5 |
| no early dose · low/low (LL) | 8 | 2 |
| early dose · LL | 4 | **3** |
| no early dose · LH | 5 | 2 |
| no early dose · HH | 3 | 0 |
| early dose · HL | 3 | 0 |

### Step 3: Generalist vs specialist models ✅ generalist
- **Question:** one model for all feeding types, or one per type?
- **Reasoning:** groups have 3-26 training batches, the outcomes don't differ between groups (step 2), and the rarest group
  ("early · LL", 4 training batches) makes up 25 % of the test set. Specialists would be weakest exactly there.
- **Decision:** one global model, with the feeding information given as *input features*.

### Step 4: Validation design ✅ team folds + test-weighted score
*Notebook: `model_playground.ipynb` §1, §4b*

- Adopted the team's 5 folds (also published on the team's results page).
- Proposed a split stratified by feeding identifier. In the team folds, fold 4 holds out 8 "HL" and 0 "LL" batches.
  - Mean scores barely change (≤ 0.1), so the folds don't bias the result.
  - The baseline's fold-to-fold SD drops from 0.99 to 0.25, so small improvements become easier to see.
- **Decision:** keep the team folds as primary (comparability) and report stratified and test-weighted scores alongside.

### Step 5: Mechanistic ODE fitted with `scipy.optimize.least_squares` (+ basin hopping) ❌
*Notebook: `model_playground.ipynb` §3-3b*

- **Model:** viable/dead cells, substrate and byproduct, with feeds as volume boluses (dilution + substrate added).
  14 shared parameters in log space, relative-error residuals, `soft_l1` loss.
- **Result:** **14.09 %**, worse than the median-trajectory **baseline (10.33 %)**. Basin hopping changed the cost by −0.0002 %, so the
  problem isn't a local minimum but the model structure:
  - the density cap stops growth too early;
  - early substrate consumption is too slow;
  - several parameters end at their bounds.
- The team's independent ODE scored 14.08 %, which suggests a limit of the ODE approach on this data, not a bug in either implementation.

### Step 6: Data-driven models on log ratios to day 0 ✅
*Notebook: `model_playground.ipynb` §3c*

- **Idea:** every batch follows the same shape, so predict each batch's *deviation from its own start*, `log(y_day / y_day0)`,
  from 36 features: day-0 values, `feed_b` per day, `feed_c` count and total, `feed_substrate` flags, and volume added per day.

| Model | vcd | tcd | substrate | byproduct | **mean** |
|---|---|---|---|---|---|
| Baseline (median ratio) | 8.33 | 8.23 | 11.54 | 13.22 | 10.33 |
| Ridge (RidgeCV, α chosen inside each training fold) | 7.71 | 7.67 | 9.08 | 10.42 | 8.72 |
| Gradient boosting (fixed, untuned settings) | 7.67 | 7.69 | 10.04 | 9.74 | 8.78 |
| **Ensemble (geometric mean)** | 7.43 | 7.40 | 9.12 | 9.63 | **8.39** |

- **Decision:** the ensemble becomes the main model. The two members make different mistakes (ridge is better on substrate, boosting on byproduct).

### Step 7: Cluster the 61 batches and train per cluster? ❌
*Notebook: `data_overview.ipynb` §8*

- **Hypothesis (team):** clusters link test batches to similar training batches; train one model per cluster.
- **Tests:**
  - Silhouette ≤ 0.31 on day-0 values, feeds or both, with no natural number of clusters.
  - Full training trajectories score 0.13.
  - Per-cluster models score **9.37-10.48** vs **8.71** for one global ridge, and nearest-neighbour averaging scores 9.6-10.1.
  - Test batches sit *inside* the training cloud (median distance to nearest training batch: 4.2 vs 4.4 between training batches).
- **Decision:** the batches form a continuum. Splitting the data costs more than local similarity gains. The global model already
  uses similarity through its features.

### Step 8: Gaussian process as a third ensemble member ❌
*Notebook: `model_playground.ipynb` §3d*

- **Result:** best kernel (linear + RBF + white noise): **8.70** on its own, but the 3-member ensemble scores **8.43** vs 8.39 without it.
- **Why:** the marginal likelihood shrinks the RBF part to almost nothing, so the GP becomes Bayesian ridge.
  Its errors correlate **1.00** with ridge's, so it adds no new information.

### Step 9: Reality check against the leaderboard 🔎
- Our pooled MAPE uses the same formula as the leaderboard.
- Expected leaderboard score at that point: ~8.5 (12-batch noise ±0.7; the test set's large "early · LL" share makes it slightly harder).
- Against the other team (7.96): we were level on VCD/TCD, but behind by ~0.9 on **substrate** and ~1.5 on **byproduct**.
- **Decision:** focus everything on substrate and byproduct.

### Step 10: Substrate caps ✅ and per-day `feed_c` features ❌
*Notebook: `model_playground.ipynb` §3e*

- **Caps:** the `feed_substrate` rules (step 2) are hard bounds on hidden values, and the feed events are known for test batches.
  A wrapper clips any substrate prediction that breaks a rule to the threshold, which is re-learned on each training fold.
  - Result: substrate 9.12 → 8.83, mean **8.39 → 8.32**.
  - The uncapped model broke the rules at 16 of 245 constrained points. ✅
- **Per-day small-`feed_c` counts** (possible pH control): 8.53 per day, 8.42 cumulative, vs 8.39 without. ❌

### Step 11: Two team hypotheses about feeds ❌
*Scratch experiments, not in the notebooks*

- **"`feed_c` is pH control driven by VCD; remove it."**
  - Small doses do respond to the culture's state: they come when VCD and byproduct are lower (within-day r = −0.26, −0.19).
    They don't predict the next day's change (|r| ≤ 0.06).
  - But a reactive input known for test batches still carries information. Removing it: 8.31 → 8.35. Kept.
- **"`feed_d` causes a metabolic shift after day 11."**
  - `feed_d` is identical in every batch (dose variation ~1 %, timing ±0.8 h), so its effect can't be separated from time.
  - Adding its amount and timing as features: 8.48 vs 8.31. Not used.

### Step 12: More mechanism for substrate and byproduct ❌
*Scratch experiments, not in the notebooks*

| Attempt | substrate | byproduct | mean |
|---|---|---|---|
| Current best at the time (ensemble + caps) | 8.84 | 9.62 | 8.31 |
| Median (L1) loss for boosting, aligned with MAPE | 8.93 | 9.89 | 8.40 |
| Chained: predicted VCD → predicted integrated VCD as a feature | 9.24 | 9.72 | 8.44 |
| ODE predictions as features (hybrid) | 8.72 | 10.01 | 8.38 |

- **Diagnosis:**
  - The remaining substrate/byproduct error is **unbiased**: median signed error ≈ 0 on every day. It's batch-to-batch scatter, not a wrong shape.
  - Days 11-13 hold ~40 % of it.
  - `feed_substrate` dose amounts are exactly proportional to volume (dose/volume constant to 0.1 %), so they carry nothing beyond high vs low.
- **Conclusion:** more equations can't help without more *information* per batch.

### Step 13: Volume shortfall, a hidden signal ✅ (current best)
*Notebook: `model_playground.ipynb` §3f*

- **Idea:** recorded volume is an allowed input. Measured volume minus (day-0 volume + cumulative recorded feeds) is the *shortfall*.
  It averages −0.43 at day 13 and varies between batches.
- **Finding:**
  - A smaller shortfall (more volume than the recorded feeds explain) goes with more byproduct on every day from 3 to 13 (r = 0.35-0.60).
  - It also goes with less substrate on days 3-7 (r down to −0.55).
  - **Mechanism:** most likely unlogged pH titrant (base). Step 14 shows where this volume enters and what it tracks.
- **Model:** the shortfall on days 1-13 is added as 13 features to the **substrate and byproduct** models only (for VCD/TCD it added noise).

| Score type | vcd | tcd | substrate | byproduct | **mean** |
|---|---|---|---|---|---|
| Team folds, pooled | 7.41 | 7.37 | 8.59 | 9.02 | **8.10** |
| Stratified folds, pooled | 7.36 | 7.29 | 8.39 | 9.36 | **8.10** |
| Team folds, test-weighted | 7.50 | 7.46 | 8.23 | 8.87 | **8.01** |
| Stratified folds, test-weighted | 7.37 | 7.30 | 8.06 | 9.07 | **7.95** |

### Step 14: Volume balance: where does the untracked volume come from? 🔎
*Notebook: `data_overview.ipynb` §9*

- **Question:** is there volume entering the reactor that no feed column records?
- **Method:** volume is recorded only at the daily samples, so the balance is closed per sampling interval:
  residual = measured ΔV − all logged feeds in the interval. Sampling and evaporation make it negative; a positive residual
  is untracked volume.
- **Finding:**
  - On every interval except one, the residual is negative in essentially every batch (about −0.016 per day early,
    −0.05 per day from day 6).
  - On the interval ending at **day 3 it is positive in all 61 batches**. Above each batch's own early loss rate the
    addition is ≈ +0.06 on day 3 and ≈ +0.01 on day 4 (84 % of batches), and gone by day 5. Total ≈ 0.07 per batch (≈ 0.7 % of volume).
  - It tracks byproduct made in the same interval (Spearman ρ = 0.56) and peak byproduct (0.47), not cell density (0.22).
  - Test batches show the same addition (0.074 vs 0.076).
- **Interpretation (hypothesis):** base pumped in by pH control while lactate (the presumed byproduct) builds up fastest,
  and not logged as a feed. It measures **early** acid production; it carries little information after day 5.
- **Decision:** the cumulative shortfall features (step 13) already contain it. Isolating the addition as separate features
  (days 3/4/5, total, early and late loss rates) scored 8.20 added to them and 8.21 instead of them, vs 8.11. Not adopted.

### Step 15: Days 11-13 and the lactate hypothesis ❌ (as a predictor), 🔎 (as biology)
*Scratch experiments, not in the notebooks*

- **Hypothesis (team):** the byproduct is lactate made from the substrate (glucose), and the late glucose feeding
  (`feed_b` halves at day 10; `feed_substrate` shots at 11.5 and 12.5) drives the large day 11-13 errors.
- **Where the error is (current best, out-of-fold):** byproduct MAPE 10.0 / 11.8 / **20.8** and substrate **19.7** / 13.4 / 14.6
  on days 11 / 12 / 13. Days 11-13 hold 36 % of all byproduct and 43 % of all substrate error. Median signed error ≈ 0: scatter, not bias.
- **The lactate reading is supported:** on days 1-2, byproduct made per unit substrate consumed is 0.83-0.90 g/g, near
  the glycolytic maximum of 1. Byproduct is made until day 4 and consumed from day 7 (the lactate shift).
- **But glucose doesn't drive the late error.** Oracle tests (true hidden values given to the substrate/byproduct models,
  on purpose, to bound what any feature could achieve):

| Information added | byproduct MAPE d11-13 | byproduct MAPE |
|---|---|---|
| none (current best) | 14.2 | 9.02 |
| true glucose, days 10-13 | 14.4 | 9.09 |
| true VCD, days 10-13 | 14.4 | 9.02 |
| true byproduct at day 2 / 4 / 6 / 8 / 10 | 14.5 / 14.0 / 13.7 / 12.2 / **11.1** | 8.69 / 8.17 / 8.30 / 7.94 / **7.64** |
| allowed late feed features (shot / volume, `feed_b` drop, late titrant increments) | 14.5 | 9.34 (mean 8.20) |

- **Diagnosis:**
  - The late byproduct error is carried over: day-10 and day 11-13 errors correlate 0.63-0.74.
  - The level is set during **consumption** (days 6-10), not production: knowing day 2-4 byproduct barely helps.
  - True glucose explains R² = 0.09 of log(B₁₃ / B₁₀); adding true B₁₀ raises it to 0.29. Late consumption depends on the
    byproduct level itself (first-order), not on glucose.
  - No allowed input explains the per-batch consumption rate: in a scan of ~60 candidates (`feed_c` timing, titrant
    per window, `feed_b` profile), the best |ρ| with the late error was 0.31, about the chance maximum for n = 49.
- **Two outliers:** Batch29 and Batch34 roughly triple their byproduct between days 12 and 13 (0.57 → 1.71, 0.65 → 1.86);
  no other batch rises more than 12 %, and neither shows extra titrant volume. Possibly measurement anomalies.
- **Decision:** the late error is information-limited. Keep the late glucose → lactate coupling for the mechanistic
  explanation (ODE), not as model features.

### Step 16: Smaller checks ❌, one candidate
*Scratch experiments, not in the notebooks. All on the team folds, against 8.11.*

| Attempt | Idea | mean | Decision |
|---|---|---|---|
| MAPE-optimal shift, per variable / per variable × day | MAPE rewards predicting below the median; learn a factor on the other folds | 8.20 / 8.37 | ❌ factors fit noise |
| `RidgeCV(alpha_per_target=True)` | late days are noisier, so let each day pick its own α | 8.16 | ❌ |
| Conditional caps | move a rule-breaking substrate prediction to the typical value on the correct side, not to the threshold | 8.10 | ❌ tie |
| Damp Batch29/34 day-13 byproduct **in training only** | keep two unexplained jumps from pulling every day-13 prediction up | **8.04** (byproduct 8.72, day 13: 20.8 → 17.2) | candidate, see below |

- **Why the caps help only a little (8.18 without → 8.11 with):** without caps, only 15 of 245 constrained substrate
  predictions break the controller rule, so the caps can only ever move those 15. The rule is also one-sided (e.g. "below
  0.967"), so it says nothing about how far below. The large day-11 substrate error sits in batches that already obey the
  rule. Moving the 15 to a better value (conditional caps) doesn't change that.
- **The outlier damping** is the only change that helped. It was chosen after looking at the data, and −0.07 is below the
  ~0.1 noise threshold, so it isn't adopted yet. A principled version is a robust loss (e.g. Huber) for the ridge member,
  which doesn't name batches.

### Step 17: Combining with the team's median ensemble ✅ (candidates)
*Scratch script, not in the notebooks. Team predictions taken from the team results page (`data.json`), same five folds.*

- **Finding:** the team's newer median ensemble (ODE S5 + 5 nearest batches + GP) scores **8.32 ± 0.31**, not the 8.56 of its
  older version. Against ens-rb v4 (8.11 ± 0.63) that is a tie under the team's ~0.4 rule. But the two are good at different
  things: the team ensemble on cells (VCD 6.61, TCD 6.56 vs 7.41, 7.37), ens-rb v4 on chemistry (substrate 8.59, byproduct 9.02
  vs 8.89, 11.16).

| Combination (per-fold Score, mean ± SD) | VCD | TCD | substrate | byproduct | Score |
|---|---|---|---|---|---|
| ens-rb v4 alone | 7.41 | 7.37 | 8.59 | 9.02 | 8.11 ± 0.63 |
| team median ensemble alone | 6.61 | 6.56 | 8.89 | 11.16 | 8.32 ± 0.31 |
| 50/50 geometric blend, all variables | 6.77 | 6.72 | 8.20 | 9.55 | **7.82 ± 0.40** |
| blend weight chosen on the other 4 folds (picked 0.40-0.50 every time) | | | | | 7.84 ± 0.41 |
| **per variable: team ensemble for VCD/TCD, ens-rb v4 for substrate/byproduct** | 6.61 | 6.56 | 8.59 | 9.02 | **7.71 ± 0.52** |
| as above, chemistry 75/25 ens-rb v4/team | 6.61 | 6.56 | 8.29 | 9.12 | 7.66 ± 0.44 (more tuned) |

- **Caveats:** the per-variable choice was made after seeing per-variable scores (2 options per variable group, so mild
  selection). The team's test predictions are assumed to come from models fitted on all 49 training batches.
- **Files:** `submission_split_team_cells_ensrb_chem.csv` (per-variable choice) and `submission_blend50_ensrb_team.csv`
  (50/50 blend, substrate caps re-applied: 3 values). Both pass the template checks.

### Step 18: `feed_c` timing as a byproduct signal ❌
*Scratch experiments, not in the notebooks. Motivation: the best team's leaderboard byproduct MAPE is 6.97 vs our 9.02 (CV).*

- **Timing structure:**
  - The 8 large doses are a fixed daily schedule on days 5-12, ~0.8 h after each sample.
  - The 3-27 small doses come a median 16.6 h after the last sample, spread over the day, with intervals that lengthen
    within a batch. They are not triggered by the daily sample (unlike `feed_substrate`), so they give no hard bounds.
    They look like an online controller (pH, foam or similar).
- **Signal:**
  - Per interval, dose count doesn't track byproduct made in the same interval (within-day ρ = −0.03).
  - Per batch, more or earlier small doses go with less byproduct (day 10/13 ρ ≈ −0.3 to −0.44), and ens-rb v4's
    byproduct error correlates +0.27 to +0.33 with the count. This is not driven by the most-dosed batches.
- **Tests (team folds, vs 8.11 / byproduct 9.02):**
  - Compact timing features (log count, first-dose time, counts on days 0-4 and 10-13, median gap days 6-10) for byproduct
    only: 8.12.
  - The same features for substrate and byproduct: 8.16.
  - Two features: 8.12. All 11 timing features: 8.13.
  - A cross-fitted one-slope correction of the byproduct prediction on log count: byproduct 9.02 → 8.95.
- **Conclusion:** real but weak information, mostly already used through `feed_c_n_small`. It does not explain the
  byproduct gap to the best team.

### Step 19: Do unlogged base and evaporation make the byproduct concentration uncertain? ❌
*Scratch experiments, not in the notebooks.*

- **Hypothesis (team):** unlogged base (in) and evaporation (out) change the volume without a logged feed, so the measured
  byproduct concentration (lactate mass / volume) shifts in ways the model can't follow.
- **Size of the effect** (log of measured volume / volume from logged feeds, in % of concentration):

| day | 1 | 3 | 5 | 7 | 9 | 11 | 13 |
|---|---|---|---|---|---|---|---|
| mean shift | −0.2 | +0.1 | −0.2 | −1.0 | −1.6 | −2.1 | −2.6 |
| SD between batches | 0.10 | 0.17 | 0.21 | 0.18 | 0.17 | 0.16 | 0.15 |
| model byproduct error SD | 4.9 | 7.8 | 11.9 | 12.4 | 11.0 | 13.0 | 33.0 |

- **Reading:**
  - Evaporation concentrates everything by up to 2.6 % by day 13, but almost identically in every batch (SD ≤ 0.2 %).
    The day-specific models absorb the shared part, and the batch-to-batch part is 50-200× smaller than the byproduct error.
  - The model's byproduct error doesn't track the dilution differences on any day (|ρ| ≤ 0.24, mostly ≈ 0).
  - The late net loss (sampling + evaporation, days 6-13) is −0.051 per interval with 3 % variation between batches. It is
    unrelated to byproduct (ρ = −0.01) and to the late model error (ρ = −0.01). Sampling and evaporation can't be
    separated, because interval lengths barely vary.
- **By-product:** subtracting each batch's own late loss rate gives a cleaner base estimate for days 2-5 (ρ = 0.62 with
  lactate made on days 1-5, vs 0.44 with the step-14 estimate).
- **Tests (vs 8.11 / byproduct 9.02 / byproduct days 11-13 14.2):**

| Variant | mean | byproduct | days 11-13 |
|---|---|---|---|
| + cleaner base estimate | 8.11 | 8.97 | 14.11 |
| + base estimate + late loss rate | 8.13 | 8.99 | 14.09 |
| byproduct modelled as mass (B·V), converted back with the measured volume | 8.09 | 8.92 | 13.86 |

- **Conclusion:** the volume effects are real but far too small and too uniform to explain the byproduct error. The
  measured volume already carries the dilution, and the cumulative shortfall features already carry the base signal.

### Score progression

| Step | Model | Mean MAPE (team folds) | Δ |
|---|---|---|---|
| 5 | ODE + least squares | 14.09 | |
| - | Median-ratio baseline | 10.33 | |
| 6 | Ridge + boosting ensemble | 8.39 | −1.94 vs baseline |
| 10 | + substrate caps | 8.32 | −0.07 |
| 13 | + volume-shortfall features (substrate/byproduct) | **8.11** | −0.21 |

The table also shows what we learned: the biggest gain came from changing the modelling approach (step 6). After that, gains
came from finding *information* in the inputs (controller thresholds, volume shortfall), not from more flexible models (GP,
clustering, neural networks were considered and not pursued for 49 batches) or more mechanistic detail.

---

## 5. The current best model, precisely

*Built in `model_playground.ipynb` §3c, §3e, §3f; refitted on all 49 training batches in §5.*

1. **Targets:** `log(y_day / y_day0)` for each variable and day 1-13. Prediction = `y_day0 × exp(predicted log ratio)`.
2. **Features (per batch, all known for test batches):**
   - day-0 sample: log of `vcd`, `tcd`, `substrate`, `byproduct`, plus start volume;
   - `feed_b` dose per day 1-12;
   - `feed_c`: number of small doses, total volume;
   - `feed_substrate`: early dose yes/no, day of the early dose, high dose at 11.5 / 12.5;
   - volume added by each sample day 1-13;
   - **substrate/byproduct models only:** volume shortfall at each sample day 1-13.
3. **Models:** for each variable, the geometric mean of
   - `RidgeCV` (multi-output over 13 days, standardised features), and
   - `HistGradientBoostingRegressor` (one row per batch and day, with `day` as a feature; `max_depth=3`, `lr=0.05`, `max_iter=300`, `min_samples_leaf=10`, `l2=1.0`).
4. **Post-processing:**
   - substrate caps (days 7-9: threshold ≈ 1.95; days 11-12: ≈ 0.967);
   - TCD raised to VCD where it came out lower (3 of 156 rows).
5. **Submission checks:** 156 rows, template format unchanged, no missing values, all positive, TCD ≥ VCD.

---

## 6. Guide for agents and teammates

**Adding a model.** In `model_playground.ipynb`, any class with this interface plugs into the harness:

```python
class MyModel:
    def fit(self, batch_ids): ...            # may use every row of these training batches
        return self
    def predict(self, batch_ids, query): ... # query: DataFrame[batch_id, elapsed_time]; return DataFrame[vcd, tcd, substrate, byproduct]
                                             # may only use the day-0 sample, feeds and volume of batch_ids
cross_validate(MyModel, "my model")          # team folds → RESULTS / OOF / MODELS / SPECS
```

To include it in the split comparison, add it to `COMPARE` in §4b. The submission cell (§5) automatically refits and writes the model
with the lowest mean MAPE in `RESULTS`. Wrap any model with `SubstrateCaps(base=MyModel)` to apply the caps.

**Rules and pitfalls**

- Never let a measurement after day 0 into a prediction. This includes standardisation and hyperparameters: fit them inside the
  training fold.
- Compare models on the same folds, and treat CV differences under ~0.1 as noise.
- The leaderboard scores 12 batches: expect ±0.7 around the cross-validated estimate. The test-weighted score is the closest estimate.
- Submissions are not limited, but each leaderboard reading is noisy (±0.7 on 12 batches). Choose between candidates
  with cross-validation and use the leaderboard as a check, not as the selector for differences below ~0.7.
- Feed columns are volumes. The `feed_substrate` controller and the volume shortfall are the two known sources of information
  about hidden values. Look for more like them before adding model complexity.

**Reproducibility notes**

- Every number in steps 1-8, 10, 13 and the step-14 balance is reproduced by running the notebooks.
- Step 9 (leaderboard noise simulation), steps 11-12, the step-14 feature test, steps 15-16, the GP kernel comparison table (§3d)
  and the per-cluster CV table (`data_overview` §8) came from one-off scratch scripts that are **not in the repo**. They're
  recorded here and in the notebook text. The step 14-16 scripts executed the `model_playground` code cells unchanged, so they
  share its folds, features and models.
- To re-execute a notebook from the command line: `jupyter nbconvert --to notebook --execute --inplace testing/<name>.ipynb`
  (needs `nbconvert`, which comes with `jupyterlab` from `environment.yml`).

---

## 7. Open questions and next steps

1. **Late byproduct (days 11-13)** remains the largest single error (day 13: 20.8 %). Step 15 shows it is information-limited:
   it comes from each batch's byproduct consumption rate on days 6-10, which no allowed input explains, and not from the
   glucose feeding. Remaining lever: robustness to the two day-13 outliers (step 16, Huber loss).
2. **Use the volume shortfall mechanistically:** step 14 shows the untracked volume enters on days 2-4 only, so it measures
   early acid (lactate) production. An ODE byproduct production term driven by it is possible, but it won't inform the late phase.
3. **The "early dose · LL" group** has 4 training batches but 3 of 12 test batches (Batch6, 35, 54). Batch6 and Batch54 are also the
   test batches farthest from any training batch. Expect the largest errors there.
4. **Not tried:** predicting TCD + viability instead of VCD (makes TCD ≥ VCD hold by construction); repeated CV for tighter estimates.
