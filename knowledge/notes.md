# Knowledge Notes: Lottery Randomness, Bias Detection, and Reproducible Inference

## 1. Scope and Corpus Composition
This corpus is intentionally mixed across (a) lottery-domain empirical studies, (b) statistical foundations for dependence/stationarity/change-point/FDR control, (c) randomness-testing standards, and (d) current operational/regulatory sources.

Core project dataset anchor: `S39` (workspace historical draw file, 1986-2026).

## 2. Formal Null Models and Combinatorial Baselines
For k-of-n draws without replacement, the canonical jackpot probability is:
- `P(jackpot) = 1 / C(n,k)` (see `S35`, `S34` for official rule anchors).

Ordered-statistics framing (`S30`) motivates rank-based descriptors invariant to permutation of drawn numbers. This is useful when testing whether observed anomalies are genuine physical/procedural effects rather than artifacts of representation.

## 3. Independence and Stationarity Diagnostics
Linear dependence diagnostics:
- Ljung-Box (`S05`): `Q = n(n+2) * sum_k rho_k^2/(n-k)`.
- Box-Pierce (`S06`): `Q_BP = n * sum_k rho_k^2`.

Nonlinear dependence:
- BDS (`S07`): `W_{m,eps} = sqrt(N)(C_m(eps)-C_1(eps)^m)/sigma_hat`.

Stationarity/unit-root complement:
- Dickey-Fuller/ADF (`S01`,`S02`): `Delta y_t = gamma y_{t-1} + ...`.
- KPSS (`S03`): stationarity-null diagnostic using partial sums.
- PP (`S04`): nonparametric correction for serial correlation/heteroskedasticity.

Cross-source synthesis:
- `S01`/`S02` and `S03` should be run jointly. If both reject their respective nulls, infer model mismatch or structural instability rather than simple stationarity/nonstationarity dichotomy.
- `S05`/`S07` combination captures both linear and nonlinear dependence.

## 4. Regime Shifts and Operational Transition Detection
Foundational break models:
- Bai-Perron (`S10`,`S11`) for multiple unknown break dates in linear models.
- CUSUM/CUSUMSQ (`S08`,`S09`) for parameter stability.
- PELT (`S12`) for scalable exact penalized segmentation.
- Practical software support (`S13`,`S37`).

Cross-paper insight:
- `S12` + `S37` is a practical implementation bridge for long lottery histories.
- `S10`/`S11` offer interpretable econometric break tests that can validate PELT findings.

## 5. Multiple Testing and False Discovery Control
Base control:
- BH (`S15`) step-up rule for FDR.

Dependence-robust and adaptive variants:
- BY (`S16`) for arbitrary dependence (conservative).
- q-value/pFDR (`S17`,`S18`) for power-oriented ranking.
- Empirical-null perspective (`S19`) when theoretical null calibration drifts.
- Recent methods (`S25`,`S26`) provide modern robust alternatives.

Actionable synthesis for this project:
1. Use BH as baseline.
2. Run BY and/or empirical-null sensitivity analyses where feature dependence is strong.
3. Report both adjusted significance and effect-size stability across held-out eras.

## 6. Randomness Test Batteries and RNG Baselines
Standards and tooling:
- NIST STS (`S20`) and TestU01 (`S21`) define complementary battery philosophies.
- Updated/usable code (`S38`) aids reproducible execution.
- Modern extractor-enhanced RNG testing (`S24`) stresses explicit entropy assumptions.

Important methodological caveat from `S20` and `S24`:
- Passing statistical tests is necessary but not sufficient for proving true randomness.

## 7. Lottery-Domain Empirical Evidence
Direct lottery-focused papers:
- Romanian 6/49 fairness study (`S29`) found no significant global marginal-frequency deviation at 0.05 in its sample.
- Ordered-statistics perspective (`S30`) provides descriptor design ideas.

Operational context and metadata sources:
- UK draw pages include machine/ball-set metadata (`S36`) useful for mechanism-level covariates.
- Regulatory/operator annual reports (`S32`,`S33`) provide governance context but are not standalone statistical proof of fairness.

## 8. Open Gaps and Risk Flags
1. High-quality peer-reviewed literature directly on physical draw-machine bias remains thin relative to generic randomness-testing literature.
2. Mandatory related-link seed (`/api/mock/files/...`) could not be fetched directly; local attachment (`S39`) was used as cached source.
3. Operational latent variables (ball wear/mass, maintenance logs, operator actions) remain unobserved.

## 9. Suggested Evidence Hierarchy for Downstream Phases
1. Primary inference on `S39` using null-consistent combinatorics + dependence/stationarity diagnostics (`S01-S07`).
2. Regime segmentation by `S10-S12` and software-backed replication (`S13`,`S37`).
3. Multiplicity control with BH/BY/q-values and robustness checks (`S15-S19`,`S25-S26`).
4. External validity checks using official sources (`S34-S36`) and recent lottery study (`S29`).
