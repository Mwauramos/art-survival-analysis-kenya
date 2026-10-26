# ART Survival Analysis — Kenya Retrospective Cohort

An epidemiology project analysing all-cause mortality among HIV-positive adults
initiating antiretroviral therapy (ART) in Kenya, using survival analysis methods.

## Study Design

| Feature | Description |
|---|---|
| Design | Retrospective cohort |
| Population | HIV-positive adults initiating first-line ART |
| Sites | Kenyatta National Hospital and Coast General Hospital |
| Period | 2010 to 2020 |
| Outcome | All-cause mortality |
| Follow-up | Up to 10 years; censored at death, loss to follow-up, or study end |
| N | 1 200 participants |

## Research Questions

1. Does baseline CD4 count at ART initiation predict survival?
2. Do survival outcomes differ by first-line ART regimen?
3. What are the independent predictors of mortality after adjusting for confounders?
4. Are there sex-based differences in survival on ART?

## Key Findings

| Analysis | Finding |
|---|---|
| Overall mortality | 16.7% over median 5-year follow-up |
| KM — CD4 groups | Significantly lower survival in CD4 < 200 group (log-rank p < 0.001) |
| KM — ART regimen | TDF/3TC/DTG associated with best survival; AZT/3TC/NVP worst |
| Cox — CD4 count | Protective (adjusted HR < 1.0); higher CD4 lowers mortality risk |
| Cox — WHO stage | Advanced stage (3/4) independently increases mortality hazard |
| Cox — DTG vs EFV | DTG-based regimen associated with lower mortality (HR approx 0.78) |
| Cox — AZT/NVP vs EFV | Older NVP-based regimen associated with higher mortality (HR approx 1.12) |
| Model fit | Concordance index above 0.70 |

## Analytical Pipeline

1. Synthetic cohort data generation (calibrated to IeDEA East Africa, NASCOP 2016)
2. Baseline characteristics table (Table 1)
3. Kaplan-Meier survival curves stratified by CD4 group, ART regimen, sex
4. Log-rank tests (pairwise and multivariate)
5. Numbers at risk table at landmark times
6. Multivariable Cox proportional hazards model
7. Forest plot of adjusted hazard ratios with 95% CI
8. Schoenfeld residual test for proportional hazards assumption
9. Survival probability at landmark times (1y, 2y, 5y) by CD4 group
10. Nelson-Aalen cumulative hazard estimator
11. Epidemiological interpretation with Kenya-specific clinical context

## Quick Start (Google Colab)

Open `ART_Survival_Analysis_Kenya.ipynb` in Google Colab and run the first cell:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "lifelines", "-q"])
```

All other libraries (numpy, pandas, scipy, matplotlib) are pre-installed in Colab.

## Tech Stack

`Python 3.10+` · `lifelines` · `numpy` · `pandas` · `scipy` · `matplotlib`

## Biological and Clinical Background

ART has transformed HIV from a fatal disease to a manageable chronic condition.
However, mortality risk on ART remains elevated among patients who initiate
treatment with advanced immunosuppression (CD4 below 200 cells/mm³) or at
WHO clinical stage 3/4. In Kenya, the NASCOP Test and Start guidelines (2016)
moved to universal ART eligibility to eliminate late initiation. This analysis
examines how baseline CD4 count, WHO staging, and regimen choice independently
predict survival, supporting the evidence base for early ART initiation and
transition to dolutegravir-based first-line therapy.

## Data

Synthetic data calibrated to:
- IeDEA East Africa cohort mortality rates (Egger et al. 2012; Brennan et al. 2013)
- Kenyan ART programme CD4 and viral load distributions (NASCOP 2016)
- ADVANCE trial efficacy data for dolutegravir-based regimens (Dorward et al. 2020)

---

*ZENIK.AI · github.com/Mwauramos*
