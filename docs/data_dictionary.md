# Data dictionary, harmonised UKHLS long panel

**Source.** UK Household Longitudinal Study (Understanding Society, UKHLS), main adult
interview, **waves 1 to 11 (a to k)**. Access: **End User Licence (EUL)**; geography =
**Government Office Region** (`gor_dv`). Restricted microdata, held in the git-ignored `data/`
directory, never committed.

**Form.** A **long person by wave panel**: 476,187 rows and 87,905 persons. UKHLS sentinel
missing codes (-1, -2, -7, -8, -9) are mapped to `NaN`; source wave-prefixed variables are
harmonised to the friendly names below. Loader: `src/estimation/load_ukhls.py`.

> This file contains **only column metadata and aggregates**, no individual records.

| Column | Meaning | Units / coding | Model role | Notes |
|---|---|---|---|---|
| `pidp` | person id | int | panel key | stable across waves |
| `wave` | wave number | 1 to 11 (a to k) | panel key / tick | 1 tick = 1 year |
| `hidp` | household id | int | household grouping | wave-specific |
| `income_net` | total net monthly personal income | GBP/month | income for utility and EV | market plus benefit (plus investment and pension in the tail); top values winsorised |
| `income_gross` | total gross monthly personal income | GBP/month | reference | a different base from net |
| `income_labour_net` / `income_labour_gross` | labour income | GBP/month | earnings model | |
| `y_market` | market-income component | GBP/month | dynamics | |
| `y_benefit` | state-benefit income component | GBP/month | income-support lever | broad (includes pensions and child benefit) |
| `y_market_is_pretransfer` | flag | bool | bookkeeping | |
| `health_pcs` / `health_mcs` | SF-12 physical / mental component summary | approx 0 to 100 | health stock construction | item-level SF-12 not in extract; 14.4% missing |
| `ghq` | GHQ-12 Likert | 0 to 36 (higher = worse) | wellbeing anchor | reverse-scored to `wellbeing_ghq`; about 14% missing |
| `ghq_caseness` | GHQ-12 caseness | 0 to 12 | mental-health flag | |
| `srh` | self-rated general health | 1 (excellent) to 5 (poor) | health covariate | 1.9% missing |
| `health_state` | three-band recode of `srh` | 0/1/2 | not a utility index | ordinal collapse of srh |
| `disability` / `disability_type` | disability flag / type | bool / 0 to 3 | need, health | |
| `jbstat` | economic activity | 1 to 13, 97 | maps to `employed` | see `JBSTAT_LABELS` |
| `employed` | in work (jbstat in {1,2}) | bool | employment state | derived; 16 to 64 weighted rate about 68.9% |
| `hours` | usual weekly hours | hours | earnings | 44% missing (only workers asked) |
| `dvage` | age | 16 to 104 | demographics | |
| `sex` | sex | 1 = male, 2 = female | demographics | |
| `mastat_dv` | marital status | 0 to 10 | household type | |
| `hiqual_dv` | highest qualification | 1 to 9 | education | 1.5% missing |
| `ethn_dv` | ethnicity | 1 to 97 | demographics | |
| `nkids_dv` | number of children | count | childcare extension | empty in this extract |
| `gor_dv` | Government Office Region | 1 to 12 | region / space | maps to `region` name; 12 = Northern Ireland |
| `urban_dv` | urban / rural | 1 / 2 | context | |
| `weight_xsec` | cross-sectional weight | non-negative | margins / initialisation | all waves; `design_weight` is identical |
| `design_weight` | design weight | non-negative | duplicate of `weight_xsec` | maximum absolute difference 0 |
| `weight_long` | longitudinal weight | non-negative | transition models | missing at wave 1 (expected) |

## Derived columns added by the loader
- `region`, the region name from `gor_dv`.
- `income_for_utility`, `income_net` winsorised to the [p1, p99.5] range of positive income.
- `log_income`, the natural log of `income_for_utility` (the utility and EV income term, ODD
  Section 7.7); `NaN` where income is missing.
- `wellbeing_ghq`, `36 - ghq` (higher = better); the wellbeing anchor.
- `health_index`, `clip(PCS / 100, 0, 1)`, the physical-health stock in [0,1] from the SF-12
  physical component.

## Modelling notes
- **Wellbeing anchor.** Life satisfaction (`sclfsato`) is not collected in this extract; the
  wellbeing anchor is reverse-scored GHQ-12.
- **Health stock.** Item-level SF-12 is not available, so the health stock `h` in [0,1] is
  constructed from the SF-12 physical component (PCS).
- **`nkids_dv`** is empty in this extract, so child counts are not used; childcare is a later
  extension.
- **`y_benefit`** is a broad state-benefit measure (it includes pensions); income-support
  targeting focuses on working-age means-tested benefits.
