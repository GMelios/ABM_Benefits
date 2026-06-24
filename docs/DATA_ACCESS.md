# Data access and governance

## Summary

This repository contains the full source code for the BENEFITS social-services
agent-based model, but it does **not** contain the underlying survey microdata. That
microdata (the UK Household Longitudinal Study) is restricted and may not be redistributed.
This document explains what data the model uses, why it is not included here, how to obtain
it, and the governance rules that keep restricted data out of the repository.

## The restricted dataset

The model's behavioural relationships and synthetic population are derived from the
**UK Household Longitudinal Study (Understanding Society, UKHLS)**, a large nationally
representative panel survey of UK households led by the Institute for Social and Economic
Research (ISER) at the University of Essex and funded by the Economic and Social Research
Council. The estimation uses the main adult interview across waves 1 to 11.

UKHLS individual-level data is made available by the **UK Data Service** under its **End User
Licence (EUL)**. Under that licence the data may be used for research by registered users but
**may not be passed on to third parties or published**, including in a code repository.
Geography below Government Office Region (for example Local Authority District) is available
only under a **Special Licence** or through the **Secure Lab**, with additional conditions.

## Why it is not in this repository

Redistributing UKHLS microdata, or any individual-level data derived from it, would breach the
UK Data Service End User Licence. Accordingly:

- the data directory (`data/`) is excluded from version control by `.gitignore`;
- no individual-level records appear in any tracked file, including code, comments, tests, or
  documentation;
- only **aggregates, estimated coefficients, and synthetic data** are committed, consistent
  with the licence.

## What this repository ships instead

So that the model can be inspected and, with appropriate data, reproduced, the repository
includes:

- **`params/`**: the estimated model coefficients (in JSON). These are aggregate statistical
  outputs, not microdata, and they are what the behaviour and valuation layers consume at run
  time. Sharing them is permitted under the licence.
- **`src/estimation/`**: the scripts that regenerate those coefficients from UKHLS, so the
  estimation is fully transparent and re-runnable by any licensed user.
- **`docs/data_dictionary.md`**: the expected schema (variables, coding, and provenance) of
  the harmonised panel the code reads.

## How to obtain the data

1. Register for a free account with the **UK Data Service** at
   <https://ukdataservice.ac.uk> and complete the required usage declaration.
2. Locate **Understanding Society (UKHLS)**, UK Data Service Study Number **SN 6614**, and
   accept the End User Licence. For analyses requiring sub-regional geography, apply instead
   for the relevant **Special Licence** edition.
3. Download the waves used by this model (waves 1 to 11) and the variables listed in
   `docs/data_dictionary.md`.
4. Construct the harmonised long panel expected by the code (see below), or adapt
   `src/estimation/load_ukhls.py` to your local extract, and place the file at:

   ```
   data/ukhls_panel_long.parquet
   ```

   The directory is git-ignored, so the file will never be staged for commit.

## Expected file and schema

The code expects a single Parquet file at `data/ukhls_panel_long.parquet` in **long
(person x wave) format**, with sentinel missing-value codes converted to nulls and the
harmonised, friendly column names documented in `docs/data_dictionary.md` (identifiers,
income components, SF-12 health, GHQ-12, employment, demographics, region, and survey
weights). The loader (`src/estimation/load_ukhls.py`) validates the schema on read and will
fail clearly if a required column is missing.

If you do not have access to UKHLS, you can still examine the code and run the logic-only
tests; the data-dependent tests skip automatically when the file is absent.

## Required citation and acknowledgement

Any use of UKHLS must cite the study and acknowledge the data provider. Use the citation for
the exact edition you download, in the form:

> University of Essex, Institute for Social and Economic Research. *Understanding Society:
> Waves 1 to [N], [years]* [data collection]. [Edition]. UK Data Service. SN: 6614,
> DOI: 10.5255/UKDA-SN-6614-[edition].

Suggested acknowledgement:

> Understanding Society is an initiative funded by the Economic and Social Research Council
> and various UK Government departments, with scientific leadership by the Institute for
> Social and Economic Research, University of Essex, and survey delivery by NatCen Social
> Research and Kantar Public. The data were accessed via the UK Data Service.

The UK Data Service and the data producers bear no responsibility for the analysis or
interpretation presented in this repository.

## Governance rules for contributors

- Never commit individual-level data, or any file derived from it at the individual level, in
  any format (`.parquet`, `.dta`, `.sav`, `.csv`, and so on). These patterns are git-ignored;
  do not override the ignore rules.
- Before committing, verify that no restricted data is staged:

  ```bash
  git check-ignore data/*                                  # should echo the paths
  git add -A -n | grep -iE "\.parquet|\.dta|\.sav|/data/"  # should return nothing
  ```

- Only aggregates, estimated coefficients, and synthetic data may enter the repository. If in
  doubt, do not commit, and consult the data licence terms.
