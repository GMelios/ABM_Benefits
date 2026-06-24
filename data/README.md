# data/

This directory holds the restricted research data, which is **not** included in the
repository. Its contents are git-ignored; only this README is tracked.

Place the harmonised UKHLS panel here:

```
data/ukhls_panel_long.parquet
```

The UK Household Longitudinal Study (Understanding Society) is provided by the UK Data Service
under its End User Licence and may not be redistributed. For how to obtain it, the expected
file schema, and the required citation, see [`../docs/DATA_ACCESS.md`](../docs/DATA_ACCESS.md).

Without this file, the code still imports and the logic-only tests run; the data-dependent
steps and tests skip automatically.
