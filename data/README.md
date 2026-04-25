# Data

This Letter uses **public end-of-day market data only**.

## Files in this directory

| File                              | Contents                                                      |
|-----------------------------------|---------------------------------------------------------------|
| `forensic_dotcom_2000.json`       | Per-day geometric indicators (D, R, Ric, TSS, FCI, Dp) + crisis events |
| `forensic_gfc_2007.json`          | Same, for the 2008 global financial crisis                    |
| `forensic_covid_2020.json`        | Same, for the COVID shock                                     |
| `forensic_tradewar_2026.json`     | Same, for the 2026 trade-war crisis                           |
| `sp500_close.csv`                 | Frozen S&P 500 daily-close series 1998–2026 (Yahoo Finance ^GSPC), used by `verify.py` Test 5 (rupture-trigger robustness) |
| `data_manifest.csv`               | Per-crisis metadata: window, rupture event, lead days, source |

The forensic JSON files contain the **computed outputs** of the URF
geometric pipeline applied to the public market data sources listed below;
the implementation pipeline itself is proprietary and not part of this
academic archive (see top-level `README.md`).

## Sources

- **Yahoo Finance** — https://finance.yahoo.com/
  Daily close prices for S&P 500, NASDAQ 100, Euro Stoxx 50, FTSE 100, GICS
  sector sub-indices, VIX, WTI crude oil, gold.

- **FRED (St. Louis Fed)** — https://fred.stlouisfed.org/
  US 2-year and 10-year Treasury yields, investment-grade and high-yield
  credit spreads, Fed liquidity indices.

## No proprietary data

No proprietary, licensed, or private data is used at any stage of the
analysis. Every empirical claim of the Letter is reproducible from the
files in this directory using `verify.py` at the repository root.
