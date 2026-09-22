# Wastewater Surveillance

Pulls virus concentration data from public wastewater monitoring programs, maps them into a shared schema, and renders an interactive chart page for the general public.

## What's in this repo

| File | Purpose |
|---|---|
| `wastewater_ingest.py` | Data ingestion — fetches, deduplicates, and standardizes raw data from CDC and California |
| `wastewater_watch.html` | Self-contained interactive chart — open in any browser, no server needed |

## Data sources

| Source | Dataset | What it covers |
|---|---|---|
| [CDC NWSS — Concentrations](https://data.cdc.gov/Public-Health-Surveillance/NWSS-Public-SARS-CoV-2-Concentration-in-Wastewater/g653-rqe2) | `g653-rqe2` | Per-sample virus concentrations across US treatment plants |
| [CDC NWSS — Site Metadata](https://data.cdc.gov/Public-Health-Surveillance/NWSS-Public-SARS-CoV-2-Wastewater-Metric-Data/2ew6-ywp6) | `2ew6-ywp6` | Treatment plant IDs, jurisdictions, population served — joined to concentrations on `key_plot_id` |
| [CA CDPH Wastewater Surveillance](https://data.chhs.ca.gov/dataset/a6ca879a-6014-4b72-9ea6-07ef8b87ae83) | CSV download | Multi-pathogen (COVID-19, Flu A, RSV, Bird Flu H5) across California sites, 2023–present |

## Setup

```bash
pip install requests pandas
```

## Running the ingest script

```bash
python wastewater_ingest.py
```

Prints a standardized table with columns: `site_id`, `jurisdiction`, `collection_date`, `target_name`, `concentration_value`, `concentration_unit`, `normalization_method`, `data_source`.

To save to CSV, add one line before the script exits:

```python
df.to_csv("wastewater_combined.csv", index=False)
```

## Viewing the chart

**Live:** https://kiranmanjunath-ai.github.io/wastewater/

Or open `index.html` locally in any browser. The page contains two charts built from an embedded data snapshot (no live API calls):

1. **COVID-19 in US wastewater, 2021–2025** — monthly median concentration across CDC-monitored plants, with variant wave annotations
2. **What's circulating in California** — weekly levels for COVID-19, Flu A, RSV, and Bird Flu H5, each normalized to % of its own historical peak

Hover over any point for the exact date and value. Works in light and dark mode.

## Deduplication note

California's dataset includes rows sourced from the CDC/Verily commercial contract — the same measurements that appear in the CDC national feed. The ingest script drops those rows to avoid double-counting.

## Built with AI assistance

Developed using [Claude Code](https://claude.com/claude-code).
