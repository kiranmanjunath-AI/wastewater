"""
Wastewater data ingestion
=========================
Pull raw data from disparate public sources, then map each one into a
single shared ("canonical") schema so they can be compared side by side.

No AI/LLM mapping here -- explicit, hand-written field-mapping dicts.
That's the right starting point: get the pipeline working end-to-end on
known sources first. An LLM mapper can replace these dicts later, for
NEW, unfamiliar sources.

All column names below were verified against the live endpoints on
2026-08-21. Sources drift -- re-run inspect_sources() if things break.

Requirements: pip install requests pandas
"""

import io
import requests
import pandas as pd


CANONICAL_FIELDS = [
    "site_id",
    "jurisdiction",
    "collection_date",
    "target_name",
    "concentration_value",
    "concentration_unit",
    "normalization_method",
    "data_source",
]


# ---------------------------------------------------------------
# Shared normalizers
# ---------------------------------------------------------------
# Every source spells the same thing differently. Normalize once,
# here, so the mapping dicts stay declarative.

def iso_date(value):
    """Coerce any source date into ISO YYYY-MM-DD. CDC ships ISO already;
    CA ships MM/DD/YYYY. Returns None on unparseable input."""
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def clean_unit(value):
    """CA mixes 'copies/l wastewater' and 'copies/L wastewater' (same
    unit, different case) plus 'copies/g dry sludge' (NOT the same unit
    -- liquid vs solids matrix, do not compare naively)."""
    if not isinstance(value, str):
        return None
    return value.strip().lower()


def to_float(value):
    return pd.to_numeric(value, errors="coerce")


# ---------------------------------------------------------------
# STEP 1: Fetch raw data from each source
# ---------------------------------------------------------------

CDC_CONC_URL = "https://data.cdc.gov/resource/g653-rqe2.json"    # concentrations
CDC_META_URL = "https://data.cdc.gov/resource/2ew6-ywp6.json"    # site metadata

CA_CSV_URL = (
    "https://data.chhs.ca.gov/dataset/a6ca879a-6014-4b72-9ea6-07ef8b87ae83"
    "/resource/2742b824-3736-4292-90a9-7fad98e94c06/download"
    "/wastewatersurveillancecalifornia.csv"
)


def _fetch_cdc_site_metadata(key_plot_ids, chunk_size=50):
    """Look up site attributes for a set of key_plot_ids.

    The concentration dataset carries ONLY (key_plot_id, date,
    pcr_conc_lin, normalization) -- no site id, no jurisdiction. Those
    live in the separate metric dataset, so the canonical schema can
    only be filled by joining the two. Site attributes are static per
    key_plot_id, so we $group to get one row each.
    """
    ids = sorted({k for k in key_plot_ids if isinstance(k, str)})
    fields = "key_plot_id,wwtp_id,reporting_jurisdiction,county_names,population_served"
    frames = []

    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:start + chunk_size]
        quoted = ",".join("'" + k.replace("'", "''") + "'" for k in chunk)
        params = {
            "$select": fields,
            "$group": fields,
            "$where": "key_plot_id in({})".format(quoted),
            "$limit": chunk_size,
        }
        resp = requests.get(CDC_META_URL, params=params, timeout=60)
        resp.raise_for_status()
        frames.append(pd.DataFrame(resp.json()))

    if not frames:
        return pd.DataFrame(columns=fields.split(","))
    return pd.concat(frames, ignore_index=True)


def fetch_cdc_nwss(limit=1000):
    """CDC NWSS public wastewater data, via the Socrata API.

    Returns concentration rows enriched with site metadata. The join is
    on key_plot_id; the concentration table's `date` corresponds to the
    metric table's `date_end` (the close of its 15-day window).
    """
    resp = requests.get(CDC_CONC_URL, params={"$limit": limit}, timeout=60)
    resp.raise_for_status()
    conc = pd.DataFrame(resp.json())
    if conc.empty:
        return conc

    meta = _fetch_cdc_site_metadata(conc["key_plot_id"].unique())
    return conc.merge(meta, on="key_plot_id", how="left")


def fetch_ca_chhs(limit=1000, target="sars-cov-2"):
    """CDPH Wastewater Surveillance Data, California.

    The full CSV is ~430 MB, so we stream it and stop after `limit`
    matching rows instead of pulling the whole file into memory.

    This source is multi-pathogen (sars-cov-2, rsv, fluav, flubv, nvo,
    fluav h1/h3/h5, ...), so a target filter is required -- otherwise
    you silently mix influenza rows into a COVID comparison.
    """
    resp = requests.get(CA_CSV_URL, stream=True, timeout=180)
    resp.raise_for_status()
    resp.raw.decode_content = True

    chunks = []
    rows = 0
    reader = pd.read_csv(
        io.TextIOWrapper(resp.raw, encoding="utf-8", errors="replace"),
        chunksize=50_000,
        low_memory=False,
    )
    try:
        for chunk in reader:
            if target is not None:
                chunk = chunk[
                    chunk["pcr_target"].astype(str).str.strip().str.lower() == target
                ]
            # Drop rows already covered by CDC_NWSS to avoid double-counting
            chunk = chunk[
                chunk["data_source"] != "CDC NWSS Commercial Contract (Verily)"
            ]
            if chunk.empty:
                continue
            chunks.append(chunk)
            rows += len(chunk)
            if rows >= limit:
                break
    finally:
        resp.close()

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).head(limit)


# ---------------------------------------------------------------
# STEP 2: Map each source's raw fields into the canonical schema
# ---------------------------------------------------------------
# Keys are CANONICAL field names. Values are lambdas that pull and
# transform the source's raw columns.

CDC_NWSS_MAPPING = {
    # wwtp_id is the stable plant identifier; key_plot_id additionally
    # encodes sample location, so it is the true per-series key.
    "site_id":              lambda row: row.get("wwtp_id"),
    "jurisdiction":         lambda row: row.get("reporting_jurisdiction"),
    "collection_date":      lambda row: iso_date(row.get("date")),
    "target_name":          lambda row: "sars-cov-2",
    "concentration_value":  lambda row: to_float(row.get("pcr_conc_lin")),
    # Unit depends on normalization: flow-population rows are a genuine
    # per-litre concentration; microbial rows are normalized against a
    # fecal marker (e.g. PMMoV) and are a RATIO, not copies/L. Labelling
    # both "copies/L" would make them falsely comparable.
    "concentration_unit":   lambda row: (
        "copies/l wastewater"
        if row.get("normalization") == "flow-population"
        else "ratio (microbial-normalized)"
    ),
    "normalization_method": lambda row: row.get("normalization"),
    "data_source":          lambda row: "CDC_NWSS",
}

CA_CHHS_MAPPING = {
    "site_id":              lambda row: row.get("site_id"),
    "jurisdiction":         lambda row: row.get("reporting_jurisdiction"),
    "collection_date":      lambda row: iso_date(row.get("sample_collect_date")),
    "target_name":          lambda row: str(row.get("pcr_target", "")).strip().lower(),
    "concentration_value":  lambda row: to_float(row.get("pcr_target_avg_conc")),
    "concentration_unit":   lambda row: clean_unit(row.get("pcr_target_units")),
    # CA has no single normalization_method column. Solids vs liquid is
    # carried by the unit; other_norm_name names an alternate normalizer
    # when one was used.
    "normalization_method": lambda row: row.get("other_norm_name"),
    "data_source":          lambda row: "STATE_CA_CHHS",
}


def apply_mapping(raw_df, mapping):
    """Turn a raw DataFrame into canonical-schema rows using a mapping dict."""
    if raw_df.empty:
        return pd.DataFrame(columns=CANONICAL_FIELDS)
    records = []
    for _, row in raw_df.iterrows():
        records.append({field: fn(row) for field, fn in mapping.items()})
    return pd.DataFrame(records, columns=CANONICAL_FIELDS)


# ---------------------------------------------------------------
# STEP 3: Combine sources into one standardized table
# ---------------------------------------------------------------

def report_overlap(raw_ca):
    """CA's own `data_source` column shows most of its rows originate
    from 'CDC NWSS Commercial Contract (Verily)' -- the SAME lab feed
    behind the CDC national dataset. Concatenating both sources will
    double-count those sites. Print the split so the caller can decide.
    """
    if raw_ca.empty or "data_source" not in raw_ca.columns:
        return
    print("\nCA row provenance (overlap check):")
    for program, n in raw_ca["data_source"].value_counts().items():
        print("  {:>6}  {}".format(n, program))
    print("  ^ rows from the CDC/Verily contract also appear in CDC_NWSS.")


def build_combined_dataset(cdc_limit=500, ca_limit=2000):
    cdc_raw = fetch_cdc_nwss(limit=cdc_limit)
    cdc_canonical = apply_mapping(cdc_raw, CDC_NWSS_MAPPING)

    # No target filter -- pull all pathogens (sars-cov-2, flu a/b, rsv, norovirus, etc.)
    # Higher limit so we get meaningful coverage across all 8 pathogen types
    ca_raw = fetch_ca_chhs(limit=ca_limit, target=None)
    ca_canonical = apply_mapping(ca_raw, CA_CHHS_MAPPING)

    combined = pd.concat([cdc_canonical, ca_canonical], ignore_index=True)
    return combined, ca_raw


def inspect_sources():
    """Print each source's real column names. Run this when a mapping
    breaks -- it is how the mappings above were derived."""
    cdc = requests.get(CDC_CONC_URL, params={"$limit": 1}, timeout=60).json()
    meta = requests.get(CDC_META_URL, params={"$limit": 1}, timeout=60).json()
    print("CDC concentration columns:", sorted(cdc[0].keys()))
    print("CDC metric columns:      ", sorted(meta[0].keys()))
    ca = fetch_ca_chhs(limit=1, target=None)
    print("CA columns:              ", sorted(ca.columns))


if __name__ == "__main__":
    df, ca_raw = build_combined_dataset()

    print(df.head(10).to_string())
    print("\nTotal standardized rows: {}".format(len(df)))
    print("Unique sites: {}".format(df["site_id"].nunique()))
    print("Unique data sources: {}".format(list(df["data_source"].unique())))

    print("\nRows per source:")
    for src, n in df["data_source"].value_counts().items():
        print("  {:>6}  {}".format(n, src))

    print("\nUnits present (must match before comparing values):")
    for unit, n in df["concentration_unit"].value_counts(dropna=False).items():
        print("  {:>6}  {}".format(n, unit))

    missing = df["concentration_value"].isna().sum()
    print("\nRows with unusable concentration: {}".format(missing))

    report_overlap(ca_raw)
