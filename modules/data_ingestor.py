import pandas as pd
import matplotlib.pyplot as plt
import os
from config import (
    REQUIRED_COLUMNS, TIMESTAMP_CANDIDATES,
    MAX_SINGLE_BAR_MOVE_PCT, MIN_PRICE,
    ALLOW_ZERO_VOLUME, OUTPUT_DIR
)


# ─────────────────────────────────────────────
# SECTION 1: LOADING
# ─────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file and return a raw DataFrame.
    Raises clear errors if file doesn't exist or can't be parsed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[TIAS] File not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"[TIAS] Failed to parse CSV: {e}")

    print(f"[TIAS] Loaded {len(df)} rows from '{filepath}'")
    print(f"[TIAS] Columns found: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# SECTION 2: COLUMN STANDARDIZATION
# ─────────────────────────────────────────────

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase.
    Detect and rename the timestamp column to 'datetime'.
    Raise error if required columns are missing.
    """
    # Lowercase all column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Detect timestamp column
    ts_col = None
    for candidate in [c.lower() for c in TIMESTAMP_CANDIDATES]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        raise ValueError(
            f"[TIAS] No timestamp column found. "
            f"Expected one of: {TIMESTAMP_CANDIDATES}\n"
            f"Found: {list(df.columns)}"
        )

    if ts_col != 'datetime':
        df = df.rename(columns={ts_col: 'datetime'})

    # Check required OHLCV columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"[TIAS] Missing required columns: {missing}")

    print(f"[TIAS] Columns standardized. Timestamp column: '{ts_col}' → 'datetime'")
    return df


# ─────────────────────────────────────────────
# SECTION 3: TIMESTAMP PARSING & SORTING
# ─────────────────────────────────────────────

def parse_and_sort_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse datetime column, set as index, sort chronologically.
    """
    try:
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    except Exception as e:
        raise ValueError(f"[TIAS] Failed to parse datetime column: {e}")

    # Check for unparseable timestamps
    null_ts = df['datetime'].isna().sum()
    if null_ts > 0:
        print(f"[TIAS] WARNING: {null_ts} rows with unparseable timestamps — dropping them.")
        df = df.dropna(subset=['datetime'])

    df = df.set_index('datetime').sort_index()

    # Check for duplicates
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        print(f"[TIAS] WARNING: {dupes} duplicate timestamps found — keeping first occurrence.")
        df = df[~df.index.duplicated(keep='first')]

    print(f"[TIAS] Timestamps parsed. Range: {df.index[0]} → {df.index[-1]}")
    return df


# ─────────────────────────────────────────────
# SECTION 4: TYPE ENFORCEMENT
# ─────────────────────────────────────────────

def enforce_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure OHLCV columns are numeric. Coerce errors to NaN.
    """
    for col in REQUIRED_COLUMNS:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        after = df[col].isna().sum()
        new_nulls = after - before
        if new_nulls > 0:
            print(f"[TIAS] WARNING: {new_nulls} non-numeric values in '{col}' → set to NaN")

    return df


# ─────────────────────────────────────────────
# SECTION 5: MISSING VALUE HANDLING
# ─────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report and handle NaN values in OHLCV columns.
    Strategy: forward-fill price columns, flag volume nulls.
    """
    for col in REQUIRED_COLUMNS:
        null_count = df[col].isna().sum()
        if null_count == 0:
            continue

        pct = (null_count / len(df)) * 100
        print(f"[TIAS] '{col}' has {null_count} missing values ({pct:.2f}%)")

        if pct > 10:
            print(f"[TIAS] WARNING: More than 10% missing in '{col}'. Data quality concern.")

        if col == 'volume':
            df[col] = df[col].fillna(0)
            print(f"[TIAS] Volume NaNs filled with 0.")
        else:
            df[col] = df[col].ffill()
            print(f"[TIAS] '{col}' NaNs forward-filled.")

    return df


# ─────────────────────────────────────────────
# SECTION 6: PRICE LOGIC VALIDATION
# ─────────────────────────────────────────────

def validate_price_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check that OHLCV values make logical sense:
    - High >= Low
    - High >= Open, High >= Close
    - Low <= Open, Low <= Close
    - All prices > MIN_PRICE
    - Volume >= 0 (or > 0 if ALLOW_ZERO_VOLUME is False)
    - No single bar moves more than MAX_SINGLE_BAR_MOVE_PCT
    """
    issues = []

    # High/Low integrity
    bad_hl = df['high'] < df['low']
    if bad_hl.sum() > 0:
        issues.append(f"  - {bad_hl.sum()} bars where High < Low")
        df = df[~bad_hl]

    # High must be the highest
    bad_high = (df['high'] < df['open']) | (df['high'] < df['close'])
    if bad_high.sum() > 0:
        issues.append(f"  - {bad_high.sum()} bars where High < Open or Close")
        df = df[~bad_high]

    # Low must be the lowest
    bad_low = (df['low'] > df['open']) | (df['low'] > df['close'])
    if bad_low.sum() > 0:
        issues.append(f"  - {bad_low.sum()} bars where Low > Open or Close")
        df = df[~bad_low]

    # Minimum price threshold
    bad_price = (df[['open', 'high', 'low', 'close']] <= MIN_PRICE).any(axis=1)
    if bad_price.sum() > 0:
        issues.append(f"  - {bad_price.sum()} bars with price ≤ {MIN_PRICE}")
        df = df[~bad_price]

    # Volume check
    if not ALLOW_ZERO_VOLUME:
        zero_vol = df['volume'] <= 0
        if zero_vol.sum() > 0:
            issues.append(f"  - {zero_vol.sum()} bars with zero/negative volume (flagged, not removed)")
            df['zero_volume_flag'] = zero_vol

    # Single bar move check
    bar_move_pct = ((df['close'] - df['open']).abs() / df['open']) * 100
    extreme_moves = bar_move_pct > MAX_SINGLE_BAR_MOVE_PCT
    if extreme_moves.sum() > 0:
        issues.append(
            f"  - {extreme_moves.sum()} bars with move > {MAX_SINGLE_BAR_MOVE_PCT}% "
            f"(flagged, not removed — could be real events)"
        )
        df['extreme_move_flag'] = extreme_moves

    if issues:
        print("[TIAS] Price logic issues found:")
        for issue in issues:
            print(issue)
    else:
        print("[TIAS] Price logic validation passed. No issues found.")

    return df


# ─────────────────────────────────────────────
# SECTION 7: TIME GAP DETECTION
# ─────────────────────────────────────────────

def detect_time_gaps(df: pd.DataFrame, expected_freq: str = None) -> pd.DataFrame:
    """
    Detect gaps in the time series.
    If expected_freq is None, we infer it from the data (median gap).
    Reports gaps but does NOT fill them — downstream modules must know gaps exist.
    """
    time_diffs = df.index.to_series().diff().dropna()
    median_gap = time_diffs.median()

    if expected_freq:
        expected_td = pd.tseries.frequencies.to_offset(expected_freq).nanos
        expected_td = pd.Timedelta(expected_td)
    else:
        expected_td = median_gap

    # A gap is anything more than 2x the expected interval
    gaps = time_diffs[time_diffs > (expected_td * 2)]

    if len(gaps) > 0:
        print(f"[TIAS] Detected {len(gaps)} time gaps (expected interval: {expected_td}):")
        for ts, gap in gaps.items():
            print(f"  - Gap of {gap} at {ts}")
        df['gap_after'] = False
        df.loc[gaps.index, 'gap_after'] = True
    else:
        print(f"[TIAS] No time gaps detected. Interval: {expected_td}")

    return df


# ─────────────────────────────────────────────
# SECTION 8: SUMMARY REPORT
# ─────────────────────────────────────────────

def generate_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a clean summary of the loaded dataset.
    """
    summary = {
        "total_bars": len(df),
        "start_date": str(df.index[0]),
        "end_date": str(df.index[-1]),
        "close_min": round(df['close'].min(), 4),
        "close_max": round(df['close'].max(), 4),
        "close_mean": round(df['close'].mean(), 4),
        "avg_volume": round(df['volume'].mean(), 2),
        "missing_values": df[REQUIRED_COLUMNS].isna().sum().to_dict(),
        "flag_columns": [c for c in df.columns if c.endswith('_flag') or c == 'gap_after'],
    }

    print("\n" + "=" * 50)
    print("  TIAS — DATA INGESTION SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
    print("=" * 50 + "\n")

    return summary


# ─────────────────────────────────────────────
# SECTION 9: VISUALIZATION
# ─────────────────────────────────────────────

def plot_raw_price(df: pd.DataFrame, title: str = "Raw Price — OHLC Overview", save: bool = True):
    """
    Plot Close price with High/Low range shaded.
    Save to outputs/ if save=True.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#0d1117')

    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#c9d1d9')
        ax.spines[:].set_color('#30363d')

    # Price panel
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], color='#58a6ff', linewidth=1.2, label='Close')
    ax1.fill_between(df.index, df['low'], df['high'], alpha=0.15, color='#58a6ff', label='High/Low Range')
    ax1.set_ylabel('Price', color='#c9d1d9')
    ax1.set_title(title, color='#e6edf3', fontsize=13, pad=10)
    ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax1.yaxis.label.set_color('#c9d1d9')

    # Flag overlays
    if 'extreme_move_flag' in df.columns:
        extremes = df[df['extreme_move_flag'] == True]
        ax1.scatter(extremes.index, extremes['close'], color='#f85149',
                    s=20, zorder=5, label='Extreme Move', marker='^')

    # Volume panel
    ax2 = axes[1]
    ax2.bar(df.index, df['volume'], color='#388bfd', alpha=0.6, width=0.8)
    ax2.set_ylabel('Volume', color='#c9d1d9')
    ax2.yaxis.label.set_color('#c9d1d9')

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "phase1_raw_price.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[TIAS] Chart saved → {path}")

    plt.show()


# ─────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────

def ingest(filepath: str, expected_freq: str = None) -> pd.DataFrame:
    """
    Full Phase 1 pipeline.
    Call this from main.py.
    Returns a clean, validated DataFrame.
    """
    print("\n[TIAS] ── Starting Phase 1: Data Ingestion & Cleaning ──\n")

    df = load_csv(filepath)
    df = standardize_columns(df)
    df = parse_and_sort_timestamps(df)
    df = enforce_numeric_types(df)
    df = handle_missing_values(df)
    df = validate_price_logic(df)
    df = detect_time_gaps(df, expected_freq=expected_freq)
    summary = generate_data_summary(df)
    plot_raw_price(df)

    print("[TIAS] ── Phase 1 Complete. Clean DataFrame ready. ──\n")
    return df