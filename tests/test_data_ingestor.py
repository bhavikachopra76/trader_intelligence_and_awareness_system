# tests/test_data_ingestor.py

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_ingestor import (
    load_csv, standardize_columns, parse_and_sort_timestamps,
    enforce_numeric_types, handle_missing_values, validate_price_logic,
    detect_time_gaps
)


# ─────────────────────────────────────────────
# HELPERS: build synthetic DataFrames for testing
# ─────────────────────────────────────────────

def make_clean_df(rows=100):
    """Generate a perfectly clean OHLCV DataFrame."""
    dates = pd.date_range(start='2023-01-01', periods=rows, freq='1D')
    close = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    high = close + np.abs(np.random.randn(rows) * 0.3)
    low = close - np.abs(np.random.randn(rows) * 0.3)
    open_ = close + np.random.randn(rows) * 0.1
    volume = np.random.randint(100000, 1000000, rows).astype(float)

    df = pd.DataFrame({
        'datetime': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    return df


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_standardize_columns_lowercase():
    df = make_clean_df()
    df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = standardize_columns(df)
    assert 'datetime' in df.columns
    assert 'close' in df.columns
    print("PASS: test_standardize_columns_lowercase")


def test_timestamp_sorting():
    df = make_clean_df(50)
    df = df.sample(frac=1)  # shuffle
    df = standardize_columns(df)
    df = parse_and_sort_timestamps(df)
    assert df.index.is_monotonic_increasing, "Index should be sorted ascending"
    print("PASS: test_timestamp_sorting")


def test_duplicate_removal():
    df = make_clean_df(50)
    df = pd.concat([df, df.iloc[:5]])  # add 5 duplicate rows
    df = standardize_columns(df)
    df = parse_and_sort_timestamps(df)
    assert not df.index.duplicated().any(), "No duplicates should remain"
    print("PASS: test_duplicate_removal")


def test_non_numeric_coercion():
    df = make_clean_df(30)
    df = standardize_columns(df)
    df = parse_and_sort_timestamps(df)
    df['close'] = df['close'].astype(str)
    df.iloc[5, df.columns.get_loc('close')] = 'bad_value'
    df = enforce_numeric_types(df)
    assert df['close'].dtype in [np.float64, np.float32]
    print("PASS: test_non_numeric_coercion")


def test_missing_value_filling():
    df = make_clean_df(50)
    df = standardize_columns(df)
    df = parse_and_sort_timestamps(df)
    df = enforce_numeric_types(df)
    df.iloc[10, df.columns.get_loc('close')] = np.nan
    df.iloc[20, df.columns.get_loc('volume')] = np.nan
    df = handle_missing_values(df)
    assert df['close'].isna().sum() == 0, "Close should have no NaNs after fill"
    assert df['volume'].isna().sum() == 0, "Volume should have no NaNs after fill"
    print("PASS: test_missing_value_filling")


def test_price_logic_bad_high_low():
    df = make_clean_df(50)
    df = standardize_columns(df)
    df = parse_and_sort_timestamps(df)
    df = enforce_numeric_types(df)
    df = handle_missing_values(df)
    original_len = len(df)
    # Inject bad bar: high < low
    df.iloc[5, df.columns.get_loc('high')] = 50.0
    df.iloc[5, df.columns.get_loc('low')] = 100.0
    df = validate_price_logic(df)
    assert len(df) < original_len, "Bad high/low bar should have been removed"
    print("PASS: test_price_logic_bad_high_low")


def test_extreme_move_flagging():
    df = make_clean_df(50)
    df = standardize_columns(df)
    df = parse_and_sort_timestamps(df)
    df = enforce_numeric_types(df)
    df = handle_missing_values(df)
    # Inject an extreme bar
    df.iloc[25, df.columns.get_loc('open')] = 100.0
    df.iloc[25, df.columns.get_loc('close')] = 130.0   # 30% move
    df.iloc[25, df.columns.get_loc('high')] = 131.0
    df.iloc[25, df.columns.get_loc('low')] = 99.0
    df = validate_price_logic(df)
    assert 'extreme_move_flag' in df.columns, "Extreme move flag column should exist"
    assert df['extreme_move_flag'].sum() >= 1, "At least 1 extreme move should be flagged"
    print("PASS: test_extreme_move_flagging")


def test_gap_detection():
    df = make_clean_df(50)
    df = standardize_columns(df)
    df = parse_and_sort_timestamps(df)
    df = enforce_numeric_types(df)
    df = handle_missing_values(df)
    df = validate_price_logic(df)
    # Drop rows 20-22 to create a gap
    df = df.drop(df.index[20:23])
    df = detect_time_gaps(df)
    assert 'gap_after' in df.columns, "gap_after column should exist"
    assert df['gap_after'].sum() >= 1, "At least one gap should be detected"
    print("PASS: test_gap_detection")


def test_missing_required_column():
    df = make_clean_df(30)
    df = df.drop(columns=['close'])
    try:
        standardize_columns(df)
        print("FAIL: Should have raised ValueError for missing 'close'")
    except ValueError as e:
        print(f"PASS: test_missing_required_column — caught: {e}")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  TIAS — Phase 1 Test Suite")
    print("="*50 + "\n")
    np.random.seed(42)

    test_standardize_columns_lowercase()
    test_timestamp_sorting()
    test_duplicate_removal()
    test_non_numeric_coercion()
    test_missing_value_filling()
    test_price_logic_bad_high_low()
    test_extreme_move_flagging()
    test_gap_detection()
    test_missing_required_column()

    print("\n" + "="*50)
    print("  All tests completed.")
    print("="*50)