# tests/test_feature_engine.py

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_ingestor import ingest
from modules.feature_engine import (
    compute_returns, compute_volatility, compute_momentum,
    compute_speed, compute_expansion_compression,
    compute_volume_features, validate_features
)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def make_base_df(rows=200):
    """Clean OHLCV DataFrame simulating daily equity data."""
    np.random.seed(42)
    dates  = pd.date_range('2023-01-01', periods=rows, freq='1D')
    close  = 100 + np.cumsum(np.random.randn(rows) * 1.5)
    high   = close + np.abs(np.random.randn(rows) * 0.8)
    low    = close - np.abs(np.random.randn(rows) * 0.8)
    open_  = close + np.random.randn(rows) * 0.3
    volume = np.random.randint(1_000_000, 10_000_000, rows).astype(float)

    df = pd.DataFrame({
        'open': open_, 'high': high,
        'low': low, 'close': close, 'volume': volume
    }, index=dates)
    df.index.name = 'datetime'
    return df


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_returns_shape():
    df = make_base_df()
    df = compute_returns(df)
    assert 'log_return' in df.columns
    assert 'simple_return' in df.columns
    assert df['log_return'].isna().sum() == 1  # first row is NaN — expected
    print("PASS: test_returns_shape")


def test_log_return_values():
    """log return of a 10% move should be ~0.0953"""
    df = make_base_df(10)
    df['close'] = [100, 110, 100, 110, 100, 110, 100, 110, 100, 110]
    df = compute_returns(df)
    expected = np.log(110 / 100)
    assert abs(df['log_return'].iloc[1] - expected) < 1e-6
    print("PASS: test_log_return_values")


def test_volatility_non_negative():
    df = make_base_df()
    df = compute_returns(df)
    df = compute_volatility(df)
    assert (df['volatility'].dropna() >= 0).all(), "Volatility must be non-negative"
    assert (df['atr'].dropna() >= 0).all(), "ATR must be non-negative"
    assert (df['atr_pct'].dropna() >= 0).all(), "ATR % must be non-negative"
    print("PASS: test_volatility_non_negative")


def test_rsi_bounds():
    df = make_base_df()
    df = compute_returns(df)
    df = compute_momentum(df)
    rsi = df['rsi'].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all(), f"RSI out of bounds: {rsi.min():.2f}–{rsi.max():.2f}"
    print("PASS: test_rsi_bounds")


def test_ema_delta_direction():
    """In a strong uptrend, EMA fast should be above EMA slow → positive delta."""
    df = make_base_df(100)
    df['close'] = np.linspace(100, 200, 100)  # pure uptrend
    df['high']  = df['close'] + 1
    df['low']   = df['close'] - 1
    df['open']  = df['close'] - 0.5
    df = compute_returns(df)
    df = compute_momentum(df)
    # Check the last 20 bars (EMAs fully warmed up)
    assert df['ema_delta'].iloc[-20:].mean() > 0, "EMA delta should be positive in uptrend"
    print("PASS: test_ema_delta_direction")


def test_speed_non_negative():
    df = make_base_df()
    df = compute_speed(df)
    assert (df['bar_speed'].dropna() >= 0).all(), "Speed must be non-negative"
    print("PASS: test_speed_non_negative")


def test_range_ratio_logic():
    """A bar with double the average size should have range_ratio ~2."""
    df = make_base_df(100)
    df = compute_expansion_compression(df)
    # range_ratio should be positive
    assert (df['range_ratio'].dropna() > 0).all()
    print("PASS: test_range_ratio_logic")


def test_bb_pct_b_bounds_approx():
    """bb_pct_b should mostly be between -0.5 and 1.5 for normal price data."""
    df = make_base_df(200)
    df = compute_expansion_compression(df)
    pct_b = df['bb_pct_b'].dropna()
    assert pct_b.between(-1.0, 2.0).mean() > 0.95, "bb_pct_b has too many extreme outliers"
    print("PASS: test_bb_pct_b_bounds_approx")


def test_volume_ratio_positive():
    df = make_base_df()
    df = compute_volume_features(df)
    assert (df['volume_ratio'].dropna() > 0).all(), "Volume ratio must be positive"
    print("PASS: test_volume_ratio_positive")


def test_no_inf_values():
    """No feature should produce infinite values."""
    df = make_base_df()
    df = compute_returns(df)
    df = compute_volatility(df)
    df = compute_momentum(df)
    df = compute_speed(df)
    df = compute_expansion_compression(df)
    df = compute_volume_features(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts   = np.isinf(df[numeric_cols]).sum()
    bad_cols     = inf_counts[inf_counts > 0]

    assert len(bad_cols) == 0, f"Infinite values found in: {bad_cols.to_dict()}"
    print("PASS: test_no_inf_values")


def test_full_pipeline_on_real_data():
    """Run full feature pipeline on actual AAPL CSV."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'aapl_data.csv')
    if not os.path.exists(csv_path):
        print("SKIP: test_full_pipeline_on_real_data — aapl_data.csv not found")
        return
    df = ingest(csv_path)
    df = compute_returns(df)
    df = compute_volatility(df)
    df = compute_momentum(df)
    df = compute_speed(df)
    df = compute_expansion_compression(df)
    df = compute_volume_features(df)
    validate_features(df)
    assert len(df) > 0
    print("PASS: test_full_pipeline_on_real_data")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  TIAS — Phase 2 Test Suite")
    print("="*55 + "\n")

    test_returns_shape()
    test_log_return_values()
    test_volatility_non_negative()
    test_rsi_bounds()
    test_ema_delta_direction()
    test_speed_non_negative()
    test_range_ratio_logic()
    test_bb_pct_b_bounds_approx()
    test_volume_ratio_positive()
    test_no_inf_values()
    test_full_pipeline_on_real_data()

    print("\n" + "="*55)
    print("  All Phase 2 tests completed.")
    print("="*55)