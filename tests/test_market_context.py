# tests/test_market_context.py

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_ingestor import ingest
from modules.feature_engine import engineer_features
from modules.market_context import (
    prepare_regime_features, fit_kmeans, interpret_clusters,
    assign_regimes, validate_regimes, current_regime_report
)
from config import REGIME_FEATURES, N_REGIMES


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def make_feature_df(rows=200):
    np.random.seed(42)
    dates  = pd.date_range('2023-01-01', periods=rows, freq='1D')
    close  = 100 + np.cumsum(np.random.randn(rows) * 1.5)
    high   = close + np.abs(np.random.randn(rows) * 0.8)
    low    = close - np.abs(np.random.randn(rows) * 0.8)
    open_  = close + np.random.randn(rows) * 0.3
    volume = np.random.randint(1_000_000, 10_000_000, rows).astype(float)
    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume
    }, index=dates)
    df.index.name = 'datetime'
    return engineer_features(df)


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_feature_preparation_no_nan():
    df = make_feature_df()
    X_scaled, valid_idx, _ = prepare_regime_features(df)
    assert not np.isnan(X_scaled).any(), "Scaled features must have no NaNs"
    assert len(valid_idx) <= len(df)
    print("PASS: test_feature_preparation_no_nan")


def test_feature_preparation_shape():
    df = make_feature_df()
    X_scaled, valid_idx, _ = prepare_regime_features(df)
    assert X_scaled.shape[1] == len(REGIME_FEATURES), "Feature count mismatch"
    assert X_scaled.shape[0] == len(valid_idx)
    print("PASS: test_feature_preparation_shape")


def test_kmeans_label_count():
    df = make_feature_df()
    X_scaled, valid_idx, _ = prepare_regime_features(df)
    km = fit_kmeans(X_scaled)
    assert len(np.unique(km.labels_)) == N_REGIMES, f"Expected {N_REGIMES} clusters"
    print("PASS: test_kmeans_label_count")


def test_cluster_label_map_completeness():
    df = make_feature_df()
    X_scaled, valid_idx, _ = prepare_regime_features(df)
    km = fit_kmeans(X_scaled)
    cluster_label_map, _ = interpret_clusters(df, valid_idx, km.labels_)
    assert len(cluster_label_map) == N_REGIMES, "Every cluster must have a label"
    valid_names = {'CALM_TRENDING', 'ACTIVE_TRENDING', 'CHOPPY', 'CHAOTIC'}
    for k, v in cluster_label_map.items():
        assert v in valid_names, f"Invalid regime name: {v}"
    print("PASS: test_cluster_label_map_completeness")


def test_regime_column_exists():
    df = make_feature_df()
    X_scaled, valid_idx, _ = prepare_regime_features(df)
    km = fit_kmeans(X_scaled)
    cluster_label_map, _ = interpret_clusters(df, valid_idx, km.labels_)
    df = assign_regimes(df, valid_idx, km.labels_, cluster_label_map)
    assert 'regime' in df.columns
    assert 'regime_cluster' in df.columns
    print("PASS: test_regime_column_exists")


def test_no_unknown_in_valid_rows():
    df = make_feature_df()
    X_scaled, valid_idx, _ = prepare_regime_features(df)
    km = fit_kmeans(X_scaled)
    cluster_label_map, _ = interpret_clusters(df, valid_idx, km.labels_)
    df = assign_regimes(df, valid_idx, km.labels_, cluster_label_map)
    valid_regimes = df.loc[valid_idx, 'regime']
    assert (valid_regimes != 'UNKNOWN').all(), "Valid rows must not have UNKNOWN regime"
    print("PASS: test_no_unknown_in_valid_rows")


def test_regime_distribution_reasonable():
    """No single regime should cover more than 80% of bars."""
    df = make_feature_df()
    X_scaled, valid_idx, _ = prepare_regime_features(df)
    km = fit_kmeans(X_scaled)
    cluster_label_map, _ = interpret_clusters(df, valid_idx, km.labels_)
    df = assign_regimes(df, valid_idx, km.labels_, cluster_label_map)
    dist = df[df['regime'] != 'UNKNOWN']['regime'].value_counts(normalize=True)
    assert dist.max() < 0.85, f"One regime dominates: {dist.idxmax()} at {dist.max():.1%}"
    print("PASS: test_regime_distribution_reasonable")


def test_current_regime_report_keys():
    df = make_feature_df()
    X_scaled, valid_idx, _ = prepare_regime_features(df)
    km = fit_kmeans(X_scaled)
    cluster_label_map, _ = interpret_clusters(df, valid_idx, km.labels_)
    df = assign_regimes(df, valid_idx, km.labels_, cluster_label_map)
    report = current_regime_report(df)
    for key in ['current_regime', 'as_of', 'volatility', 'rsi', 'ema_delta']:
        assert key in report, f"Missing key in report: {key}"
    print("PASS: test_current_regime_report_keys")


def test_full_pipeline_on_real_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'aapl_data.csv')
    if not os.path.exists(csv_path):
        print("SKIP: test_full_pipeline_on_real_data — aapl_data.csv not found")
        return
    df = ingest(csv_path)
    df = engineer_features(df)
    X_scaled, valid_idx, _ = prepare_regime_features(df)
    km = fit_kmeans(X_scaled)
    cluster_label_map, _ = interpret_clusters(df, valid_idx, km.labels_)
    df = assign_regimes(df, valid_idx, km.labels_, cluster_label_map)
    validate_regimes(df)
    report = current_regime_report(df)
    assert report['current_regime'] in {'CALM_TRENDING','ACTIVE_TRENDING','CHOPPY','CHAOTIC'}
    print(f"PASS: test_full_pipeline_on_real_data — Current regime: {report['current_regime']}")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  TIAS — Phase 3 Test Suite")
    print("="*55 + "\n")

    test_feature_preparation_no_nan()
    test_feature_preparation_shape()
    test_kmeans_label_count()
    test_cluster_label_map_completeness()
    test_regime_column_exists()
    test_no_unknown_in_valid_rows()
    test_regime_distribution_reasonable()
    test_current_regime_report_keys()
    test_full_pipeline_on_real_data()

    print("\n" + "="*55)
    print("  All Phase 3 tests completed.")
    print("="*55)