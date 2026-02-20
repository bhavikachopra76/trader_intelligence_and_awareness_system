# modules/market_context.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from config import (
    REGIME_FEATURES, N_REGIMES, RANDOM_SEED, OUTPUT_DIR
)


# ─────────────────────────────────────────────
# REGIME COLOR MAP — used across all visuals
# ─────────────────────────────────────────────

REGIME_COLORS = {
    'CALM_TRENDING':   '#3fb950',   # green
    'ACTIVE_TRENDING': '#ffa657',   # orange
    'CHOPPY':          '#79c0ff',   # blue
    'CHAOTIC':         '#f85149',   # red
    'UNKNOWN':         '#6e7681',   # grey
}


# ─────────────────────────────────────────────
# SECTION 1: PREPARE FEATURES FOR CLUSTERING
# ─────────────────────────────────────────────

def prepare_regime_features(df: pd.DataFrame) -> tuple:
    """
    Extract and scale the features used for regime detection.
    Uses RobustScaler — resistant to outliers (important for financial data).
    Returns:
        X_scaled: numpy array ready for clustering
        valid_idx: index of rows that had no NaNs (we skip NaN rows)
    """
    feature_df = df[REGIME_FEATURES].copy()

    # Drop rows where any regime feature is NaN
    valid_mask = feature_df.notna().all(axis=1)
    feature_df = feature_df[valid_mask]
    valid_idx  = feature_df.index

    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"[TIAS] Dropped {n_dropped} rows with NaN in regime features (warmup period)")

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(feature_df)

    print(f"[TIAS] Regime features prepared: {len(valid_idx)} bars, {len(REGIME_FEATURES)} features")
    return X_scaled, valid_idx, scaler


# ─────────────────────────────────────────────
# SECTION 2: FIND OPTIMAL K (ELBOW + SILHOUETTE)
# ─────────────────────────────────────────────

def evaluate_cluster_counts(X_scaled: np.ndarray, max_k: int = 8, save: bool = True):
    """
    Test k=2 through max_k and report:
    - Inertia (elbow method)
    - Silhouette score (cluster quality)
    This helps validate that N_REGIMES=4 is a reasonable choice.
    """
    inertias    = []
    silhouettes = []
    k_range     = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor('#0d1117')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#c9d1d9')
        ax.spines[:].set_color('#30363d')
        ax.grid(color='#21262d', linewidth=0.5, linestyle='--')

    ax1.plot(list(k_range), inertias, 'o-', color='#58a6ff', lw=1.5)
    ax1.axvline(N_REGIMES, color='#f85149', lw=1.0, linestyle='--', label=f'Chosen k={N_REGIMES}')
    ax1.set_title('Elbow Method — Inertia', color='#e6edf3', fontsize=10)
    ax1.set_xlabel('Number of Clusters (k)', color='#c9d1d9')
    ax1.set_ylabel('Inertia', color='#c9d1d9')
    ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=8)

    ax2.plot(list(k_range), silhouettes, 'o-', color='#3fb950', lw=1.5)
    ax2.axvline(N_REGIMES, color='#f85149', lw=1.0, linestyle='--', label=f'Chosen k={N_REGIMES}')
    ax2.set_title('Silhouette Score — Cluster Quality', color='#e6edf3', fontsize=10)
    ax2.set_xlabel('Number of Clusters (k)', color='#c9d1d9')
    ax2.set_ylabel('Silhouette Score', color='#c9d1d9')
    ax2.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=8)

    plt.suptitle('TIAS — Regime Cluster Evaluation', color='#e6edf3', fontsize=11)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "phase3_cluster_eval.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[TIAS] Cluster eval chart saved → {path}")

    plt.show()

    print(f"\n[TIAS] Silhouette scores by k:")
    for k, s in zip(k_range, silhouettes):
        marker = " ← chosen" if k == N_REGIMES else ""
        print(f"  k={k}: {s:.4f}{marker}")

    return silhouettes


# ─────────────────────────────────────────────
# SECTION 3: FIT KMEANS
# ─────────────────────────────────────────────

def fit_kmeans(X_scaled: np.ndarray) -> KMeans:
    """
    Fit KMeans with N_REGIMES clusters.
    n_init=20: run 20 times with different seeds, keep best result.
    """
    km = KMeans(
        n_clusters=N_REGIMES,
        random_state=RANDOM_SEED,
        n_init=20,
        max_iter=500
    )
    km.fit(X_scaled)
    print(f"[TIAS] KMeans fitted. Inertia: {km.inertia_:.2f}")
    return km


# ─────────────────────────────────────────────
# SECTION 4: INTERPRET & LABEL CLUSTERS
# ─────────────────────────────────────────────

def interpret_clusters(df: pd.DataFrame, valid_idx, labels: np.ndarray) -> dict:
    """
    For each cluster, compute the mean of each regime feature.
    Use those means to assign a human-readable regime name.

    Logic:
    - High volatility + high atr_pct + extreme roc → CHAOTIC
    - High volatility + moderate momentum (ema_delta strong) → ACTIVE_TRENDING
    - Low volatility + strong ema_delta → CALM_TRENDING
    - High volatility + weak/mixed ema_delta + low roc → CHOPPY
    """
    feature_df = df.loc[valid_idx, REGIME_FEATURES].copy()
    feature_df['cluster'] = labels

    cluster_means = feature_df.groupby('cluster')[REGIME_FEATURES].mean()

    print("\n[TIAS] Cluster mean feature values:")
    print(cluster_means.round(4).to_string())

    # Rank clusters by volatility and momentum to assign names
    cluster_means['vol_rank']  = cluster_means['volatility'].rank()
    cluster_means['mom_score'] = (
        cluster_means['ema_delta'].abs() +
        cluster_means['roc'].abs() / 10
    )
    cluster_means['mom_rank']  = cluster_means['mom_score'].rank()

    cluster_label_map = {}

    for cluster_id, row in cluster_means.iterrows():
        vol_rank = row['vol_rank']
        mom_rank = row['mom_rank']
        n        = N_REGIMES

        if vol_rank == n:
            # Highest volatility cluster
            name = 'CHAOTIC'
        elif vol_rank >= n - 1 and mom_rank <= 2:
            # High vol, low momentum = choppy
            name = 'CHOPPY'
        elif vol_rank >= n - 1 and mom_rank >= n - 1:
            # High vol, high momentum = active trending
            name = 'ACTIVE_TRENDING'
        elif vol_rank <= 2 and mom_rank >= n - 1:
            # Low vol, high momentum = calm trending
            name = 'CALM_TRENDING'
        elif vol_rank <= 2:
            name = 'CALM_TRENDING'
        else:
            name = 'CHOPPY'

        cluster_label_map[cluster_id] = name
        print(f"  Cluster {cluster_id} → {name}  (vol_rank={vol_rank:.0f}, mom_rank={mom_rank:.0f})")

    return cluster_label_map, cluster_means


# ─────────────────────────────────────────────
# SECTION 5: ASSIGN REGIMES TO DATAFRAME
# ─────────────────────────────────────────────

def assign_regimes(df: pd.DataFrame, valid_idx, labels: np.ndarray,
                   cluster_label_map: dict) -> pd.DataFrame:
    """
    Add regime_cluster (int) and regime (string label) columns to df.
    Rows outside valid_idx (NaN warmup rows) get regime = 'UNKNOWN'.
    """
    df['regime_cluster'] = np.nan
    df['regime']         = 'UNKNOWN'

    df.loc[valid_idx, 'regime_cluster'] = labels

    for cluster_id, regime_name in cluster_label_map.items():
        mask = df['regime_cluster'] == cluster_id
        df.loc[mask, 'regime'] = regime_name

    # Regime distribution
    dist = df['regime'].value_counts()
    print("\n[TIAS] Regime distribution:")
    for regime, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {regime:<20}: {count:>4} bars  ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────
# SECTION 6: VALIDATE REGIMES LOGICALLY
# ─────────────────────────────────────────────

def validate_regimes(df: pd.DataFrame) -> None:
    """
    Sanity checks on regime assignments:
    - No regime should be 0% or 100% of all bars
    - CHAOTIC should not be the most common regime
    - Feature means per regime should follow expected ordering
    """
    print("\n[TIAS] Regime Validation:")
    issues = []

    valid_df   = df[df['regime'] != 'UNKNOWN']
    regime_pct = valid_df['regime'].value_counts(normalize=True) * 100

    # Check no regime dominates completely
    if regime_pct.max() > 80:
        issues.append(f"  WARNING: '{regime_pct.idxmax()}' dominates at {regime_pct.max():.1f}% — consider adjusting N_REGIMES")

    # Chaotic should not be dominant
    chaotic_pct = regime_pct.get('CHAOTIC', 0)
    if chaotic_pct > 40:
        issues.append(f"  WARNING: CHAOTIC regime is {chaotic_pct:.1f}% of all bars — unusually high")

    # Volatility ordering check
    regime_vol = valid_df.groupby('regime')['volatility'].mean().sort_values()
    print(f"  Volatility by regime (should increase: CALM → ACTIVE → CHOPPY → CHAOTIC):")
    for r, v in regime_vol.items():
        print(f"    {r:<20}: {v:.4f}")

    # RSI check — CHAOTIC and CHOPPY should have wider RSI spread
    regime_rsi_std = valid_df.groupby('regime')['rsi'].std()
    print(f"\n  RSI std by regime (CHOPPY/CHAOTIC should be higher):")
    for r, v in regime_rsi_std.sort_values(ascending=False).items():
        print(f"    {r:<20}: {v:.2f}")

    if issues:
        print("\n  Issues found:")
        for issue in issues:
            print(issue)
    else:
        print("\n  Regime validation passed.")


# ─────────────────────────────────────────────
# SECTION 7: VISUALIZATION
# ─────────────────────────────────────────────

def plot_regimes(df: pd.DataFrame, X_scaled: np.ndarray,
                 valid_idx, labels: np.ndarray, save: bool = True):
    """
    3-panel chart:
    1. Price colored by regime
    2. PCA scatter — do clusters separate visually?
    3. Regime timeline — how did regime evolve over time?
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                            height_ratios=[2, 2, 1])

    text_color = '#c9d1d9'
    grid_color = '#21262d'

    def style_ax(ax, title):
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors=text_color, labelsize=8)
        ax.spines[:].set_color('#30363d')
        ax.set_title(title, color='#e6edf3', fontsize=10, pad=6, loc='left')
        ax.grid(color=grid_color, linewidth=0.5, linestyle='--')

    # ── Panel 1: Price colored by regime ──
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, "Price  |  Colored by Market Regime")

    # Plot grey base line first
    ax1.plot(df.index, df['close'], color='#30363d', lw=1.0, zorder=1)

    # Overlay colored segments per regime
    for regime, color in REGIME_COLORS.items():
        mask = df['regime'] == regime
        if mask.sum() == 0:
            continue
        ax1.scatter(df.index[mask], df['close'][mask],
                    color=color, s=18, zorder=2, label=regime, alpha=0.9)

    legend_patches = [
        mpatches.Patch(color=c, label=r)
        for r, c in REGIME_COLORS.items()
        if r in df['regime'].values
    ]
    ax1.legend(handles=legend_patches, facecolor='#161b22',
               edgecolor='#30363d', labelcolor=text_color, fontsize=8,
               loc='upper left')

    # ── Panel 2: PCA scatter ──
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, "PCA Scatter  |  Cluster Separation")
    ax2.grid(False)

    pca    = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca  = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_ * 100

    regime_labels_for_valid = df.loc[valid_idx, 'regime'].values
    for regime, color in REGIME_COLORS.items():
        mask = regime_labels_for_valid == regime
        if mask.sum() == 0:
            continue
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=color, s=20, alpha=0.75, label=regime)

    ax2.set_xlabel(f'PC1 ({var_explained[0]:.1f}% var)', color=text_color, fontsize=8)
    ax2.set_ylabel(f'PC2 ({var_explained[1]:.1f}% var)', color=text_color, fontsize=8)
    ax2.legend(facecolor='#161b22', edgecolor='#30363d',
               labelcolor=text_color, fontsize=7)

    # ── Panel 3: Regime feature heatmap ──
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, "Regime Feature Profile  |  Mean Values (Scaled)")
    ax3.grid(False)

    valid_df       = df[df['regime'] != 'UNKNOWN']
    regime_means   = valid_df.groupby('regime')[REGIME_FEATURES].mean()

    # Normalize each feature 0-1 for heatmap readability
    norm_means = (regime_means - regime_means.min()) / (regime_means.max() - regime_means.min() + 1e-9)

    im = ax3.imshow(norm_means.values, aspect='auto', cmap='RdYlGn',
                    interpolation='nearest', vmin=0, vmax=1)

    ax3.set_xticks(range(len(REGIME_FEATURES)))
    ax3.set_xticklabels(REGIME_FEATURES, rotation=45, ha='right',
                        color=text_color, fontsize=7)
    ax3.set_yticks(range(len(norm_means.index)))
    ax3.set_yticklabels(norm_means.index, color=text_color, fontsize=8)
    plt.colorbar(im, ax=ax3, fraction=0.03).ax.tick_params(colors=text_color)

    # ── Panel 4: Regime timeline ──
    ax4 = fig.add_subplot(gs[2, :])
    style_ax(ax4, "Regime Timeline  |  How Market Context Evolved")
    ax4.grid(False)

    regime_order = ['CALM_TRENDING', 'ACTIVE_TRENDING', 'CHOPPY', 'CHAOTIC', 'UNKNOWN']
    regime_num   = {r: i for i, r in enumerate(regime_order)}

    regime_series = df['regime'].map(regime_num).fillna(4)
    colors_series = [REGIME_COLORS.get(r, '#6e7681') for r in df['regime']]

    ax4.bar(df.index, [1] * len(df), color=colors_series,
            width=1.5, alpha=0.85)
    ax4.set_yticks([])
    ax4.set_ylim(0, 1)

    # Add legend patches
    ax4.legend(handles=legend_patches, facecolor='#161b22',
               edgecolor='#30363d', labelcolor=text_color,
               fontsize=7, loc='upper right',
               ncol=len(legend_patches))

    plt.suptitle("TIAS — Phase 3: Market Context Engine",
                 color='#e6edf3', fontsize=13, y=1.01)

    if save:
        path = os.path.join(OUTPUT_DIR, "phase3_regimes.png")
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"[TIAS] Regime chart saved → {path}")

    plt.show()


# ─────────────────────────────────────────────
# SECTION 8: REGIME SUMMARY — CURRENT STATE
# ─────────────────────────────────────────────

def current_regime_report(df: pd.DataFrame) -> dict:
    """
    Report the current regime (last bar) and recent regime history.
    This is what a trader actually wants to know.
    """
    last = df.iloc[-1]
    recent = df['regime'].iloc[-10:]

    print("\n" + "="*55)
    print("  TIAS — CURRENT MARKET CONTEXT")
    print("="*55)
    print(f"  Current Regime     : {last['regime']}")
    print(f"  As of              : {df.index[-1].date()}")
    print(f"  Volatility (ann.)  : {last['volatility']:.4f}")
    print(f"  ATR %              : {last['atr_pct']:.4f}")
    print(f"  RSI                : {last['rsi']:.2f}")
    print(f"  EMA Delta %        : {last['ema_delta']:.4f}")
    print(f"  Range Ratio        : {last['range_ratio']:.4f}")
    print(f"  Volume Ratio       : {last['volume_ratio']:.4f}")
    print(f"\n  Last 10 Regimes    :")
    for dt, reg in recent.items():
        print(f"    {str(dt.date()):<14} {reg}")
    print("="*55 + "\n")

    return {
        'current_regime': last['regime'],
        'as_of': str(df.index[-1].date()),
        'volatility': round(last['volatility'], 4),
        'rsi': round(last['rsi'], 2),
        'ema_delta': round(last['ema_delta'], 4),
    }


# ─────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────

def detect_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full Phase 3 pipeline.
    Takes feature DataFrame from Phase 2.
    Returns DataFrame with regime columns added.
    """
    print("[TIAS] ── Starting Phase 3: Market Context Engine ──\n")

    X_scaled, valid_idx, scaler = prepare_regime_features(df)
    evaluate_cluster_counts(X_scaled)
    km                          = fit_kmeans(X_scaled)
    cluster_label_map, _        = interpret_clusters(df, valid_idx, km.labels_)
    df                          = assign_regimes(df, valid_idx, km.labels_, cluster_label_map)
    validate_regimes(df)
    plot_regimes(df, X_scaled, valid_idx, km.labels_)
    current_regime_report(df)

    print("[TIAS] ── Phase 3 Complete. Regime DataFrame ready. ──\n")
    return df