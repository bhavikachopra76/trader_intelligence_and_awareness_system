# modules/feature_engine.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from config import (
    VOLATILITY_WINDOW, ATR_WINDOW, MOMENTUM_WINDOW,
    EMA_FAST, EMA_SLOW, BB_WINDOW, BB_STD,
    VOLUME_MA_WINDOW, OUTPUT_DIR
)


# ─────────────────────────────────────────────
# SECTION 1: RETURNS
# ─────────────────────────────────────────────

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns: more statistically stable than simple returns.
    Also compute simple returns for momentum calculations.
    """
    df['log_return']    = np.log(df['close'] / df['close'].shift(1))
    df['simple_return'] = df['close'].pct_change()
    return df


# ─────────────────────────────────────────────
# SECTION 2: VOLATILITY FEATURES
# ─────────────────────────────────────────────

def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling volatility: std of log returns over VOLATILITY_WINDOW bars.
    Normalized volatility: how does current vol compare to its own history?
    ATR: Average True Range — measures bar-by-bar price range.
    ATR %: ATR as a % of close — makes it comparable across price levels.
    """
    # Rolling volatility (annualized for daily data)
    df['volatility'] = (
        df['log_return']
        .rolling(VOLATILITY_WINDOW)
        .std() * np.sqrt(252)
    )

    # Normalized volatility — z-score of volatility vs its own history
    vol_mean = df['volatility'].rolling(VOLATILITY_WINDOW * 3).mean()
    vol_std  = df['volatility'].rolling(VOLATILITY_WINDOW * 3).std()
    df['volatility_zscore'] = (df['volatility'] - vol_mean) / (vol_std + 1e-9)

    # True Range
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs()
    ], axis=1).max(axis=1)

    df['atr']   = tr.rolling(ATR_WINDOW).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100  # as % of price

    return df


# ─────────────────────────────────────────────
# SECTION 3: MOMENTUM FEATURES
# ─────────────────────────────────────────────

def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rate of Change (ROC): % move over N bars — pure directional force.
    EMA crossover delta: fast EMA minus slow EMA, normalized by price.
      Positive = bullish alignment. Negative = bearish.
    RSI: momentum oscillator — overbought / oversold awareness.
    """
    # Rate of Change
    df['roc'] = df['close'].pct_change(periods=MOMENTUM_WINDOW) * 100

    # EMA fast and slow
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['ema_delta'] = (df['ema_fast'] - df['ema_slow']) / df['close'] * 100

    # RSI
    delta = df['close'].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    return df


# ─────────────────────────────────────────────
# SECTION 4: SPEED FEATURES
# ─────────────────────────────────────────────

def compute_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Speed = how fast is price moving bar to bar?
    This is different from momentum (direction) — speed is magnitude only.

    bar_speed: absolute close-to-close move per bar
    speed_ma:  smoothed speed over N bars
    speed_zscore: is current speed unusual vs recent history?
    """
    df['bar_speed']     = df['close'].diff().abs()
    df['speed_ma']      = df['bar_speed'].rolling(VOLATILITY_WINDOW).mean()
    speed_std           = df['bar_speed'].rolling(VOLATILITY_WINDOW).std()
    df['speed_zscore']  = (df['bar_speed'] - df['speed_ma']) / (speed_std + 1e-9)

    return df


# ─────────────────────────────────────────────
# SECTION 5: EXPANSION / COMPRESSION FEATURES
# ─────────────────────────────────────────────

def compute_expansion_compression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expansion: bars are getting larger — market is waking up.
    Compression: bars are getting smaller — market is coiling.

    bar_range: High - Low of each bar (raw bar size)
    range_ma: smoothed average bar size
    range_ratio: current bar size vs its average
      > 1.0 = expansion (larger than normal)
      < 1.0 = compression (smaller than normal)

    Bollinger Band Width: another measure of compression/expansion.
      Squeeze = BB narrowing = low bandwidth = coiling.
    """
    df['bar_range']   = df['high'] - df['low']
    df['range_ma']    = df['bar_range'].rolling(VOLATILITY_WINDOW).mean()
    df['range_ratio'] = df['bar_range'] / (df['range_ma'] + 1e-9)

    # Bollinger Bands
    bb_mid            = df['close'].rolling(BB_WINDOW).mean()
    bb_std            = df['close'].rolling(BB_WINDOW).std()
    df['bb_upper']    = bb_mid + BB_STD * bb_std
    df['bb_lower']    = bb_mid - BB_STD * bb_std
    df['bb_width']    = (df['bb_upper'] - df['bb_lower']) / (bb_mid + 1e-9)

    # BB %B — where is price within the bands? 0=at lower, 1=at upper
    df['bb_pct_b']    = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)

    return df


# ─────────────────────────────────────────────
# SECTION 6: VOLUME FEATURES
# ─────────────────────────────────────────────

def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume tells you conviction behind price moves.
    High volume on a move = real. Low volume = suspicious.

    volume_ma: smoothed average volume
    volume_ratio: today's volume vs average (>1 = above average activity)
    volume_zscore: how unusual is today's volume?
    """
    df['volume_ma']     = df['volume'].rolling(VOLUME_MA_WINDOW).mean()
    df['volume_ratio']  = df['volume'] / (df['volume_ma'] + 1e-9)
    vol_std             = df['volume'].rolling(VOLUME_MA_WINDOW).std()
    df['volume_zscore'] = (df['volume'] - df['volume_ma']) / (vol_std + 1e-9)

    return df


# ─────────────────────────────────────────────
# SECTION 7: FEATURE VALIDATION
# ─────────────────────────────────────────────

def validate_features(df: pd.DataFrame) -> None:
    """
    Sanity check every feature we built.
    Prints a clean report — no silent failures.
    """
    feature_cols = [
        'log_return', 'simple_return',
        'volatility', 'volatility_zscore', 'atr', 'atr_pct',
        'roc', 'ema_delta', 'rsi',
        'bar_speed', 'speed_zscore',
        'bar_range', 'range_ratio', 'bb_width', 'bb_pct_b',
        'volume_ratio', 'volume_zscore'
    ]

    print("\n" + "="*60)
    print("  TIAS — FEATURE VALIDATION REPORT")
    print("="*60)
    print(f"  {'Feature':<22} {'NaNs':>6} {'Min':>10} {'Max':>10} {'Mean':>10}")
    print("-"*60)

    issues = []

    for col in feature_cols:
        if col not in df.columns:
            issues.append(f"  MISSING FEATURE: {col}")
            continue

        series  = df[col].dropna()
        n_nan   = df[col].isna().sum()
        mn      = series.min()
        mx      = series.max()
        mean    = series.mean()

        print(f"  {col:<22} {n_nan:>6} {mn:>10.4f} {mx:>10.4f} {mean:>10.4f}")

        # Logical checks
        if col == 'rsi' and (mx > 100 or mn < 0):
            issues.append(f"  RSI out of range: min={mn:.2f}, max={mx:.2f}")
        if col == 'volatility' and mn < 0:
            issues.append(f"  Negative volatility detected")
        if col == 'atr' and mn < 0:
            issues.append(f"  Negative ATR detected")
        if col == 'volume_ratio' and mn < 0:
            issues.append(f"  Negative volume ratio")
        if col == 'range_ratio' and mn < 0:
            issues.append(f"  Negative range ratio")
        if n_nan > len(df) * 0.5:
            issues.append(f"  '{col}' has >50% NaNs — check window size vs data length")

    print("="*60)

    if issues:
        print("\n  ISSUES FOUND:")
        for issue in issues:
            print(issue)
    else:
        print("\n  All features passed validation. No issues found.")

    print("="*60 + "\n")


# ─────────────────────────────────────────────
# SECTION 8: VISUALIZATION
# ─────────────────────────────────────────────

def plot_features(df: pd.DataFrame, save: bool = True):
    """
    Multi-panel chart showing each feature category.
    Dark trading-desk style.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(6, 1, figure=fig, hspace=0.5)

    panel_style = dict(facecolor='#0d1117')
    text_color  = '#c9d1d9'
    grid_color  = '#21262d'

    def style_ax(ax, title):
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors=text_color, labelsize=8)
        ax.spines[:].set_color('#30363d')
        ax.set_title(title, color='#e6edf3', fontsize=10, pad=6, loc='left')
        ax.yaxis.label.set_color(text_color)
        ax.grid(color=grid_color, linewidth=0.5, linestyle='--')

    idx = df.index

    # Panel 1 — Price + EMAs
    ax1 = fig.add_subplot(gs[0])
    style_ax(ax1, "Price  |  EMA Fast vs Slow")
    ax1.plot(idx, df['close'],    color='#58a6ff', lw=1.2, label='Close')
    ax1.plot(idx, df['ema_fast'], color='#f0883e', lw=0.9, label=f'EMA {EMA_FAST}', alpha=0.85)
    ax1.plot(idx, df['ema_slow'], color='#bc8cff', lw=0.9, label=f'EMA {EMA_SLOW}', alpha=0.85)
    ax1.fill_between(idx, df['bb_upper'], df['bb_lower'], alpha=0.07, color='#58a6ff')
    ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=text_color, fontsize=7)

    # Panel 2 — Volatility
    ax2 = fig.add_subplot(gs[1])
    style_ax(ax2, "Volatility  |  Annualized Rolling Std  +  ATR %")
    ax2.plot(idx, df['volatility'], color='#ffa657', lw=1.0, label='Volatility (ann.)')
    ax2_r = ax2.twinx()
    ax2_r.plot(idx, df['atr_pct'], color='#ff7b72', lw=0.8, alpha=0.7, label='ATR %')
    ax2_r.tick_params(colors=text_color, labelsize=8)
    ax2_r.spines[:].set_color('#30363d')
    ax2.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=text_color, fontsize=7, loc='upper left')
    ax2_r.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=text_color, fontsize=7, loc='upper right')

    # Panel 3 — Momentum
    ax3 = fig.add_subplot(gs[2])
    style_ax(ax3, "Momentum  |  RSI  +  EMA Delta")
    ax3.plot(idx, df['rsi'],       color='#79c0ff', lw=1.0, label='RSI')
    ax3.axhline(70, color='#f85149', lw=0.7, linestyle='--', alpha=0.7)
    ax3.axhline(30, color='#3fb950', lw=0.7, linestyle='--', alpha=0.7)
    ax3.axhline(50, color='#6e7681', lw=0.5, linestyle=':')
    ax3_r = ax3.twinx()
    ax3_r.plot(idx, df['ema_delta'], color='#f0883e', lw=0.8, alpha=0.8, label='EMA Delta %')
    ax3_r.axhline(0, color='#6e7681', lw=0.5)
    ax3_r.tick_params(colors=text_color, labelsize=8)
    ax3_r.spines[:].set_color('#30363d')
    ax3.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=text_color, fontsize=7, loc='upper left')
    ax3_r.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=text_color, fontsize=7, loc='upper right')

    # Panel 4 — Speed
    ax4 = fig.add_subplot(gs[3])
    style_ax(ax4, "Speed  |  Bar-to-Bar Price Movement")
    ax4.plot(idx, df['bar_speed'], color='#a5d6ff', lw=0.8, alpha=0.7, label='Bar Speed')
    ax4.plot(idx, df['speed_ma'],  color='#ffa657', lw=1.1, label='Speed MA')
    ax4.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=text_color, fontsize=7)

    # Panel 5 — Expansion / Compression
    ax5 = fig.add_subplot(gs[4])
    style_ax(ax5, "Expansion / Compression  |  Range Ratio  +  BB Width")
    ax5.plot(idx, df['range_ratio'], color='#56d364', lw=0.9, label='Range Ratio')
    ax5.axhline(1.0, color='#6e7681', lw=0.6, linestyle='--')
    ax5_r = ax5.twinx()
    ax5_r.plot(idx, df['bb_width'], color='#bc8cff', lw=0.8, alpha=0.8, label='BB Width')
    ax5_r.tick_params(colors=text_color, labelsize=8)
    ax5_r.spines[:].set_color('#30363d')
    ax5.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=text_color, fontsize=7, loc='upper left')
    ax5_r.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=text_color, fontsize=7, loc='upper right')

    # Panel 6 — Volume
    ax6 = fig.add_subplot(gs[5])
    style_ax(ax6, "Volume  |  Volume Ratio vs Average")
    colors = ['#3fb950' if v >= 1.0 else '#f85149' for v in df['volume_ratio'].fillna(1.0)]
    ax6.bar(idx, df['volume_ratio'], color=colors, alpha=0.7, width=1.0)
    ax6.axhline(1.0, color='#ffa657', lw=0.8, linestyle='--', label='Average (1.0)')
    ax6.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=text_color, fontsize=7)

    plt.suptitle("TIAS — Phase 2: Feature Engineering Dashboard",
                 color='#e6edf3', fontsize=13, y=0.995)

    if save:
        path = os.path.join(OUTPUT_DIR, "phase2_features.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[TIAS] Feature chart saved → {path}")

    plt.show()


# ─────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full Phase 2 pipeline.
    Takes clean DataFrame from Phase 1.
    Returns DataFrame enriched with all behavioral features.
    """
    print("[TIAS] ── Starting Phase 2: Feature Engineering ──\n")

    df = compute_returns(df)
    df = compute_volatility(df)
    df = compute_momentum(df)
    df = compute_speed(df)
    df = compute_expansion_compression(df)
    df = compute_volume_features(df)

    validate_features(df)
    plot_features(df)

    print("[TIAS] ── Phase 2 Complete. Feature DataFrame ready. ──\n")
    return df