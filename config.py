#Expected CSV Columns (case-insensitive)
REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

#Different brokers name time columns differently.
TIMESTAMP_CANDIDATES = ['datetime', 'date', 'time', 'timestamp', 'Date', 'Datetime']

#Flag if any single bar moves more than 20%
MAX_SINGLE_BAR_MOVE_PCT = 20.0

#Prices below this are invalid
MIN_PRICE = 0.0001

#Set True for indices/synthetic instruments
ALLOW_ZERO_VOLUME = False

# Output paths
OUTPUT_DIR = "outputs/"
DATA_DIR = "data/"

# ── Phase 2: Feature Engineering ──────────────────

# Lookback windows
VOLATILITY_WINDOW    = 20   # rolling std of returns
ATR_WINDOW           = 14   # Average True Range
MOMENTUM_WINDOW      = 14   # Rate of change
EMA_FAST             = 9
EMA_SLOW             = 21
BB_WINDOW            = 20   # Bollinger Band window
BB_STD               = 2.0  # Bollinger Band std multiplier
VOLUME_MA_WINDOW     = 20   # volume moving average

# ── Phase 3: Market Context Engine ────────────────────

# Features used for regime clustering
REGIME_FEATURES = [
    'volatility',
    'atr_pct',
    'rsi',
    'ema_delta',
    'roc',
    'range_ratio',
    'bb_width',
    'volume_ratio'
]

# Number of clusters — we use 4 regimes
N_REGIMES = 4

# Random seed for reproducibility
RANDOM_SEED = 42