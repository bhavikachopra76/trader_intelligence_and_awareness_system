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