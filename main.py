# main.py

from modules.data_ingestor import ingest
from modules.feature_engine import engineer_features
from modules.market_context import detect_regimes

if __name__ == "__main__":
    df = ingest("data/aapl_data.csv")
    df = engineer_features(df)
    df = detect_regimes(df)

    # Quick look at recent regime history
    print(df[['close', 'volatility', 'rsi', 'regime']].tail(15).to_string())