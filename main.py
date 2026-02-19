# main.py

from modules.data_ingestor import ingest

if __name__ == "__main__":
    # Replace with your actual CSV path
    df = ingest("data/aapl_data.csv")

    # Quick sanity peek
    print(df.head())
    print(df.dtypes)
    print(df.shape)