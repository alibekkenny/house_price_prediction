import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def brief_info(df, n=5):
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Daat preview:")
    print(df.head(n))