import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["registration_time"] = pd.to_datetime(df["registration_time"])
    return df

def load_bound_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df