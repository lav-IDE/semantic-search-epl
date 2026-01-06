import pandas as pd

def load_csv_path(csv_path):
    
    df = pd.read_csv(csv_path)
    return df