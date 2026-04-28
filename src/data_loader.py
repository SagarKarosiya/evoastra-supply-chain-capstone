import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    
    print("Shape:", df.shape)
    print(df.head())
    
    return df