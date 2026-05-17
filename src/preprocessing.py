import pandas as pd

def preprocess_data(df):
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.ffill()
    
    # Convert date safely
    if "year" in df.columns:
        df['date'] = pd.to_datetime(df['year'], format='%Y')

    return df