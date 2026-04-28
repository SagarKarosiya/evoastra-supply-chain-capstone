import pandas as pd

def clean_data(df):
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.ffill()
    
    # Convert date
    df['date'] = pd.to_datetime(df['year'], format='%Y')
    
    return df