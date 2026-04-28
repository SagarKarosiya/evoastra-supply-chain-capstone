def create_features(df):
    
    # Lag features
    df['lag_1'] = df['demand'].shift(1)
    df['lag_7'] = df['demand'].shift(7)

    # Rolling
    df['rolling_mean'] = df['demand'].rolling(7).mean()

    # Time features
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.dayofweek

    return df