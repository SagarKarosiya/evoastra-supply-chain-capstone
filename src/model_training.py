from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(df):
    
    df = df.dropna()

    X = df[['lag_1','lag_7','rolling_mean']]
    y = df['demand']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test