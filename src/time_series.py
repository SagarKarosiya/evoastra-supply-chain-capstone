from statsmodels.tsa.arima.model import ARIMA

def run_arima(df):
    
    model = ARIMA(df['demand'], order=(2,1,2))
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=10)
    
    print(forecast)