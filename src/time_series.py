from statsmodels.tsa.arima.model import ARIMA

def forecast_demand(series):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=10)
    return forecast