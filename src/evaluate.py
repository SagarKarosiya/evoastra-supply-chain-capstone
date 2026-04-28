from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate(model, X_test, y_test):
    
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print("RMSE:", rmse)