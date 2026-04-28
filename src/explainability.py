import shap

def explain(model, X):
    
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    shap.plots.bar(shap_values)