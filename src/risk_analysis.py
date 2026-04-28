import numpy as np

def monte_carlo(df):
    
    simulations = 1000
    results = []

    for _ in range(simulations):
        demand = np.random.normal(df['demand'].mean(), df['demand'].std())
        risk = demand * 0.1
        
        results.append(risk)

    print("Expected Risk:", np.mean(results))