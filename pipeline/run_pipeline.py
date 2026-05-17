from pipeline import SupplyChainPipeline

class Config:
    DATA_PATH = "data/processed/data.csv"
    TARGET = "demand_quantity"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

if __name__ == "__main__":
    pipeline = SupplyChainPipeline(Config())
    result = pipeline.run()

    print("Best Model:", result["best_model_name"])