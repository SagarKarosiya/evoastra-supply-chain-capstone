class Config:
    DATA_PATH = "data/processed/final_dataset.csv"

    TARGET = "demand_quantity"
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    MODEL_PATH = "output/models/model.pkl"
    OUTPUT_DIR = "output"