from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features, apply_pca
from src.visualization import Visualizer
from src.risk_analysis import RiskAnalyzer
from src.stats_analysis import StatisticalAnalyzer
from src.model_training import ModelTrainer
from src.evaluation_system import EvaluationSystem
from src.feature_analysis import FeatureAnalyzer
from src.time_series import forecast_demand
from src.hyperparameter_tuning import HyperparameterTuner
from src.explainability import ExplainabilityEngine
from src.fairness import FairnessAnalyzer
from evidently.metric_preset import TargetDriftPreset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import pandas as pd


class SupplyChainPipeline:

    def __init__(self, config):
        self.config = config
        self.viz = Visualizer()
        
        
    monitor = DriftMonitor(reference_df=train_data, current_df=new_data)
    monitor.generate_drift_report()
    
    report = Report(metrics=[TargetDriftPreset()])
    
    metrics_log = {
    "rmse": current_rmse,
    "timestamp": pd.Timestamp.now()
}
    metrics_log.csv
    
    if current_rmse > baseline_rmse * 1.2:
     print("⚠️ ALERT: Model degradation detected")
    
    
    if drift_score > 0.3:
     print("⚠️ Data drift detected")
     
    def retrain_if_needed():

     if drift_detected or performance_drop:
        print("🔄 Retraining model...")
        os.system("python run_pipeline.py")
    
    
    def run(self):

        # -----------------------------
        # 1. Load Data
        # -----------------------------
        df = load_data(self.config.DATA_PATH)

        # -----------------------------
        # 2. Preprocess
        # -----------------------------
        df = preprocess_data(df)

        # -----------------------------
        # 3. Feature Engineering
        # -----------------------------
        df = create_features(df)

        pca_df = apply_pca(df)
        df = pd.concat([df, pca_df], axis=1)

        # -----------------------------
        # 4. Feature Analysis
        # -----------------------------
        fa = FeatureAnalyzer()

        X = df.drop(columns=[self.config.TARGET])
        y = df[self.config.TARGET]

        fa.correlation_analysis(df)
        mi_df = fa.mutual_information(X, y)
        fa.rank_features(mi_df)
        fa.plot_interactive_importance(mi_df)

        # -----------------------------
        # 5. Statistical Analysis
        # -----------------------------
        stats = StatisticalAnalyzer(df)

        stats.save_report({
            "t_test": stats.run_test(
                "t_test",
                col1="supplier_rating",
                col2="delivery_delay",
                threshold=4
            ),
            "anova": stats.run_test(
                "anova",
                category_col="product_category",
                value_col="demand_quantity"
            )
        })

        # -----------------------------
        # 6. Train-Test Split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )

        # -----------------------------
        # 7. Model Training (Baseline + Advanced)
        # -----------------------------
        trainer = ModelTrainer()

        # Baseline models
        base_model, model_results = trainer.train_regression_models(
            X_train, X_test, y_train, y_test
        )

        # Advanced models
        advanced_results = trainer.train_advanced_models(
            X_train, X_test, y_train, y_test
        )

        # Merge all results
        all_results = {}

        for k, v in model_results.items():
            all_results[k] = v

        for k, v in advanced_results.items():
            all_results[k] = v

        # Safety check
        if len(all_results) == 0:
            raise ValueError("No models trained successfully")

        # Select best model (based on RMSE)
        best_model_name = min(
            all_results,
            key=lambda x: all_results[x]["RMSE"]
        )

        best_model = all_results[best_model_name]["model"]

        print(f"🏆 Best Model: {best_model_name}")
        
        # -----------------------------
        # HYPERPARAMETER TUNING
        # -----------------------------
        tuner = HyperparameterTuner(X_train, y_train)

        # Grid Search (RF)
        rf_best = tuner.grid_search_rf()

        # Random Search (XGB)
        xgb_best = tuner.random_search_xgb()

        # Optuna (Advanced)
        optuna_best = tuner.optuna_xgb(trials=20)

        print("✅ Hyperparameter tuning completed")
        
        # -----------------------------
        # 8. Model Evaluation
        # -----------------------------
        evaluator = EvaluationSystem()

        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso()
        }

        comparison_df = evaluator.compare_models(
            models, X_train, X_test, y_train, y_test
        )

        evaluator.save_report(comparison_df)

        # -----------------------------
        # 9. Explainability
        # -----------------------------
        explainer = ExplainabilityEngine(
        best_model,
        X_train
         )

        X_sample = X_test.sample(min(100, len(X_test)))

        # SHAP
        explainer.shap_summary(X_sample)
        explainer.shap_force_plot(X_sample, index=0)

        # Dependency
        explainer.shap_dependence_plot(
          feature=X.columns[0],
          X_sample=X_sample
           )

         # LIME
        explainer.lime_explanation(X_sample, index=0)

        # Business Report
        explainer.generate_business_explanation(X_sample, index=0)

        # -----------------------------
        # 10. Risk Analysis
        # -----------------------------
        risk = RiskAnalyzer(df)

        df = risk.calculate_risk_score()
        df = risk.categorize_risk()
        df = risk.calculate_financial_risk()

        risk.supplier_risk_exposure()
        risk.top_risk_nodes()
        risk.scenario_analysis_advanced()
        risk.generate_executive_summary()

        self.viz.plot_risk_distribution(df)

        # -----------------------------
        # 11. Forecasting
        # -----------------------------
        forecast = forecast_demand(df[self.config.TARGET])
        
        # -----------------------------
        # 12. FAIRNESS ANALYSIS
        # -----------------------------
        # Choose sensitive feature (example)
        sensitive_feature = X_test["region"] if "region" in X_test.columns else None

        if sensitive_feature is not None:

         fairness = FairnessAnalyzer(
          best_model,
           X_test,
             y_test,
              sensitive_feature
               )

        dp = fairness.demographic_parity()
        eo = fairness.equal_opportunity()
        gap = fairness.fairness_gap()

        print("Demographic Parity:\n", dp)
        print("Equal Opportunity:\n", eo)
        print("Fairness Gap:", gap)

        fairness.generate_report()
        
        

        # -----------------------------
        # DONE
        # -----------------------------
        print("✅ Pipeline executed successfully")

        return {
            "model": best_model,
            "best_model_name": best_model_name,
            "data": df,
            "forecast": forecast,
            "comparison": comparison_df,
            "all_model_results": all_results
        }