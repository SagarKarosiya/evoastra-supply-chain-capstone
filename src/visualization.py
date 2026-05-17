import os
import shap
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO)

class Visualizer:

    def __init__(self, output_dir="output/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_feature_importance(self, model, feature_names):
        importance = model.feature_importances_

        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Importance", y="Feature")
        plt.title("Feature Importance")

        plt.savefig(os.path.join(self.output_dir, "feature_importance.png"))
        plt.close()

    def plot_shap_summary(self, model, X_sample):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)

        plt.savefig(os.path.join(self.output_dir, "shap_summary.png"))
        plt.close()

    def plot_correlation_heatmap(self, df):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), cmap="coolwarm")
        plt.title("Correlation Heatmap")

        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"))
        plt.close()

    def plot_residuals(self, residual_df):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=residual_df["Predicted"],
            y=residual_df["Residuals"]
        )
        plt.axhline(0, linestyle="--")
        plt.title("Residual Analysis")

        plt.savefig(os.path.join(self.output_dir, "residual_plot.png"))
        plt.close()

    def plot_risk_distribution(self, df):
        plt.figure(figsize=(8, 5))
        sns.countplot(x="risk_level", data=df)
        plt.title("Risk Distribution")

        plt.savefig(os.path.join(self.output_dir, "risk_distribution.png"))
        plt.close()