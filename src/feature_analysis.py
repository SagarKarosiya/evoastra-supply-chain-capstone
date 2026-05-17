import os
import logging
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.feature_selection import mutual_info_regression
import shap

logging.basicConfig(level=logging.INFO)


class FeatureAnalyzer:

    def __init__(self, output_dir="output/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # CORRELATION ANALYSIS
    # ---------------------------------------------------------

    def correlation_analysis(self, df):

        numeric_df = df.select_dtypes(include=[np.number])

        pearson = numeric_df.corr(method="pearson")
        spearman = numeric_df.corr(method="spearman")

        # Save heatmaps
        plt.figure(figsize=(10, 6))
        sns.heatmap(pearson, cmap="coolwarm", annot=False)
        plt.title("Pearson Correlation")
        plt.savefig(f"{self.output_dir}/pearson_corr.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.heatmap(spearman, cmap="viridis", annot=False)
        plt.title("Spearman Correlation")
        plt.savefig(f"{self.output_dir}/spearman_corr.png")
        plt.close()

        return pearson, spearman

    # ---------------------------------------------------------
    # MUTUAL INFORMATION (VERY IMPORTANT)
    # ---------------------------------------------------------

    def mutual_information(self, X, y):

        mi = mutual_info_regression(X, y)

        mi_df = pd.DataFrame({
            "Feature": X.columns,
            "MI Score": mi
        }).sort_values(by="MI Score", ascending=False)

        mi_df.to_csv(f"{self.output_dir}/mutual_info.csv", index=False)

        return mi_df

    # ---------------------------------------------------------
    # FEATURE RANKING
    # ---------------------------------------------------------

    def rank_features(self, mi_df):

        ranked = mi_df.copy()
        ranked["Rank"] = range(1, len(ranked) + 1)

        ranked.to_csv(f"{self.output_dir}/feature_ranking.csv", index=False)

        return ranked

    # ---------------------------------------------------------
    # SHAP EXPLAINABILITY
    # ---------------------------------------------------------

    def shap_analysis(self, model, X_sample):

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)

        plt.savefig(f"{self.output_dir}/shap_summary.png")
        plt.close()

    # ---------------------------------------------------------
    # PLOTLY INTERACTIVE FEATURE IMPORTANCE
    # ---------------------------------------------------------

    def plot_interactive_importance(self, mi_df):

        fig = px.bar(
            mi_df.head(20),
            x="MI Score",
            y="Feature",
            orientation="h",
            title="Top Features by Mutual Information"
        )

        fig.write_html(f"{self.output_dir}/interactive_feature_importance.html")