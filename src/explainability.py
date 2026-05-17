import os
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)


class ExplainabilityEngine:

    def __init__(self, model, X_train, output_dir="outputs/explainability"):

        self.model = model
        self.X_train = X_train
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # SHAP explainer
        self.shap_explainer = shap.Explainer(model, X_train)

    # ---------------------------------------------------------
    # GLOBAL EXPLANATION (SHAP SUMMARY)
    # ---------------------------------------------------------
    def shap_summary(self, X_sample):

        shap_values = self.shap_explainer(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)

        path = os.path.join(self.output_dir, "shap_summary.png")
        plt.savefig(path)
        plt.close()

        logging.info("SHAP summary saved")

    # ---------------------------------------------------------
    # LOCAL EXPLANATION (FORCE PLOT)
    # ---------------------------------------------------------
    def shap_force_plot(self, X_sample, index=0):

        shap_values = self.shap_explainer(X_sample)

        force_plot = shap.plots.force(
            shap_values[index],
            matplotlib=True,
            show=False
        )

        path = os.path.join(self.output_dir, f"force_plot_{index}.png")
        plt.savefig(path)
        plt.close()

        logging.info("Force plot saved")

    # ---------------------------------------------------------
    # DEPENDENCY PLOT
    # ---------------------------------------------------------
    def shap_dependence_plot(self, feature, X_sample):

        shap_values = self.shap_explainer(X_sample)

        shap.dependence_plot(
            feature,
            shap_values.values,
            X_sample,
            show=False
        )

        path = os.path.join(self.output_dir, f"dependence_{feature}.png")
        plt.savefig(path)
        plt.close()

        logging.info(f"Dependence plot for {feature} saved")

    # ---------------------------------------------------------
    # LIME (LOCAL INTERPRETABILITY)
    # ---------------------------------------------------------
    def lime_explanation(self, X_sample, index=0):

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.X_train),
            feature_names=self.X_train.columns.tolist(),
            mode="regression"
        )

        exp = explainer.explain_instance(
            X_sample.iloc[index].values,
            self.model.predict
        )

        path = os.path.join(self.output_dir, f"lime_{index}.html")
        exp.save_to_file(path)

        logging.info("LIME explanation saved")

    # ---------------------------------------------------------
    # STAKEHOLDER REPORT (TEXT)
    # ---------------------------------------------------------
    def generate_business_explanation(self, X_sample, index=0):

        shap_values = self.shap_explainer(X_sample)

        contributions = shap_values.values[index]
        features = X_sample.columns

        explanation = sorted(
            zip(features, contributions),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        report_path = os.path.join(
            self.output_dir,
            f"business_explanation_{index}.txt"
        )

        with open(report_path, "w") as f:

            f.write("BUSINESS EXPLANATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            for feature, impact in explanation[:10]:
                f.write(f"{feature} → Impact: {round(impact, 4)}\n")

        logging.info("Business explanation generated")