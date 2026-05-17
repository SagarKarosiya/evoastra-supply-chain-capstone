import pandas as pd
import numpy as np
import logging

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate
)

logging.basicConfig(level=logging.INFO)


class FairnessAnalyzer:

    def __init__(self, model, X_test, y_test, sensitive_features):

        self.model = model
        self.X = X_test
        self.y = y_test
        self.sensitive = sensitive_features

        self.preds = model.predict(X_test)

    # ---------------------------------------------------------
    # DEMOGRAPHIC PARITY
    # ---------------------------------------------------------
    def demographic_parity(self):

        mf = MetricFrame(
            metrics=selection_rate,
            y_true=self.y,
            y_pred=self.preds,
            sensitive_features=self.sensitive
        )

        result = mf.by_group

        logging.info("Demographic Parity calculated")

        return result

    # ---------------------------------------------------------
    # EQUAL OPPORTUNITY
    # ---------------------------------------------------------
    def equal_opportunity(self):

        mf = MetricFrame(
            metrics=true_positive_rate,
            y_true=self.y,
            y_pred=self.preds,
            sensitive_features=self.sensitive
        )

        result = mf.by_group

        logging.info("Equal Opportunity calculated")

        return result

    # ---------------------------------------------------------
    # FAIRNESS GAP
    # ---------------------------------------------------------
    def fairness_gap(self):

        dp = self.demographic_parity()

        gap = dp.max() - dp.min()

        return {"fairness_gap": gap}

    # ---------------------------------------------------------
    # GENERATE REPORT
    # ---------------------------------------------------------
    def generate_report(self, output_path="reports/fairness_report.txt"):

        dp = self.demographic_parity()
        eo = self.equal_opportunity()
        gap = self.fairness_gap()

        with open(output_path, "w") as f:

            f.write("FAIRNESS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("Demographic Parity:\n")
            f.write(str(dp) + "\n\n")

            f.write("Equal Opportunity:\n")
            f.write(str(eo) + "\n\n")

            f.write("Fairness Gap:\n")
            f.write(str(gap) + "\n\n")

        logging.info("Fairness report generated")