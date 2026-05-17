import os
import logging
import pandas as pd
import numpy as np

from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from statsmodels.tsa.stattools import grangercausalitytests

logging.basicConfig(level=logging.INFO)


class StatisticalAnalyzer:

    def __init__(self, df,
                 output_dir="output/plots",
                 report_dir="output/reports"):

        self.df = df.copy()
        self.output_dir = output_dir
        self.report_dir = report_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    # ---------------------------------------------------------
    # MAIN CONTROLLER (USER SELECT TEST)
    # ---------------------------------------------------------

    def run_test(self, test_name, **kwargs):

        if test_name == "t_test":
            return self.t_test(**kwargs)

        elif test_name == "anova":
            return self.anova(**kwargs)

        elif test_name == "chi_square":
            return self.chi_square(**kwargs)

        elif test_name == "granger":
            return self.granger_test(**kwargs)

        else:
            raise ValueError("Invalid test selected")

    # ---------------------------------------------------------
    # T-TEST
    # ---------------------------------------------------------

    def t_test(self, col1, col2, threshold=None):

        if threshold is not None:
            group1 = self.df[self.df[col1] >= threshold][col2]
            group2 = self.df[self.df[col1] < threshold][col2]
        else:
            raise ValueError("Threshold required for t-test grouping")

        t_stat, p_val = ttest_ind(
            group1.dropna(),
            group2.dropna(),
            equal_var=False
        )

        return {
            "test": "t-test",
            "t_stat": t_stat,
            "p_value": p_val,
            "significant": p_val < 0.05
        }

    # ---------------------------------------------------------
    # ANOVA
    # ---------------------------------------------------------

    def anova(self, category_col, value_col):

        groups = [
            self.df[self.df[category_col] == c][value_col].dropna()
            for c in self.df[category_col].dropna().unique()
        ]

        if len(groups) < 2:
            return {"error": "Not enough groups for ANOVA"}

        f_stat, p_val = f_oneway(*groups)

        return {
            "test": "anova",
            "f_stat": f_stat,
            "p_value": p_val,
            "significant": p_val < 0.05
        }

    # ---------------------------------------------------------
    # CHI-SQUARE
    # ---------------------------------------------------------

    def chi_square(self, col1, col2):

        contingency = pd.crosstab(self.df[col1], self.df[col2])

        chi2, p, dof, _ = chi2_contingency(contingency)

        return {
            "test": "chi-square",
            "chi2_stat": chi2,
            "p_value": p,
            "significant": p < 0.05
        }

    # ---------------------------------------------------------
    # GRANGER CAUSALITY
    # ---------------------------------------------------------

    def granger_test(self, col1, col2, max_lag=5):

        data = self.df[[col1, col2]].dropna()

        result = grangercausalitytests(data, max_lag, verbose=False)

        p_values = [
            result[i+1][0]['ssr_ftest'][1]
            for i in range(max_lag)
        ]

        return {
            "test": "granger",
            "p_values_by_lag": p_values,
            "causality_detected": any(p < 0.05 for p in p_values)
        }

    # ---------------------------------------------------------
    # REPORT GENERATION
    # ---------------------------------------------------------

    def save_report(self, results, filename="stat_test_report.txt"):

        path = os.path.join(self.report_dir, filename)

        with open(path, "w") as f:
            f.write("STATISTICAL TEST REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(str(results))

        return path