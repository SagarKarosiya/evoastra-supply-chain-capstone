import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

class RiskAnalyzer:

    def __init__(self, dataframe, output_dir="output/reports"):
        self.df = dataframe.copy()
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # CALCULATE RISK SCORE
    # ---------------------------------------------------------
    def calculate_risk_score(self):

        numeric_df = self.df.select_dtypes(include=["number"])

        # Drop target if present
        if "demand_quantity" in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=["demand_quantity"])

        self.df["risk_score"] = numeric_df.mean(axis=1)

        # Normalize safely
        self.df["risk_score"] = (
            self.df["risk_score"] - self.df["risk_score"].min()
        ) / (
            self.df["risk_score"].max() - self.df["risk_score"].min() + 1e-8
        )

        return self.df

    # ---------------------------------------------------------
    # CATEGORIZE RISK
    # ---------------------------------------------------------
    def categorize_risk(self):

        def risk_label(score):
            if score <= 0.3:
                return "Low"
            elif score <= 0.7:
                return "Medium"
            return "High"

        self.df["risk_level"] = self.df["risk_score"].apply(risk_label)

        return self.df

    # ---------------------------------------------------------
    # SCENARIO ANALYSIS
    # ---------------------------------------------------------
    def scenario_analysis(self):

        scenarios = pd.DataFrame()

        base = self.df["risk_score"]

        scenarios["base_case"] = base
        scenarios["best_case"] = base * 0.8
        scenarios["likely_case"] = base
        scenarios["worst_case"] = base * 1.2

        path = os.path.join(self.output_dir, "scenario_analysis.csv")
        scenarios.to_csv(path, index=False)

        return scenarios

    # ---------------------------------------------------------
    # RISK MATRIX
    # ---------------------------------------------------------
    def generate_risk_matrix(self):

        cols = [c for c in ["country", "region"] if c in self.df.columns]
        cols += ["risk_score", "risk_level"]

        matrix = self.df[cols]

        path = os.path.join(self.output_dir, "risk_matrix.csv")
        matrix.to_csv(path, index=False)

        return matrix

    # ---------------------------------------------------------
    # BUSINESS REPORT
    # ---------------------------------------------------------
    def generate_business_risk_report(self):

        path = os.path.join(self.output_dir, "business_risk_report.txt")

        with open(path, "w") as f:
            f.write("BUSINESS RISK REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total Records: {len(self.df)}\n")
            f.write(f"High Risk: {(self.df['risk_level']=='High').sum()}\n")
            f.write(f"Medium Risk: {(self.df['risk_level']=='Medium').sum()}\n")
            f.write(f"Low Risk: {(self.df['risk_level']=='Low').sum()}\n")

        return path
    
    
    def calculate_financial_risk(self):

      required_cols = ["demand_quantity", "unit_price", "risk_score"]

      for col in required_cols:
        if col not in self.df.columns:
            raise ValueError(f"{col} missing for financial risk")

      # Revenue at risk
      self.df["revenue"] = self.df["demand_quantity"] * self.df["unit_price"]

      self.df["financial_risk"] = self.df["revenue"] * self.df["risk_score"]

      path = os.path.join(self.output_dir, "financial_risk.csv")
      self.df.to_csv(path, index=False)

      return self.df
  
  
    def supplier_risk_exposure(self):

     if "supplier_id" not in self.df.columns:
        return None

     supplier_risk = self.df.groupby("supplier_id")["financial_risk"].sum()

     supplier_risk = supplier_risk.sort_values(ascending=False)

     path = os.path.join(self.output_dir, "supplier_risk_exposure.csv")
     supplier_risk.to_csv(path)

     return supplier_risk
 
    def top_risk_nodes(self, top_n=10):

     top = self.df.sort_values(
        by="financial_risk",
        ascending=False
    ).head(top_n)

     path = os.path.join(self.output_dir, "top_risk_nodes.csv")
     top.to_csv(path, index=False)

     return top
 
    
    def scenario_analysis_advanced(self):

     scenarios = pd.DataFrame()

     base = self.df["financial_risk"]

     scenarios["base_case"] = base
     scenarios["best_case"] = base * 0.7   # mitigation success
     scenarios["likely_case"] = base
     scenarios["worst_case"] = base * 1.5  # disruption spike

     path = os.path.join(self.output_dir, "advanced_scenarios.csv")
     scenarios.to_csv(path, index=False)

     return scenarios
 
    
    def generate_executive_summary(self):

     total_risk = self.df["financial_risk"].sum()

     high_risk_count = (self.df["risk_level"] == "High").sum()

     top_supplier = None
     if "supplier_id" in self.df.columns:
         top_supplier = (
            self.df.groupby("supplier_id")["financial_risk"]
            .sum()
            .idxmax()
        )

     report_path = os.path.join(self.output_dir, "executive_summary.txt")

     with open(report_path, "w") as f:
        f.write("EXECUTIVE RISK SUMMARY\n")
        f.write("="*50 + "\n\n")

        f.write(f"Total Financial Risk Exposure: {total_risk:.2f}\n")
        f.write(f"High Risk Nodes: {high_risk_count}\n")
        f.write(f"Top Risk Supplier: {top_supplier}\n")

        f.write("\nRecommendations:\n")
        f.write("- Diversify high-risk suppliers\n")
        f.write("- Increase safety stock for volatile demand\n")
        f.write("- Monitor lead-time variability\n")

     return report_path