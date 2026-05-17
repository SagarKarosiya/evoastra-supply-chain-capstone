import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class DriftMonitor:

    def __init__(self, reference_df, current_df):
        self.reference = reference_df
        self.current = current_df

    def generate_drift_report(self):

        report = Report(metrics=[DataDriftPreset()])

        report.run(
            reference_data=self.reference,
            current_data=self.current
        )

        report.save_html("reports/data_drift_report.html")

        return report