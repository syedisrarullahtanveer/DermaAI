"""
Clinical HTML-to-PDF report generation template engine for diagnostic documentation.
"""
class PDFReportGenerator:
    def generate_html_report(self, case_id: str, patient_info: dict, predictions: list, recommendation: str) -> str:
        preds_html = "".join([f"<li><strong>{p['name']}:</strong> {p['probability']:.1%}</li>" for p in predictions])
        html = f"""
<!DOCTYPE html>
<html>
<head><title>DermaAI Report - {case_id}</title></head>
<body>
  <h1>Dermatology Clinical Decision Support Report</h1>
  <p><strong>Case ID:</strong> {case_id}</p>
  <p><strong>Age:</strong> {patient_info.get('age', 'N/A')} | <strong>Site:</strong> {patient_info.get('site', 'N/A')}</p>
  <h3>Differential Diagnoses:</h3>
  <ul>{preds_html}</ul>
  <h3>Clinical Recommendation:</h3>
  <p>{recommendation}</p>
</body>
</html>
"""
        return html.strip()
