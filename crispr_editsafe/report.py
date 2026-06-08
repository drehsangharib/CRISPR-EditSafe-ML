from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def generate_html_report(metrics_path: str | Path, predictions_path: str | Path, output_path: str | Path) -> None:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    preds = pd.read_csv(predictions_path)
    top = preds.sort_values("off_target_risk", ascending=False).head(20)
    rows_list = []
    for _, r in top.iterrows():
        rows_list.append(
            "<tr>"
            f"<td>{r.get('sgRNA', '')}</td>"
            f"<td>{r.get('target', '')}</td>"
            f"<td>{r.get('label', '')}</td>"
            f"<td>{float(r.get('off_target_risk', 0.0)):.4f}</td>"
            "</tr>"
        )
    rows = "\n".join(rows_list)
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>CRISPR-EditSafe-ML Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.5; }}
    code, pre {{ background: #f5f5f5; padding: 2px 4px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
    th {{ background: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>CRISPR-EditSafe-ML Example Report</h1>
  <p><strong>Disclaimer:</strong> Research/education use only. Not for clinical decision-making.</p>
  <h2>Validation metrics</h2>
  <pre>{json.dumps(metrics, indent=2)}</pre>
  <h2>Top predicted off-target candidates</h2>
  <table>
    <tr><th>sgRNA</th><th>target</th><th>label</th><th>off_target_risk</th></tr>
    {rows}
  </table>
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
