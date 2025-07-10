"""Comprehensive Report Generator

Generates detailed evaluation reports for model comparison and performance analysis.
Supports HTML, JSON, and text formats with visualizations and statistics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Comprehensive report generator for evaluation results.

    Generates detailed reports with metrics, comparisons, and recommendations
    for model evaluation and production deployment decisions.
    """

    def __init__(self, config: Any):
        """Initialize report generator."""
        self.config = config

        # Report configuration
        self.report_config = {
            "include_detailed_metrics": True,
            "include_error_analysis": True,
            "include_recommendations": True,
            "include_visualizations": True,
            "decimal_precision": 3,
        }

        logger.info(
            "ReportGenerator initialized for comprehensive evaluation reporting",
        )

    def generate_model_comparison_report(
        self,
        comparison_results: dict[str, Any],
        output_path: str | Path | None = None,
        format_type: str = "html",
    ) -> str:
        """Generate comprehensive model comparison report.

        Args:
            comparison_results: Results from model comparison
            output_path: Optional path to save report
            format_type: Report format (html, json, text)

        Returns:
            Report content as string

        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if format_type.lower() == "html":
            report_content = self._generate_html_comparison_report(comparison_results)
            if output_path:
                output_file = Path(output_path) / f"model_comparison_{timestamp}.html"
                with output_file.open("w", encoding="utf-8") as f:
                    f.write(report_content)
                logger.info(f"HTML comparison report saved to: {output_file}")

        elif format_type.lower() == "json":
            report_content = self._generate_json_comparison_report(comparison_results)
            if output_path:
                output_file = Path(output_path) / f"model_comparison_{timestamp}.json"
                with output_file.open("w", encoding="utf-8") as f:
                    f.write(report_content)
                logger.info(f"JSON comparison report saved to: {output_file}")

        else:  # text format
            report_content = self._generate_text_comparison_report(comparison_results)
            if output_path:
                output_file = Path(output_path) / f"model_comparison_{timestamp}.txt"
                with output_file.open("w", encoding="utf-8") as f:
                    f.write(report_content)
                logger.info(f"Text comparison report saved to: {output_file}")

        return report_content

    def generate_dataset_evaluation_report(
        self,
        evaluation_result: Any,
        output_path: str | Path | None = None,
        format_type: str = "html",
    ) -> str:
        """Generate detailed dataset evaluation report.

        Args:
            evaluation_result: Dataset evaluation result
            output_path: Optional path to save report
            format_type: Report format (html, json, text)

        Returns:
            Report content as string

        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if format_type.lower() == "html":
            report_content = self._generate_html_dataset_report(evaluation_result)
            if output_path:
                output_file = Path(output_path) / f"dataset_evaluation_{timestamp}.html"
                with output_file.open("w", encoding="utf-8") as f:
                    f.write(report_content)
                logger.info(f"HTML dataset report saved to: {output_file}")

        elif format_type.lower() == "json":
            report_content = self._generate_json_dataset_report(evaluation_result)
            if output_path:
                output_file = Path(output_path) / f"dataset_evaluation_{timestamp}.json"
                with output_file.open("w", encoding="utf-8") as f:
                    f.write(report_content)
                logger.info(f"JSON dataset report saved to: {output_file}")

        else:  # text format
            report_content = self._generate_text_dataset_report(evaluation_result)
            if output_path:
                output_file = Path(output_path) / f"dataset_evaluation_{timestamp}.txt"
                with output_file.open("w", encoding="utf-8") as f:
                    f.write(report_content)
                logger.info(f"Text dataset report saved to: {output_file}")

        return report_content

    def _generate_html_comparison_report(
        self,
        comparison_results: dict[str, Any],
    ) -> str:
        """Generate HTML model comparison report."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report - Unified Vision Processor</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .model-card {{
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 20px;
            background: #f8f9fa;
        }}
        .model-card h3 {{
            margin-top: 0;
            color: #495057;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: white;
            border-radius: 4px;
        }}
        .metric-label {{
            font-weight: 500;
        }}
        .metric-value {{
            font-weight: bold;
            color: #28a745;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .comparison-table th,
        .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .comparison-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .best-score {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .recommendation {{
            background-color: #e7f3ff;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 20px 0;
        }}
        .error-analysis {{
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 20px 0;
        }}
        .timestamp {{
            color: #6c757d;
            font-size: 0.9em;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Model Comparison Report</h1>
        <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        
        <h2>📊 Executive Summary</h2>
        {self._generate_html_executive_summary(comparison_results)}
        
        <h2>📈 Detailed Metrics Comparison</h2>
        {self._generate_html_metrics_table(comparison_results)}
        
        <h2>🎯 Model Performance Cards</h2>
        <div class="summary-grid">
            {self._generate_html_model_cards(comparison_results)}
        </div>
        
        <h2>🏭 Production Readiness Assessment</h2>
        {self._generate_html_production_assessment(comparison_results)}
        
        <h2>⚠️ Error Analysis</h2>
        {self._generate_html_error_analysis(comparison_results)}
        
        <h2>💡 Recommendations</h2>
        {self._generate_html_recommendations(comparison_results)}
        
        <h2>🔍 Technical Details</h2>
        {self._generate_html_technical_details(comparison_results)}
    </div>
</body>
</html>"""
        return html_content

    def _generate_html_executive_summary(
        self,
        comparison_results: dict[str, Any],
    ) -> str:
        """Generate executive summary section."""
        if not comparison_results:
            return "<p>No comparison results available.</p>"

        # Find best performing model
        best_model = max(
            comparison_results.keys(),
            key=lambda k: comparison_results[k].average_f1_score,
        )
        best_result = comparison_results[best_model]

        # Safe formatting to handle Mock objects in tests
        f1_score = getattr(best_result, "average_f1_score", 0.0)
        ready_rate = getattr(best_result, "production_ready_rate", 0.0)
        proc_time = getattr(best_result, "average_processing_time", 0.0)
        success_count = getattr(best_result, "successful_extractions", 0)
        total_count = getattr(best_result, "total_documents", 0)

        # Format numbers safely
        try:
            f1_formatted = f"{float(f1_score):.3f}"
            rate_formatted = f"{float(ready_rate):.1%}"
            time_formatted = f"{float(proc_time):.2f}s"
        except (ValueError, TypeError):
            f1_formatted = str(f1_score)
            rate_formatted = str(ready_rate)
            time_formatted = str(proc_time)

        summary = f"""
        <div class="recommendation">
            <h3>🏆 Best Performing Model: {best_model}</h3>
            <p><strong>F1 Score:</strong> {f1_formatted}</p>
            <p><strong>Production Ready Rate:</strong> {rate_formatted}</p>
            <p><strong>Average Processing Time:</strong> {time_formatted}</p>
            <p><strong>Documents Processed:</strong> {success_count}/{total_count}</p>
        </div>
        """

        return summary

    def _generate_html_metrics_table(self, comparison_results: dict[str, Any]) -> str:
        """Generate metrics comparison table."""
        if not comparison_results:
            return "<p>No metrics available.</p>"

        headers = [
            "Model",
            "F1 Score",
            "Precision",
            "Recall",
            "Confidence",
            "Production Ready",
            "Avg Time (s)",
        ]

        table_html = '<table class="comparison-table">\n<thead>\n<tr>'
        for header in headers:
            table_html += f"<th>{header}</th>"
        table_html += "</tr>\n</thead>\n<tbody>\n"

        # Find best scores for highlighting
        best_f1 = max(result.average_f1_score for result in comparison_results.values())
        best_precision = max(
            result.average_precision for result in comparison_results.values()
        )
        best_recall = max(
            result.average_recall for result in comparison_results.values()
        )
        best_confidence = max(
            result.average_confidence for result in comparison_results.values()
        )
        best_production_rate = max(
            result.production_ready_rate for result in comparison_results.values()
        )

        for model_name, result in comparison_results.items():
            table_html += "<tr>"
            table_html += f"<td><strong>{model_name}</strong></td>"

            # F1 Score
            css_class = (
                ' class="best-score"' if result.average_f1_score == best_f1 else ""
            )
            table_html += f"<td{css_class}>{result.average_f1_score:.3f}</td>"

            # Precision
            css_class = (
                ' class="best-score"'
                if result.average_precision == best_precision
                else ""
            )
            table_html += f"<td{css_class}>{result.average_precision:.3f}</td>"

            # Recall
            css_class = (
                ' class="best-score"' if result.average_recall == best_recall else ""
            )
            table_html += f"<td{css_class}>{result.average_recall:.3f}</td>"

            # Confidence
            css_class = (
                ' class="best-score"'
                if result.average_confidence == best_confidence
                else ""
            )
            table_html += f"<td{css_class}>{result.average_confidence:.3f}</td>"

            # Production Ready
            css_class = (
                ' class="best-score"'
                if result.production_ready_rate == best_production_rate
                else ""
            )
            table_html += f"<td{css_class}>{result.production_ready_rate:.1%}</td>"

            # Processing Time
            table_html += f"<td>{result.average_processing_time:.2f}</td>"

            table_html += "</tr>"

        table_html += "</tbody>\n</table>"
        return table_html

    def _generate_html_model_cards(self, comparison_results: dict[str, Any]) -> str:
        """Generate model performance cards."""
        cards_html = ""

        for model_name, result in comparison_results.items():
            cards_html += f"""
            <div class="model-card">
                <h3>📱 {model_name}</h3>
                <div class="metric">
                    <span class="metric-label">Documents Processed:</span>
                    <span class="metric-value">{result.successful_extractions}/{result.total_documents}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">{result.successful_extractions / result.total_documents:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">F1 Score:</span>
                    <span class="metric-value">{result.average_f1_score:.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Production Ready:</span>
                    <span class="metric-value">{result.production_ready_rate:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">AWK Fallback Rate:</span>
                    <span class="metric-value">{result.awk_fallback_rate:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Processing Time:</span>
                    <span class="metric-value">{result.average_processing_time:.2f}s</span>
                </div>
            </div>
            """

        return cards_html

    def _generate_html_production_assessment(
        self,
        comparison_results: dict[str, Any],
    ) -> str:
        """Generate production readiness assessment."""
        assessment_html = ""

        for model_name, result in comparison_results.items():
            # Calculate production readiness grade
            if result.production_ready_rate >= 0.9:
                grade = "🟢 Excellent"
                grade_color = "#28a745"
            elif result.production_ready_rate >= 0.7:
                grade = "🟡 Good"
                grade_color = "#ffc107"
            elif result.production_ready_rate >= 0.5:
                grade = "🟠 Fair"
                grade_color = "#fd7e14"
            else:
                grade = "🔴 Poor"
                grade_color = "#dc3545"

            assessment_html += f"""
            <div class="model-card">
                <h3>{model_name}</h3>
                <div class="metric">
                    <span class="metric-label">Production Grade:</span>
                    <span class="metric-value" style="color: {grade_color};">{grade}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Ready Documents:</span>
                    <span class="metric-value">{result.production_ready_count}/{result.total_documents}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Quality Distribution:</span>
                    <span class="metric-value">{", ".join(f"{k}: {v}" for k, v in result.quality_distribution.items())}</span>
                </div>
            </div>
            """

        return assessment_html

    def _generate_html_error_analysis(self, comparison_results: dict[str, Any]) -> str:
        """Generate error analysis section."""
        error_html = ""

        for model_name, result in comparison_results.items():
            if result.failed_documents or result.error_analysis:
                error_html += f"""
                <div class="error-analysis">
                    <h4>⚠️ {model_name} Error Analysis</h4>
                    <p><strong>Failed Documents:</strong> {len(result.failed_documents)}</p>
                    {f"<p><strong>Failed Files:</strong> {', '.join(result.failed_documents[:5])}" + ("..." if len(result.failed_documents) > 5 else "") + "</p>" if result.failed_documents else ""}
                    {f"<p><strong>Error Types:</strong> {', '.join(f'{k}: {v}' for k, v in result.error_analysis.items())}</p>" if result.error_analysis else ""}
                </div>
                """

        return error_html or "<p>No errors detected in the evaluation.</p>"

    def _generate_html_recommendations(self, comparison_results: dict[str, Any]) -> str:
        """Generate recommendations section."""
        recommendations = self._generate_model_recommendations(comparison_results)

        recommendations_html = ""
        for category, recs in recommendations.items():
            recommendations_html += f"<h4>{category}</h4><ul>"
            for rec in recs:
                recommendations_html += f"<li>{rec}</li>"
            recommendations_html += "</ul>"

        return f'<div class="recommendation">{recommendations_html}</div>'

    def _generate_html_technical_details(
        self,
        comparison_results: dict[str, Any],
    ) -> str:
        """Generate technical details section."""
        details_html = ""

        for model_name, result in comparison_results.items():
            details_html += f"""
            <h4>🔧 {model_name} Technical Details</h4>
            <ul>
                <li><strong>Dataset:</strong> {result.dataset_name}</li>
                <li><strong>Total Processing Time:</strong> {result.total_processing_time:.2f}s</li>
                <li><strong>Highlight Detection Rate:</strong> {result.highlight_detection_rate:.1%}</li>
                <li><strong>AWK Fallback Usage:</strong> {result.awk_fallback_rate:.1%}</li>
                <li><strong>Document Types Processed:</strong> {len(set(doc.document_type for doc in result.document_results))}</li>
            </ul>
            """

        return details_html

    def _generate_text_comparison_report(
        self,
        comparison_results: dict[str, Any],
    ) -> str:
        """Generate text-based comparison report."""
        report_lines = [
            "=" * 80,
            "MODEL COMPARISON REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
        ]

        # Find best model
        if comparison_results:
            best_model = max(
                comparison_results.keys(),
                key=lambda k: comparison_results[k].average_f1_score,
            )
            best_result = comparison_results[best_model]

            report_lines.extend(
                [
                    f"Best Performing Model: {best_model}",
                    f"F1 Score: {best_result.average_f1_score:.3f}",
                    f"Production Ready Rate: {best_result.production_ready_rate:.1%}",
                    f"Average Processing Time: {best_result.average_processing_time:.2f}s",
                    "",
                ],
            )

        # Detailed metrics
        report_lines.extend(
            [
                "DETAILED METRICS",
                "-" * 40,
            ],
        )

        for model_name, result in comparison_results.items():
            report_lines.extend(
                [
                    f"{model_name}:",
                    f"  F1 Score: {result.average_f1_score:.3f}",
                    f"  Precision: {result.average_precision:.3f}",
                    f"  Recall: {result.average_recall:.3f}",
                    f"  Confidence: {result.average_confidence:.3f}",
                    f"  Production Ready: {result.production_ready_rate:.1%}",
                    f"  Processing Time: {result.average_processing_time:.2f}s",
                    f"  Success Rate: {result.successful_extractions / result.total_documents:.1%}",
                    "",
                ],
            )

        return "\n".join(report_lines)

    def _generate_json_comparison_report(
        self,
        comparison_results: dict[str, Any],
    ) -> str:
        """Generate JSON comparison report."""
        report_data = {
            "report_type": "model_comparison",
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "detailed_results": {},
            "recommendations": self._generate_model_recommendations(comparison_results),
        }

        # Convert results to JSON-serializable format
        for model_name, result in comparison_results.items():
            report_data["detailed_results"][model_name] = {
                "dataset_name": result.dataset_name,
                "total_documents": result.total_documents,
                "successful_extractions": result.successful_extractions,
                "average_f1_score": result.average_f1_score,
                "average_precision": result.average_precision,
                "average_recall": result.average_recall,
                "average_confidence": result.average_confidence,
                "production_ready_rate": result.production_ready_rate,
                "processing_time": result.average_processing_time,
                "awk_fallback_rate": result.awk_fallback_rate,
                "quality_distribution": result.quality_distribution,
                "failed_documents": result.failed_documents,
                "error_analysis": result.error_analysis,
            }

        return json.dumps(report_data, indent=2, default=str)

    def _generate_text_dataset_report(self, evaluation_result: Any) -> str:
        """Generate text dataset evaluation report."""
        report_lines = [
            "=" * 80,
            "DATASET EVALUATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {evaluation_result.model_name}",
            f"Dataset: {evaluation_result.dataset_name}",
            "",
            "SUMMARY METRICS",
            "-" * 40,
            f"Total Documents: {evaluation_result.total_documents}",
            f"Successful Extractions: {evaluation_result.successful_extractions}",
            f"Success Rate: {evaluation_result.successful_extractions / evaluation_result.total_documents:.1%}",
            f"F1 Score: {evaluation_result.average_f1_score:.3f}",
            f"Production Ready Rate: {evaluation_result.production_ready_rate:.1%}",
            "",
        ]

        return "\n".join(report_lines)

    def save_report_to_file(
        self, report_content: str, output_path: str | Path, format_type: str = "html"
    ) -> None:
        """Save report content to file.

        Args:
            report_content: Report content to save
            output_path: Path where to save the report
            format_type: Format type for file extension
        """
        output_path = Path(output_path)
        if format_type == "html" and not output_path.suffix:
            output_path = output_path.with_suffix(".html")
        elif format_type == "json" and not output_path.suffix:
            output_path = output_path.with_suffix(".json")
        elif format_type == "text" and not output_path.suffix:
            output_path = output_path.with_suffix(".txt")

        with output_path.open("w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Report saved to: {output_path}")

    def _generate_json_dataset_report(self, evaluation_result: Any) -> str:
        """Generate JSON dataset evaluation report."""
        report_data = {
            "report_type": "dataset_evaluation",
            "generated_at": datetime.now().isoformat(),
            "model_name": evaluation_result.model_name,
            "dataset_name": evaluation_result.dataset_name,
            "summary_metrics": {
                "total_documents": evaluation_result.total_documents,
                "successful_extractions": evaluation_result.successful_extractions,
                "average_f1_score": evaluation_result.average_f1_score,
                "production_ready_rate": evaluation_result.production_ready_rate,
            },
        }

        return json.dumps(report_data, indent=2, default=str)

    def _generate_html_dataset_report(self, evaluation_result: Any) -> str:
        """Generate HTML dataset evaluation report."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ margin: 10px 0; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Dataset Evaluation Report</h1>
    <div class="summary">
        <h2>{evaluation_result.model_name} - {evaluation_result.dataset_name}</h2>
        <div class="metric">Total Documents: {evaluation_result.total_documents}</div>
        <div class="metric">Successful Extractions: {evaluation_result.successful_extractions}</div>
        <div class="metric">F1 Score: {evaluation_result.average_f1_score:.3f}</div>
        <div class="metric">Production Ready Rate: {evaluation_result.production_ready_rate:.1%}</div>
    </div>
</body>
</html>"""

    def _generate_model_recommendations(
        self,
        comparison_results: dict[str, Any],
    ) -> dict[str, list[str]]:
        """Generate model-specific recommendations."""
        recommendations = {
            "Production Deployment": [],
            "Performance Optimization": [],
            "Quality Improvement": [],
            "Technical Considerations": [],
        }

        if not comparison_results:
            return recommendations

        # Find best model
        best_model = max(
            comparison_results.keys(),
            key=lambda k: comparison_results[k].average_f1_score,
        )

        best_result = comparison_results[best_model]

        # Production deployment recommendations
        if best_result.production_ready_rate >= 0.9:
            recommendations["Production Deployment"].append(
                f"✅ {best_model} is ready for production deployment with {best_result.production_ready_rate:.1%} ready rate",
            )
        elif best_result.production_ready_rate >= 0.7:
            recommendations["Production Deployment"].append(
                f"⚠️ {best_model} requires monitoring in production with {best_result.production_ready_rate:.1%} ready rate",
            )
        else:
            recommendations["Production Deployment"].append(
                f"❌ {best_model} requires significant improvement before production deployment",
            )

        # Performance optimization
        for model_name, result in comparison_results.items():
            if result.awk_fallback_rate > 0.3:
                recommendations["Performance Optimization"].append(
                    f"🔄 {model_name}: High AWK fallback rate ({result.awk_fallback_rate:.1%}) - consider improving primary extraction",
                )

            if result.average_processing_time > 10.0:
                recommendations["Performance Optimization"].append(
                    f"⏱️ {model_name}: Slow processing ({result.average_processing_time:.2f}s avg) - consider optimization",
                )

        # Quality improvement
        for model_name, result in comparison_results.items():
            if result.average_f1_score < 0.8:
                recommendations["Quality Improvement"].append(
                    f"📈 {model_name}: F1 score below threshold ({result.average_f1_score:.3f}) - review extraction accuracy",
                )

        # Technical considerations
        recommendations["Technical Considerations"].extend(
            [
                "🔧 All models use identical Llama 7-step pipeline for fair comparison",
                "📊 Consider ensemble approach combining best features from multiple models",
                "🎯 Focus on Australian tax document compliance and ATO requirements",
            ],
        )

        return recommendations
