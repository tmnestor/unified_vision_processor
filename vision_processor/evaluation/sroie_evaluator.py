"""SROIE Dataset Evaluator

Enhanced SROIE (Scanned Receipts OCR and Information Extraction) dataset evaluation
with Australian tax document adaptations and cross-model comparison capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .metrics_calculator import MetricsCalculator
from .unified_evaluator import UnifiedEvaluator

logger = logging.getLogger(__name__)


class SROIEEvaluator:
    """Enhanced SROIE dataset evaluator.

    Provides specialized evaluation for SROIE dataset with Australian tax
    document adaptations and standardized metrics for model comparison.
    """

    def __init__(self, config: Any):
        """Initialize SROIE evaluator."""
        self.config = config
        self.unified_evaluator = UnifiedEvaluator(config)
        self.metrics_calculator = MetricsCalculator(config)

        # SROIE field mapping to standard fields
        # Tests expect simple string mapping, not lists
        self.sroie_field_mapping = {
            "company": "supplier_name",
            "date": "date",
            "address": "address",
            "total": "total_amount",
        }

        # Full mapping with alternatives for internal use
        self.sroie_field_mapping_full = {
            "company": ["supplier_name", "business_name", "company_name"],
            "date": ["date", "date_value", "transaction_date"],
            "address": ["address", "business_address", "supplier_address"],
            "total": ["total_amount", "total", "amount", "total_value"],
        }

        # Enhanced field mapping for Australian tax documents
        self.enhanced_field_mapping = {
            "company": ["supplier_name", "business_name", "company_name", "store_name"],
            "date": ["date", "date_value", "transaction_date", "invoice_date"],
            "address": ["address", "business_address", "supplier_address", "location"],
            "total": ["total_amount", "total", "amount", "total_value", "final_total"],
            "tax": ["gst_amount", "tax", "tax_value", "gst"],
            "abn": ["abn", "abn_number", "australian_business_number"],
            "subtotal": ["subtotal", "sub_total", "amount_before_tax"],
        }

        logger.info("SROIEEvaluator initialized with enhanced Australian tax support")

        # For backward compatibility with tests, provide simple field mapping access
        self.simple_sroie_field_mapping = {
            "company": "supplier_name",
            "date": "date",
            "address": "address",
            "total": "total_amount",
        }

    def evaluate_sroie_dataset(
        self,
        dataset_path: str | Path,
        ground_truth_path: str | Path,
        model_name: str,
        use_enhanced_fields: bool = True,
        max_documents: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate model on SROIE dataset with enhanced metrics.

        Args:
            dataset_path: Path to SROIE dataset images
            ground_truth_path: Path to SROIE ground truth files
            model_name: Name of model being evaluated
            use_enhanced_fields: Use enhanced field mapping for Australian tax docs
            max_documents: Optional limit on number of documents

        Returns:
            Enhanced SROIE evaluation results

        """
        dataset_path = Path(dataset_path)
        ground_truth_path = Path(ground_truth_path)

        logger.info(
            f"Starting enhanced SROIE evaluation: {dataset_path} with {model_name}",
        )

        # Find all images and ground truth files
        image_files = list(dataset_path.glob("*.jpg")) + list(
            dataset_path.glob("*.png"),
        )
        if max_documents:
            image_files = image_files[:max_documents]

        results = {
            "dataset_info": {
                "dataset_path": str(dataset_path),
                "model_name": model_name,
                "total_documents": len(image_files),
                "enhanced_fields": use_enhanced_fields,
            },
            "sroie_metrics": {
                "company": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "date": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "address": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "total": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            },
            "enhanced_metrics": {},
            "overall_metrics": {},
            "document_results": [],
            "failed_documents": [],
            "processing_statistics": {},
        }

        if use_enhanced_fields:
            results["enhanced_metrics"] = {
                "tax": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "abn": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "subtotal": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }

        # Process each document
        field_mapping = (
            self.enhanced_field_mapping
            if use_enhanced_fields
            else self.sroie_field_mapping
        )

        for i, image_file in enumerate(image_files, 1):
            logger.info(
                f"Processing SROIE document {i}/{len(image_files)}: {image_file.name}",
            )

            # Find corresponding ground truth
            gt_file = ground_truth_path / f"{image_file.stem}.json"
            if not gt_file.exists():
                logger.warning(f"No ground truth found for {image_file.name}")
                results["failed_documents"].append(image_file.name)
                continue

            # Load and process ground truth
            with Path(gt_file).open("r", encoding="utf-8") as f:
                ground_truth_data = json.load(f)

            # Convert SROIE format to standard format
            standardized_gt = self._standardize_sroie_ground_truth(
                ground_truth_data,
                field_mapping,
            )

            # Evaluate document using unified evaluator
            doc_result = self.unified_evaluator.evaluate_document(
                image_file,
                standardized_gt,
                model_name,
            )

            # Calculate SROIE-specific metrics
            sroie_metrics = self._calculate_sroie_field_metrics(
                doc_result.extracted_fields,
                standardized_gt,
                field_mapping,
            )

            # Add SROIE metrics to document result
            doc_result_dict = {
                "image_file": image_file.name,
                "success": doc_result.success,
                "sroie_field_metrics": sroie_metrics,
                "overall_f1": doc_result.f1_score,
                "confidence": doc_result.confidence_score,
                "processing_time": doc_result.processing_time,
            }

            results["document_results"].append(doc_result_dict)

            if not doc_result.success:
                results["failed_documents"].append(image_file.name)

        # Calculate aggregated SROIE metrics
        results["sroie_metrics"] = self._aggregate_sroie_metrics(
            results["document_results"],
            list(field_mapping.keys()),
        )

        # Calculate overall metrics
        results["overall_metrics"] = self._calculate_overall_sroie_metrics(
            results["document_results"],
        )

        # Calculate processing statistics
        results["processing_statistics"] = self._calculate_processing_statistics(
            results["document_results"],
        )

        logger.info(
            f"SROIE evaluation completed: {len(results['document_results'])} documents processed",
        )

        return results

    def get_sroie_field_mapping(self, field_name: str) -> str:
        """Get simple field mapping for SROIE field (for test compatibility)."""
        return self.simple_sroie_field_mapping.get(field_name, field_name)

    def compare_models_on_sroie(
        self,
        dataset_path: str | Path,
        ground_truth_path: str | Path,
        model_names: list[str],
        use_enhanced_fields: bool = True,
        max_documents: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compare multiple models on SROIE dataset.

        Args:
            dataset_path: Path to SROIE dataset images
            ground_truth_path: Path to SROIE ground truth files
            model_names: List of model names to compare
            use_enhanced_fields: Use enhanced field mapping
            max_documents: Optional limit on number of documents

        Returns:
            Dictionary mapping model names to SROIE evaluation results

        """
        logger.info(f"Starting SROIE model comparison: {model_names}")

        comparison_results = {}

        for model_name in model_names:
            logger.info(f"Evaluating {model_name} on SROIE dataset")

            result = self.evaluate_sroie_dataset(
                dataset_path,
                ground_truth_path,
                model_name,
                use_enhanced_fields,
                max_documents,
            )

            comparison_results[model_name] = result

            logger.info(
                f"Model {model_name} SROIE results: "
                f"Overall F1={result['overall_metrics']['average_f1']:.3f}, "
                f"Success Rate={result['overall_metrics']['success_rate']:.1%}",
            )

        return comparison_results

    def generate_sroie_leaderboard(
        self,
        comparison_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate SROIE leaderboard from comparison results.

        Args:
            comparison_results: Results from SROIE model comparison

        Returns:
            SROIE leaderboard with rankings and metrics

        """
        leaderboard = {
            "rankings": [],
            "field_performance": {},
            "summary_statistics": {},
        }

        # Create rankings based on overall F1 score
        model_scores = []
        for model_name, result in comparison_results.items():
            overall_f1 = result["overall_metrics"]["average_f1"]
            model_scores.append((model_name, overall_f1, result))

        # Sort by F1 score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)

        # Create leaderboard rankings
        for rank, (model_name, f1_score, result) in enumerate(model_scores, 1):
            ranking_entry = {
                "rank": rank,
                "model_name": model_name,
                "overall_f1": f1_score,
                "success_rate": result["overall_metrics"]["success_rate"],
                "avg_processing_time": result["processing_statistics"][
                    "avg_processing_time"
                ],
                "total_documents": result["dataset_info"]["total_documents"],
                "sroie_field_scores": result["sroie_metrics"],
            }

            leaderboard["rankings"].append(ranking_entry)

        # Calculate field performance comparison
        fields = ["company", "date", "address", "total"]
        if comparison_results and any(
            "enhanced_metrics" in result for result in comparison_results.values()
        ):
            fields.extend(["tax", "abn", "subtotal"])

        for field in fields:
            field_scores = []
            for model_name, result in comparison_results.items():
                if field in result["sroie_metrics"]:
                    field_f1 = result["sroie_metrics"][field]["f1"]
                    field_scores.append((model_name, field_f1))
                elif field in result.get("enhanced_metrics", {}):
                    field_f1 = result["enhanced_metrics"][field]["f1"]
                    field_scores.append((model_name, field_f1))

            if field_scores:
                # Sort by F1 score (descending)
                field_scores.sort(key=lambda x: x[1], reverse=True)
                leaderboard["field_performance"][field] = field_scores

        # Calculate summary statistics
        if comparison_results:
            all_f1_scores = [
                result["overall_metrics"]["average_f1"]
                for result in comparison_results.values()
            ]
            all_processing_times = [
                result["processing_statistics"]["avg_processing_time"]
                for result in comparison_results.values()
            ]

            leaderboard["summary_statistics"] = {
                "num_models": len(comparison_results),
                "best_f1_score": max(all_f1_scores),
                "worst_f1_score": min(all_f1_scores),
                "average_f1_score": sum(all_f1_scores) / len(all_f1_scores),
                "fastest_avg_time": min(all_processing_times),
                "slowest_avg_time": max(all_processing_times),
            }

        return leaderboard

    def _standardize_sroie_ground_truth(
        self,
        ground_truth_data: dict[str, Any],
        field_mapping: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Convert SROIE ground truth format to standardized format."""
        standardized = {}

        for sroie_field, standard_fields in field_mapping.items():
            if sroie_field in ground_truth_data:
                # Use the first standard field name as the key
                standard_key = standard_fields[0]
                standardized[standard_key] = ground_truth_data[sroie_field]

        return standardized

    def _calculate_sroie_field_metrics(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        field_mapping: dict[str, list[str]],
    ) -> dict[str, dict[str, float]]:
        """Calculate field-specific metrics for SROIE evaluation."""
        field_metrics = {}

        for sroie_field, standard_fields in field_mapping.items():
            # Find extracted value for this field
            extracted_value = None
            for field_name in standard_fields:
                if extracted.get(field_name):
                    extracted_value = extracted[field_name]
                    break

            # Find ground truth value for this field
            gt_value = None
            for field_name in standard_fields:
                if ground_truth.get(field_name):
                    gt_value = ground_truth[field_name]
                    break

            # Calculate metrics for this field
            if gt_value is not None:
                if extracted_value is not None:
                    # Check exact match
                    if self._field_exact_match(extracted_value, gt_value, sroie_field):
                        precision = recall = f1 = 1.0
                    else:
                        precision = recall = f1 = 0.0
                else:
                    precision = recall = f1 = 0.0
            # No ground truth for this field
            elif extracted_value is not None:
                precision = 0.0  # False positive
                recall = 0.0
                f1 = 0.0
            else:
                precision = recall = f1 = 1.0  # True negative

            field_metrics[sroie_field] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        return field_metrics

    def _field_exact_match(
        self,
        extracted_value: Any,
        gt_value: Any,
        field_type: str,
    ) -> bool:
        """Check if extracted value exactly matches ground truth for specific field type."""
        extracted_str = str(extracted_value).strip().lower()
        gt_str = str(gt_value).strip().lower()

        if field_type == "total":
            # For amounts, normalize and compare numerically
            try:
                import re

                extracted_amount = float(re.sub(r"[$,\s]", "", extracted_str))
                gt_amount = float(re.sub(r"[$,\s]", "", gt_str))
                return abs(extracted_amount - gt_amount) < 0.01
            except ValueError:
                return extracted_str == gt_str

        elif field_type == "date":
            # Normalize date formats for comparison
            extracted_normalized = re.sub(r"[-/\s]", "", extracted_str)
            gt_normalized = re.sub(r"[-/\s]", "", gt_str)
            return extracted_normalized == gt_normalized

        else:
            # Standard string comparison for company, address, etc.
            return extracted_str == gt_str

    def _aggregate_sroie_metrics(
        self,
        document_results: list[dict[str, Any]],
        fields: list[str],
    ) -> dict[str, dict[str, float]]:
        """Aggregate SROIE metrics across all documents."""
        aggregated = {}

        for field in fields:
            precisions = []
            recalls = []
            f1s = []

            for doc_result in document_results:
                if doc_result["success"] and field in doc_result["sroie_field_metrics"]:
                    metrics = doc_result["sroie_field_metrics"][field]
                    precisions.append(metrics["precision"])
                    recalls.append(metrics["recall"])
                    f1s.append(metrics["f1"])

            if precisions:
                aggregated[field] = {
                    "precision": sum(precisions) / len(precisions),
                    "recall": sum(recalls) / len(recalls),
                    "f1": sum(f1s) / len(f1s),
                    "sample_count": len(precisions),
                }
            else:
                aggregated[field] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "sample_count": 0,
                }

        return aggregated

    def _calculate_overall_sroie_metrics(
        self,
        document_results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Calculate overall SROIE metrics."""
        successful_results = [r for r in document_results if r["success"]]

        if not successful_results:
            return {
                "average_f1": 0.0,
                "success_rate": 0.0,
                "total_documents": len(document_results),
                "successful_documents": 0,
            }

        # Calculate average F1 across all successful documents
        f1_scores = [r["overall_f1"] for r in successful_results]
        average_f1 = sum(f1_scores) / len(f1_scores)

        # Calculate success rate
        success_rate = len(successful_results) / len(document_results)

        return {
            "average_f1": average_f1,
            "success_rate": success_rate,
            "total_documents": len(document_results),
            "successful_documents": len(successful_results),
        }

    def _calculate_processing_statistics(
        self,
        document_results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Calculate processing time statistics."""
        successful_results = [r for r in document_results if r["success"]]

        if not successful_results:
            return {
                "avg_processing_time": 0.0,
                "min_processing_time": 0.0,
                "max_processing_time": 0.0,
                "total_processing_time": 0.0,
            }

        processing_times = [r["processing_time"] for r in successful_results]

        return {
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "total_processing_time": sum(processing_times),
        }
