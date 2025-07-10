"""Advanced Metrics Calculator

Comprehensive metrics computation for document processing evaluation.
Supports precision, recall, F1, exact match, and Australian tax-specific metrics.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Advanced metrics calculator for document processing evaluation.

    Supports standard NLP metrics and Australian tax document-specific
    evaluation criteria.
    """

    def __init__(self, config: Any):
        """Initialize metrics calculator."""
        self.config = config

        # Australian-specific validation patterns
        self.abn_pattern = re.compile(r"^\d{2}\s?\d{3}\s?\d{3}\s?\d{3}$")
        self.bsb_pattern = re.compile(r"^\d{3}-?\d{3}$")
        self.date_patterns = [
            re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"),
            re.compile(
                r"^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$",
            ),
        ]

        # Amount validation
        self.amount_pattern = re.compile(r"^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$")

        logger.info("MetricsCalculator initialized with Australian tax validation")

    def calculate_prf_metrics(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> tuple[float, float, float]:
        """Calculate precision, recall, and F1 score.

        Args:
            extracted: Extracted fields from model
            ground_truth: Ground truth fields

        Returns:
            Tuple of (precision, recall, f1_score)

        """
        if not ground_truth:
            return 0.0, 0.0, 0.0

        # Get sets of extracted and ground truth fields
        extracted_fields = set(extracted.keys())
        ground_truth_fields = set(ground_truth.keys())

        # Calculate field-level metrics
        true_positives = len(extracted_fields & ground_truth_fields)
        false_positives = len(extracted_fields - ground_truth_fields)
        false_negatives = len(ground_truth_fields - extracted_fields)

        # Calculate precision and recall
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        # Calculate F1 score
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return precision, recall, f1_score

    def calculate_exact_match(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> float:
        """Calculate exact match score for extracted values.

        Args:
            extracted: Extracted fields from model
            ground_truth: Ground truth fields

        Returns:
            Exact match score (0.0 to 1.0)

        """
        if not ground_truth:
            return 0.0

        total_fields = len(ground_truth)
        exact_matches = 0

        for field, true_value in ground_truth.items():
            if field in extracted:
                extracted_value = extracted[field]

                # Normalize values for comparison
                if self._exact_match_comparison(extracted_value, true_value, field):
                    exact_matches += 1

        return min(1.0, exact_matches / total_fields)

    def calculate_fuzzy_match(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        threshold: float = 0.8,
    ) -> float:
        """Calculate fuzzy match score using string similarity.

        Args:
            extracted: Extracted fields from model
            ground_truth: Ground truth fields
            threshold: Similarity threshold for match

        Returns:
            Fuzzy match score (0.0 to 1.0)

        """
        if not ground_truth:
            return 0.0

        total_fields = len(ground_truth)
        fuzzy_matches = 0

        for field, true_value in ground_truth.items():
            if field in extracted:
                extracted_value = extracted[field]

                # Calculate similarity
                similarity = self._calculate_string_similarity(
                    str(extracted_value),
                    str(true_value),
                )

                if similarity >= threshold:
                    fuzzy_matches += 1

        return min(1.0, fuzzy_matches / total_fields)

    def calculate_ato_compliance_score(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> float:
        """Calculate ATO compliance score for Australian tax documents.

        Args:
            extracted: Extracted fields from model
            ground_truth: Ground truth fields

        Returns:
            ATO compliance score (0.0 to 1.0)

        """
        compliance_score = 0.0
        total_checks = 0

        # ABN validation
        if "abn" in ground_truth:
            total_checks += 1
            if "abn" in extracted:
                if self._validate_abn_format(extracted["abn"]):
                    compliance_score += 1.0
                elif (
                    self._validate_abn_format(ground_truth["abn"])
                    and extracted["abn"] == ground_truth["abn"]
                ):
                    compliance_score += 0.8  # Correct value but poor format

        # GST calculation validation
        if "gst_amount" in ground_truth and "total_amount" in ground_truth:
            total_checks += 1
            if self._validate_gst_calculation(extracted):
                compliance_score += 1.0

        # Date format validation
        if "date" in ground_truth:
            total_checks += 1
            if "date" in extracted:
                if self._validate_australian_date_format(extracted["date"]):
                    compliance_score += 1.0

        # Amount format validation
        for amount_field in ["total_amount", "subtotal", "gst_amount"]:
            if amount_field in ground_truth:
                total_checks += 1
                if amount_field in extracted:
                    if self._validate_amount_format(extracted[amount_field]):
                        compliance_score += 1.0

        return compliance_score / total_checks if total_checks > 0 else 0.0

    def calculate_field_confidence_correlation(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        confidence_scores: dict[str, float],
    ) -> float:
        """Calculate correlation between field confidence and accuracy.

        Args:
            extracted: Extracted fields from model
            ground_truth: Ground truth fields
            confidence_scores: Per-field confidence scores

        Returns:
            Confidence correlation score

        """
        if not confidence_scores:
            return 0.0

        # Calculate accuracy for each field with confidence
        field_accuracies = []
        field_confidences = []

        for field, confidence in confidence_scores.items():
            if field in ground_truth:
                # Calculate accuracy for this field
                if field in extracted:
                    accuracy = (
                        1.0
                        if self._exact_match_comparison(
                            extracted[field],
                            ground_truth[field],
                            field,
                        )
                        else 0.0
                    )
                else:
                    accuracy = 0.0

                field_accuracies.append(accuracy)
                field_confidences.append(confidence)

        if len(field_accuracies) < 2:
            return 0.0

        # Calculate Pearson correlation coefficient
        return self._calculate_correlation(field_confidences, field_accuracies)

    def calculate_processing_efficiency(
        self,
        processing_time: float,
        document_complexity: float = 1.0,
    ) -> dict[str, float]:
        """Calculate processing efficiency metrics.

        Args:
            processing_time: Time taken to process document
            document_complexity: Complexity factor (1.0 = normal)

        Returns:
            Dictionary of efficiency metrics

        """
        # Baseline processing times (seconds)
        baseline_times = {
            "simple": 2.0,
            "normal": 5.0,
            "complex": 10.0,
        }

        # Determine complexity category
        if document_complexity <= 0.5:
            complexity_category = "simple"
        elif document_complexity <= 1.5:
            complexity_category = "normal"
        else:
            complexity_category = "complex"

        baseline_time = baseline_times[complexity_category]

        # Calculate efficiency metrics
        efficiency_ratio = (
            baseline_time / processing_time if processing_time > 0 else 0.0
        )

        # Performance categories
        if efficiency_ratio >= 2.0:
            performance_category = "excellent"
        elif efficiency_ratio >= 1.5:
            performance_category = "good"
        elif efficiency_ratio >= 1.0:
            performance_category = "acceptable"
        elif efficiency_ratio >= 0.5:
            performance_category = "poor"
        else:
            performance_category = "very_poor"

        return {
            "processing_time": processing_time,
            "baseline_time": baseline_time,
            "efficiency_ratio": efficiency_ratio,
            "performance_category": performance_category,
            "complexity_factor": document_complexity,
        }

    def _exact_match_comparison(
        self,
        extracted_value: Any,
        true_value: Any,
        field_name: str,
    ) -> bool:
        """Compare two values for exact match with field-specific logic."""
        # Convert to strings and normalize
        extracted_str = str(extracted_value).strip().lower()
        true_str = str(true_value).strip().lower()

        # Field-specific comparison logic
        if field_name in ["abn", "abn_number"]:
            # Remove spaces and hyphens for ABN comparison
            extracted_clean = re.sub(r"[\s-]", "", extracted_str)
            true_clean = re.sub(r"[\s-]", "", true_str)
            return extracted_clean == true_clean

        if field_name in ["total_amount", "amount", "subtotal", "gst_amount"]:
            # Remove currency symbols and normalize amounts
            extracted_amount = re.sub(r"[$,\s]", "", extracted_str)
            true_amount = re.sub(r"[$,\s]", "", true_str)
            try:
                return float(extracted_amount) == float(true_amount)
            except ValueError:
                return extracted_str == true_str

        elif field_name in ["date", "transaction_date", "invoice_date"]:
            # Normalize date formats
            return self._normalize_date(extracted_str) == self._normalize_date(true_str)

        else:
            # Standard string comparison
            return extracted_str == true_str

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein distance."""
        str1, str2 = str1.lower().strip(), str2.lower().strip()

        if str1 == str2:
            return 1.0

        if not str1 or not str2:
            return 0.0

        # Simple Levenshtein distance implementation
        len1, len2 = len(str1), len(str2)

        # Create distance matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Initialize first row and column
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        # Fill the matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if str1[i - 1] == str2[j - 1]:
                    cost = 0
                else:
                    cost = 1

                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # deletion
                    matrix[i][j - 1] + 1,  # insertion
                    matrix[i - 1][j - 1] + cost,  # substitution
                )

        # Calculate similarity
        max_len = max(len1, len2)
        distance = matrix[len1][len2]
        similarity = 1 - (distance / max_len) if max_len > 0 else 0

        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, similarity))

    def _validate_abn_format(self, abn: str) -> bool:
        """Validate Australian Business Number format."""
        if not abn:
            return False

        # Remove spaces and check format
        cleaned_abn = re.sub(r"\s", "", str(abn))
        return bool(re.match(r"^\d{11}$", cleaned_abn))

    def _validate_gst_calculation(self, fields: dict[str, Any]) -> bool:
        """Validate GST calculation (10% in Australia)."""
        try:
            if "total_amount" not in fields or "subtotal" not in fields:
                return False

            total = float(re.sub(r"[$,\s]", "", str(fields["total_amount"])))
            subtotal = float(re.sub(r"[$,\s]", "", str(fields["subtotal"])))

            # Calculate expected GST (10%)
            expected_gst = subtotal * 0.1
            calculated_total = subtotal + expected_gst

            # Allow 1 cent tolerance for rounding
            return abs(total - calculated_total) <= 0.01

        except (ValueError, KeyError):
            return False

    def _validate_australian_date_format(self, date_str: str) -> bool:
        """Validate Australian date format (DD/MM/YYYY)."""
        if not date_str:
            return False

        for pattern in self.date_patterns:
            if pattern.match(str(date_str).strip()):
                return True

        return False

    def _validate_amount_format(self, amount: str) -> bool:
        """Validate amount format."""
        if not amount:
            return False

        return bool(self.amount_pattern.match(str(amount).strip()))

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string for comparison."""
        # Simple normalization - could be enhanced
        normalized = re.sub(r"[-/\s]", "", date_str.lower())
        return normalized

    def _calculate_correlation(
        self,
        x_values: list[float],
        y_values: list[float],
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        n = len(x_values)

        # Calculate means
        mean_x = sum(x_values) / n
        mean_y = sum(y_values) / n

        # Calculate correlation coefficient
        numerator = sum(
            (x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n)
        )

        sum_sq_x = sum((x_values[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y_values[i] - mean_y) ** 2 for i in range(n))

        denominator = (sum_sq_x * sum_sq_y) ** 0.5

        if denominator == 0:
            return 0.0

        correlation = numerator / denominator
        return correlation

    def get_metrics_summary(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        processing_time: float = 1.0,
    ) -> dict[str, Any]:
        """Get comprehensive metrics summary.

        Args:
            extracted: Extracted fields from model
            ground_truth: Ground truth fields
            processing_time: Processing time in seconds

        Returns:
            Dictionary with comprehensive metrics
        """
        # Calculate basic metrics
        precision, recall, f1_score = self.calculate_prf_metrics(
            extracted, ground_truth
        )
        exact_match_score = self.calculate_exact_match(extracted, ground_truth)
        fuzzy_match_score = self.calculate_fuzzy_match(extracted, ground_truth)
        ato_compliance_score = self.calculate_ato_compliance_score(
            extracted, ground_truth
        )
        processing_efficiency = self.calculate_processing_efficiency(processing_time)

        # Calculate field counts
        field_count_extracted = len(extracted) if extracted else 0
        field_count_ground_truth = len(ground_truth) if ground_truth else 0
        field_coverage = (
            field_count_extracted / field_count_ground_truth
            if field_count_ground_truth > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "exact_match_score": exact_match_score,
            "fuzzy_match_score": fuzzy_match_score,
            "ato_compliance_score": ato_compliance_score,
            "processing_efficiency": processing_efficiency,
            "field_count_extracted": field_count_extracted,
            "field_count_ground_truth": field_count_ground_truth,
            "field_coverage": field_coverage,
        }
