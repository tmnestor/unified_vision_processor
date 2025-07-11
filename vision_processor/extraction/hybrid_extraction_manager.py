"""Hybrid Extraction Manager - Unified 7-Step Processing Pipeline

Implements the Llama-3.2 7-step processing pipeline as the foundation for unified vision processing.
Integrates InternVL advanced features while maintaining graceful degradation capabilities.

7-Step Pipeline:
1. Document Classification → 2. Model Inference → 3. Primary Extraction →
4. AWK Fallback → 5. Field Validation → 6. ATO Compliance → 7. Confidence Integration
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image

from ..classification import DocumentClassifier, DocumentType
from ..confidence import ConfidenceManager
from ..config.model_factory import ModelFactory
from ..config.prompt_manager import PromptManager
from ..config.unified_config import ExtractionMethod, UnifiedConfig
from ..models.base_model import BaseVisionModel, ModelResponse
from .awk_extractor import AWKExtractor
from .pipeline_components import (
    ATOComplianceHandler,
    EnhancedKeyValueParser,
    HighlightDetector,
    create_document_handler,
)

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing pipeline stages."""

    CLASSIFICATION = "classification"
    INFERENCE = "inference"
    PRIMARY_EXTRACTION = "primary_extraction"
    AWK_FALLBACK = "awk_fallback"
    VALIDATION = "validation"
    ATO_COMPLIANCE = "ato_compliance"
    CONFIDENCE_INTEGRATION = "confidence_integration"


class QualityGrade(Enum):
    """Quality assessment grades."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"


@dataclass
class ProcessingResult:
    """Comprehensive processing result from unified pipeline."""

    # Basic processing info
    model_type: str
    document_type: str
    processing_time: float

    # Model response
    raw_response: str
    model_confidence: float

    # Extraction results
    extracted_fields: dict[str, Any]
    awk_fallback_used: bool
    highlights_detected: int

    # Quality assessment
    confidence_score: float
    quality_grade: QualityGrade
    production_ready: bool

    # Compliance and validation
    ato_compliance_score: float
    validation_passed: bool

    # Processing pipeline info
    stages_completed: list[ProcessingStage]
    quality_flags: list[str]
    recommendations: list[str]

    # Technical metrics
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0

    # Error handling
    errors: list[str] = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class UnifiedExtractionManager:
    """Unified extraction manager implementing Llama 7-step pipeline.

    Serves as the foundation for model-agnostic document processing,
    integrating InternVL advanced capabilities with Llama processing excellence.
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize the unified extraction manager.

        Args:
            config: Unified configuration object

        """
        self.config = config

        # Initialize model
        self.model: BaseVisionModel | None = None
        self._initialize_model()

        # Initialize pipeline components
        self.classifier = DocumentClassifier(config)
        self.awk_extractor = AWKExtractor(config)
        self.confidence_manager = ConfidenceManager(config)
        self.ato_compliance = ATOComplianceHandler(config)
        self.prompt_manager = PromptManager()
        self.prompt_manager.initialize()

        # InternVL integrations
        self.highlight_detector = HighlightDetector(config) if config.highlight_detection else None
        self.enhanced_parser = (
            EnhancedKeyValueParser(config) if config.extraction_method == ExtractionMethod.HYBRID else None
        )

        # Initialize all components
        self._initialize_components()

        # Processing state
        self.processing_stats = {
            "total_documents": 0,
            "successful_extractions": 0,
            "awk_fallbacks": 0,
            "production_ready": 0,
        }

        logger.info(
            f"Initialized UnifiedExtractionManager with {config.model_type.value} model",
        )

    def _use_test_components(self):
        """Replace components with test mocks if available."""
        if hasattr(self, "_test_mocks"):
            if "classifier" in self._test_mocks:
                self.classifier = self._test_mocks["classifier"]
            if "confidence_manager" in self._test_mocks:
                self.confidence_manager = self._test_mocks["confidence_manager"]
            if "ato_handler" in self._test_mocks:
                self.ato_compliance = self._test_mocks["ato_handler"]

    def _initialize_model(self) -> None:
        """Initialize the vision model."""
        try:
            # Log offline mode status
            if self.config.offline_mode:
                logger.info(
                    f"Offline mode enabled - loading {self.config.model_type.value} from {self.config.model_path}",
                )

            self.model = ModelFactory.create_model(
                self.config.model_type,
                self.config.model_path,
                self.config,
            )
            logger.info(
                f"Model {self.config.model_type.value} initialized successfully",
            )
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        components = [
            self.classifier,
            self.awk_extractor,
            self.confidence_manager,
            self.ato_compliance,
            self.prompt_manager,
        ]

        if self.highlight_detector:
            components.append(self.highlight_detector)
        if self.enhanced_parser:
            components.append(self.enhanced_parser)

        for component in components:
            try:
                component.ensure_initialized()
            except Exception as e:
                logger.warning(
                    f"Failed to initialize {component.__class__.__name__}: {e}",
                )

    def process_document(
        self,
        image_path: str | Path | Image.Image,
        document_type: str | None = None,
        **_kwargs,
    ) -> ProcessingResult:
        """Process document using unified 7-step Llama pipeline.

        Args:
            image_path: Path to image file or PIL Image
            document_type: Optional pre-classified document type
            **kwargs: Additional processing parameters

        Returns:
            ProcessingResult with comprehensive analysis

        """
        # Use test components if available (for testing)
        self._use_test_components()

        start_time = time.time()
        stages_completed = []
        quality_flags = []
        recommendations = []
        errors = []
        warnings = []

        try:
            # Convert image path to Path object if needed
            if isinstance(image_path, str):
                image_path = Path(image_path)

            # =================================================
            # STEP 1: DOCUMENT CLASSIFICATION
            # =================================================
            logger.info("Step 1: Document Classification")
            stages_completed.append(ProcessingStage.CLASSIFICATION)

            if document_type:
                classified_type = DocumentType(document_type)
                classification_confidence = 1.0
                classification_evidence = []
                logger.info(f"Using provided document type: {classified_type}")
            else:
                # Use classify_with_evidence for consistent interface
                classified_type, classification_confidence, classification_evidence = (
                    self.classifier.classify_with_evidence(image_path)
                )
                logger.info(
                    f"Classification: {classified_type.value} (confidence: {classification_confidence:.2f})",
                )

            # Graceful degradation: proceed even with lower confidence
            if classification_confidence < self.config.confidence_threshold:
                quality_flags.append("low_classification_confidence")
                if classification_confidence < 0.3:
                    recommendations.append(
                        "Manual document type verification recommended",
                    )

            # =================================================
            # STEP 2: MODEL INFERENCE
            # =================================================
            logger.info("Step 2: Model Inference")
            stages_completed.append(ProcessingStage.INFERENCE)

            # Get prompt from YAML-based prompt manager
            classification_text = " ".join(classification_evidence) if classification_evidence else None
            prompt = self.prompt_manager.get_prompt_for_document_type(
                classified_type,
                model_type=self.config.model_type.value,
                classification_response=classification_text,
            )

            logger.info(f"DEBUG: Prompt being sent to model (first 200 chars): {prompt[:200]}")

            # Process with model
            model_response = self._process_with_model(image_path, prompt)

            # InternVL Integration: Computer Vision Processing (placeholder for Phase 4)
            highlights = []
            if (
                self.config.computer_vision
                and self.highlight_detector
                and classified_type == DocumentType.BANK_STATEMENT
            ):
                highlights = []  # Placeholder - will implement highlight detection
                logger.info(f"Detected {len(highlights)} highlights")

            # =================================================
            # STEP 3: HANDLER SELECTION AND PRIMARY EXTRACTION
            # =================================================
            logger.info("Step 3: Primary Extraction")
            stages_completed.append(ProcessingStage.PRIMARY_EXTRACTION)

            # Get document handler and perform primary extraction
            handler = self._get_handler(classified_type)
            handler.ensure_initialized()

            # DEBUG: Log raw model response for troubleshooting
            logger.info(f"Raw model response (first 500 chars): {model_response.raw_text[:500]}")

            extracted_fields = handler.extract_fields_primary(model_response.raw_text)

            # DEBUG: Log extracted fields
            logger.info(f"Extracted fields: {extracted_fields}")

            # InternVL Integration: Enhanced Key-Value Parser
            if self.config.extraction_method == ExtractionMethod.HYBRID and self.enhanced_parser:
                enhanced_fields = self.enhanced_parser.parse(model_response.raw_text)
                extracted_fields = self._merge_extractions(
                    extracted_fields,
                    enhanced_fields,
                )

            # =================================================
            # STEP 4: AWK FALLBACK
            # =================================================
            logger.info("Step 4: AWK Fallback Assessment")
            stages_completed.append(ProcessingStage.AWK_FALLBACK)

            awk_used = False
            awk_enabled = self.config.awk_fallback
            quality_insufficient = self._extraction_quality_insufficient(extracted_fields)

            # Preliminary quality grading for AWK fallback decision
            preliminary_grade = self._assess_preliminary_quality_grade(extracted_fields)

            # Trigger AWK fallback for insufficient quality OR Fair/Poor grades
            should_use_awk = quality_insufficient or preliminary_grade in [
                QualityGrade.FAIR,
                QualityGrade.POOR,
                QualityGrade.VERY_POOR,
            ]

            logger.info(
                f"AWK check: enabled={awk_enabled}, quality_insufficient={quality_insufficient}, "
                f"preliminary_grade={preliminary_grade.value}, should_use_awk={should_use_awk}"
            )

            if awk_enabled and should_use_awk:
                # Use AWK extractor component
                logger.info("Triggering AWK fallback extraction")
                awk_fields = self.awk_extractor.extract(
                    model_response.raw_text,
                    classified_type,
                )
                logger.info(f"AWK extracted fields: {awk_fields}")
                extracted_fields = self._merge_extractions(extracted_fields, awk_fields)
                awk_used = True
                self.processing_stats["awk_fallbacks"] += 1
                quality_flags.append("awk_fallback_used")
                logger.info("AWK fallback extraction applied")
            else:
                logger.info(
                    f"AWK fallback skipped: enabled={awk_enabled}, quality_insufficient={quality_insufficient}, "
                    f"preliminary_grade={preliminary_grade.value}, should_use_awk={should_use_awk}"
                )

            # =================================================
            # STEP 5: FIELD VALIDATION
            # =================================================
            logger.info("Step 5: Field Validation")
            stages_completed.append(ProcessingStage.VALIDATION)

            # Use handler for field validation
            validated_fields = handler.validate_fields(extracted_fields)
            validation_passed = len(validated_fields) > 0 and not any(
                v is None for v in validated_fields.values()
            )

            # InternVL Integration: Highlight Enhancement (placeholder for Phase 4)
            if highlights and classified_type == DocumentType.BANK_STATEMENT:
                validated_fields = self._enhance_with_highlights(
                    validated_fields,
                    highlights,
                )

            # =================================================
            # STEP 6: ATO COMPLIANCE ASSESSMENT
            # =================================================
            logger.info("Step 6: ATO Compliance Assessment")
            stages_completed.append(ProcessingStage.ATO_COMPLIANCE)

            # Use ATO compliance component
            compliance_result = self.ato_compliance.assess_compliance(
                validated_fields,
                classified_type,
            )
            ato_compliance_score = compliance_result.compliance_score

            # Populate warnings and errors from compliance assessment
            if hasattr(compliance_result, "violations") and compliance_result.violations:
                errors.extend(compliance_result.violations)
            if hasattr(compliance_result, "warnings") and compliance_result.warnings:
                warnings.extend(compliance_result.warnings)

            # =================================================
            # STEP 7: CONFIDENCE INTEGRATION AND PRODUCTION READINESS
            # =================================================
            logger.info("Step 7: Confidence Integration")
            stages_completed.append(ProcessingStage.CONFIDENCE_INTEGRATION)

            # 4-component confidence scoring using confidence manager
            try:
                confidence_result = self.confidence_manager.assess_document_confidence(
                    model_response.raw_text,
                    validated_fields,
                    compliance_result,
                    classification_confidence,
                    bool(highlights),
                )
            except Exception as e:
                logger.error(f"Confidence assessment failed: {e}")
                # Create fallback confidence result
                from ..confidence import ConfidenceResult

                confidence_result = ConfidenceResult(
                    overall_confidence=0.5,
                    quality_grade=QualityGrade.FAIR,
                    production_ready=False,
                    component_scores={},
                    quality_flags=["confidence_assessment_failed"],
                    recommendations=["Manual review recommended"],
                )

            # Update processing stats
            self.processing_stats["total_documents"] += 1
            if confidence_result.overall_confidence > self.config.confidence_threshold:
                self.processing_stats["successful_extractions"] += 1
            if confidence_result.production_ready:
                self.processing_stats["production_ready"] += 1

            processing_time = time.time() - start_time

            # Get memory usage
            memory_usage = self.model.get_memory_usage() if self.model else 0.0

            return ProcessingResult(
                model_type=self.config.model_type.value,
                document_type=classified_type.value,
                processing_time=processing_time,
                raw_response=model_response.raw_text,
                model_confidence=model_response.confidence,
                extracted_fields=validated_fields,
                awk_fallback_used=awk_used,
                highlights_detected=len(highlights),
                confidence_score=confidence_result.overall_confidence,
                quality_grade=(
                    QualityGrade(confidence_result.quality_grade)
                    if isinstance(confidence_result.quality_grade, str)
                    else confidence_result.quality_grade
                ),
                production_ready=confidence_result.production_ready,
                ato_compliance_score=ato_compliance_score,
                validation_passed=validation_passed,
                stages_completed=stages_completed,
                quality_flags=quality_flags + confidence_result.quality_flags,
                recommendations=recommendations + confidence_result.recommendations,
                memory_usage_mb=memory_usage,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(
                f"Processing failed at stage {stages_completed[-1] if stages_completed else 'initialization'}: {e}",
            )
            errors.append(str(e))

            # Return partial result on error
            processing_time = time.time() - start_time
            return ProcessingResult(
                model_type=self.config.model_type.value,
                document_type="error",
                processing_time=processing_time,
                raw_response="",
                model_confidence=0.0,
                extracted_fields={},
                awk_fallback_used=False,
                highlights_detected=0,
                confidence_score=0.0,
                quality_grade=QualityGrade.VERY_POOR,
                production_ready=False,
                ato_compliance_score=0.0,
                validation_passed=False,
                stages_completed=stages_completed,
                quality_flags=["processing_error"],
                recommendations=["Review error logs and retry"],
                errors=errors,
                warnings=warnings,
            )

    def _process_with_model(
        self,
        image_path: Path | Image.Image,
        prompt: str,
    ) -> ModelResponse:
        """Process image with the vision model."""
        if not self.model:
            raise RuntimeError("Model not initialized")

        return self.model.process_image(image_path, prompt)

    def _extraction_quality_insufficient(self, fields: dict[str, Any]) -> bool:
        """Assess if extraction quality is insufficient for AWK fallback.

        Checks both field count and value quality to determine if AWK fallback is needed.
        """
        # Filter out metadata fields to count only actual extracted data
        data_fields = {
            k: v
            for k, v in fields.items()
            if k not in ["extracted_by", "fields_count", "extraction_method", "handler_type"]
            and v is not None
            and str(v).strip()
        }

        # Check 1: Insufficient field count
        if len(data_fields) < 3:
            logger.info(f"Quality insufficient: only {len(data_fields)} fields extracted")
            return True

        # Check 2: Critical field quality assessment
        quality_issues = []

        # Check total_amount field - should be reasonable for business receipts
        if "total_amount" in data_fields:
            total_str = str(data_fields["total_amount"]).replace("$", "").replace(",", "").strip()
            try:
                total_value = float(total_str)
                # Flag obviously wrong totals (too small for typical business receipts)
                if total_value < 5.0:
                    quality_issues.append(f"total_amount too small: {total_value}")
            except (ValueError, TypeError):
                quality_issues.append(f"total_amount not numeric: {total_str}")

        # Check receipt_number field - should not be nonsensical single words
        if "receipt_number" in data_fields:
            receipt_str = str(data_fields["receipt_number"]).strip()
            # Flag single letters or common meaningless extractions
            if len(receipt_str) <= 2 or receipt_str.lower() in [
                "is",
                "it",
                "to",
                "of",
                "and",
                "the",
                "a",
                "an",
            ]:
                quality_issues.append(f"receipt_number invalid: '{receipt_str}'")

        # Check supplier_name field - should not be single characters
        if "supplier_name" in data_fields:
            supplier_str = str(data_fields["supplier_name"]).strip()
            if len(supplier_str) <= 2:
                quality_issues.append(f"supplier_name too short: '{supplier_str}'")

        # Check date fields - should follow DD/MM/YYYY pattern or similar
        date_fields = ["invoice_date", "transaction_date", "date"]
        for field in date_fields:
            if field in data_fields:
                date_str = str(data_fields[field]).strip()
                # Basic date validation - should contain numbers and separators
                if not any(char.isdigit() for char in date_str) or len(date_str) < 8:
                    quality_issues.append(f"{field} invalid format: '{date_str}'")

        if quality_issues:
            logger.info(f"Quality insufficient: {len(quality_issues)} issues found: {quality_issues}")
            return True

        # Check 3: Overall value meaningfulness
        meaningless_values = 0
        for _key, value in data_fields.items():
            value_str = str(value).strip().lower()
            # Count values that are clearly extraction errors
            if len(value_str) <= 1 or value_str in [
                "n/a",
                "na",
                "not available",
                "not visible",
                "none",
                "",
                "-",
            ]:
                meaningless_values += 1

        # If more than half the fields are meaningless, quality is insufficient
        if meaningless_values > len(data_fields) // 2:
            logger.info(
                f"Quality insufficient: {meaningless_values}/{len(data_fields)} fields are meaningless"
            )
            return True

        # Quality is sufficient
        logger.info(f"Quality sufficient: {len(data_fields)} fields with good values")
        return False

    def _assess_preliminary_quality_grade(self, fields: dict[str, Any]) -> QualityGrade:
        """Assess preliminary quality grade for AWK fallback decision.

        This is a lightweight version of quality assessment that runs before
        the full confidence manager assessment.
        """
        # Filter out metadata fields
        data_fields = {
            k: v
            for k, v in fields.items()
            if k not in ["extracted_by", "fields_count", "extraction_method", "handler_type"]
            and v is not None
            and str(v).strip()
        }

        field_count = len(data_fields)
        quality_issues = 0

        # Count quality issues using similar logic to _extraction_quality_insufficient

        # Check critical fields
        if "total_amount" in data_fields:
            total_str = str(data_fields["total_amount"]).replace("$", "").replace(",", "").strip()
            try:
                total_value = float(total_str)
                if total_value < 5.0:
                    quality_issues += 1
            except (ValueError, TypeError):
                quality_issues += 1

        if "receipt_number" in data_fields:
            receipt_str = str(data_fields["receipt_number"]).strip()
            if len(receipt_str) <= 2 or receipt_str.lower() in [
                "is",
                "it",
                "to",
                "of",
                "and",
                "the",
                "a",
                "an",
            ]:
                quality_issues += 1

        if "supplier_name" in data_fields:
            supplier_str = str(data_fields["supplier_name"]).strip()
            if len(supplier_str) <= 2:
                quality_issues += 1

        # Check date fields
        date_fields = ["invoice_date", "transaction_date", "date"]
        for field in date_fields:
            if field in data_fields:
                date_str = str(data_fields[field]).strip()
                if not any(char.isdigit() for char in date_str) or len(date_str) < 8:
                    quality_issues += 1

        # Count meaningless values
        meaningless_values = 0
        for _key, value in data_fields.items():
            value_str = str(value).strip().lower()
            if len(value_str) <= 1 or value_str in [
                "n/a",
                "na",
                "not available",
                "not visible",
                "none",
                "",
                "-",
            ]:
                meaningless_values += 1

        # Grade based on field count, quality issues, and meaningless values
        if field_count < 3 or quality_issues >= 3:
            return QualityGrade.VERY_POOR
        elif field_count < 5 or quality_issues >= 2 or meaningless_values > field_count // 2:
            return QualityGrade.POOR
        elif field_count < 7 or quality_issues >= 1 or meaningless_values > field_count // 3:
            return QualityGrade.FAIR
        elif field_count >= 8 and quality_issues == 0 and meaningless_values == 0:
            return QualityGrade.EXCELLENT
        else:
            return QualityGrade.GOOD

    def _merge_extractions(
        self,
        primary: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge primary and fallback extractions."""
        merged = primary.copy()
        for key, value in fallback.items():
            if key not in merged or not merged[key]:
                merged[key] = value
        return merged

    def _enhance_with_highlights(
        self,
        fields: dict[str, Any],
        highlights: list,
    ) -> dict[str, Any]:
        """Enhance fields with highlight information (placeholder)."""
        # This will be implemented in Phase 4
        fields["highlights_processed"] = len(highlights)
        return fields

    def process_batch(
        self,
        image_paths: list[str | Path | Image.Image],
        document_types: list[str] | None = None,
        **kwargs,
    ) -> list[ProcessingResult]:
        """Process multiple documents in batch.

        Args:
            image_paths: List of image paths or PIL Images
            document_types: Optional list of document types
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessingResult objects

        """
        if document_types and len(document_types) != len(image_paths):
            raise ValueError("document_types length must match image_paths length")

        results = []

        for i, image_path in enumerate(image_paths):
            document_type = document_types[i] if document_types else None

            try:
                result = self.process_document(image_path, document_type, **kwargs)
                results.append(result)
                logger.info(
                    f"Processed document {i + 1}/{len(image_paths)}: {result.quality_grade.value}",
                )
            except Exception as e:
                logger.error(f"Failed to process document {i + 1}: {e}")
                # Add error result
                error_result = ProcessingResult(
                    model_type=self.config.model_type.value,
                    document_type="error",
                    processing_time=0.0,
                    raw_response="",
                    model_confidence=0.0,
                    extracted_fields={},
                    awk_fallback_used=False,
                    highlights_detected=0,
                    confidence_score=0.0,
                    quality_grade=QualityGrade.VERY_POOR,
                    production_ready=False,
                    ato_compliance_score=0.0,
                    validation_passed=False,
                    stages_completed=[],
                    quality_flags=["batch_processing_error"],
                    recommendations=["Review error logs"],
                    errors=[str(e)],
                )
                results.append(error_result)

        return results

    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()

        if stats["total_documents"] > 0:
            stats["success_rate"] = stats["successful_extractions"] / stats["total_documents"]
            stats["awk_fallback_rate"] = stats["awk_fallbacks"] / stats["total_documents"]
            stats["production_ready_rate"] = stats["production_ready"] / stats["total_documents"]
        else:
            stats["success_rate"] = 0.0
            stats["awk_fallback_rate"] = 0.0
            stats["production_ready_rate"] = 0.0

        if self.model:
            stats["model_info"] = self.model.get_device_info()

        return stats

    def _get_handler(self, document_type: DocumentType):
        """Get document handler for the specified document type.

        Args:
            document_type: The type of document to get a handler for

        Returns:
            Document handler instance for the specified type
        """
        return create_document_handler(document_type, self.config)

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "total_documents": 0,
            "successful_extractions": 0,
            "awk_fallbacks": 0,
            "production_ready": 0,
        }
        logger.info("Processing statistics reset")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.model:
            if hasattr(self.model, "__exit__"):
                self.model.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        # Handle both enum and string values defensively
        model_value = getattr(self.config.model_type, "value", self.config.model_type)
        pipeline_value = getattr(self.config.processing_pipeline, "value", self.config.processing_pipeline)
        return (
            f"UnifiedExtractionManager("
            f"model={model_value}, "
            f"pipeline={pipeline_value}, "
            f"processed={self.processing_stats['total_documents']})"
        )
