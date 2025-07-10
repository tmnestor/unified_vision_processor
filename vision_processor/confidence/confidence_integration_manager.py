"""Confidence Integration Manager - 4-Component Confidence Scoring System

This module implements the Llama-3.2 4-component confidence scoring system
for production readiness assessment of Australian tax document processing.

Components:
1. Classification confidence (25%) - Document type classification accuracy
2. Extraction quality (35%) - Field extraction completeness and quality
3. ATO compliance (25%) - Australian Tax Office compliance indicators
4. Business recognition (15%) - Australian business context recognition
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    """Result from ATO compliance assessment."""

    compliance_score: float
    compliance_passed: bool
    passed: bool  # Alias for compliance_passed
    issues: list[str]
    recommendations: list[str]
    violations: list[str] = None  # Additional field for violations
    warnings: list[str] = None  # Additional field for warnings

    def __post_init__(self):
        """Post-initialization to set defaults and aliases."""
        if self.violations is None:
            self.violations = []
        if self.warnings is None:
            self.warnings = []
        # Set passed as alias for compliance_passed
        self.passed = self.compliance_passed


@dataclass
class ConfidenceResult:
    """Result from confidence assessment."""

    overall_confidence: float
    quality_grade: str  # Will use QualityGrade enum
    production_ready: bool
    component_scores: dict[str, float]
    quality_flags: list[str]
    recommendations: list[str]


class BasePipelineComponent(ABC):
    """Base class for pipeline components."""

    def __init__(self, config: Any):
        self.config = config
        self.initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component."""

    def ensure_initialized(self) -> None:
        """Ensure component is initialized."""
        if not self.initialized:
            self.initialize()
            self.initialized = True


class ConfidenceManager(BasePipelineComponent):
    """4-component confidence scoring system for production readiness assessment."""

    def initialize(self) -> None:
        """Initialize confidence manager with 4-component scoring."""
        # Component weight configuration (Llama 4-component system)
        self.component_weights = {
            "classification": 0.25,  # 25% - Document type classification
            "extraction": 0.35,  # 35% - Field extraction quality
            "ato_compliance": 0.25,  # 25% - ATO compliance indicators
            "business_recognition": 0.15,  # 15% - Australian business recognition
        }

        # Production readiness thresholds
        self.readiness_thresholds = {
            "excellent": 0.90,  # 90%+ confidence
            "good": 0.70,  # 70-89% confidence
            "fair": 0.50,  # 50-69% confidence
            "poor": 0.30,  # 30-49% confidence
            "very_poor": 0.0,  # <30% confidence
        }

        # Quality control thresholds
        self.quality_thresholds = {
            "minimum_production_confidence": 0.70,
            "minimum_extraction_fields": 3,
            "minimum_ato_compliance": 0.60,
            "minimum_classification_confidence": 0.60,
        }

        # Processing decision rules
        self.processing_rules = {
            "auto_approve_threshold": 0.90,
            "manual_review_threshold": 0.70,
            "reject_threshold": 0.30,
            "ato_compliance_required": 0.80,
        }

        logger.info("ConfidenceManager initialized with 4-component scoring system")

    def assess_document_confidence(
        self,
        raw_text: str,
        extracted_fields: dict[str, Any],
        compliance_result: ComplianceResult,
        classification_confidence: float,
        highlights_detected: bool,
    ) -> ConfidenceResult:
        """Assess document confidence using 4-component scoring.

        Returns:
            ConfidenceResult with comprehensive assessment

        """
        # Component 1: Classification confidence (25%)
        classification_score = classification_confidence

        # Component 2: Extraction quality (35%)
        extraction_score = self._assess_extraction_quality(extracted_fields, raw_text)

        # Component 3: ATO compliance (25%)
        ato_compliance_score = compliance_result.compliance_score

        # Component 4: Business recognition (15%)
        business_recognition_score = self._assess_business_recognition(
            raw_text,
            extracted_fields,
            highlights_detected,
        )

        # Calculate overall confidence using weighted average
        component_scores = {
            "classification": classification_score,
            "extraction": extraction_score,
            "ato_compliance": ato_compliance_score,
            "business_recognition": business_recognition_score,
        }

        overall_confidence = sum(
            component_scores[component] * self.component_weights[component]
            for component in self.component_weights
        )

        # Determine quality grade
        quality_grade = self._determine_quality_grade(overall_confidence)

        # Determine production readiness
        production_ready = (
            overall_confidence
            >= self.quality_thresholds["minimum_production_confidence"]
        )

        # Generate quality flags
        quality_flags = self._generate_quality_flags(
            component_scores,
            extracted_fields,
            compliance_result,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            component_scores,
            quality_flags,
            overall_confidence,
        )

        return ConfidenceResult(
            overall_confidence=overall_confidence,
            quality_grade=quality_grade,
            production_ready=production_ready,
            component_scores=component_scores,
            quality_flags=quality_flags,
            recommendations=recommendations,
        )

    def _assess_extraction_quality(
        self,
        extracted_fields: dict[str, Any],
        _raw_text: str,
    ) -> float:
        """Assess extraction quality based on fields extracted."""
        if not extracted_fields:
            return 0.0

        # Base score from number of fields
        field_count = len(
            [v for v in extracted_fields.values() if v is not None and v != ""],
        )
        field_score = min(field_count / 6.0, 1.0)  # Expect around 6 fields

        # Quality indicators
        quality_indicators = 0.0
        max_indicators = 5.0

        # Check for key fields
        key_fields = ["date", "total_amount", "supplier_name", "business_name"]
        for field in key_fields:
            if extracted_fields.get(field):
                quality_indicators += 1.0

        # Check for structured data
        if extracted_fields.get("gst_amount"):
            quality_indicators += 0.5

        # Normalize quality indicators
        quality_score = quality_indicators / max_indicators

        # Combine field count and quality
        extraction_score = (field_score * 0.6) + (quality_score * 0.4)

        return min(extraction_score, 1.0)

    def _assess_business_recognition(
        self,
        raw_text: str,
        extracted_fields: dict[str, Any],
        highlights_detected: bool,
    ) -> float:
        """Assess Australian business recognition."""
        score = 0.0

        # Check for Australian business names in text
        australian_businesses = [
            "woolworths",
            "coles",
            "aldi",
            "target",
            "kmart",
            "bunnings",
            "bp",
            "shell",
            "caltex",
            "ampol",
            "mobil",
            "7-eleven",
            "anz",
            "commonwealth bank",
            "westpac",
            "nab",
            "ing",
            "macquarie",
            "qantas",
            "jetstar",
            "virgin australia",
            "tigerair",
        ]

        text_lower = raw_text.lower()
        for business in australian_businesses:
            if business in text_lower:
                score += 0.3

        # Check for Australian business indicators
        if "abn" in text_lower or extracted_fields.get("abn"):
            score += 0.2

        if "gst" in text_lower or extracted_fields.get("gst_amount"):
            score += 0.2

        # Check for Australian address indicators
        au_indicators = [
            "australia",
            "sydney",
            "melbourne",
            "brisbane",
            "perth",
            "adelaide",
        ]
        for indicator in au_indicators:
            if indicator in text_lower:
                score += 0.1

        # Bonus for highlights detected (InternVL feature)
        if highlights_detected:
            score += 0.1

        return min(score, 1.0)

    def _determine_quality_grade(self, overall_confidence: float) -> str:
        """Determine quality grade based on confidence."""
        if overall_confidence >= self.readiness_thresholds["excellent"]:
            return "excellent"
        if overall_confidence >= self.readiness_thresholds["good"]:
            return "good"
        if overall_confidence >= self.readiness_thresholds["fair"]:
            return "fair"
        if overall_confidence >= self.readiness_thresholds["poor"]:
            return "poor"
        return "very_poor"

    def _generate_quality_flags(
        self,
        component_scores: dict[str, float],
        extracted_fields: dict[str, Any],
        compliance_result: ComplianceResult,
    ) -> list[str]:
        """Generate quality flags based on component scores."""
        flags = []

        # Classification flags
        if (
            component_scores["classification"]
            < self.quality_thresholds["minimum_classification_confidence"]
        ):
            flags.append("low_classification_confidence")

        # Extraction flags
        field_count = len(
            [v for v in extracted_fields.values() if v is not None and v != ""],
        )
        if field_count < self.quality_thresholds["minimum_extraction_fields"]:
            flags.append("insufficient_fields")

        # ATO compliance flags
        if (
            component_scores["ato_compliance"]
            < self.quality_thresholds["minimum_ato_compliance"]
        ):
            flags.append("low_ato_compliance")

        if not compliance_result.compliance_passed:
            flags.append("ato_compliance_failed")

        # Business recognition flags
        if component_scores["business_recognition"] < 0.3:
            flags.append("low_business_recognition")

        # Overall quality flags
        overall_confidence = sum(
            component_scores[component] * self.component_weights[component]
            for component in self.component_weights
        )

        if (
            overall_confidence
            < self.quality_thresholds["minimum_production_confidence"]
        ):
            flags.append("below_production_threshold")

        return flags

    def _generate_recommendations(
        self,
        component_scores: dict[str, float],
        quality_flags: list[str],
        overall_confidence: float,
    ) -> list[str]:
        """Generate recommendations based on confidence assessment."""
        recommendations = []

        # Component-specific recommendations
        if component_scores["classification"] < 0.6:
            recommendations.append("Consider manual document type verification")

        if component_scores["extraction"] < 0.6:
            recommendations.append("Review extracted fields for completeness")

        if component_scores["ato_compliance"] < 0.6:
            recommendations.append("Verify ATO compliance requirements")

        if component_scores["business_recognition"] < 0.5:
            recommendations.append("Verify Australian business context")

        # Quality flag recommendations
        if "below_production_threshold" in quality_flags:
            recommendations.append("Document confidence below production threshold")

        if "ato_compliance_failed" in quality_flags:
            recommendations.append("ATO compliance validation failed")

        if "insufficient_fields" in quality_flags:
            recommendations.append(
                "Insufficient fields extracted - consider manual review",
            )

        # Processing recommendations
        if overall_confidence >= self.processing_rules["auto_approve_threshold"]:
            recommendations.append("Document ready for automated processing")
        elif overall_confidence >= self.processing_rules["manual_review_threshold"]:
            recommendations.append("Document requires manual review")
        else:
            recommendations.append("Document requires manual processing")

        return list(set(recommendations))  # Remove duplicates
