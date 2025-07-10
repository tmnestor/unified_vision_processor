"""
Base ATO Handler

Foundation handler implementing the Llama-3.2 7-step processing pipeline
for Australian Tax Office document processing, enhanced with InternVL features.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HandlerResult:
    """Result from document handler processing."""

    extracted_fields: Dict[str, Any]
    validation_passed: bool
    validation_issues: List[str]
    confidence_score: float
    processing_notes: List[str]
    highlight_enhanced: bool = False


class BaseATOHandler(ABC):
    """
    Base Australian Tax Office document handler implementing Llama 7-step pipeline.

    This handler provides the foundation for all 11 Australian tax document types,
    following the sophisticated 7-step processing approach from Llama-3.2 with
    InternVL feature integrations.

    7-Step Pipeline:
    1. Document Classification (handled by classifier)
    2. Model Inference (handled by model)
    3. Primary Field Extraction (implemented by subclasses)
    4. AWK Fallback (if needed)
    5. Field Validation (implemented by subclasses)
    6. ATO Compliance Assessment (delegated to compliance module)
    7. Confidence Integration (delegated to confidence module)
    """

    def __init__(self, config: Any):
        self.config = config
        self.initialized = False

        # InternVL feature integration flags
        self.supports_highlights = getattr(config, "highlight_detection", False)
        self.supports_enhanced_parsing = getattr(config, "enhanced_parsing", False)

        # ATO compliance requirements
        self.required_fields: List[str] = []
        self.optional_fields: List[str] = []
        self.validation_rules: Dict[str, Any] = {}

    def initialize(self) -> None:
        """Initialize handler with document-specific configuration."""
        if self.initialized:
            return

        # Load document-specific requirements
        self._load_field_requirements()
        self._load_validation_rules()
        self._load_australian_patterns()

        # Initialize InternVL integrations if enabled
        if self.supports_highlights:
            self._initialize_highlight_integration()

        logger.info(f"{self.__class__.__name__} initialized with ATO compliance")
        self.initialized = True

    @abstractmethod
    def _load_field_requirements(self) -> None:
        """Load required and optional fields for this document type."""
        pass

    @abstractmethod
    def _load_validation_rules(self) -> None:
        """Load validation rules specific to this document type."""
        pass

    def _load_australian_patterns(self) -> None:
        """Load Australian business and format patterns."""
        # Australian date patterns (DD/MM/YYYY)
        self.date_patterns = [
            r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
            r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b",
            r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b",
        ]

        # Australian currency patterns
        self.currency_patterns = [
            r"\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
            r"(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|aud|au\$)",
        ]

        # ABN patterns (Australian Business Number)
        self.abn_patterns = [
            r"\b(?:abn\s*:?\s*)?(\d{2}\s+\d{3}\s+\d{3}\s+\d{3})\b",
            r"\b(?:abn\s*:?\s*)?(\d{11})\b",
        ]

        # GST patterns (10% in Australia)
        self.gst_patterns = [
            r"(?:gst|tax)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
            r"(?:goods\s+and\s+services\s+tax)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
        ]

    def _initialize_highlight_integration(self) -> None:
        """Initialize InternVL highlight detection integration."""
        # This will be enhanced when computer vision module is fully integrated
        logger.info("Highlight integration initialized for enhanced extraction")

    def extract_fields_primary(self, raw_text: str) -> Dict[str, Any]:
        """
        Step 3: Primary field extraction using document-specific logic.

        Args:
            raw_text: Raw text from model response

        Returns:
            Dictionary of extracted fields
        """
        if not self.initialized:
            self.initialize()

        extracted_fields = {}

        # Extract common Australian tax fields
        extracted_fields.update(self._extract_common_fields(raw_text))

        # Extract document-specific fields (implemented by subclasses)
        document_fields = self._extract_document_specific_fields(raw_text)
        extracted_fields.update(document_fields)

        # Apply InternVL enhanced parsing if enabled
        if self.supports_enhanced_parsing:
            enhanced_fields = self._apply_enhanced_parsing(raw_text)
            extracted_fields = self._merge_field_extractions(
                extracted_fields, enhanced_fields
            )

        return extracted_fields

    def _extract_common_fields(self, text: str) -> Dict[str, Any]:
        """Extract fields common to all Australian tax documents."""
        fields = {}

        # Extract date
        date_match = self._extract_first_match(text, self.date_patterns)
        if date_match:
            fields["date"] = date_match

        # Extract total amount
        total_patterns = [
            r"(?:total|amount)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
            r"(?:grand\s+total)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
        ]
        total_match = self._extract_first_match(text, total_patterns)
        if total_match:
            fields["total_amount"] = float(total_match)

        # Extract GST
        gst_match = self._extract_first_match(text, self.gst_patterns)
        if gst_match:
            fields["gst_amount"] = float(gst_match)

        # Extract ABN
        abn_match = self._extract_first_match(text, self.abn_patterns)
        if abn_match:
            fields["abn"] = abn_match

        # Extract business name (Australian businesses)
        business_name = self._extract_australian_business_name(text)
        if business_name:
            fields["business_name"] = business_name

        return fields

    @abstractmethod
    def _extract_document_specific_fields(self, text: str) -> Dict[str, Any]:
        """Extract fields specific to this document type."""
        pass

    def _apply_enhanced_parsing(self, _text: str) -> Dict[str, Any]:
        """Apply InternVL enhanced parsing techniques."""
        # Placeholder for enhanced parsing integration
        # This will be implemented when InternVL features are fully integrated
        return {}

    def _merge_field_extractions(
        self, primary: Dict[str, Any], enhanced: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge primary and enhanced field extractions with conflict resolution."""
        merged = primary.copy()

        for key, value in enhanced.items():
            if key not in merged or not merged[key]:
                merged[key] = value
            elif (
                value and isinstance(value, str) and len(value) > len(str(merged[key]))
            ):
                # Prefer longer, more detailed extractions
                merged[key] = value

        return merged

    def validate_fields(self, fields: Dict[str, Any]) -> HandlerResult:
        """
        Step 5: Validate extracted fields using ATO compliance rules.

        Args:
            fields: Extracted fields to validate

        Returns:
            HandlerResult with validation status and issues
        """
        if not self.initialized:
            self.initialize()

        validation_issues = []
        processing_notes = []

        # Check required fields
        for required_field in self.required_fields:
            if required_field not in fields or not fields[required_field]:
                validation_issues.append(f"Missing required field: {required_field}")

        # Apply document-specific validation
        document_issues = self._validate_document_specific_fields(fields)
        validation_issues.extend(document_issues)

        # Apply Australian tax validation
        tax_issues = self._validate_australian_tax_fields(fields)
        validation_issues.extend(tax_issues)

        # Calculate confidence based on field completeness and validation
        confidence_score = self._calculate_field_confidence(fields, validation_issues)

        return HandlerResult(
            extracted_fields=fields,
            validation_passed=len(validation_issues) == 0,
            validation_issues=validation_issues,
            confidence_score=confidence_score,
            processing_notes=processing_notes,
        )

    @abstractmethod
    def _validate_document_specific_fields(self, fields: Dict[str, Any]) -> List[str]:
        """Validate fields specific to this document type."""
        pass

    def _validate_australian_tax_fields(self, fields: Dict[str, Any]) -> List[str]:
        """Validate Australian tax-specific fields."""
        issues = []

        # Validate ABN format if present
        if "abn" in fields and fields["abn"]:
            abn = str(fields["abn"]).replace(" ", "")
            if len(abn) != 11 or not abn.isdigit():
                issues.append("Invalid ABN format (must be 11 digits)")

        # Validate GST calculation if present
        if all(
            field in fields and fields[field]
            for field in ["total_amount", "gst_amount"]
        ):
            try:
                total = float(fields["total_amount"])
                gst = float(fields["gst_amount"])
                expected_gst = total * 0.10 / 1.10  # Extract GST from inclusive amount
                gst_difference = abs(gst - expected_gst)

                if gst_difference > 0.05:  # 5 cent tolerance
                    issues.append(
                        f"GST amount {gst:.2f} may be incorrect (expected ~{expected_gst:.2f})"
                    )
            except (ValueError, TypeError):
                issues.append("Invalid numeric values for GST calculation")

        # Validate date format (Australian DD/MM/YYYY)
        if "date" in fields and fields["date"]:
            date_str = str(fields["date"])
            if not re.match(r"\d{1,2}/\d{1,2}/\d{4}", date_str):
                issues.append("Date should be in DD/MM/YYYY format")

        return issues

    def _calculate_field_confidence(
        self, fields: Dict[str, Any], validation_issues: List[str]
    ) -> float:
        """Calculate confidence score based on field extraction and validation."""
        # Base confidence from field completeness
        total_fields = len(self.required_fields) + len(self.optional_fields)
        if total_fields == 0:
            field_completeness = 1.0
        else:
            extracted_count = sum(
                1
                for field in self.required_fields + self.optional_fields
                if field in fields and fields[field]
            )
            field_completeness = extracted_count / total_fields

        # Penalty for validation issues
        validation_penalty = min(len(validation_issues) * 0.1, 0.5)

        # Bonus for high-quality extractions
        quality_bonus = 0.0
        if "abn" in fields and fields["abn"]:
            quality_bonus += 0.1
        if "gst_amount" in fields and fields["gst_amount"]:
            quality_bonus += 0.1

        confidence = max(
            0.0, min(1.0, field_completeness - validation_penalty + quality_bonus)
        )
        return confidence

    def enhance_with_highlights(
        self, fields: Dict[str, Any], highlights: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enhance extracted fields using InternVL highlight detection.

        Args:
            fields: Currently extracted fields
            highlights: Detected highlights from computer vision

        Returns:
            Enhanced fields dictionary
        """
        if not highlights or not self.supports_highlights:
            return fields

        enhanced_fields = fields.copy()

        # Process highlights to extract additional information
        for highlight in highlights:
            # Extract text from highlight region
            if "text" in highlight and highlight["text"]:
                highlight_text = highlight["text"]

                # Try to extract missing fields from highlighted text
                highlight_fields = self._extract_common_fields(highlight_text)
                for key, value in highlight_fields.items():
                    if key not in enhanced_fields or not enhanced_fields[key]:
                        enhanced_fields[key] = value
                        logger.info(f"Enhanced field {key} from highlight: {value}")

        return enhanced_fields

    def _extract_first_match(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extract the first match from a list of regex patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return the first capture group or the whole match
                return match.group(1) if match.groups() else match.group(0)
        return None

    def _extract_australian_business_name(self, text: str) -> Optional[str]:
        """Extract Australian business names from text."""
        # Common Australian business names
        australian_businesses = [
            "woolworths",
            "coles",
            "aldi",
            "bunnings",
            "target",
            "kmart",
            "officeworks",
            "harvey norman",
            "jb hi-fi",
            "chemist warehouse",
            "bp",
            "shell",
            "caltex",
            "ampol",
            "7-eleven",
            "mcdonald's",
            "kfc",
            "subway",
        ]

        text_lower = text.lower()
        for business in australian_businesses:
            if business in text_lower:
                # Find the actual case version in the original text
                pattern = re.compile(re.escape(business), re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    return match.group(0)

        return None

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of handler processing capabilities."""
        return {
            "document_type": self.__class__.__name__.replace("Handler", ""),
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "supports_highlights": self.supports_highlights,
            "supports_enhanced_parsing": self.supports_enhanced_parsing,
            "validation_rules_count": len(self.validation_rules),
        }
