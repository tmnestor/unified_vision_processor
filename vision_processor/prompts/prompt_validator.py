"""Prompt Validator

Validation system for prompt compatibility and effectiveness across models.
"""

import logging
import re
from typing import Any

from ..extraction.pipeline_components import DocumentType

logger = logging.getLogger(__name__)


class PromptValidator:
    """Validate prompt compatibility and effectiveness across different models.

    Features:
    - Cross-model compatibility checking
    - Prompt structure validation
    - Australian tax compliance verification
    - Performance impact assessment
    """

    def __init__(self, config: Any = None):
        self.config = config or {}
        self.initialized = False

        # Validation criteria
        self.max_prompt_length = 4000  # Maximum reasonable prompt length
        self.min_prompt_length = 100  # Minimum effective prompt length
        self.required_australian_terms = ["australian", "ato", "gst", "abn", "business"]

        # Model compatibility requirements
        self.model_requirements = {
            "internvl3": {
                "supports_highlights": True,
                "supports_detailed_analysis": True,
                "optimal_length_range": (500, 2000),
                "required_keywords": ["extract", "analyze", "image"],
            },
            "llama32_vision": {
                "supports_highlights": False,
                "supports_detailed_analysis": True,
                "optimal_length_range": (300, 1500),
                "required_keywords": ["compliance", "ato", "business"],
            },
        }

    def initialize(self) -> None:
        """Initialize the prompt validator."""
        if self.initialized:
            return

        logger.info("PromptValidator initialized")
        self.initialized = True

    def validate_prompt(
        self,
        prompt: str,
        document_type: DocumentType,
        target_model: str = "both",
    ) -> dict[str, Any]:
        """Validate a prompt for effectiveness and compatibility.

        Args:
            prompt: Prompt text to validate
            document_type: Document type the prompt is designed for
            target_model: Target model (internvl3, llama32_vision, both)

        Returns:
            Validation results with scores and recommendations

        """
        if not self.initialized:
            self.initialize()

        validation_result = {
            "prompt_length": len(prompt),
            "document_type": document_type.value,
            "target_model": target_model,
            "validation_score": 0.0,
            "compatibility_score": 0.0,
            "effectiveness_score": 0.0,
            "issues": [],
            "recommendations": [],
            "passed": False,
        }

        # Basic structure validation
        structure_score = self._validate_structure(prompt, validation_result)

        # Content validation
        content_score = self._validate_content(prompt, document_type, validation_result)

        # Model compatibility validation
        compatibility_score = self._validate_compatibility(
            prompt,
            target_model,
            validation_result,
        )

        # Australian tax compliance validation
        compliance_score = self._validate_australian_compliance(
            prompt,
            validation_result,
        )

        # Calculate overall scores
        validation_result["effectiveness_score"] = (
            structure_score + content_score + compliance_score
        ) / 3
        validation_result["compatibility_score"] = compatibility_score
        validation_result["validation_score"] = (
            validation_result["effectiveness_score"] + compatibility_score
        ) / 2

        # Determine if validation passed
        validation_result["passed"] = (
            validation_result["validation_score"] >= 0.7
            and len(
                [
                    issue
                    for issue in validation_result["issues"]
                    if issue["severity"] == "critical"
                ],
            )
            == 0
        )

        return validation_result

    def _validate_structure(self, prompt: str, result: dict[str, Any]) -> float:
        """Validate basic prompt structure."""
        score = 1.0

        # Length validation
        if len(prompt) < self.min_prompt_length:
            result["issues"].append(
                {
                    "type": "structure",
                    "severity": "critical",
                    "message": f"Prompt too short ({len(prompt)} < {self.min_prompt_length} characters)",
                },
            )
            score -= 0.5

        if len(prompt) > self.max_prompt_length:
            result["issues"].append(
                {
                    "type": "structure",
                    "severity": "warning",
                    "message": f"Prompt very long ({len(prompt)} > {self.max_prompt_length} characters)",
                },
            )
            score -= 0.2

        # Structure elements
        if not re.search(r"(?:extract|analyze|process)", prompt, re.IGNORECASE):
            result["issues"].append(
                {
                    "type": "structure",
                    "severity": "warning",
                    "message": "Missing clear instruction verbs (extract, analyze, process)",
                },
            )
            score -= 0.1

        # Check for clear sections
        if not re.search(
            r"(?:requirements?|fields?|information):",
            prompt,
            re.IGNORECASE,
        ):
            result["issues"].append(
                {
                    "type": "structure",
                    "severity": "minor",
                    "message": "Consider adding clear section headers for requirements",
                },
            )
            score -= 0.05

        return max(score, 0.0)

    def _validate_content(
        self,
        prompt: str,
        document_type: DocumentType,
        result: dict[str, Any],
    ) -> float:
        """Validate prompt content relevance and completeness."""
        score = 1.0

        # Document type specific validation
        document_keywords = {
            DocumentType.FUEL_RECEIPT: [
                "fuel",
                "petrol",
                "diesel",
                "litres",
                "station",
            ],
            DocumentType.TAX_INVOICE: ["tax invoice", "abn", "gst", "supplier"],
            DocumentType.BUSINESS_RECEIPT: ["receipt", "business", "items", "total"],
            DocumentType.BANK_STATEMENT: [
                "bank",
                "statement",
                "transaction",
                "balance",
            ],
            DocumentType.MEAL_RECEIPT: ["meal", "restaurant", "food", "dining"],
            DocumentType.ACCOMMODATION: ["hotel", "accommodation", "booking", "stay"],
            DocumentType.TRAVEL_DOCUMENT: ["travel", "flight", "transport", "journey"],
            DocumentType.PARKING_TOLL: ["parking", "toll", "vehicle", "duration"],
            DocumentType.PROFESSIONAL_SERVICES: [
                "professional",
                "services",
                "consulting",
                "legal",
            ],
            DocumentType.EQUIPMENT_SUPPLIES: [
                "equipment",
                "supplies",
                "tools",
                "assets",
            ],
        }

        expected_keywords = document_keywords.get(document_type, [])
        found_keywords = sum(
            1
            for keyword in expected_keywords
            if re.search(rf"\b{keyword}\b", prompt, re.IGNORECASE)
        )

        if expected_keywords:
            keyword_coverage = found_keywords / len(expected_keywords)
            if keyword_coverage < 0.5:
                result["issues"].append(
                    {
                        "type": "content",
                        "severity": "warning",
                        "message": f"Low document-specific keyword coverage ({found_keywords}/{len(expected_keywords)})",
                    },
                )
                score -= 0.3
            elif keyword_coverage < 0.8:
                result["issues"].append(
                    {
                        "type": "content",
                        "severity": "minor",
                        "message": f"Moderate document-specific keyword coverage ({found_keywords}/{len(expected_keywords)})",
                    },
                )
                score -= 0.1

        # Check for output format specification
        if not re.search(r"(?:format|structure|json|output)", prompt, re.IGNORECASE):
            result["issues"].append(
                {
                    "type": "content",
                    "severity": "warning",
                    "message": "Consider specifying desired output format",
                },
            )
            score -= 0.1

        return max(score, 0.0)

    def _validate_compatibility(
        self,
        prompt: str,
        target_model: str,
        result: dict[str, Any],
    ) -> float:
        """Validate model compatibility."""
        if target_model not in self.model_requirements and target_model != "both":
            result["issues"].append(
                {
                    "type": "compatibility",
                    "severity": "warning",
                    "message": f"Unknown target model: {target_model}",
                },
            )
            return 0.5

        if target_model == "both":
            # Validate for both models
            internvl_score = self._validate_single_model_compatibility(
                prompt,
                "internvl3",
                result,
            )
            llama_score = self._validate_single_model_compatibility(
                prompt,
                "llama32_vision",
                result,
            )
            return (internvl_score + llama_score) / 2
        return self._validate_single_model_compatibility(
            prompt,
            target_model,
            result,
        )

    def _validate_single_model_compatibility(
        self,
        prompt: str,
        model: str,
        result: dict[str, Any],
    ) -> float:
        """Validate compatibility with a specific model."""
        if model not in self.model_requirements:
            return 0.5

        requirements = self.model_requirements[model]
        score = 1.0

        # Length validation for model
        min_length, max_length = requirements["optimal_length_range"]
        if not (min_length <= len(prompt) <= max_length):
            result["issues"].append(
                {
                    "type": "compatibility",
                    "severity": "minor",
                    "message": f"Prompt length ({len(prompt)}) outside optimal range for {model} ({min_length}-{max_length})",
                },
            )
            score -= 0.1

        # Required keywords for model
        required_keywords = requirements["required_keywords"]
        found_required = sum(
            1
            for keyword in required_keywords
            if re.search(rf"\b{keyword}\b", prompt, re.IGNORECASE)
        )

        if found_required < len(required_keywords):
            result["issues"].append(
                {
                    "type": "compatibility",
                    "severity": "warning",
                    "message": f"Missing {model} keywords: {found_required}/{len(required_keywords)} found",
                },
            )
            score -= 0.2

        # Highlight support validation
        if model == "llama32_vision" and "highlight" in prompt.lower():
            result["issues"].append(
                {
                    "type": "compatibility",
                    "severity": "warning",
                    "message": f"{model} does not support highlight detection features",
                },
            )
            score -= 0.2

        return max(score, 0.0)

    def _validate_australian_compliance(
        self,
        prompt: str,
        result: dict[str, Any],
    ) -> float:
        """Validate Australian tax compliance elements."""
        score = 1.0

        # Check for Australian context
        australian_terms_found = sum(
            1
            for term in self.required_australian_terms
            if re.search(rf"\b{term}\b", prompt, re.IGNORECASE)
        )

        if australian_terms_found < 2:
            result["issues"].append(
                {
                    "type": "compliance",
                    "severity": "warning",
                    "message": f"Limited Australian tax context ({australian_terms_found}/{len(self.required_australian_terms)} terms)",
                },
            )
            score -= 0.3

        # Check for GST rate
        if "gst" in prompt.lower() and "10%" not in prompt:
            result["issues"].append(
                {
                    "type": "compliance",
                    "severity": "minor",
                    "message": "Consider specifying Australian GST rate (10%)",
                },
            )
            score -= 0.1

        # Check for date format
        if "date" in prompt.lower() and "dd/mm/yyyy" not in prompt.lower():
            result["issues"].append(
                {
                    "type": "compliance",
                    "severity": "minor",
                    "message": "Consider specifying Australian date format (DD/MM/YYYY)",
                },
            )
            score -= 0.1

        # Check for ABN format
        if "abn" in prompt.lower() and "xx xxx xxx xxx" not in prompt.lower():
            result["issues"].append(
                {
                    "type": "compliance",
                    "severity": "minor",
                    "message": "Consider specifying ABN format (XX XXX XXX XXX)",
                },
            )
            score -= 0.1

        return max(score, 0.0)

    def validate_prompt_library(self, prompts: dict[str, str]) -> dict[str, Any]:
        """Validate an entire prompt library."""
        library_results = {
            "total_prompts": len(prompts),
            "passed_prompts": 0,
            "failed_prompts": 0,
            "average_score": 0.0,
            "validation_results": {},
            "summary_issues": {},
            "recommendations": [],
        }

        total_score = 0.0
        issue_counts = {}

        for prompt_id, prompt_text in prompts.items():
            # Try to determine document type from prompt_id
            document_type = self._infer_document_type(prompt_id)

            validation = self.validate_prompt(prompt_text, document_type)
            library_results["validation_results"][prompt_id] = validation

            total_score += validation["validation_score"]

            if validation["passed"]:
                library_results["passed_prompts"] += 1
            else:
                library_results["failed_prompts"] += 1

            # Count issue types
            for issue in validation["issues"]:
                issue_type = f"{issue['type']}_{issue['severity']}"
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        if library_results["total_prompts"] > 0:
            library_results["average_score"] = (
                total_score / library_results["total_prompts"]
            )

        library_results["summary_issues"] = issue_counts

        # Generate library-level recommendations
        if library_results["average_score"] < 0.7:
            library_results["recommendations"].append(
                "Overall prompt quality below threshold - review and improve prompts",
            )

        if library_results["failed_prompts"] > library_results["passed_prompts"]:
            library_results["recommendations"].append(
                "More prompts failed than passed - significant improvements needed",
            )

        # Top issue recommendations
        if issue_counts:
            top_issue = max(issue_counts.items(), key=lambda x: x[1])
            library_results["recommendations"].append(
                f"Most common issue: {top_issue[0]} ({top_issue[1]} occurrences)",
            )

        return library_results

    def _infer_document_type(self, prompt_id: str) -> DocumentType:
        """Infer document type from prompt identifier."""
        prompt_id_lower = prompt_id.lower()

        if "fuel" in prompt_id_lower:
            return DocumentType.FUEL_RECEIPT
        if "tax_invoice" in prompt_id_lower:
            return DocumentType.TAX_INVOICE
        if "business" in prompt_id_lower:
            return DocumentType.BUSINESS_RECEIPT
        if "bank" in prompt_id_lower:
            return DocumentType.BANK_STATEMENT
        if "meal" in prompt_id_lower:
            return DocumentType.MEAL_RECEIPT
        if "accommodation" in prompt_id_lower:
            return DocumentType.ACCOMMODATION
        if "travel" in prompt_id_lower:
            return DocumentType.TRAVEL_DOCUMENT
        if "parking" in prompt_id_lower or "toll" in prompt_id_lower:
            return DocumentType.PARKING_TOLL
        if "professional" in prompt_id_lower:
            return DocumentType.PROFESSIONAL_SERVICES
        if "equipment" in prompt_id_lower or "supplies" in prompt_id_lower:
            return DocumentType.EQUIPMENT_SUPPLIES
        return DocumentType.OTHER

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get validation system statistics."""
        return {
            "validation_criteria": {
                "min_prompt_length": self.min_prompt_length,
                "max_prompt_length": self.max_prompt_length,
                "required_australian_terms": len(self.required_australian_terms),
                "supported_models": list(self.model_requirements.keys()),
            },
            "compliance_checks": [
                "Australian tax context",
                "GST rate specification (10%)",
                "Date format (DD/MM/YYYY)",
                "ABN format (XX XXX XXX XXX)",
            ],
            "compatibility_features": [
                "Cross-model validation",
                "Highlight detection support",
                "Optimal length ranges",
                "Required keyword validation",
            ],
        }
