"""ATO Compliance Validator

This module provides comprehensive Australian Taxation Office compliance validation
combining features from both InternVL and Llama-3.2 systems.
"""

import logging
from typing import Any

from ..classification import DocumentType
from ..extraction.pipeline_components import ComplianceResult
from .australian_business_registry import AustralianBusinessRegistry
from .field_validators import (
    ABNValidator,
    AmountValidator,
    BSBValidator,
    DateValidator,
    GSTValidator,
)

logger = logging.getLogger(__name__)


class ATOComplianceValidator:
    """Comprehensive ATO compliance validator for Australian tax documents.

    Features:
    - ABN validation with checksum verification
    - GST calculation validation (10% rate)
    - Australian business recognition and validation
    - Date format validation (DD/MM/YYYY)
    - BSB validation for banking
    - Document-specific compliance requirements
    - Field completeness assessment
    """

    def __init__(self, config: Any):
        self.config = config
        self.initialized = False

        # Initialize validators
        self.abn_validator = ABNValidator()
        self.bsb_validator = BSBValidator()
        self.date_validator = DateValidator()
        self.gst_validator = GSTValidator()
        self.amount_validator = AmountValidator()
        self.business_registry = AustralianBusinessRegistry()

        # Compliance thresholds
        self.compliance_thresholds = {
            "minimum_fields_required": 3,
            "minimum_confidence_score": 0.60,
            "gst_calculation_tolerance": 0.02,
            "date_range_years": 10,
        }

        # Document-specific requirements
        self.document_requirements = {
            DocumentType.TAX_INVOICE: {
                "required_fields": ["date", "total_amount", "supplier_name", "abn"],
                "optional_fields": ["gst_amount", "invoice_number", "customer_name"],
                "must_have_gst": True,
                "must_have_abn": True,
            },
            DocumentType.BUSINESS_RECEIPT: {
                "required_fields": ["date", "total_amount", "supplier_name"],
                "optional_fields": ["gst_amount", "receipt_number"],
                "must_have_gst": False,
                "must_have_abn": False,
            },
            DocumentType.FUEL_RECEIPT: {
                "required_fields": [
                    "date",
                    "total_amount",
                    "supplier_name",
                    "fuel_type",
                ],
                "optional_fields": ["litres", "price_per_litre", "gst_amount"],
                "must_have_gst": False,
                "must_have_abn": False,
            },
            DocumentType.BANK_STATEMENT: {
                "required_fields": ["date", "account_number", "bsb"],
                "optional_fields": [
                    "statement_period",
                    "opening_balance",
                    "closing_balance",
                ],
                "must_have_gst": False,
                "must_have_abn": False,
            },
            DocumentType.MEAL_RECEIPT: {
                "required_fields": ["date", "total_amount", "supplier_name"],
                "optional_fields": ["gst_amount", "receipt_number"],
                "must_have_gst": False,
                "must_have_abn": False,
            },
            DocumentType.ACCOMMODATION: {
                "required_fields": ["date", "total_amount", "supplier_name"],
                "optional_fields": [
                    "check_in_date",
                    "check_out_date",
                    "nights",
                    "gst_amount",
                ],
                "must_have_gst": False,
                "must_have_abn": False,
            },
            DocumentType.TRAVEL_DOCUMENT: {
                "required_fields": ["date", "total_amount", "supplier_name"],
                "optional_fields": ["flight_number", "passenger_name", "gst_amount"],
                "must_have_gst": False,
                "must_have_abn": False,
            },
            DocumentType.PARKING_TOLL: {
                "required_fields": ["date", "total_amount"],
                "optional_fields": ["location", "duration", "vehicle_registration"],
                "must_have_gst": False,
                "must_have_abn": False,
            },
            DocumentType.PROFESSIONAL_SERVICES: {
                "required_fields": [
                    "date",
                    "total_amount",
                    "supplier_name",
                    "description",
                ],
                "optional_fields": ["hours", "rate", "gst_amount", "abn"],
                "must_have_gst": True,
                "must_have_abn": True,
            },
            DocumentType.EQUIPMENT_SUPPLIES: {
                "required_fields": [
                    "date",
                    "total_amount",
                    "supplier_name",
                    "description",
                ],
                "optional_fields": ["quantity", "unit_price", "gst_amount", "abn"],
                "must_have_gst": False,
                "must_have_abn": False,
            },
        }

    def initialize(self) -> None:
        """Initialize the ATO compliance validator."""
        if self.initialized:
            return

        # Load configuration overrides
        if hasattr(self.config, "ato_compliance_config"):
            self.compliance_thresholds.update(self.config.ato_compliance_config)

        # Initialize business registry
        self.business_registry.initialize()

        logger.info("ATO Compliance Validator initialized")

    def validate_abn(self, abn: str) -> Any:
        """Validate ABN format and checksum - returns validation result object."""
        if not abn:
            from dataclasses import dataclass

            @dataclass
            class ABNValidationResult:
                is_valid: bool
                errors: list[str]
                formatted_abn: str = ""
                normalized_abn: str = ""

            return ABNValidationResult(is_valid=False, errors=["ABN is required"])

        is_valid, formatted_abn, errors = self.abn_validator.validate(abn)
        from dataclasses import dataclass

        @dataclass
        class ABNValidationResult:
            is_valid: bool
            errors: list[str]
            formatted_abn: str = ""
            normalized_abn: str = ""

        return ABNValidationResult(
            is_valid=is_valid,
            errors=errors,
            formatted_abn=formatted_abn,
            normalized_abn=formatted_abn,
        )

    def validate_gst_calculation(
        self, fields_or_subtotal, gst_amount=None, total=None
    ) -> Any:
        """Validate GST calculation correctness - returns validation result object."""
        # Handle case where test passes fields dictionary
        if isinstance(fields_or_subtotal, dict):
            fields = fields_or_subtotal
            try:
                subtotal = float(
                    fields.get("subtotal", "0").replace("$", "").replace(",", "")
                )
                gst_amount = float(
                    fields.get("gst_amount", "0").replace("$", "").replace(",", "")
                )
                total = float(
                    fields.get("total_amount", "0").replace("$", "").replace(",", "")
                )
            except (ValueError, AttributeError):
                return self._create_gst_result(
                    False, {}, ["Invalid numeric values in fields"]
                )
        else:
            # Handle separate parameters
            subtotal = fields_or_subtotal
            if gst_amount is None and total is None:
                return self._create_gst_result(True, {}, [])

        is_valid, calculated_values, errors = (
            self.gst_validator.validate_gst_calculation(subtotal, gst_amount, total)
        )
        return self._create_gst_result(is_valid, calculated_values, errors)

    def _create_gst_result(
        self, is_valid: bool, calculated_values: dict, errors: list[str]
    ) -> Any:
        """Create GST validation result object."""
        from dataclasses import dataclass

        @dataclass
        class GSTValidationResult:
            is_valid: bool
            errors: list[str]
            calculated_values: dict = None

        return GSTValidationResult(
            is_valid=is_valid, errors=errors, calculated_values=calculated_values
        )

    def validate_date_format(self, date_str: str) -> Any:
        """Validate Australian date format - returns validation result object."""
        is_valid, parsed_date, formatted_date, errors = self.date_validator.validate(
            date_str
        )
        from dataclasses import dataclass

        @dataclass
        class DateValidationResult:
            is_valid: bool
            errors: list[str]
            parsed_date: Any = None
            formatted_date: str = ""
            normalized_date: str = ""

        return DateValidationResult(
            is_valid=is_valid,
            errors=errors,
            parsed_date=parsed_date,
            formatted_date=formatted_date,
            normalized_date=formatted_date,
        )

    def validate_amount_format(self, amount_str: str) -> Any:
        """Validate and parse Australian currency amount - returns validation result object."""
        is_valid, parsed_amount, formatted_amount, errors = (
            self.amount_validator.validate(amount_str)
        )
        from dataclasses import dataclass

        @dataclass
        class AmountValidationResult:
            is_valid: bool
            errors: list[str]
            parsed_amount: float | None = None
            formatted_amount: str = ""
            normalized_amount: str = ""

        # For normalized_amount, return the numeric value without $ symbol for comparison
        normalized_value = ""
        if parsed_amount is not None:
            normalized_value = str(parsed_amount)
        elif formatted_amount:
            # Strip $ symbol for normalized value
            normalized_value = formatted_amount.replace("$", "").replace(",", "")

        return AmountValidationResult(
            is_valid=is_valid,
            errors=errors,
            parsed_amount=parsed_amount,
            formatted_amount=formatted_amount,
            normalized_amount=normalized_value,
        )

    def assess_compliance(
        self,
        extracted_fields: dict[str, Any],
        document_type: DocumentType,
        raw_text: str = "",
        _classification_confidence: float = 0.7,
    ) -> ComplianceResult:
        """Assess ATO compliance for extracted fields.

        Args:
            extracted_fields: Fields extracted from document
            document_type: Type of document being processed
            raw_text: Original document text for business recognition
            classification_confidence: Confidence in document classification

        Returns:
            ComplianceResult with comprehensive compliance assessment

        """
        if not self.initialized:
            self.initialize()

        compliance_issues = []
        compliance_score = 0.0
        recommendations = []

        # Component 1: Field completeness validation (30%)
        field_score, field_issues = self._validate_field_completeness(
            extracted_fields,
            document_type,
        )
        compliance_issues.extend(field_issues)
        compliance_score += field_score * 0.30

        # Component 2: Field format validation (25%)
        format_score, format_issues = self._validate_field_formats(extracted_fields)
        compliance_issues.extend(format_issues)
        compliance_score += format_score * 0.25

        # Component 3: Business validation (20%)
        business_score, business_issues, business_recs = (
            self._validate_business_context(extracted_fields, document_type, raw_text)
        )
        compliance_issues.extend(business_issues)
        recommendations.extend(business_recs)
        compliance_score += business_score * 0.20

        # Component 4: Tax calculation validation (15%)
        tax_score, tax_issues = self._validate_tax_calculations(extracted_fields)
        compliance_issues.extend(tax_issues)
        compliance_score += tax_score * 0.15

        # Component 5: Document-specific validation (10%)
        doc_score, doc_issues = self._validate_document_specific_requirements(
            extracted_fields,
            document_type,
        )
        compliance_issues.extend(doc_issues)
        compliance_score += doc_score * 0.10

        # Determine compliance status
        compliance_passed = (
            compliance_score >= self.compliance_thresholds["minimum_confidence_score"]
            and len(compliance_issues) == 0
        )

        # Generate recommendations if compliance failed
        if not compliance_passed:
            recommendations.extend(
                self._generate_compliance_recommendations(
                    compliance_score,
                    compliance_issues,
                    document_type,
                ),
            )

        logger.info(
            f"ATO compliance assessed: {compliance_score:.2f} score, "
            f"{len(compliance_issues)} issues, passed: {compliance_passed}",
        )

        return ComplianceResult(
            compliance_score=compliance_score,
            passed=compliance_passed,
            violations=compliance_issues,
            warnings=[],
            recommendations=recommendations,
            field_results={},
        )

    def _validate_field_completeness(
        self,
        extracted_fields: dict[str, Any],
        document_type: DocumentType,
    ) -> tuple[float, list[str]]:
        """Validate field completeness for document type."""
        issues = []

        # Get requirements for document type
        requirements = self.document_requirements.get(document_type)
        if not requirements:
            return 0.5, ["Unknown document type compliance requirements"]

        required_fields = requirements["required_fields"]

        # Check required fields
        missing_required = []
        for field in required_fields:
            field_value = extracted_fields.get(field)
            if not field_value or (
                isinstance(field_value, str) and not field_value.strip()
            ):
                missing_required.append(field)

        if missing_required:
            issues.append(f"Missing required fields: {', '.join(missing_required)}")

        # Calculate completeness score
        total_required = len(required_fields)
        found_required = total_required - len(missing_required)

        if total_required == 0:
            completeness_score = 1.0
        else:
            completeness_score = found_required / total_required

        # Bonus for optional fields
        optional_fields = requirements.get("optional_fields", [])
        found_optional = sum(
            1
            for field in optional_fields
            if extracted_fields.get(field) and str(extracted_fields[field]).strip()
        )

        if optional_fields:
            optional_bonus = (found_optional / len(optional_fields)) * 0.2
            completeness_score = min(completeness_score + optional_bonus, 1.0)

        return completeness_score, issues

    def _validate_field_formats(
        self,
        extracted_fields: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """Validate field formats using specialized validators."""
        issues = []
        validation_scores = []

        # Validate ABN if present
        if extracted_fields.get("abn"):
            is_valid, formatted_abn, abn_issues = self.abn_validator.validate(
                extracted_fields["abn"],
            )
            if not is_valid:
                issues.extend([f"ABN: {issue}" for issue in abn_issues])
                validation_scores.append(0.0)
            else:
                validation_scores.append(1.0)
                # Update with formatted ABN
                extracted_fields["abn"] = formatted_abn

        # Validate BSB if present
        if extracted_fields.get("bsb"):
            is_valid, formatted_bsb, bsb_issues, bank_name = (
                self.bsb_validator.validate(extracted_fields["bsb"])
            )
            if not is_valid:
                issues.extend([f"BSB: {issue}" for issue in bsb_issues])
                validation_scores.append(0.0)
            else:
                validation_scores.append(1.0)
                # Update with formatted BSB and bank info
                extracted_fields["bsb"] = formatted_bsb
                if bank_name:
                    extracted_fields["bank_name"] = bank_name

        # Validate dates
        date_fields = ["date", "check_in_date", "check_out_date", "invoice_date"]
        for field in date_fields:
            if extracted_fields.get(field):
                is_valid, parsed_date, formatted_date, date_issues = (
                    self.date_validator.validate(extracted_fields[field])
                )
                if not is_valid:
                    issues.extend([f"{field}: {issue}" for issue in date_issues])
                    validation_scores.append(0.0)
                else:
                    validation_scores.append(1.0)
                    # Update with formatted date and add financial year
                    extracted_fields[field] = formatted_date
                    if parsed_date and field == "date":
                        extracted_fields["financial_year"] = (
                            self.date_validator.get_financial_year(parsed_date)
                        )

        # Validate amounts (basic format check)
        amount_fields = ["total_amount", "subtotal", "gst_amount", "unit_price"]
        for field in amount_fields:
            if extracted_fields.get(field):
                try:
                    amount = float(
                        str(extracted_fields[field]).replace("$", "").replace(",", ""),
                    )
                    if amount < 0:
                        issues.append(f"{field}: Amount cannot be negative")
                        validation_scores.append(0.0)
                    else:
                        validation_scores.append(1.0)
                        # Update with cleaned amount
                        extracted_fields[field] = f"{amount:.2f}"
                except ValueError:
                    issues.append(f"{field}: Invalid amount format")
                    validation_scores.append(0.0)

        # Calculate overall format validation score
        if validation_scores:
            format_score = sum(validation_scores) / len(validation_scores)
        else:
            format_score = 1.0  # No validations required

        return format_score, issues

    def _validate_business_context(
        self,
        extracted_fields: dict[str, Any],
        document_type: DocumentType,
        raw_text: str,
    ) -> tuple[float, list[str], list[str]]:
        """Validate Australian business context."""
        issues = []
        recommendations = []

        # Recognize businesses in text
        recognized_business = self.business_registry.recognize_business(raw_text)

        business_score = 0.0

        if recognized_business.is_recognized:
            # Use the recognized business
            business_name = extracted_fields.get("supplier_name", "")
            is_valid, context_issues, context_recs = (
                self.business_registry.validate_business_context(
                    business_name,
                    document_type.value,
                    extracted_fields,
                )
            )

            if is_valid:
                business_score = recognized_business.confidence_score
            else:
                business_score = recognized_business.confidence_score * 0.5
                issues.extend(context_issues)
                recommendations.extend(context_recs)

            # Add business information to extracted fields
            extracted_fields["recognized_business"] = (
                recognized_business.normalized_name
            )
            extracted_fields["business_industry"] = recognized_business.industry

        # No recognized business - check if we have supplier name
        elif extracted_fields.get("supplier_name"):
            business_score = 0.3  # Partial credit for having supplier name
            issues.append("Supplier not recognized as major Australian business")
            recommendations.append("Verify supplier name and business details")
        else:
            business_score = 0.0
            issues.append("No supplier information found")

        return business_score, issues, recommendations

    def _validate_tax_calculations(
        self,
        extracted_fields: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """Validate GST and tax calculations."""
        issues = []

        # Check if we have the necessary fields for GST validation
        subtotal = extracted_fields.get("subtotal")
        gst_amount = extracted_fields.get("gst_amount")
        total_amount = extracted_fields.get("total_amount")

        if not (subtotal and gst_amount and total_amount):
            # Try to extract from total if GST is missing
            if total_amount and not gst_amount and not subtotal:
                try:
                    total = float(str(total_amount).replace("$", "").replace(",", ""))
                    gst_calc = self.gst_validator.extract_gst_from_total(total)

                    extracted_fields["subtotal_calculated"] = str(gst_calc["subtotal"])
                    extracted_fields["gst_amount_calculated"] = str(
                        gst_calc["gst_amount"],
                    )

                    return 0.7, []  # Partial score for calculated GST

                except ValueError:
                    return 0.5, [
                        "Cannot validate GST - insufficient amount information",
                    ]

            return 0.8, []  # No GST validation required

        try:
            # Convert to float for validation
            subtotal_val = float(str(subtotal).replace("$", "").replace(",", ""))
            gst_val = float(str(gst_amount).replace("$", "").replace(",", ""))
            total_val = float(str(total_amount).replace("$", "").replace(",", ""))

            # Validate GST calculation
            is_valid, calc_values, gst_issues = (
                self.gst_validator.validate_gst_calculation(
                    subtotal_val,
                    gst_val,
                    total_val,
                )
            )

            if is_valid:
                return 1.0, []
            issues.extend(gst_issues)
            return 0.3, issues

        except ValueError:
            issues.append("Invalid amount format for GST validation")
            return 0.2, issues

    def _validate_document_specific_requirements(
        self,
        extracted_fields: dict[str, Any],
        document_type: DocumentType,
    ) -> tuple[float, list[str]]:
        """Validate document-specific ATO requirements."""
        issues = []

        requirements = self.document_requirements.get(document_type)
        if not requirements:
            return 1.0, []

        score = 1.0

        # Check GST requirement
        if requirements["must_have_gst"]:
            if not extracted_fields.get("gst_amount"):
                issues.append(f"{document_type.value} must include GST amount")
                score -= 0.4

        # Check ABN requirement
        if requirements["must_have_abn"]:
            if not extracted_fields.get("abn"):
                issues.append(f"{document_type.value} must include ABN")
                score -= 0.4

        # Document-specific validations
        if document_type == DocumentType.FUEL_RECEIPT:
            if not extracted_fields.get("fuel_type"):
                issues.append("Fuel receipt must specify fuel type")
                score -= 0.2

        elif document_type == DocumentType.BANK_STATEMENT:
            if not (
                extracted_fields.get("account_number") and extracted_fields.get("bsb")
            ):
                issues.append("Bank statement must include account number and BSB")
                score -= 0.3

        elif document_type == DocumentType.PROFESSIONAL_SERVICES:
            if not extracted_fields.get("description"):
                issues.append("Professional services must include service description")
                score -= 0.2

        return max(score, 0.0), issues

    def _generate_compliance_recommendations(
        self,
        compliance_score: float,
        issues: list[str],
        document_type: DocumentType,
    ) -> list[str]:
        """Generate specific recommendations for compliance improvement."""
        recommendations = []

        if compliance_score < 0.3:
            recommendations.append(
                "Document requires significant manual review for ATO compliance",
            )
        elif compliance_score < 0.6:
            recommendations.append(
                "Document needs additional validation before submission",
            )

        # Issue-specific recommendations
        if any("ABN" in issue for issue in issues):
            recommendations.append("Verify ABN format and validity with ABR lookup")

        if any("GST" in issue for issue in issues):
            recommendations.append(
                "Check GST calculations and ensure 10% rate is applied correctly",
            )

        if any("date" in issue.lower() for issue in issues):
            recommendations.append(
                "Verify date format follows DD/MM/YYYY Australian standard",
            )

        if any("supplier" in issue.lower() for issue in issues):
            recommendations.append(
                "Ensure supplier name matches Australian business registry",
            )

        # Document-specific recommendations
        if document_type == DocumentType.TAX_INVOICE:
            recommendations.append(
                "Tax invoices require ABN, GST amount, and complete supplier details",
            )
        elif document_type == DocumentType.FUEL_RECEIPT:
            recommendations.append(
                "Fuel receipts should include litres, fuel type, and price per litre",
            )

        return list(set(recommendations))  # Remove duplicates
