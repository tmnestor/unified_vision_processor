"""
Unit Tests for ATO Compliance System

Tests the unified ATO compliance validation system that combines
validation rules from both InternVL and Llama systems.
"""

import pytest

from vision_processor.classification import DocumentType
from vision_processor.compliance.ato_compliance_validator import ATOComplianceValidator
from vision_processor.compliance.australian_business_registry import (
    AustralianBusinessRegistry,
)
from vision_processor.compliance.field_validators import (
    ABNValidator,
    AmountValidator,
    DateValidator,
    GSTValidator,
)


class TestATOComplianceValidator:
    """Test suite for ATO Compliance Validator."""

    @pytest.fixture
    def ato_validator(self, test_config):
        """Create an ATO compliance validator for testing."""
        return ATOComplianceValidator(test_config)

    def test_abn_validation_valid_formats(self, ato_validator):
        """Test ABN validation with valid formats."""
        valid_abns = [
            "88 000 014 675",  # Woolworths (with spaces) - verified valid
            "88000014675",  # Woolworths (no spaces) - verified valid
            "64 067 190 363",  # Test ABN 1 (with spaces)
            "64067190363",  # Test ABN 1 (no spaces)
            "61 291 347 325",  # Test ABN 2 (with spaces)
            "61291347325",  # Test ABN 2 (no spaces)
        ]

        for abn in valid_abns:
            result = ato_validator.validate_abn(abn)
            assert result.is_valid is True
            assert result.normalized_abn is not None
            assert len(result.normalized_abn.replace(" ", "")) == 11

    def test_abn_validation_invalid_formats(self, ato_validator):
        """Test ABN validation with invalid formats."""
        invalid_abns = [
            "12345",  # Too short
            "123456789012",  # Too long
            "abc123def456",  # Contains letters
            "88 000 014 999",  # Invalid check digit (last digit changed from 5 to 9)
            "",  # Empty
            None,  # None
            "51 004 085 616",  # Invalid checksum (verified invalid)
        ]

        for abn in invalid_abns:
            result = ato_validator.validate_abn(abn)
            assert result.is_valid is False
            if abn:  # Non-empty ABNs should have error messages
                assert len(result.errors) > 0

    def test_gst_calculation_validation(self, ato_validator):
        """Test GST calculation validation (10% in Australia)."""
        test_cases = [
            # (subtotal, gst, total, expected_valid)
            (100.00, 10.00, 110.00, True),  # Perfect 10%
            (45.45, 4.55, 50.00, True),  # Rounded correctly
            (23.63, 2.36, 25.99, True),  # Within tolerance
            (100.00, 5.00, 105.00, False),  # Wrong GST rate
            (100.00, 10.00, 115.00, False),  # Wrong total
            (100.00, 0.00, 100.00, True),  # No GST (valid)
        ]

        for subtotal, gst, total, expected in test_cases:
            fields = {
                "subtotal": str(subtotal),
                "gst_amount": str(gst),
                "total_amount": str(total),
            }

            result = ato_validator.validate_gst_calculation(fields)
            assert result.is_valid == expected

    def test_date_format_validation_australian(self, ato_validator):
        """Test Australian date format validation (DD/MM/YYYY)."""
        valid_dates = [
            "25/03/2024",  # Standard format
            "1/1/2024",  # Single digits
            "31/12/2023",  # End of year
            "29/02/2024",  # Leap year
            "25-03-2024",  # Dash separator
            "25 Mar 2024",  # Month name
            "25 March 2024",  # Full month name
        ]

        for date_str in valid_dates:
            result = ato_validator.validate_date_format(date_str)
            assert result.is_valid is True
            assert result.normalized_date is not None

    def test_date_format_validation_invalid(self, ato_validator):
        """Test invalid date format validation."""
        invalid_dates = [
            "2024/03/25",  # US format
            "2024-03-25",  # ISO format
            "25/13/2024",  # Invalid month
            "32/03/2024",  # Invalid day
            "29/02/2023",  # Non-leap year
            "invalid",  # Not a date
            "",  # Empty
            None,  # None
        ]

        for date_str in invalid_dates:
            result = ato_validator.validate_date_format(date_str)
            assert result.is_valid is False

    def test_amount_format_validation(self, ato_validator):
        """Test amount format validation."""
        valid_amounts = [
            "$45.67",  # With currency symbol
            "45.67",  # Without currency symbol
            "$1,234.56",  # With thousands separator
            "1234.56",  # Without separator
            "$0.50",  # Small amount
            "0.50",  # Small amount no symbol
            "$10",  # Whole dollars
            "10",  # Whole dollars no symbol
        ]

        for amount in valid_amounts:
            result = ato_validator.validate_amount_format(amount)
            assert result.is_valid is True
            assert result.normalized_amount is not None
            assert float(result.normalized_amount) >= 0

    def test_amount_format_validation_invalid(self, ato_validator):
        """Test invalid amount format validation."""
        invalid_amounts = [
            "$45.6789",  # Too many decimal places
            "45..67",  # Double decimal point
            "$-45.67",  # Negative amount
            "abc",  # Not a number
            "",  # Empty
            None,  # None
            "$1,23.45",  # Wrong thousands separator position
        ]

        for amount in invalid_amounts:
            result = ato_validator.validate_amount_format(amount)
            assert result.is_valid is False

    def test_comprehensive_document_compliance(
        self, ato_validator, sample_extracted_fields
    ):
        """Test comprehensive document compliance assessment."""
        result = ato_validator.assess_compliance(
            sample_extracted_fields, DocumentType.TAX_INVOICE
        )

        assert hasattr(result, "compliance_score")
        assert hasattr(result, "passed")
        assert hasattr(result, "violations")
        assert hasattr(result, "warnings")
        assert hasattr(result, "field_results")

        assert 0.0 <= result.compliance_score <= 1.0
        assert isinstance(result.passed, bool)
        assert isinstance(result.violations, list)
        assert isinstance(result.warnings, list)

    def test_business_receipt_specific_compliance(self, ato_validator):
        """Test business receipt specific compliance rules."""
        business_receipt_fields = {
            "supplier_name": "Woolworths",
            "date": "25/03/2024",
            "total_amount": "$45.67",
            "gst_amount": "$4.15",
            "items": ["Groceries", "Household items"],
        }

        result = ato_validator.assess_compliance(
            business_receipt_fields, DocumentType.BUSINESS_RECEIPT
        )

        assert result.compliance_score > 0.7  # Should be reasonably compliant
        assert result.passed is True

    def test_fuel_receipt_specific_compliance(self, ato_validator):
        """Test fuel receipt specific compliance rules."""
        fuel_receipt_fields = {
            "supplier_name": "Shell",
            "date": "25/03/2024",
            "total_amount": "$65.00",
            "fuel_type": "Unleaded",
            "litres": "45.5",
            "vehicle_registration": "ABC123",
        }

        result = ato_validator.assess_compliance(
            fuel_receipt_fields, DocumentType.FUEL_RECEIPT
        )

        # Fuel receipts should pass basic compliance
        assert result.compliance_score > 0.6

    def test_bank_statement_specific_compliance(self, ato_validator):
        """Test bank statement specific compliance rules."""
        bank_statement_fields = {
            "bank_name": "Commonwealth Bank",
            "account_number": "123456789",
            "bsb": "062-001",
            "statement_period": "March 2024",
            "transactions": [
                {
                    "date": "25/03/2024",
                    "description": "Office supplies",
                    "amount": "-45.67",
                }
            ],
        }

        result = ato_validator.assess_compliance(
            bank_statement_fields, DocumentType.BANK_STATEMENT
        )

        # Bank statements have different compliance requirements
        assert result.compliance_score >= 0.0


class TestAustralianBusinessRegistry:
    """Test suite for Australian Business Registry."""

    @pytest.fixture
    def business_registry(self, test_config):
        """Create an Australian business registry for testing."""
        return AustralianBusinessRegistry(test_config)

    def test_known_business_recognition(self, business_registry):
        """Test recognition of known Australian businesses."""
        known_businesses = [
            "Woolworths",
            "Coles",
            "JB Hi-Fi",
            "Harvey Norman",
            "Big W",
            "Bunnings Warehouse",
            "Kmart",
            "Target",
            "Officeworks",
            "Chemist Warehouse",
        ]

        for business in known_businesses:
            result = business_registry.recognize_business(business)
            assert result.is_recognized is True
            assert result.business_name == business
            assert result.confidence_score > 0.8

    def test_business_name_variations(self, business_registry):
        """Test recognition of business name variations."""
        variations = [
            ("woolworths", "Woolworths"),
            ("WOOLWORTHS", "Woolworths"),
            ("Woolworth", "Woolworths"),  # Close match
            ("JB HIFI", "JB Hi-Fi"),
            ("jb hi-fi", "JB Hi-Fi"),
            ("Bunnings", "Bunnings Warehouse"),
        ]

        for input_name, expected_name in variations:
            result = business_registry.recognize_business(input_name)
            assert result.is_recognized is True
            assert result.normalized_name == expected_name

    def test_unknown_business_handling(self, business_registry):
        """Test handling of unknown businesses."""
        unknown_businesses = [
            "Unknown Store XYZ",
            "Local Cafe",
            "Small Business Name",
            "Random Text",
        ]

        for business in unknown_businesses:
            result = business_registry.recognize_business(business)
            assert result.is_recognized is False
            assert result.confidence_score < 0.5

    def test_abn_lookup_integration(self, business_registry):
        """Test ABN lookup integration with business recognition."""
        # Test with known ABN
        woolworths_abn = "88 000 014 675"
        result = business_registry.lookup_business_by_abn(woolworths_abn)

        assert result.is_found is True
        assert "woolworths" in result.business_name.lower()
        assert result.abn == woolworths_abn

    def test_business_category_classification(self, business_registry):
        """Test business category classification."""
        category_tests = [
            ("Woolworths", "supermarket"),
            ("JB Hi-Fi", "electronics"),
            ("Harvey Norman", "furniture_electronics"),
            ("Shell", "fuel_station"),
            ("McDonald's", "restaurant"),
        ]

        for business, expected_category in category_tests:
            result = business_registry.get_business_category(business)
            assert result.category == expected_category
            assert result.confidence > 0.7


class TestFieldValidators:
    """Test suite for individual field validators."""

    def test_abn_validator_checksum(self):
        """Test ABN validator checksum calculation."""
        validator = ABNValidator()

        # Known valid ABNs with correct checksums
        valid_abns = [
            "88000014675",  # Woolworths (verified valid)
            "64067190363",  # Test ABN 1 (generated valid)
            "61291347325",  # Test ABN 2 (generated valid)
        ]

        for abn in valid_abns:
            assert validator._validate_checksum(abn) is True

    def test_abn_validator_format_normalization(self):
        """Test ABN format normalization."""
        validator = ABNValidator()

        test_cases = [
            ("88 000 014 675", "88000014675"),
            ("88-000-014-675", "88000014675"),
            ("88  000  014  675", "88000014675"),
        ]

        for input_abn, expected in test_cases:
            normalized = validator._normalize_abn(input_abn)
            assert normalized == expected

    def test_gst_validator_tolerance(self):
        """Test GST validator tolerance for rounding."""
        validator = GSTValidator()

        # Test cases with rounding tolerance
        test_cases = [
            (23.63, 2.36, 25.99, True),  # 23.63 * 0.1 = 2.363, rounded to 2.36
            (
                23.64,
                2.36,
                26.00,
                True,
            ),  # 23.64 * 0.1 = 2.364, rounded to 2.36 (correct)
            (45.45, 4.55, 50.00, True),  # 45.45 * 0.1 = 4.545, rounded to 4.55
            (100.00, 10.01, 110.01, False),  # Outside tolerance
        ]

        for subtotal, gst, total, expected in test_cases:
            result = validator.validate_calculation(subtotal, gst, total)
            assert result == expected

    def test_date_validator_australian_formats(self):
        """Test date validator for Australian date formats."""
        validator = DateValidator()

        valid_formats = [
            "25/03/2024",
            "1/1/2024",
            "31/12/2023",
            "25-03-2024",
            "25.03.2024",
        ]

        for date_str in valid_formats:
            result = validator.validate_format(date_str)
            assert result.is_valid is True
            assert result.parsed_date is not None

    def test_amount_validator_currency_handling(self):
        """Test amount validator currency symbol handling."""
        validator = AmountValidator()

        test_cases = [
            ("$45.67", 45.67),
            ("45.67", 45.67),
            ("$1,234.56", 1234.56),
            ("1,234.56", 1234.56),
            ("$0.50", 0.50),
            ("$10", 10.00),
        ]

        for input_amount, expected_value in test_cases:
            result = validator.validate_and_parse(input_amount)
            assert result.is_valid is True
            assert abs(result.parsed_amount - expected_value) < 0.01
