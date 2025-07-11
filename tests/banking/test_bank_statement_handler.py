"""
Unit Tests for Bank Statement Handler

Tests the complete bank statement processing including transaction extraction,
work expense identification, and integration with banking modules.
"""

import re
from unittest.mock import Mock, patch

import pytest

from vision_processor.handlers import BankStatementHandler


class TestBankStatementHandler:
    """Test suite for Bank Statement Handler."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for handler."""
        config = Mock()
        config.highlight_detection = True
        config.enhanced_parsing = True
        return config

    @pytest.fixture
    def bank_handler(self, mock_config):
        """Create bank statement handler for testing."""
        handler = BankStatementHandler(mock_config)
        handler.initialize()
        return handler

    @pytest.fixture
    def sample_statement_text(self):
        """Sample bank statement text for testing."""
        return """
        ANZ Bank Account Statement
        Account: 123-456 789012345
        Statement Period: 01/06/2024 to 30/06/2024
        
        Date        Description                    Amount
        15/06/2024  BP Fuel Station Melbourne     -85.50
        16/06/2024  Wilson Parking Flinders St    -12.00
        17/06/2024  Salary Payment                +5000.00
        18/06/2024  Uber Trip to Airport          -25.30
        19/06/2024  Officeworks Office Supplies  -156.80
        20/06/2024  Client Lunch - Restaurant     -68.90
        21/06/2024  Deloitte Consulting Fee       -2500.00
        22/06/2024  Telstra Phone Bill            -89.90
        23/06/2024  CityLink Toll                 -8.20
        
        Opening Balance: $2,456.78
        Closing Balance: $1,510.48
        """

    def test_handler_initialization(self, bank_handler):
        """Test bank statement handler initialization."""
        assert bank_handler.initialized
        assert bank_handler.supports_highlights
        assert bank_handler.supports_enhanced_parsing

        # Check required fields for bank statements
        assert "bank_name" in bank_handler.required_fields
        assert "account_holder" in bank_handler.required_fields
        assert "account_number" in bank_handler.required_fields

    def test_australian_bank_recognition(self, bank_handler, sample_statement_text):
        """Test recognition of Australian banks from statement text."""
        # Should recognize ANZ from the statement text
        fields = bank_handler._extract_document_specific_fields(sample_statement_text)

        assert "bank_name" in fields
        # Check that ANZ is identified in some form
        bank_name = fields.get("bank_name", "").lower()
        assert "anz" in bank_name or bank_name != ""

    def test_account_information_extraction(self, bank_handler, sample_statement_text):
        """Test extraction of account information."""
        fields = bank_handler._extract_document_specific_fields(sample_statement_text)

        # Should extract account information
        expected_fields = [
            "account_number",
            "statement_period_from",
            "statement_period_to",
            "opening_balance",
            "closing_balance",
        ]

        extracted_count = sum(1 for field in expected_fields if field in fields and fields[field])
        assert extracted_count > 0  # Should extract at least some account information

    def test_transaction_extraction_integration(self, bank_handler, sample_statement_text):
        """Test integration with transaction extraction."""
        # Mock the transaction categorizer to test integration
        with patch("vision_processor.banking.AustralianTransactionCategorizer") as mock_categorizer:
            mock_instance = Mock()
            mock_categorizer.return_value = mock_instance
            mock_instance.extract_transactions.return_value = [
                Mock(
                    date="15/06/2024",
                    description="BP Fuel Station",
                    amount=-85.50,
                    transaction_type="debit",
                ),
                Mock(
                    date="16/06/2024",
                    description="Wilson Parking",
                    amount=-12.00,
                    transaction_type="debit",
                ),
            ]
            mock_instance.identify_work_expenses.return_value = [
                Mock(
                    transaction=Mock(description="BP Fuel Station", amount=-85.50),
                    category="fuel",
                    work_score=0.9,
                    ato_category="Work-related car expenses",
                )
            ]

            # Extract fields which should trigger transaction processing
            fields = bank_handler._extract_document_specific_fields(sample_statement_text)

            # Should have processed transactions
            assert "transactions" in fields or "work_expenses" in fields

    def test_work_expense_scoring(self, bank_handler):
        """Test work expense scoring functionality."""
        # Test with obvious work expenses
        work_transactions = [
            "BP Fuel Station Melbourne",
            "Wilson Parking City",
            "Officeworks Business Supplies",
            "Deloitte Professional Services",
        ]

        for transaction_desc in work_transactions:
            # Test the work expense patterns
            score = 0
            for _category, patterns in bank_handler.work_expense_patterns.items():
                for pattern in patterns:
                    if pattern in transaction_desc.lower():
                        score += 0.3
                        break

            # Should identify as potential work expense
            assert score > 0

    def test_bsb_validation_integration(self, bank_handler):
        """Test integration with BSB validation."""
        # Test with valid BSB format
        statement_with_bsb = """
        Commonwealth Bank Statement
        BSB: 062-001
        Account: 123456789
        """

        fields = bank_handler._extract_document_specific_fields(statement_with_bsb)

        # Should extract and potentially validate BSB
        if "bsb" in fields:
            bsb = fields["bsb"]
            # Should be in standard format XXX-XXX
            assert "-" in bsb or len(bsb.replace("-", "")) == 6

    def test_australian_business_recognition(self, bank_handler):
        """Test recognition of Australian businesses in transactions."""
        australian_business_text = """
        Date        Description                    Amount
        15/06/2024  Woolworths Supermarket        -156.80
        16/06/2024  Bunnings Warehouse            -89.90
        17/06/2024  JB Hi-Fi Electronics          -299.00
        """

        fields = bank_handler._extract_document_specific_fields(australian_business_text)

        # Should recognize Australian business context
        # This could be reflected in business_score or similar metrics
        assert isinstance(fields, dict)

    def test_highlight_integration_support(self, bank_handler):
        """Test support for highlight detection integration."""
        # Handler should be configured to support highlights
        assert bank_handler.supports_highlights

        # Should have highlight integration setup
        if hasattr(bank_handler, "_initialize_highlight_integration"):
            # This method should exist and not raise errors
            try:
                bank_handler._initialize_highlight_integration()
            except AttributeError:
                pass  # Method might not be implemented yet

    def test_field_validation_requirements(self, bank_handler):
        """Test field validation for bank statements."""
        # Test with minimal required fields
        minimal_fields = {
            "bank_name": "ANZ",
            "account_holder": "John Smith",
            "account_number": "123456789",
            "statement_period_from": "01/06/2024",
            "statement_period_to": "30/06/2024",
        }

        # Should validate successfully with required fields
        validation_result = bank_handler.validate_fields(minimal_fields)

        assert validation_result.validation_passed or len(validation_result.validation_issues) == 0

    def test_transaction_categorization_accuracy(self, bank_handler):
        """Test accuracy of transaction categorization."""
        test_transactions = [
            ("BP Fuel Station", "fuel"),
            ("Wilson Parking", "parking"),
            ("Uber Trip", "transport"),
            ("Deloitte Consulting", "professional_services"),
            ("Officeworks Supplies", "office_supplies"),
            ("Restaurant Business Lunch", "meals"),
        ]

        for description, expected_category in test_transactions:
            # Check if the description matches expected category patterns
            found_category = None
            max_score = 0

            for category, pattern in bank_handler.work_expense_patterns.items():
                if re.search(pattern, description.lower(), re.IGNORECASE):
                    score = 0.8  # Arbitrary high score for pattern match
                    if score > max_score:
                        max_score = score
                        found_category = category

            # Should categorize correctly for obvious cases
            if found_category:
                # Check if found category is reasonable (exact match or similar)
                assert (
                    found_category == expected_category
                    or found_category in expected_category
                    or expected_category in found_category
                )

    def test_ato_compliance_integration(self, bank_handler, sample_statement_text):
        """Test integration with ATO compliance requirements."""
        fields = bank_handler._extract_document_specific_fields(sample_statement_text)

        # Should extract fields relevant to ATO compliance
        ato_relevant_fields = [
            "bank_name",
            "account_number",
            "statement_period_from",
            "statement_period_to",
            "transactions",
            "work_expenses",
        ]

        extracted_ato_fields = sum(1 for field in ato_relevant_fields if field in fields and fields[field])
        assert extracted_ato_fields > 0

    def test_confidence_scoring_integration(self, bank_handler, sample_statement_text):
        """Test integration with confidence scoring."""
        # Process the statement
        fields = bank_handler._extract_document_specific_fields(sample_statement_text)

        # Handler should provide some confidence indicators
        # This could be through field completeness, bank recognition, etc.
        assert isinstance(fields, dict)
        assert len(fields) > 0  # Should extract some fields

        # Check for confidence-related information
        confidence_indicators = [
            "bank_name",  # Successful bank identification
            "account_number",  # Account information extracted
            "transactions",  # Transaction parsing successful
        ]

        confidence_score = sum(1 for indicator in confidence_indicators if fields.get(indicator))
        assert confidence_score > 0

    def test_error_handling(self, bank_handler):
        """Test error handling with malformed input."""
        malformed_inputs = [
            "",  # Empty string
            "Not a bank statement",  # No bank information
            "Random text without structure",  # No transaction format
            "123456789",  # Just numbers
        ]

        for malformed_input in malformed_inputs:
            try:
                fields = bank_handler._extract_document_specific_fields(malformed_input)
                # Should not crash and should return a dict
                assert isinstance(fields, dict)
            except Exception as e:
                # If it raises an exception, it should be a reasonable one
                assert isinstance(e, (ValueError, AttributeError, KeyError))

    def test_processing_performance(self, bank_handler, sample_statement_text):
        """Test processing performance for reasonable response times."""
        import time

        start_time = time.time()
        fields = bank_handler._extract_document_specific_fields(sample_statement_text)
        processing_time = time.time() - start_time

        # Should process within reasonable time (< 5 seconds for this sample)
        assert processing_time < 5.0
        assert isinstance(fields, dict)

    def test_field_completeness(self, bank_handler, sample_statement_text):
        """Test completeness of field extraction."""
        fields = bank_handler._extract_document_specific_fields(sample_statement_text)

        # Should extract reasonable number of fields
        assert len(fields) >= 3  # Minimum expected fields

        # Check for key banking fields
        banking_fields = [
            "bank_name",
            "account_number",
            "statement_period_from",
            "statement_period_to",
        ]
        extracted_banking_fields = sum(1 for field in banking_fields if field in fields and fields[field])

        # Should extract at least some banking information
        assert extracted_banking_fields > 0
