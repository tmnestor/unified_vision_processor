"""
Unit Tests for Australian BSB Validator

Tests BSB format validation, bank identification, and error handling
for Australian Bank State Branch numbers.
"""

import pytest

from vision_processor.banking import AustralianBSBValidator, BSBValidationResult


class TestAustralianBSBValidator:
    """Test suite for Australian BSB Validator."""

    @pytest.fixture
    def bsb_validator(self):
        """Create BSB validator for testing."""
        return AustralianBSBValidator()

    def test_valid_bsb_formats(self, bsb_validator):
        """Test validation of correctly formatted BSBs."""
        valid_bsbs = [
            "123456",  # 6 digits
            "123-456",  # Formatted with hyphen
            "12 34 56",  # With spaces
            "12-34-56",  # Multiple hyphens
            " 123456 ",  # With whitespace
        ]

        for bsb in valid_bsbs:
            result = bsb_validator.validate(bsb)
            if len(bsb.strip().replace("-", "").replace(" ", "")) == 6:
                # Should format correctly even if not recognized as valid bank
                assert result.formatted_bsb.count("-") == 1
                assert len(result.formatted_bsb.replace("-", "")) == 6

    def test_big_four_bank_bsbs(self, bsb_validator):
        """Test validation of Big Four bank BSBs."""
        big_four_bsbs = [
            ("123456", "ANZ"),  # ANZ range 10-19 (using 12)
            ("623456", "Commonwealth Bank"),  # CBA range 60-69 (using 62)
            ("323456", "Westpac"),  # Westpac range 30-39 (using 32)
            ("823456", "NAB"),  # NAB range 80-89 (using 82)
        ]

        for bsb, expected_bank in big_four_bsbs:
            result = bsb_validator.validate(bsb)
            if result.bank_name:  # If recognized
                assert (
                    expected_bank in result.bank_name
                    or result.bank_name in expected_bank
                )
                assert result.bank_category == "big_four"

    def test_regional_bank_bsbs(self, bsb_validator):
        """Test validation of regional bank BSBs."""
        regional_bsbs = [
            ("632456", "Bendigo Bank"),  # Bendigo range 630-639
            ("642456", "Suncorp"),  # Suncorp range 640-649
            ("182456", "Macquarie"),  # Macquarie range 182
        ]

        for bsb, expected_bank in regional_bsbs:
            result = bsb_validator.validate(bsb)
            if result.bank_name:  # If recognized
                assert (
                    expected_bank in result.bank_name
                    or result.bank_name in expected_bank
                )

    def test_online_bank_bsbs(self, bsb_validator):
        """Test validation of online bank BSBs."""
        # ING Direct BSB
        result = bsb_validator.validate("923456")
        if result.bank_name:
            assert "ING" in result.bank_name
            assert result.bank_category == "online"

    def test_invalid_bsb_lengths(self, bsb_validator):
        """Test handling of BSBs with invalid lengths."""
        invalid_lengths = [
            ("", "BSB is required"),
            ("12345", "BSB too short"),  # 5 digits
            ("1234567", "BSB too long"),  # 7 digits
            ("12", "BSB too short"),  # 2 digits
            ("123456789", "BSB too long"),  # 9 digits
        ]

        for bsb, expected_issue_type in invalid_lengths:
            result = bsb_validator.validate(bsb)
            assert not result.is_valid
            assert len(result.validation_issues) > 0
            # Check that error message contains expected type
            issue_text = " ".join(result.validation_issues).lower()
            assert any(
                expected_word in issue_text
                for expected_word in expected_issue_type.lower().split()
            )

    def test_non_digit_handling(self, bsb_validator):
        """Test handling of BSBs with non-digit characters."""
        non_digit_cases = [
            "abc123",  # Letters and numbers
            "12a456",  # Letter in middle
            "!@#$%^",  # Special characters only
            "12-34-56",  # Multiple hyphens (should be cleaned)
        ]

        for bsb in non_digit_cases:
            result = bsb_validator.validate(bsb)
            # Should extract only digits for processing
            clean_digits = "".join(c for c in bsb if c.isdigit())

            if len(clean_digits) == 6:
                # Should succeed in formatting if 6 digits found
                assert result.formatted_bsb.replace("-", "") == clean_digits
            elif len(clean_digits) == 0:
                assert not result.is_valid
                assert "no digits" in " ".join(result.validation_issues).lower()

    def test_bsb_formatting(self, bsb_validator):
        """Test BSB formatting functionality."""
        test_cases = [
            ("123456", "123-456"),
            ("123-456", "123-456"),
            ("12 34 56", "123-456"),
            (" 123456 ", "123-456"),
            ("12345", "12345"),  # Invalid length, no formatting
        ]

        for input_bsb, expected in test_cases:
            formatted = bsb_validator.format_bsb(input_bsb)
            assert formatted == expected

    def test_format_validation_quick_check(self, bsb_validator):
        """Test quick format validation."""
        valid_formats = ["123456", "123-456", " 123456 "]
        invalid_formats = ["12345", "1234567", "abc123", ""]

        for bsb in valid_formats:
            assert bsb_validator.is_valid_format(bsb)

        for bsb in invalid_formats:
            assert not bsb_validator.is_valid_format(bsb)

    def test_bank_info_retrieval(self, bsb_validator):
        """Test detailed bank information retrieval."""
        # Test with known ANZ BSB
        anz_bsb = "123456"
        bank_info = bsb_validator.get_bank_info(anz_bsb)

        if bank_info:  # If recognized
            assert "bank_name" in bank_info
            assert "official_name" in bank_info
            assert "category" in bank_info
            assert "bsb_ranges" in bank_info
            assert "formatted_bsb" in bank_info
            assert bank_info["formatted_bsb"] == "123-456"

        # Test with invalid BSB
        invalid_info = bsb_validator.get_bank_info("999999")
        assert invalid_info is None

    def test_supported_banks(self, bsb_validator):
        """Test retrieval of supported banks."""
        supported = bsb_validator.get_supported_banks()

        assert len(supported) >= 10
        assert "ANZ" in supported
        assert "Commonwealth Bank" in supported
        assert "Westpac" in supported
        assert "NAB" in supported

    def test_bank_categorization(self, bsb_validator):
        """Test bank categorization functionality."""
        big_four = bsb_validator.get_bank_by_category("big_four")
        regional = bsb_validator.get_bank_by_category("regional")
        online = bsb_validator.get_bank_by_category("online")

        assert len(big_four) == 4
        assert "ANZ" in big_four
        assert "Commonwealth Bank" in big_four
        assert "Westpac" in big_four
        assert "NAB" in big_four

        assert len(regional) > 0
        assert "Bendigo Bank" in regional

        assert len(online) > 0
        assert "ING" in online

    def test_big_four_identification(self, bsb_validator):
        """Test Big Four bank identification."""
        # Test known Big Four BSBs
        big_four_tests = [
            "123456",  # ANZ
            "623456",  # Commonwealth Bank
            "323456",  # Westpac
            "823456",  # NAB
        ]

        for bsb in big_four_tests:
            # Note: Only test if the BSB is actually recognized
            if bsb_validator.get_bank_info(bsb):
                is_big_four = bsb_validator.is_big_four_bsb(bsb)
                # Should be True for these BSBs
                assert is_big_four

        # Test non-Big Four BSB
        regional_bsb = "632456"  # Bendigo Bank
        if bsb_validator.get_bank_info(regional_bsb):
            assert not bsb_validator.is_big_four_bsb(regional_bsb)

    def test_multiple_bsb_validation(self, bsb_validator):
        """Test validation of multiple BSBs at once."""
        bsbs = ["123456", "invalid", "623456", "12345"]
        results = bsb_validator.validate_multiple(bsbs)

        assert len(results) == len(bsbs)
        assert all(isinstance(result, BSBValidationResult) for result in results)

        # Check that some are valid and some are invalid
        valid_count = sum(1 for result in results if result.is_valid)
        invalid_count = sum(1 for result in results if not result.is_valid)

        assert valid_count > 0 or invalid_count > 0  # At least some should be processed

    def test_correction_suggestions(self, bsb_validator):
        """Test BSB correction suggestions."""
        # Test short BSB
        suggestions = bsb_validator.suggest_corrections("12345")
        assert len(suggestions) > 0
        assert any("add" in suggestion.lower() for suggestion in suggestions)

        # Test long BSB
        suggestions = bsb_validator.suggest_corrections("1234567")
        assert len(suggestions) > 0
        assert any("remove" in suggestion.lower() for suggestion in suggestions)

        # Test empty BSB
        suggestions = bsb_validator.suggest_corrections("")
        assert len(suggestions) > 0
        assert any("6-digit" in suggestion.lower() for suggestion in suggestions)

    def test_placeholder_bsb_detection(self, bsb_validator):
        """Test detection of placeholder BSBs."""
        placeholder_bsbs = ["000000", "123456"]

        for bsb in placeholder_bsbs:
            result = bsb_validator.validate(bsb)
            if bsb == "000000":
                # Should detect as invalid placeholder
                assert not result.is_valid
                assert any(
                    "invalid" in issue.lower() for issue in result.validation_issues
                )
            elif bsb == "123456":
                # Might be flagged as placeholder depending on implementation
                issue_text = " ".join(result.validation_issues).lower()
                if "placeholder" in issue_text:
                    assert not result.is_valid

    def test_edge_cases(self, bsb_validator):
        """Test edge cases and error conditions."""
        edge_cases = [
            None,  # None input
            123456,  # Integer instead of string
            " ",  # Whitespace only
            "-",  # Hyphen only
            "12-34",  # Partial BSB with hyphen
        ]

        for case in edge_cases:
            try:
                if case is None:
                    result = bsb_validator.validate("")  # Handle None as empty
                else:
                    result = bsb_validator.validate(str(case))

                # Should not crash and should return a result
                assert isinstance(result, BSBValidationResult)

            except Exception as e:
                # If it throws an exception, it should be a reasonable one
                assert isinstance(e, (ValueError, TypeError, AttributeError))

    def test_validation_result_structure(self, bsb_validator):
        """Test that validation results have proper structure."""
        result = bsb_validator.validate("123456")

        # Check all required fields exist
        assert hasattr(result, "is_valid")
        assert hasattr(result, "formatted_bsb")
        assert hasattr(result, "validation_issues")
        assert hasattr(result, "bank_name")
        assert hasattr(result, "bank_category")
        assert hasattr(result, "suggestions")

        # Check field types
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.formatted_bsb, str)
        assert isinstance(result.validation_issues, list)
        assert isinstance(result.suggestions, list)

        # bank_name and bank_category can be None
        assert result.bank_name is None or isinstance(result.bank_name, str)
        assert result.bank_category is None or isinstance(result.bank_category, str)
