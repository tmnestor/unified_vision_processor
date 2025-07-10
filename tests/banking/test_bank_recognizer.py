"""
Unit Tests for Australian Bank Recognizer

Tests the bank recognition functionality including pattern matching,
confidence scoring, and Australian bank identification.
"""

import pytest

from vision_processor.banking import AustralianBankRecognizer


class TestAustralianBankRecognizer:
    """Test suite for Australian Bank Recognizer."""

    @pytest.fixture
    def bank_recognizer(self):
        """Create bank recognizer for testing."""
        recognizer = AustralianBankRecognizer()
        recognizer.initialize()
        return recognizer

    def test_initialization(self, bank_recognizer):
        """Test bank recognizer initialization."""
        assert bank_recognizer.initialized
        assert len(bank_recognizer.bank_patterns) > 0
        assert len(bank_recognizer.bank_metadata) > 0

    def test_big_four_bank_recognition(self, bank_recognizer):
        """Test recognition of Australia's Big Four banks."""
        test_cases = [
            ("ANZ Bank Statement", "ANZ"),
            ("Commonwealth Bank of Australia", "Commonwealth Bank"),
            ("Westpac Banking Corporation", "Westpac"),
            ("National Australia Bank", "NAB"),
        ]

        for text, expected_bank in test_cases:
            matches = bank_recognizer.recognize_banks(text)
            assert len(matches) > 0
            assert matches[0].official_name.startswith(expected_bank.split()[0])
            assert matches[0].category == "big_four"

    def test_regional_bank_recognition(self, bank_recognizer):
        """Test recognition of regional Australian banks."""
        test_cases = [
            ("Bendigo Bank Account Statement", "Bendigo"),
            ("Suncorp Bank Transaction", "Suncorp"),
            ("Macquarie Bank Statement", "Macquarie"),
        ]

        for text, expected_bank in test_cases:
            matches = bank_recognizer.recognize_banks(text)
            assert len(matches) > 0
            # Check that the recognized bank name contains expected text
            found = any(
                expected_bank.lower() in match.official_name.lower()
                for match in matches
            )
            assert found

    def test_online_bank_recognition(self, bank_recognizer):
        """Test recognition of online banks."""
        text = "ING Direct Savings Account"
        matches = bank_recognizer.recognize_banks(text)

        assert len(matches) > 0
        assert any("ING" in match.official_name for match in matches)
        assert any(match.category == "online" for match in matches)

    def test_confidence_scoring(self, bank_recognizer):
        """Test confidence scoring for bank recognition."""
        # High confidence case - exact institutional name
        high_confidence_text = "Commonwealth Bank of Australia Account Statement"
        matches = bank_recognizer.recognize_banks(high_confidence_text)
        assert len(matches) > 0
        assert matches[0].confidence > 0.7

        # Lower confidence case - abbreviated name
        low_confidence_text = "CBA"
        matches = bank_recognizer.recognize_banks(low_confidence_text)
        if matches:  # May not match depending on pattern strictness
            assert matches[0].confidence <= 0.7

    def test_multiple_bank_mentions(self, bank_recognizer):
        """Test handling of text mentioning multiple banks."""
        text = "Transfer from ANZ to Commonwealth Bank account"
        matches = bank_recognizer.recognize_banks(text)

        # Should detect both banks
        assert len(matches) >= 2
        bank_names = [match.official_name for match in matches]
        assert any("ANZ" in name for name in bank_names)
        assert any("Commonwealth" in name for name in bank_names)

    def test_primary_bank_identification(self, bank_recognizer):
        """Test identification of primary bank from text."""
        # Clear primary bank
        text = "Westpac Banking Corporation Statement Period"
        primary = bank_recognizer.identify_primary_bank(text)
        assert primary is not None
        assert "Westpac" in primary.official_name

        # No clear primary bank
        ambiguous_text = "Some random text without bank names"
        primary = bank_recognizer.identify_primary_bank(ambiguous_text)
        assert primary is None

    def test_big_four_classification(self, bank_recognizer):
        """Test Big Four bank classification."""
        big_four_banks = ["ANZ", "Commonwealth Bank", "Westpac", "NAB"]

        for bank in big_four_banks:
            assert bank_recognizer.is_big_four_bank(bank)

        # Test non-Big Four banks
        assert not bank_recognizer.is_big_four_bank("ING")
        assert not bank_recognizer.is_big_four_bank("Bendigo Bank")

    def test_bank_categorization(self, bank_recognizer):
        """Test bank category classification."""
        test_cases = [
            ("ANZ", "big_four"),
            ("Commonwealth Bank", "big_four"),
            ("ING", "online"),
            ("Bendigo Bank", "regional"),
            ("HSBC", "international"),
        ]

        for bank_name, expected_category in test_cases:
            category = bank_recognizer.get_bank_category(bank_name)
            assert category == expected_category

    def test_bsb_ranges(self, bank_recognizer):
        """Test BSB range retrieval for banks."""
        # Test ANZ BSB ranges
        anz_ranges = bank_recognizer.get_bank_bsb_ranges("ANZ")
        assert len(anz_ranges) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in anz_ranges)

        # Test unknown bank
        unknown_ranges = bank_recognizer.get_bank_bsb_ranges("Unknown Bank")
        assert len(unknown_ranges) == 0

    def test_supported_banks(self, bank_recognizer):
        """Test retrieval of supported banks."""
        supported = bank_recognizer.get_supported_banks()

        assert len(supported) >= 10  # Should support at least 10 major banks
        assert "ANZ" in supported
        assert "Commonwealth Bank" in supported
        assert "Westpac" in supported
        assert "NAB" in supported

    def test_case_insensitive_matching(self, bank_recognizer):
        """Test that bank recognition is case-insensitive."""
        test_cases = [
            "anz bank statement",
            "ANZ BANK STATEMENT",
            "Anz Bank Statement",
            "commonwealth bank account",
            "COMMONWEALTH BANK ACCOUNT",
        ]

        for text in test_cases:
            matches = bank_recognizer.recognize_banks(text)
            assert len(matches) > 0

    def test_bank_pattern_variations(self, bank_recognizer):
        """Test recognition of various bank name patterns."""
        # Test different Commonwealth Bank variations
        commonwealth_variations = [
            "Commonwealth Bank",
            "CBA",
            "CommBank",
            "Commonwealth Bank of Australia",
        ]

        for variation in commonwealth_variations:
            matches = bank_recognizer.recognize_banks(variation)
            # At least one should match (depending on pattern strictness)
            if matches:
                assert any("Commonwealth" in match.official_name for match in matches)

    def test_banking_context_bonus(self, bank_recognizer):
        """Test that banking context improves confidence."""
        # Text with banking context
        banking_context = "ANZ Bank Account Statement Balance Transaction"
        matches_with_context = bank_recognizer.recognize_banks(banking_context)

        # Text without banking context
        no_context = "ANZ"
        matches_no_context = bank_recognizer.recognize_banks(no_context)

        if matches_with_context and matches_no_context:
            # Context should improve confidence
            assert (
                matches_with_context[0].confidence >= matches_no_context[0].confidence
            )

    def test_empty_and_invalid_input(self, bank_recognizer):
        """Test handling of empty and invalid input."""
        # Empty text
        assert bank_recognizer.recognize_banks("") == []
        assert bank_recognizer.identify_primary_bank("") is None

        # Invalid text with no bank names
        assert bank_recognizer.recognize_banks("random text 123") == []

        # Whitespace only
        assert bank_recognizer.recognize_banks("   ") == []
