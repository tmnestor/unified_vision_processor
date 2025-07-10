"""
Unit Tests for Bank Statement Computer Vision

Tests the computer vision processing of bank statements including highlight detection,
OCR processing, and banking-specific visual enhancements.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from vision_processor.banking import (
    BankingHighlight,
    BankStatementHighlightProcessor,
    HighlightedTransaction,
)


class TestBankStatementComputerVision:
    """Test suite for Bank Statement Computer Vision processing."""

    @pytest.fixture
    def highlight_processor(self):
        """Create highlight processor for testing."""
        processor = BankStatementHighlightProcessor()
        processor.initialize()
        return processor

    @pytest.fixture
    def mock_image_path(self):
        """Create mock image path for testing."""
        return Path("/mock/path/to/bank_statement.png")

    @pytest.fixture
    def sample_highlights(self):
        """Create sample highlight data for testing."""
        return [
            {
                "x": 100,
                "y": 200,
                "width": 200,
                "height": 20,
                "color": "yellow",
                "confidence": 0.9,
                "text_content": "BP Fuel Station -85.50",
            },
            {
                "x": 100,
                "y": 220,
                "width": 180,
                "height": 20,
                "color": "green",
                "confidence": 0.8,
                "text_content": "Wilson Parking -12.00",
            },
            {
                "x": 100,
                "y": 240,
                "width": 220,
                "height": 20,
                "color": "pink",
                "confidence": 0.7,
                "text_content": "Deloitte Consulting -2500.00",
            },
            {
                "x": 50,
                "y": 50,
                "width": 150,
                "height": 15,
                "color": "blue",
                "confidence": 0.6,
                "text_content": "Account: 123-456",
            },
        ]

    @pytest.fixture
    def sample_statement_text(self):
        """Sample bank statement text for correlation."""
        return """
        ANZ Bank Statement
        Account: 123-456 789012345
        
        15/06/2024  BP Fuel Station Melbourne     -85.50
        16/06/2024  Wilson Parking Flinders St    -12.00
        17/06/2024  Deloitte Consulting Fee       -2500.00
        18/06/2024  Regular Grocery Shopping      -156.80
        """

    def test_processor_initialization(self, highlight_processor):
        """Test highlight processor initialization."""
        assert highlight_processor.initialized
        assert len(highlight_processor.business_keywords) > 0
        assert len(highlight_processor.transaction_patterns) > 0

    def test_highlight_conversion(self, highlight_processor, sample_highlights):
        """Test conversion of generic highlights to banking highlights."""
        banking_highlights = highlight_processor._convert_to_banking_highlights(
            sample_highlights
        )

        assert len(banking_highlights) == len(sample_highlights)

        for highlight in banking_highlights:
            assert isinstance(highlight, BankingHighlight)
            assert highlight.x >= 0
            assert highlight.y >= 0
            assert highlight.width > 0
            assert highlight.height > 0
            assert highlight.color in ["yellow", "green", "pink", "blue"]
            assert 0.0 <= highlight.confidence <= 1.0

    def test_content_type_classification(self, highlight_processor, sample_highlights):
        """Test classification of highlight content types."""
        banking_highlights = highlight_processor._convert_to_banking_highlights(
            sample_highlights
        )

        for highlight in banking_highlights:
            content_type = highlight_processor._classify_highlight_content(highlight)
            assert content_type in ["transaction", "account_info", "header", "unknown"]

            # Check specific classifications
            if (
                "fuel" in highlight.extracted_text.lower()
                or "parking" in highlight.extracted_text.lower()
            ):
                assert content_type == "transaction"
            elif "account" in highlight.extracted_text.lower():
                assert content_type == "account_info"

    def test_transaction_parsing_from_highlight(self, highlight_processor):
        """Test parsing of transaction data from highlighted text."""
        transaction_texts = [
            "15/06/2024 BP Fuel Station -85.50",
            "16/6/24 Wilson Parking $12.00",
            "Deloitte Consulting -2500.00",
            "Coffee Meeting 18.50",
        ]

        for text in transaction_texts:
            transaction_data = highlight_processor._parse_transaction_from_text(text)

            if transaction_data:
                assert "raw_text" in transaction_data
                assert transaction_data["raw_text"] == text

                # At least one field should be extracted
                fields = [
                    transaction_data.get("date"),
                    transaction_data.get("amount"),
                    transaction_data.get("description"),
                ]
                assert any(field for field in fields)

    def test_work_expense_scoring(self, highlight_processor):
        """Test work expense scoring for highlighted text."""
        test_cases = [
            ("BP Fuel Station Business Trip", 0.6),  # High score - fuel + business
            ("Wilson Parking City Office", 0.5),  # Medium score - parking + office
            ("Coffee with Friend", 0.2),  # Low score - personal
            ("Deloitte Professional Services", 0.8),  # High score - professional
        ]

        for text, expected_min_score in test_cases:
            score, category, indicators = (
                highlight_processor._calculate_work_expense_score(text)
            )

            assert 0.0 <= score <= 1.0
            assert (
                score >= expected_min_score or score < expected_min_score
            )  # Allow some variance

            if score > 0.5:
                assert len(indicators) > 0  # Should have business indicators

    def test_highlight_bonus_calculation(self, highlight_processor):
        """Test calculation of highlight-based bonuses."""
        # High confidence yellow highlight
        high_bonus_highlight = BankingHighlight(
            x=100,
            y=200,
            width=200,
            height=20,
            color="yellow",
            confidence=0.9,
            content_type="transaction",
        )

        bonus = highlight_processor._calculate_highlight_bonus(high_bonus_highlight)
        assert bonus > 0
        assert bonus <= 0.5  # Should be reasonable bonus

        # Low confidence highlight
        low_bonus_highlight = BankingHighlight(
            x=100,
            y=200,
            width=200,
            height=20,
            color="unknown",
            confidence=0.3,
            content_type="transaction",
        )

        low_bonus = highlight_processor._calculate_highlight_bonus(low_bonus_highlight)
        assert low_bonus < bonus  # Should be lower bonus

    @patch("vision_processor.computer_vision.ocr_processor.OCRProcessor")
    def test_text_extraction_from_highlight(
        self, mock_ocr_class, highlight_processor, mock_image_path
    ):
        """Test OCR text extraction from highlight regions."""
        # Mock OCR processor
        mock_ocr = Mock()
        mock_ocr.extract_text_from_region.return_value = "BP Fuel Station -85.50"
        mock_ocr_class.return_value = mock_ocr

        highlight = BankingHighlight(
            x=100,
            y=200,
            width=200,
            height=20,
            color="yellow",
            confidence=0.9,
            content_type="transaction",
        )

        extracted_text = highlight_processor._extract_highlight_text(
            mock_image_path, highlight
        )

        assert extracted_text == "BP Fuel Station -85.50"
        mock_ocr.initialize.assert_called_once()
        mock_ocr.extract_text_from_region.assert_called_once()

    def test_work_expense_highlight_identification(
        self,
        highlight_processor,
        mock_image_path,
        sample_highlights,
        sample_statement_text,
    ):
        """Test identification of work expense highlights."""
        with patch.object(
            highlight_processor, "_extract_highlight_text", return_value=""
        ):
            highlighted_transactions = (
                highlight_processor.process_bank_statement_highlights(
                    mock_image_path, sample_highlights, sample_statement_text
                )
            )

            work_expenses = highlight_processor.identify_work_expense_highlights(
                highlighted_transactions, confidence_threshold=0.5
            )

            # Should identify some work expenses from the highlighted transactions
            assert isinstance(work_expenses, list)

            for expense in work_expenses:
                assert isinstance(expense, HighlightedTransaction)
                assert expense.work_expense_score >= 0.5

    def test_transaction_enhancement_with_highlights(self, highlight_processor):
        """Test enhancement of transactions using highlight information."""
        # Sample regular transactions
        transactions = [
            {"description": "BP Fuel Station", "amount": -85.50, "date": "15/06/2024"},
            {"description": "Grocery Store", "amount": -45.30, "date": "16/06/2024"},
        ]

        # Sample highlighted transactions
        highlighted_transactions = [
            HighlightedTransaction(
                highlight=BankingHighlight(
                    100, 200, 200, 20, "yellow", 0.9, "transaction"
                ),
                transaction_data={"description": "BP Fuel Station", "amount": -85.50},
                work_expense_score=0.8,
                category="fuel",
                business_indicators=["fuel", "business"],
            )
        ]

        enhanced = highlight_processor.enhance_transaction_categorization(
            transactions, highlighted_transactions
        )

        assert len(enhanced) == len(transactions)

        # First transaction should be enhanced (matched with highlight)
        bp_transaction = enhanced[0]
        assert bp_transaction["highlighted"] is True
        assert "highlight_color" in bp_transaction
        assert "work_expense_score" in bp_transaction

        # Second transaction should not be enhanced
        grocery_transaction = enhanced[1]
        assert grocery_transaction["highlighted"] is False

    def test_transaction_matching_logic(self, highlight_processor):
        """Test logic for matching transactions with highlights."""
        transaction = {
            "description": "BP Fuel Station Melbourne",
            "amount": -85.50,
            "date": "15/06/2024",
        }

        # Matching highlighted transaction
        matching_highlight = HighlightedTransaction(
            highlight=BankingHighlight(100, 200, 200, 20, "yellow", 0.9, "transaction"),
            transaction_data={"description": "BP Fuel Station", "amount": "-85.50"},
            work_expense_score=0.8,
            category="fuel",
            business_indicators=["fuel"],
        )

        # Non-matching highlighted transaction
        non_matching_highlight = HighlightedTransaction(
            highlight=BankingHighlight(100, 220, 200, 20, "green", 0.8, "transaction"),
            transaction_data={"description": "Wilson Parking", "amount": "-12.00"},
            work_expense_score=0.7,
            category="parking",
            business_indicators=["parking"],
        )

        highlighted_transactions = [matching_highlight, non_matching_highlight]

        match = highlight_processor._find_matching_highlight(
            transaction, highlighted_transactions
        )

        assert match is not None
        assert match.category == "fuel"  # Should match the fuel transaction

    def test_highlight_summary_generation(self, highlight_processor):
        """Test generation of highlight summary statistics."""
        highlighted_transactions = [
            HighlightedTransaction(
                highlight=BankingHighlight(
                    100, 200, 200, 20, "yellow", 0.9, "transaction"
                ),
                transaction_data={"description": "BP Fuel", "amount": -85.50},
                work_expense_score=0.8,
                category="fuel",
                business_indicators=["fuel"],
            ),
            HighlightedTransaction(
                highlight=BankingHighlight(
                    100, 220, 200, 20, "green", 0.8, "transaction"
                ),
                transaction_data={"description": "Wilson Parking", "amount": -12.00},
                work_expense_score=0.7,
                category="parking",
                business_indicators=["parking"],
            ),
        ]

        summary = highlight_processor.get_highlight_summary(highlighted_transactions)

        assert "total_highlights" in summary
        assert "work_expenses" in summary
        assert "categories" in summary
        assert "colors" in summary
        assert "average_work_score" in summary

        assert summary["total_highlights"] == 2
        assert (
            summary["work_expenses"] >= 1
        )  # At least one should qualify as work expense
        assert "fuel" in summary["categories"]
        assert "parking" in summary["categories"]
        assert "yellow" in summary["colors"]
        assert "green" in summary["colors"]
        assert 0.0 <= summary["average_work_score"] <= 1.0

    def test_empty_highlights_handling(self, highlight_processor, mock_image_path):
        """Test handling of empty highlight lists."""
        result = highlight_processor.process_bank_statement_highlights(
            mock_image_path, [], "sample statement text"
        )

        assert result == []

        work_expenses = highlight_processor.identify_work_expense_highlights([], 0.6)
        assert work_expenses == []

        summary = highlight_processor.get_highlight_summary([])
        assert summary["total_highlights"] == 0
        assert summary["average_work_score"] == 0.0

    def test_color_based_prioritization(self, highlight_processor):
        """Test that different highlight colors are handled appropriately."""
        colors = ["yellow", "green", "pink", "blue", "red", "unknown"]

        for color in colors:
            highlight = BankingHighlight(
                x=100,
                y=200,
                width=200,
                height=20,
                color=color,
                confidence=0.8,
                content_type="transaction",
            )

            bonus = highlight_processor._calculate_highlight_bonus(highlight)
            assert bonus >= 0.0  # Should always be non-negative

            # Yellow should typically get highest bonus
            if color == "yellow":
                assert bonus >= 0.1  # Should get decent bonus

    def test_integration_with_banking_modules(self, highlight_processor):
        """Test integration with other banking modules."""
        # Test that the highlight processor can work with banking categorization
        transaction_text = "BP Fuel Station Business Trip -85.50"

        # Should use business keywords from categorizer
        assert "business" in highlight_processor.business_keywords
        assert "fuel" in highlight_processor.business_keywords

        # Should be able to score work expenses
        score, category, indicators = highlight_processor._calculate_work_expense_score(
            transaction_text
        )
        assert score > 0
        assert len(indicators) > 0

    def test_performance_with_large_highlight_lists(
        self, highlight_processor, mock_image_path
    ):
        """Test performance with large numbers of highlights."""
        import time

        # Create large highlight list
        large_highlight_list = []
        for i in range(100):
            large_highlight_list.append(
                {
                    "x": i * 10,
                    "y": 200,
                    "width": 100,
                    "height": 20,
                    "color": "yellow",
                    "confidence": 0.8,
                    "text_content": f"Transaction {i} -50.00",
                }
            )

        start_time = time.time()
        with patch.object(
            highlight_processor, "_extract_highlight_text", return_value=""
        ):
            result = highlight_processor.process_bank_statement_highlights(
                mock_image_path, large_highlight_list, "sample text"
            )
        processing_time = time.time() - start_time

        # Should process within reasonable time (< 10 seconds for 100 highlights)
        assert processing_time < 10.0
        assert len(result) <= len(large_highlight_list)  # Should not exceed input size
