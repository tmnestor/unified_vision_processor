"""Banking Highlight Processor - Bank Statement Visual Enhancement Integration

This module provides specialized highlight processing for bank statements,
integrating with InternVL's computer vision capabilities to identify and
process highlighted transactions and account information.

Features:
- Bank statement specific highlight detection
- Transaction highlight correlation
- Account information highlight processing
- Work expense highlight identification
- Integration with transaction categorization
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BankingHighlight:
    """Represents a banking-related highlight region."""

    x: int
    y: int
    width: int
    height: int
    color: str
    confidence: float
    content_type: str  # "transaction", "account_info", "balance", "header"
    extracted_text: str | None = None
    business_relevance: float = 0.0


@dataclass
class HighlightedTransaction:
    """Represents a highlighted transaction with business context."""

    highlight: BankingHighlight
    transaction_data: dict[str, Any]
    work_expense_score: float
    category: str | None
    business_indicators: list[str]


class BankStatementHighlightProcessor:
    """Specialized highlight processor for bank statements.

    Processes highlighted regions in bank statements to identify business
    transactions, account information, and work-related expenses with
    enhanced accuracy through visual highlighting cues.
    """

    def __init__(self):
        self.initialized = False
        self.business_keywords = []
        self.transaction_patterns = {}

    def initialize(self) -> None:
        """Initialize banking highlight processor."""
        if self.initialized:
            return

        # Business-related keywords that boost highlight relevance
        self.business_keywords = [
            "business",
            "office",
            "work",
            "professional",
            "corporate",
            "fuel",
            "parking",
            "toll",
            "taxi",
            "uber",
            "hotel",
            "restaurant",
            "software",
            "equipment",
            "supplies",
            "training",
            "conference",
            "legal",
            "accounting",
            "consulting",
            "insurance",
            "bank fee",
        ]

        # Transaction highlight patterns for different content types
        self.transaction_patterns = {
            "date": r"\d{1,2}/\d{1,2}/\d{2,4}",
            "amount": r"[\-\+]?\$?\d+(?:\.\d{2})?",
            "reference": r"[A-Z0-9]{6,20}",
            "account": r"\d{3}-\d{3}|\d{6,10}",
        }

        logger.info("BankStatementHighlightProcessor initialized")
        self.initialized = True

    def process_bank_statement_highlights(
        self,
        image_path: Path,
        highlights: list[dict[str, Any]],
        statement_text: str,
    ) -> list[HighlightedTransaction]:
        """Process highlights in bank statement for business transaction identification.

        Args:
            image_path: Path to bank statement image
            highlights: List of detected highlight regions
            statement_text: Extracted text from statement

        Returns:
            List of HighlightedTransaction objects with business context

        """
        self.initialize()

        if not highlights:
            return []

        banking_highlights = self._convert_to_banking_highlights(highlights)
        highlighted_transactions = []

        for highlight in banking_highlights:
            # Extract text from highlight region if not provided
            if not highlight.extracted_text:
                highlight.extracted_text = self._extract_highlight_text(
                    image_path,
                    highlight,
                )

            # Determine content type
            highlight.content_type = self._classify_highlight_content(highlight)

            # Process transaction highlights specifically
            if highlight.content_type == "transaction":
                transaction = self._process_transaction_highlight(
                    highlight,
                    statement_text,
                )
                if transaction:
                    highlighted_transactions.append(transaction)

        # Sort by business relevance (highest first)
        highlighted_transactions.sort(
            key=lambda x: x.work_expense_score,
            reverse=True,
        )

        return highlighted_transactions

    def identify_work_expense_highlights(
        self,
        highlighted_transactions: list[HighlightedTransaction],
        confidence_threshold: float = 0.6,
    ) -> list[HighlightedTransaction]:
        """Filter highlighted transactions for likely work expenses.

        Args:
            highlighted_transactions: List of highlighted transactions
            confidence_threshold: Minimum work expense score

        Returns:
            List of transactions likely to be work expenses

        """
        work_expenses = [
            transaction
            for transaction in highlighted_transactions
            if transaction.work_expense_score >= confidence_threshold
        ]

        logger.info(f"Identified {len(work_expenses)} work expense highlights")
        return work_expenses

    def enhance_transaction_categorization(
        self,
        transactions: list[dict[str, Any]],
        highlighted_transactions: list[HighlightedTransaction],
    ) -> list[dict[str, Any]]:
        """Enhance transaction categorization using highlight information.

        Args:
            transactions: List of extracted transactions
            highlighted_transactions: List of highlighted transactions

        Returns:
            Enhanced transactions with highlight-based improvements

        """
        enhanced_transactions = []

        for transaction in transactions:
            enhanced = transaction.copy()

            # Find matching highlighted transaction
            matching_highlight = self._find_matching_highlight(
                transaction,
                highlighted_transactions,
            )

            if matching_highlight:
                # Boost work expense confidence if highlighted
                if "work_expense_score" in enhanced:
                    enhanced["work_expense_score"] = min(
                        enhanced["work_expense_score"] + 0.3,
                        1.0,
                    )
                else:
                    enhanced["work_expense_score"] = matching_highlight.work_expense_score

                # Add highlight metadata
                enhanced["highlighted"] = True
                enhanced["highlight_color"] = matching_highlight.highlight.color
                enhanced["highlight_confidence"] = matching_highlight.highlight.confidence
                enhanced["business_indicators"] = matching_highlight.business_indicators

                # Use highlighted category if more specific
                if matching_highlight.category:
                    enhanced["category"] = matching_highlight.category
            else:
                enhanced["highlighted"] = False

            enhanced_transactions.append(enhanced)

        return enhanced_transactions

    def _convert_to_banking_highlights(
        self,
        highlights: list[dict[str, Any]],
    ) -> list[BankingHighlight]:
        """Convert generic highlights to banking-specific highlights."""
        banking_highlights = []

        for highlight in highlights:
            banking_highlight = BankingHighlight(
                x=highlight.get("x", 0),
                y=highlight.get("y", 0),
                width=highlight.get("width", 0),
                height=highlight.get("height", 0),
                color=highlight.get("color", "unknown"),
                confidence=highlight.get("confidence", 0.0),
                content_type="unknown",
                extracted_text=highlight.get("text_content"),
            )

            banking_highlights.append(banking_highlight)

        return banking_highlights

    def _extract_highlight_text(
        self,
        image_path: Path,
        highlight: BankingHighlight,
    ) -> str:
        """Extract text from highlighted region using OCR."""
        try:
            # Import OCR processor from computer vision module
            from ..computer_vision.ocr_processor import OCRProcessor

            ocr = OCRProcessor()
            ocr.initialize()

            # Extract text from specific region
            region = (highlight.x, highlight.y, highlight.width, highlight.height)
            extracted_text = ocr.extract_text_from_region(image_path, region)

            return extracted_text.strip() if extracted_text else ""

        except Exception as e:
            logger.warning(f"Failed to extract text from highlight: {e}")
            return ""

    def _classify_highlight_content(self, highlight: BankingHighlight) -> str:
        """Classify what type of content the highlight contains."""
        if not highlight.extracted_text:
            return "unknown"

        text = highlight.extracted_text.lower()

        # Check for transaction patterns
        if any(pattern in text for pattern in ["$", "debit", "credit", "payment"]):
            return "transaction"

        # Check for account information
        if any(pattern in text for pattern in ["account", "bsb", "balance"]):
            return "account_info"

        # Check for header information
        if any(pattern in text for pattern in ["statement", "period", "from", "to"]):
            return "header"

        # Default to transaction if contains amount-like patterns
        import re

        if re.search(r"\d+\.\d{2}", text):
            return "transaction"

        return "unknown"

    def _process_transaction_highlight(
        self,
        highlight: BankingHighlight,
        _statement_text: str,
    ) -> HighlightedTransaction | None:
        """Process a transaction highlight to extract business context."""
        if not highlight.extracted_text:
            return None

        # Parse transaction data from highlighted text
        transaction_data = self._parse_transaction_from_text(highlight.extracted_text)

        if not transaction_data:
            return None

        # Calculate work expense score
        work_score, category, indicators = self._calculate_work_expense_score(
            highlight.extracted_text,
        )

        # Enhance with highlight-specific factors
        highlight_bonus = self._calculate_highlight_bonus(highlight)
        work_score = min(work_score + highlight_bonus, 1.0)

        return HighlightedTransaction(
            highlight=highlight,
            transaction_data=transaction_data,
            work_expense_score=work_score,
            category=category,
            business_indicators=indicators,
        )

    def _parse_transaction_from_text(self, text: str) -> dict[str, Any] | None:
        """Parse transaction information from highlighted text."""
        import re

        # Extract date
        date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", text)
        date = date_match.group(1) if date_match else None

        # Extract amount
        amount_match = re.search(r"([\-\+]?\$?\d+(?:\.\d{2})?)", text)
        amount = amount_match.group(1) if amount_match else None

        # Extract description (remaining text after removing date and amount)
        description = text
        if date_match:
            description = description.replace(date_match.group(1), "")
        if amount_match:
            description = description.replace(amount_match.group(1), "")
        description = description.strip()

        if not (date or amount or description):
            return None

        return {
            "date": date,
            "amount": amount,
            "description": description,
            "raw_text": text,
        }

    def _calculate_work_expense_score(
        self,
        text: str,
    ) -> tuple[float, str | None, list[str]]:
        """Calculate work expense score for highlighted text."""
        text_lower = text.lower()
        score = 0.0
        category = None
        indicators = []

        # Business keyword scoring
        for keyword in self.business_keywords:
            if keyword in text_lower:
                score += 0.2
                indicators.append(keyword)

        # Category-specific scoring
        categories = {
            "fuel": ["bp", "shell", "caltex", "ampol", "fuel", "petrol"],
            "parking": ["parking", "wilson", "secure"],
            "transport": ["uber", "taxi", "train", "bus"],
            "meals": ["restaurant", "cafe", "lunch", "dinner"],
            "office": ["officeworks", "supplies", "equipment"],
        }

        max_category_score = 0.0
        for cat, keywords in categories.items():
            cat_score = sum(0.3 for kw in keywords if kw in text_lower)
            if cat_score > max_category_score:
                max_category_score = cat_score
                category = cat

        score += min(max_category_score, 0.6)

        return min(score, 1.0), category, indicators

    def _calculate_highlight_bonus(self, highlight: BankingHighlight) -> float:
        """Calculate bonus score for being highlighted."""
        bonus = 0.0

        # Color-based bonuses (some colors indicate importance)
        color_bonuses = {
            "yellow": 0.2,
            "green": 0.15,
            "pink": 0.1,
            "blue": 0.05,
        }

        bonus += color_bonuses.get(highlight.color.lower(), 0.0)

        # Confidence bonus
        if highlight.confidence > 0.8:
            bonus += 0.1
        elif highlight.confidence > 0.6:
            bonus += 0.05

        return bonus

    def _find_matching_highlight(
        self,
        transaction: dict[str, Any],
        highlighted_transactions: list[HighlightedTransaction],
    ) -> HighlightedTransaction | None:
        """Find highlighted transaction that matches a regular transaction."""
        transaction_desc = transaction.get("description", "").lower()
        transaction_amount = str(transaction.get("amount", ""))

        for highlighted in highlighted_transactions:
            highlight_desc = highlighted.transaction_data.get("description", "").lower()
            highlight_amount = str(highlighted.transaction_data.get("amount", ""))

            # Check for description match (fuzzy)
            if (
                transaction_desc
                and highlight_desc
                and len(set(transaction_desc.split()) & set(highlight_desc.split())) >= 2
            ):
                return highlighted

            # Check for amount match
            if transaction_amount and highlight_amount and transaction_amount in highlight_amount:
                return highlighted

        return None

    def get_highlight_summary(
        self,
        highlighted_transactions: list[HighlightedTransaction],
    ) -> dict[str, Any]:
        """Generate summary of highlighted transactions."""
        if not highlighted_transactions:
            return {
                "total_highlights": 0,
                "work_expenses": 0,
                "categories": {},
                "colors": {},
                "average_work_score": 0.0,
            }

        work_expenses = [t for t in highlighted_transactions if t.work_expense_score >= 0.6]

        categories = {}
        colors = {}

        for transaction in highlighted_transactions:
            # Category breakdown
            cat = transaction.category or "other"
            categories[cat] = categories.get(cat, 0) + 1

            # Color breakdown
            color = transaction.highlight.color
            colors[color] = colors.get(color, 0) + 1

        avg_score = sum(t.work_expense_score for t in highlighted_transactions) / len(
            highlighted_transactions
        )

        return {
            "total_highlights": len(highlighted_transactions),
            "work_expenses": len(work_expenses),
            "categories": categories,
            "colors": colors,
            "average_work_score": avg_score,
        }
