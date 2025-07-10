"""Bank Statement Computer Vision Processing

This module provides specialized computer vision processing for bank statements,
combining highlight detection, OCR, and transaction parsing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .highlight_detector import HighlightDetector, HighlightRegion
from .ocr_processor import OCRProcessor, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Represents a parsed transaction from bank statement."""

    date: str | None
    description: str
    amount: float
    balance: float | None = None
    reference: str | None = None
    transaction_type: str | None = None  # debit, credit
    highlight_region: HighlightRegion | None = None
    confidence: float = 0.0

    def __post_init__(self):
        """Clean up transaction data."""
        self.description = self.description.strip()
        if self.reference:
            self.reference = self.reference.strip()


@dataclass
class BankStatementResult:
    """Complete result from bank statement processing."""

    account_info: dict[str, Any]
    transactions: list[Transaction]
    highlights_detected: int
    highlights_processed: int
    total_amount: float
    statement_period: str | None = None
    opening_balance: float | None = None
    closing_balance: float | None = None
    processing_metadata: dict[str, Any] | None = None


class BankStatementCV:
    """Specialized computer vision processor for bank statements.

    Features:
    - Bank statement layout detection
    - Transaction row identification
    - Highlight-based transaction extraction
    - Account information parsing
    - Balance and period extraction
    """

    def __init__(self, config: Any):
        self.config = config
        self.initialized = False

        # Bank statement specific configuration
        self.bank_config = {
            "min_transaction_width": 200,  # Minimum width for transaction rows
            "min_transaction_height": 20,  # Minimum height for transaction rows
            "max_transaction_height": 100,  # Maximum height for transaction rows
            "transaction_confidence_threshold": 0.4,
            "account_info_regions": {
                "top_section": 0.3,  # Top 30% for account info
                "transaction_section": 0.7,  # Bottom 70% for transactions
            },
        }

        # Australian bank patterns
        self.bank_patterns = {
            "bsb": r"\b\d{3}[-\s]?\d{3}\b",
            "account": r"\b\d{6,12}\b",
            "date": [
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                r"\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}\b",
            ],
            "amount": r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
            "reference": r"(?:ref|reference)[\s:]*([a-zA-Z0-9-]+)",
        }

        # Initialize sub-components
        self.highlight_detector = None
        self.ocr_processor = None

    def initialize(self) -> None:
        """Initialize bank statement CV processor."""
        if self.initialized:
            return

        # Initialize components
        self.highlight_detector = HighlightDetector(self.config)
        self.highlight_detector.initialize()

        self.ocr_processor = OCRProcessor(self.config)
        self.ocr_processor.initialize()

        logger.info("BankStatementCV initialized")
        self.initialized = True

    def process_bank_statement(
        self,
        image_path: str | Path | Image.Image,
    ) -> BankStatementResult:
        """Process a complete bank statement.

        Args:
            image_path: Path to bank statement image

        Returns:
            Complete bank statement processing result

        """
        if not self.initialized:
            self.initialize()

        try:
            # Step 1: Detect highlights
            highlights = self.highlight_detector.detect_bank_statement_highlights(
                image_path,
            )
            logger.info(f"Detected {len(highlights)} potential transaction highlights")

            # Step 2: Filter highlights for transactions
            transaction_highlights = self._filter_transaction_highlights(
                highlights,
                image_path,
            )
            logger.info(
                f"Filtered to {len(transaction_highlights)} transaction highlights",
            )

            # Step 3: Process OCR on highlights
            ocr_data = self.ocr_processor.process_bank_statement_highlights(
                image_path,
                transaction_highlights,
            )

            # Step 4: Parse transactions
            transactions = self._parse_transactions(ocr_data["ocr_results"])

            # Step 5: Extract account information
            account_info = self._extract_account_information(image_path)
            account_info.update(ocr_data.get("account_info", {}))

            # Step 6: Calculate totals and balances
            total_amount = sum(abs(t.amount) for t in transactions)

            # Step 7: Detect statement period
            statement_period = self._detect_statement_period(image_path)

            # Step 8: Extract opening/closing balances
            opening_balance, closing_balance = self._extract_balances(image_path)

            result = BankStatementResult(
                account_info=account_info,
                transactions=transactions,
                highlights_detected=len(highlights),
                highlights_processed=len(transaction_highlights),
                total_amount=total_amount,
                statement_period=statement_period,
                opening_balance=opening_balance,
                closing_balance=closing_balance,
                processing_metadata={
                    "ocr_results_count": len(ocr_data["ocr_results"]),
                    "processing_method": "highlight_based",
                    "confidence_threshold": self.bank_config[
                        "transaction_confidence_threshold"
                    ],
                },
            )

            logger.info(
                f"Successfully processed bank statement: {len(transactions)} transactions, "
                f"${total_amount:.2f} total amount",
            )

            return result

        except Exception as e:
            logger.error(f"Error processing bank statement: {e}")
            return BankStatementResult(
                account_info={},
                transactions=[],
                highlights_detected=0,
                highlights_processed=0,
                total_amount=0.0,
                processing_metadata={"error": str(e)},
            )

    def _filter_transaction_highlights(
        self,
        highlights: list[HighlightRegion],
        image_path: str | Path | Image.Image,
    ) -> list[HighlightRegion]:
        """Filter highlights to keep only those likely to be transactions."""
        if not highlights:
            return []

        try:
            # Load image to get dimensions
            if isinstance(image_path, (str, Path)):
                image = cv2.imread(str(image_path))
                if image is None:
                    return highlights
            elif isinstance(image_path, Image.Image):
                image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
            else:
                return highlights

            height, width, _ = image.shape

            # Filter based on transaction characteristics
            transaction_highlights = []

            for highlight in highlights:
                # Check dimensions
                if (
                    highlight.width >= self.bank_config["min_transaction_width"]
                    and self.bank_config["min_transaction_height"]
                    <= highlight.height
                    <= self.bank_config["max_transaction_height"]
                ):
                    # Check aspect ratio (transactions are usually wide)
                    aspect_ratio = highlight.width / highlight.height
                    if 3.0 <= aspect_ratio <= 25.0:
                        # Check position (transactions are usually in the lower part of statement)
                        relative_y = highlight.y / height
                        if relative_y >= 0.2:  # Not in the very top section
                            # Check confidence
                            if (
                                highlight.confidence
                                >= self.bank_config["transaction_confidence_threshold"]
                            ):
                                transaction_highlights.append(highlight)

            return transaction_highlights

        except Exception as e:
            logger.error(f"Error filtering transaction highlights: {e}")
            return highlights

    def _parse_transactions(self, ocr_results: list[OCRResult]) -> list[Transaction]:
        """Parse transactions from OCR results."""
        transactions = []

        for ocr_result in ocr_results:
            transaction = self._parse_single_transaction(ocr_result)
            if transaction:
                transactions.append(transaction)

        # Sort transactions by position (top to bottom)
        transactions.sort(
            key=lambda t: t.highlight_region.y if t.highlight_region else 0,
        )

        return transactions

    def _parse_single_transaction(self, ocr_result: OCRResult) -> Transaction | None:
        """Parse a single transaction from OCR result."""
        import re

        text = ocr_result.text

        # Extract components
        date = self._extract_date(text)
        amount = self._extract_amount(text)
        reference = self._extract_reference(text)

        if amount is None:
            return None  # Must have an amount to be a valid transaction

        # Extract description (remove parsed components)
        description = text
        if date:
            description = re.sub(self.bank_patterns["date"][0], "", description)
            description = re.sub(
                self.bank_patterns["date"][1],
                "",
                description,
                flags=re.IGNORECASE,
            )
        if reference:
            description = re.sub(
                self.bank_patterns["reference"],
                "",
                description,
                flags=re.IGNORECASE,
            )

        # Remove amount patterns
        description = re.sub(self.bank_patterns["amount"], "", description)

        # Clean up description
        description = re.sub(r"\s+", " ", description).strip()
        if not description:
            description = "Transaction"

        # Determine transaction type
        transaction_type = "debit" if amount < 0 else "credit"

        return Transaction(
            date=date,
            description=description,
            amount=abs(amount),  # Store as positive, type indicates direction
            reference=reference,
            transaction_type=transaction_type,
            highlight_region=ocr_result.highlight_region,
            confidence=ocr_result.confidence,
        )

    def _extract_date(self, text: str) -> str | None:
        """Extract date from text."""
        import re

        for pattern in self.bank_patterns["date"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    def _extract_amount(self, text: str) -> float | None:
        """Extract amount from text."""
        import re

        amounts = re.findall(self.bank_patterns["amount"], text)
        if not amounts:
            return None

        # Find the most likely amount (usually the last one or the largest)
        parsed_amounts = []
        for amount_str in amounts:
            try:
                # Remove $ and commas
                clean_amount = amount_str.replace("$", "").replace(",", "")
                amount = float(clean_amount)
                parsed_amounts.append(amount)
            except ValueError:
                continue

        if not parsed_amounts:
            return None

        # Return the largest amount (most likely to be the transaction amount)
        return max(parsed_amounts)

    def _extract_reference(self, text: str) -> str | None:
        """Extract reference number from text."""
        import re

        match = re.search(self.bank_patterns["reference"], text, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_account_information(
        self,
        image_path: str | Path | Image.Image,
    ) -> dict[str, Any]:
        """Extract account information from bank statement."""
        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = cv2.imread(str(image_path))
                if image is None:
                    return {}
            elif isinstance(image_path, Image.Image):
                image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
            else:
                return {}

            height, width, _ = image.shape

            # Extract top section for account information
            top_section_height = int(
                height * self.bank_config["account_info_regions"]["top_section"],
            )
            account_section = image[:top_section_height, :]

            account_info = {}

            # Process all text from account section
            if (
                hasattr(self.ocr_processor, "tesseract")
                and self.ocr_processor.tesseract
            ):
                try:
                    # Run OCR on entire account section
                    account_text = self.ocr_processor.tesseract.image_to_string(
                        account_section,
                    )

                    # Extract account information using patterns
                    account_info.update(self._parse_account_text(account_text))

                except Exception as e:
                    logger.error(f"Error extracting account information: {e}")

            return account_info

        except Exception as e:
            logger.error(f"Error processing account information: {e}")
            return {}

    def _parse_account_text(self, text: str) -> dict[str, Any]:
        """Parse account information from text."""
        import re

        account_info = {}

        # BSB
        bsb_match = re.search(self.bank_patterns["bsb"], text)
        if bsb_match:
            account_info["bsb"] = bsb_match.group(0)

        # Account number
        account_matches = re.findall(self.bank_patterns["account"], text)
        if account_matches:
            # Take the longest match (most likely to be account number)
            account_info["account_number"] = max(account_matches, key=len)

        # Account holder name (heuristic: look for capitalized words)
        name_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
        name_matches = re.findall(name_pattern, text)
        if name_matches:
            # Take the first reasonable match
            for name in name_matches:
                if 10 <= len(name) <= 50:  # Reasonable name length
                    account_info["account_holder"] = name
                    break

        return account_info

    def _detect_statement_period(
        self,
        _image_path: str | Path | Image.Image,
    ) -> str | None:
        """Detect statement period from bank statement."""
        try:
            # This would typically look for "Statement Period" text
            # For now, return None - can be enhanced with specific OCR
            return None
        except Exception:
            return None

    def _extract_balances(
        self,
        _image_path: str | Path | Image.Image,
    ) -> tuple[float | None, float | None]:
        """Extract opening and closing balances."""
        try:
            # This would typically look for balance information
            # For now, return None - can be enhanced with specific OCR
            return None, None
        except Exception:
            return None, None

    def analyze_transaction_patterns(
        self,
        transactions: list[Transaction],
    ) -> dict[str, Any]:
        """Analyze patterns in transactions."""
        if not transactions:
            return {"error": "No transactions to analyze"}

        analysis = {
            "total_transactions": len(transactions),
            "total_debits": sum(
                1 for t in transactions if t.transaction_type == "debit"
            ),
            "total_credits": sum(
                1 for t in transactions if t.transaction_type == "credit"
            ),
            "total_debit_amount": sum(
                t.amount for t in transactions if t.transaction_type == "debit"
            ),
            "total_credit_amount": sum(
                t.amount for t in transactions if t.transaction_type == "credit"
            ),
            "average_transaction_amount": sum(t.amount for t in transactions)
            / len(transactions),
            "confidence_scores": {
                "average": sum(t.confidence for t in transactions) / len(transactions),
                "minimum": min(t.confidence for t in transactions),
                "maximum": max(t.confidence for t in transactions),
            },
        }

        # Work-related expense analysis
        work_keywords = [
            "uber",
            "taxi",
            "fuel",
            "parking",
            "toll",
            "hotel",
            "flight",
            "restaurant",
            "cafe",
            "office",
            "supplies",
            "equipment",
        ]

        work_related = []
        for transaction in transactions:
            description_lower = transaction.description.lower()
            if any(keyword in description_lower for keyword in work_keywords):
                work_related.append(transaction)

        analysis["work_related"] = {
            "count": len(work_related),
            "total_amount": sum(t.amount for t in work_related),
            "percentage": (len(work_related) / len(transactions)) * 100
            if transactions
            else 0,
        }

        return analysis

    def export_transaction_data(
        self,
        result: BankStatementResult,
        output_path: str | Path,
    ) -> Path:
        """Export transaction data to CSV format."""
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "date",
                "description",
                "amount",
                "transaction_type",
                "reference",
                "confidence",
                "highlight_x",
                "highlight_y",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for transaction in result.transactions:
                row = {
                    "date": transaction.date or "",
                    "description": transaction.description,
                    "amount": transaction.amount,
                    "transaction_type": transaction.transaction_type or "",
                    "reference": transaction.reference or "",
                    "confidence": f"{transaction.confidence:.3f}",
                    "highlight_x": transaction.highlight_region.x
                    if transaction.highlight_region
                    else "",
                    "highlight_y": transaction.highlight_region.y
                    if transaction.highlight_region
                    else "",
                }
                writer.writerow(row)

        logger.info(
            f"Exported {len(result.transactions)} transactions to {output_path}",
        )
        return output_path
