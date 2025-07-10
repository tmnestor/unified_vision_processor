"""Transaction Categorizer - Australian Business Expense Classification

This module provides comprehensive transaction categorization for Australian
business expenses, identifying work-related transactions from bank statements
and categorizing them according to ATO expense types.

Features:
- Work expense pattern matching for 10+ categories
- Business expense scoring and confidence assessment
- ATO-aligned expense categorization
- Transaction deduplication and validation
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Represents a bank transaction."""

    date: str
    description: str
    amount: float
    transaction_type: str  # "debit" or "credit"


@dataclass
class WorkExpense:
    """Represents a categorized work expense."""

    transaction: Transaction
    category: str
    subcategory: str | None
    work_score: float
    confidence: float
    business_indicators: list[str]
    ato_category: str


class AustralianTransactionCategorizer:
    """Comprehensive transaction categorizer for Australian business expenses.

    Identifies and categorizes work-related expenses from bank statement
    transactions using pattern matching and business context analysis.
    """

    def __init__(self):
        self.initialized = False
        self.work_expense_patterns = {}
        self.business_keywords = []
        self.ato_categories = {}

    def initialize(self) -> None:
        """Initialize transaction categorization patterns."""
        if self.initialized:
            return

        # Work expense patterns for business transaction identification
        self.work_expense_patterns = {
            "fuel": {
                "patterns": [
                    r"\b(?:bp|shell|caltex|ampol|mobil|7-eleven|fuel|petrol|diesel|unleaded)\b"
                ],
                "ato_category": "Work-related car expenses",
                "confidence_weight": 0.9,
            },
            "parking": {
                "patterns": [
                    r"\b(?:parking|secure parking|wilson|ace park|meter|space)\b"
                ],
                "ato_category": "Work-related car expenses",
                "confidence_weight": 0.8,
            },
            "toll": {
                "patterns": [r"\b(?:toll|citylink|eastlink|etag|westconnex|tunnel)\b"],
                "ato_category": "Work-related car expenses",
                "confidence_weight": 0.9,
            },
            "transport": {
                "patterns": [
                    r"\b(?:uber|taxi|cab|train|bus|metro|opal|myki|translink)\b"
                ],
                "ato_category": "Work-related travel expenses",
                "confidence_weight": 0.7,
            },
            "accommodation": {
                "patterns": [
                    r"\b(?:hotel|motel|accommodation|booking\.com|airbnb|hilton|marriott)\b"
                ],
                "ato_category": "Work-related travel expenses",
                "confidence_weight": 0.8,
            },
            "meals": {
                "patterns": [
                    r"\b(?:restaurant|cafe|lunch|dinner|coffee|meal|food|catering)\b"
                ],
                "ato_category": "Meal entertainment expenses",
                "confidence_weight": 0.6,  # Lower confidence as personal meals are common
            },
            "office_supplies": {
                "patterns": [
                    r"\b(?:officeworks|staples|office supplies|stationery|paper|ink|pens)\b"
                ],
                "ato_category": "Office expenses",
                "confidence_weight": 0.8,
            },
            "professional_services": {
                "patterns": [
                    r"\b(?:legal|lawyer|accountant|consultant|professional|advisory|pwc|deloitte|kpmg)\b"
                ],
                "ato_category": "Professional fees",
                "confidence_weight": 0.9,
            },
            "equipment": {
                "patterns": [
                    r"\b(?:computer|laptop|software|equipment|tools|machinery|printer|scanner)\b"
                ],
                "ato_category": "Business equipment",
                "confidence_weight": 0.8,
            },
            "communications": {
                "patterns": [
                    r"\b(?:telstra|optus|vodafone|phone|internet|mobile|telecommunications)\b"
                ],
                "ato_category": "Phone and internet expenses",
                "confidence_weight": 0.7,
            },
            "insurance": {
                "patterns": [
                    r"\b(?:insurance|allianz|aami|nrma|professional indemnity|public liability)\b"
                ],
                "ato_category": "Insurance expenses",
                "confidence_weight": 0.8,
            },
            "banking": {
                "patterns": [
                    r"\b(?:bank fee|transaction fee|merchant fee|account fee|banking)\b"
                ],
                "ato_category": "Bank charges",
                "confidence_weight": 0.9,
            },
        }

        # Business context keywords that boost work expense confidence
        self.business_keywords = [
            "business",
            "company",
            "pty",
            "ltd",
            "enterprise",
            "corporate",
            "office",
            "work",
            "professional",
            "commercial",
            "industrial",
            "client",
            "customer",
            "project",
            "contract",
            "invoice",
            "meeting",
            "conference",
            "training",
            "workshop",
            "seminar",
            "subscription",
            "license",
            "software",
            "service",
            "maintenance",
        ]

        # ATO expense category metadata
        self.ato_categories = {
            "Work-related car expenses": {
                "description": "Vehicle expenses for work purposes",
                "deductible": True,
                "requires_logbook": False,
                "gst_applicable": True,
            },
            "Work-related travel expenses": {
                "description": "Travel expenses for work purposes",
                "deductible": True,
                "requires_receipts": True,
                "gst_applicable": True,
            },
            "Meal entertainment expenses": {
                "description": "Business meals and entertainment",
                "deductible": "limited",  # 50% deductible
                "requires_business_purpose": True,
                "gst_applicable": True,
            },
            "Office expenses": {
                "description": "Office supplies and equipment",
                "deductible": True,
                "requires_receipts": True,
                "gst_applicable": True,
            },
            "Professional fees": {
                "description": "Legal, accounting, consulting fees",
                "deductible": True,
                "requires_invoices": True,
                "gst_applicable": True,
            },
            "Business equipment": {
                "description": "Equipment and tools for business use",
                "deductible": True,
                "depreciation_applicable": True,
                "gst_applicable": True,
            },
            "Phone and internet expenses": {
                "description": "Communication expenses for work",
                "deductible": "partial",  # Work percentage only
                "requires_usage_records": True,
                "gst_applicable": True,
            },
            "Insurance expenses": {
                "description": "Business insurance premiums",
                "deductible": True,
                "requires_policy_documents": True,
                "gst_applicable": True,
            },
            "Bank charges": {
                "description": "Business banking fees and charges",
                "deductible": True,
                "requires_statements": True,
                "gst_applicable": False,
            },
        }

        logger.info(
            "AustralianTransactionCategorizer initialized with 12 expense categories"
        )
        self.initialized = True

    def extract_transactions(self, text: str) -> list[Transaction]:
        """Extract transaction details from bank statement text.

        Args:
            text: Bank statement text content

        Returns:
            List of Transaction objects

        """
        transactions = []
        lines = text.split("\n")

        # Australian transaction line pattern (date, description, amount)
        # Supports DD/MM/YY and DD/MM/YYYY formats
        transaction_pattern = r"^(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+(.{5,50}?)\s+([\-\+]?\$?\d+(?:\.\d{2})?)\s*$"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(transaction_pattern, line)
            if match:
                date = match.group(1)
                description = match.group(2).strip()
                amount_str = match.group(3).replace("$", "").replace(",", "")

                try:
                    amount = float(amount_str)
                    transaction = Transaction(
                        date=date,
                        description=description,
                        amount=amount,
                        transaction_type="debit" if amount < 0 else "credit",
                    )
                    transactions.append(transaction)
                except ValueError:
                    continue

        return transactions

    def identify_work_expenses(
        self, transactions: list[Transaction]
    ) -> list[WorkExpense]:
        """Identify and categorize work-related expenses from transactions.

        Args:
            transactions: List of Transaction objects

        Returns:
            List of WorkExpense objects for identified business expenses

        """
        self.initialize()

        if not transactions:
            return []

        work_expenses = []

        for transaction in transactions:
            # Only consider debits (expenses)
            if transaction.amount >= 0:
                continue

            work_expense = self._categorize_transaction(transaction)
            if (
                work_expense and work_expense.work_score >= 0.3
            ):  # Minimum confidence threshold
                work_expenses.append(work_expense)

        # Deduplicate similar transactions
        work_expenses = self._deduplicate_expenses(work_expenses)

        # Sort by confidence (highest first)
        work_expenses.sort(key=lambda x: x.confidence, reverse=True)

        return work_expenses

    def _categorize_transaction(self, transaction: Transaction) -> WorkExpense | None:
        """Categorize a single transaction for work expense classification."""
        description = transaction.description.lower()
        work_score = 0.0
        best_category = "other"
        best_subcategory = None
        business_indicators = []

        # Score transaction against work expense patterns
        max_category_score = 0.0

        for category, config in self.work_expense_patterns.items():
            category_score = 0.0

            for pattern in config["patterns"]:
                if re.search(pattern, description, re.IGNORECASE):
                    pattern_score = config["confidence_weight"]
                    category_score = max(category_score, pattern_score)

            if category_score > max_category_score:
                max_category_score = category_score
                best_category = category
                work_score = category_score

        # Boost score for business context keywords
        business_score = 0.0
        for keyword in self.business_keywords:
            if keyword in description:
                business_score += 0.1
                business_indicators.append(keyword)

        # Apply business context boost (max 0.3 additional points)
        business_boost = min(business_score, 0.3)
        work_score += business_boost

        # Calculate final confidence
        confidence = min(work_score, 1.0)

        # Get ATO category
        category_config = self.work_expense_patterns.get(best_category, {})
        ato_category = category_config.get("ato_category", "Other business expenses")

        # Return work expense if meets minimum threshold
        if confidence >= 0.3:
            return WorkExpense(
                transaction=transaction,
                category=best_category,
                subcategory=best_subcategory,
                work_score=work_score,
                confidence=confidence,
                business_indicators=business_indicators,
                ato_category=ato_category,
            )

        return None

    def _deduplicate_expenses(
        self, work_expenses: list[WorkExpense]
    ) -> list[WorkExpense]:
        """Remove duplicate or very similar work expenses."""
        if len(work_expenses) <= 1:
            return work_expenses

        deduplicated = []
        seen_transactions = set()

        for expense in work_expenses:
            # Create a signature for deduplication
            signature = (
                expense.transaction.date,
                expense.transaction.description[:20],  # First 20 chars
                abs(expense.transaction.amount),
            )

            if signature not in seen_transactions:
                seen_transactions.add(signature)
                deduplicated.append(expense)

        return deduplicated

    def get_expense_summary(self, work_expenses: list[WorkExpense]) -> dict[str, Any]:
        """Generate summary statistics for work expenses."""
        if not work_expenses:
            return {
                "total_expenses": 0,
                "total_amount": 0.0,
                "categories": {},
                "ato_categories": {},
                "average_confidence": 0.0,
            }

        total_amount = sum(abs(expense.transaction.amount) for expense in work_expenses)
        categories = {}
        ato_categories = {}

        for expense in work_expenses:
            # Category breakdown
            if expense.category not in categories:
                categories[expense.category] = {"count": 0, "amount": 0.0}
            categories[expense.category]["count"] += 1
            categories[expense.category]["amount"] += abs(expense.transaction.amount)

            # ATO category breakdown
            if expense.ato_category not in ato_categories:
                ato_categories[expense.ato_category] = {"count": 0, "amount": 0.0}
            ato_categories[expense.ato_category]["count"] += 1
            ato_categories[expense.ato_category]["amount"] += abs(
                expense.transaction.amount
            )

        average_confidence = sum(expense.confidence for expense in work_expenses) / len(
            work_expenses
        )

        return {
            "total_expenses": len(work_expenses),
            "total_amount": total_amount,
            "categories": categories,
            "ato_categories": ato_categories,
            "average_confidence": average_confidence,
        }

    def get_supported_categories(self) -> list[str]:
        """Get list of supported expense categories."""
        self.initialize()
        return list(self.work_expense_patterns.keys())

    def get_ato_category_info(self, ato_category: str) -> dict[str, Any]:
        """Get ATO category metadata and requirements."""
        self.initialize()
        return self.ato_categories.get(ato_category, {})
