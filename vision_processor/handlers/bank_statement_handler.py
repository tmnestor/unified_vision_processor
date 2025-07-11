"""Bank Statement Handler

Specialized handler for bank statements following the Llama 7-step pipeline
with InternVL highlight detection integration and Australian banking expertise.
"""

import logging
import re
from typing import Any

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class BankStatementHandler(BaseATOHandler):
    """Handler for bank statements with Australian banking expertise and highlight integration.

    Supports major Australian banks:
    - Big Four: ANZ, Commonwealth Bank, Westpac, NAB
    - Regional: ING, Macquarie, Bendigo Bank, Suncorp
    - Credit Unions and smaller institutions

    Features:
    - InternVL highlight detection for transaction identification
    - Transaction categorization for business expenses
    - Work-related expense scoring and validation
    - BSB and account number validation
    - Australian banking compliance

    ATO Requirements:
    - Account holder name and details
    - Bank name and BSB
    - Statement period (from/to dates)
    - Opening and closing balances
    - Transaction details with business purpose
    - Work-related transaction identification
    """

    def _load_field_requirements(self) -> None:
        """Load bank statement specific field requirements."""
        self.required_fields = [
            "bank_name",
            "account_holder",
            "account_number",
            "statement_period_from",
            "statement_period_to",
        ]

        self.optional_fields = [
            "bsb",
            "opening_balance",
            "closing_balance",
            "transactions",
            "work_expenses",
            "total_debits",
            "total_credits",
            "account_type",
            "branch_details",
        ]

    def _load_validation_rules(self) -> None:
        """Load bank statement validation rules."""
        self.validation_rules = {
            "balance_range": (-100000.0, 1000000.0),  # Reasonable account balance range
            "transaction_amount_range": (
                -50000.0,
                50000.0,
            ),  # Reasonable transaction range
            "bsb_format": r"^\d{3}-\d{3}$",  # Australian BSB format
            "account_number_length": (6, 10),  # Australian account number length
            "work_expense_keywords": [
                "office supplies",
                "fuel",
                "parking",
                "toll",
                "uber",
                "taxi",
                "hotel",
                "accommodation",
                "conference",
                "training",
                "software",
                "equipment",
                "travel",
                "meal",
                "restaurant",
                "coffee",
                "subscription",
                "professional",
                "legal",
                "accounting",
            ],
        }

        # Australian bank patterns
        self.bank_patterns = {
            "ANZ": r"\b(?:anz|australia and new zealand)\b",
            "Commonwealth Bank": r"\b(?:commonwealth|cba|commbank)\b",
            "Westpac": r"\bwestpac\b",
            "NAB": r"\b(?:nab|national australia bank)\b",
            "ING": r"\bing\s*(?:direct|bank)?\b",
            "Macquarie": r"\bmacquarie\s*(?:bank)?\b",
            "Bendigo Bank": r"\bbendigo\s*(?:bank)?\b",
            "Suncorp": r"\bsuncorp\s*(?:bank)?\b",
            "Bank of Queensland": r"\b(?:boq|bank of queensland)\b",
            "HSBC": r"\bhsbc\b",
            "Citibank": r"\bcitibank\b",
        }

        # Transaction type patterns for work expense identification
        self.work_expense_patterns = {
            "fuel": r"\b(?:bp|shell|caltex|ampol|mobil|7-eleven|fuel|petrol)\b",
            "parking": r"\b(?:parking|secure parking|wilson|ace park)\b",
            "toll": r"\b(?:toll|citylink|eastlink|etag)\b",
            "transport": r"\b(?:uber|taxi|cab|train|bus|metro|opal)\b",
            "accommodation": r"\b(?:hotel|motel|accommodation|booking\.com|airbnb)\b",
            "meals": r"\b(?:restaurant|cafe|lunch|dinner|coffee|meal)\b",
            "office": r"\b(?:officeworks|staples|office supplies|stationery)\b",
            "professional": r"\b(?:legal|lawyer|accountant|consultant|professional)\b",
            "equipment": r"\b(?:computer|laptop|software|equipment|tools)\b",
            "communications": r"\b(?:telstra|optus|vodafone|phone|internet|mobile)\b",
        }

    def _extract_document_specific_fields(self, text: str) -> dict[str, Any]:
        """Extract bank statement specific fields."""
        fields = {}

        # Extract bank name
        bank_name = self._extract_bank_name(text)
        if bank_name:
            fields["bank_name"] = bank_name

        # Extract account holder
        account_holder = self._extract_account_holder(text)
        if account_holder:
            fields["account_holder"] = account_holder

        # Extract BSB
        bsb = self._extract_bsb(text)
        if bsb:
            fields["bsb"] = bsb

        # Extract account number
        account_number = self._extract_account_number(text)
        if account_number:
            fields["account_number"] = account_number

        # Extract statement period
        period_from, period_to = self._extract_statement_period(text)
        if period_from:
            fields["statement_period_from"] = period_from
        if period_to:
            fields["statement_period_to"] = period_to

        # Extract balances
        opening_balance = self._extract_opening_balance(text)
        if opening_balance is not None:
            fields["opening_balance"] = opening_balance

        closing_balance = self._extract_closing_balance(text)
        if closing_balance is not None:
            fields["closing_balance"] = closing_balance

        # Extract transactions
        transactions = self._extract_transactions(text)
        if transactions:
            fields["transactions"] = transactions

        # Identify work expenses
        work_expenses = self._identify_work_expenses(transactions)
        if work_expenses:
            fields["work_expenses"] = work_expenses

        # Extract account type
        account_type = self._extract_account_type(text)
        if account_type:
            fields["account_type"] = account_type

        return fields

    def _extract_bank_name(self, text: str) -> str:
        """Extract Australian bank name."""
        for bank_name, pattern in self.bank_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return bank_name

        # Look for "bank" in the first few lines
        lines = text.split("\n")
        for line in lines[:5]:
            if "bank" in line.lower():
                return line.strip()

        return ""

    def _extract_account_holder(self, text: str) -> str:
        """Extract account holder name."""
        holder_patterns = [
            r"(?:account holder|customer|name)\s*:?\s*([A-Za-z\s]+)",
            r"(?:mr|mrs|ms|dr|prof)\.?\s+([A-Za-z\s]+)",
        ]

        for pattern in holder_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2 and not re.search(r"\d", name):
                    return name

        return ""

    def _extract_bsb(self, text: str) -> str:
        """Extract BSB (Bank State Branch) code."""
        bsb_patterns = [
            r"(?:bsb|bank state branch)\s*:?\s*(\d{3}-\d{3})",
            r"(?:bsb|bank state branch)\s*:?\s*(\d{6})",
        ]

        for pattern in bsb_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                bsb = match.group(1)
                # Format as XXX-XXX if not already formatted
                if len(bsb) == 6 and "-" not in bsb:
                    bsb = f"{bsb[:3]}-{bsb[3:]}"
                return bsb

        return ""

    def _extract_account_number(self, text: str) -> str:
        """Extract account number."""
        account_patterns = [
            r"(?:account|acc)\s*(?:no\.?|number)\s*:?\s*(\d{6,10})",
            r"(?:account|acc)\s*:?\s*(\d{6,10})",
        ]

        for pattern in account_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return ""

    def _extract_statement_period(self, text: str) -> tuple[str, str]:
        """Extract statement period from and to dates."""
        period_patterns = [
            r"(?:statement period|period)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})\s*(?:to|-)\s*(\d{1,2}/\d{1,2}/\d{4})",
            r"(?:from|start)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4}).*?(?:to|end)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
        ]

        for pattern in period_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1), match.group(2)

        return "", ""

    def _extract_opening_balance(self, text: str) -> float:
        """Extract opening balance."""
        balance_patterns = [
            r"(?:opening|previous|brought forward)\s*(?:balance)?\s*:?\s*\$?\s*([\-\+]?\d+(?:,\d{3})*(?:\.\d{2})?)",
            r"(?:balance\s*brought\s*forward)\s*:?\s*\$?\s*([\-\+]?\d+(?:,\d{3})*(?:\.\d{2})?)",
        ]

        return self._extract_balance_amount(text, balance_patterns)

    def _extract_closing_balance(self, text: str) -> float:
        """Extract closing balance."""
        balance_patterns = [
            r"(?:closing|final|current)\s*(?:balance)?\s*:?\s*\$?\s*([\-\+]?\d+(?:,\d{3})*(?:\.\d{2})?)",
            r"(?:balance\s*carried\s*forward)\s*:?\s*\$?\s*([\-\+]?\d+(?:,\d{3})*(?:\.\d{2})?)",
        ]

        return self._extract_balance_amount(text, balance_patterns)

    def _extract_balance_amount(self, text: str, patterns: list[str]) -> float:
        """Extract balance amount using provided patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    amount_str = match.group(1).replace(",", "")
                    return float(amount_str)
                except ValueError:
                    continue

        return 0.0

    def _extract_transactions(self, text: str) -> list[dict[str, Any]]:
        """Extract transaction details from statement."""
        transactions = []
        lines = text.split("\n")

        # Transaction line pattern (date, description, amount)
        transaction_pattern = (
            r"^(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+(.{5,50}?)\s+([\-\+]?\$?\d+(?:\.\d{2})?)\s*$"
        )

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
                    transactions.append(
                        {
                            "date": date,
                            "description": description,
                            "amount": amount,
                            "type": "debit" if amount < 0 else "credit",
                        },
                    )
                except ValueError:
                    continue

        return transactions

    def _identify_work_expenses(
        self,
        transactions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Identify work-related expenses from transactions."""
        if not transactions:
            return []

        work_expenses = []

        for transaction in transactions:
            description = transaction.get("description", "").lower()
            amount = transaction.get("amount", 0)

            # Only consider debits (expenses)
            if amount >= 0:
                continue

            work_score = 0
            expense_category = "other"

            # Score transaction for work-relatedness
            for category, pattern in self.work_expense_patterns.items():
                if re.search(pattern, description, re.IGNORECASE):
                    work_score += 1
                    expense_category = category
                    break

            # Check for general work keywords
            for keyword in self.validation_rules["work_expense_keywords"]:
                if keyword in description:
                    work_score += 0.5

            # If score suggests work expense, add to list
            if work_score > 0:
                work_expense = transaction.copy()
                work_expense.update(
                    {
                        "work_score": work_score,
                        "expense_category": expense_category,
                        "confidence": min(work_score / 2.0, 1.0),  # Normalize to 0-1
                    },
                )
                work_expenses.append(work_expense)

        return work_expenses

    def _extract_account_type(self, text: str) -> str:
        """Extract account type."""
        account_type_patterns = [
            r"(?:account type|type)\s*:?\s*(savings?|cheque|checking|transaction|business|personal)",
            r"\b(savings?|cheque|transaction|business)\s*(?:account)?\b",
        ]

        for pattern in account_type_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        return ""

    def _validate_document_specific_fields(self, fields: dict[str, Any]) -> list[str]:
        """Validate bank statement specific fields."""
        issues = []

        # Validate BSB format
        if fields.get("bsb"):
            bsb = fields["bsb"]
            if not re.match(self.validation_rules["bsb_format"], bsb):
                issues.append(f"Invalid BSB format: {bsb} (should be XXX-XXX)")

        # Validate account number length
        if fields.get("account_number"):
            account_number = str(fields["account_number"])
            min_len, max_len = self.validation_rules["account_number_length"]
            if not (min_len <= len(account_number) <= max_len):
                issues.append(
                    f"Account number length {len(account_number)} outside range ({min_len}-{max_len})",
                )

        # Validate balance ranges
        for balance_field in ["opening_balance", "closing_balance"]:
            if balance_field in fields and fields[balance_field] is not None:
                try:
                    balance = float(fields[balance_field])
                    min_balance, max_balance = self.validation_rules["balance_range"]
                    if not (min_balance <= balance <= max_balance):
                        issues.append(
                            f"{balance_field.replace('_', ' ').title()} ${balance:,.2f} outside reasonable range",
                        )
                except (ValueError, TypeError):
                    issues.append(f"Invalid {balance_field.replace('_', ' ')} format")

        # Validate transaction amounts
        if fields.get("transactions"):
            for i, transaction in enumerate(fields["transactions"]):
                if "amount" in transaction:
                    try:
                        amount = float(transaction["amount"])
                        min_amount, max_amount = self.validation_rules["transaction_amount_range"]
                        if not (min_amount <= amount <= max_amount):
                            issues.append(
                                f"Transaction {i + 1} amount ${amount:,.2f} outside reasonable range",
                            )
                    except (ValueError, TypeError):
                        issues.append(f"Invalid amount format in transaction {i + 1}")

        # Validate statement period dates
        if all(
            field in fields and fields[field] for field in ["statement_period_from", "statement_period_to"]
        ):
            from_date = fields["statement_period_from"]
            to_date = fields["statement_period_to"]

            # Basic date format validation
            date_pattern = r"\d{1,2}/\d{1,2}/\d{4}"
            if not re.match(date_pattern, from_date):
                issues.append(f"Invalid statement period from date format: {from_date}")
            if not re.match(date_pattern, to_date):
                issues.append(f"Invalid statement period to date format: {to_date}")

        # Validate work expenses scoring
        if fields.get("work_expenses"):
            work_expenses = fields["work_expenses"]
            if len(work_expenses) == 0 and "transactions" in fields:
                # Might indicate missed work expenses
                logger.info("No work expenses identified - consider manual review")

        return issues

    def enhance_with_highlights(
        self,
        fields: dict[str, Any],
        highlights: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Enhance bank statement extraction using InternVL highlight detection.

        Bank statements benefit significantly from highlight detection for:
        - Transaction identification
        - Balance highlighting
        - Work expense identification
        """
        if not highlights or not self.supports_highlights:
            return fields

        enhanced_fields = fields.copy()
        logger.info(f"Enhancing bank statement with {len(highlights)} highlights")

        # Process highlights for transaction enhancement
        highlighted_transactions = []
        for highlight in highlights:
            if highlight.get("text"):
                highlight_text = highlight["text"]

                # Try to extract transaction from highlighted region
                transaction_data = self._extract_transactions(highlight_text)
                if transaction_data:
                    # Mark as highlighted for priority
                    for transaction in transaction_data:
                        transaction["highlighted"] = True
                        transaction["highlight_confidence"] = highlight.get(
                            "confidence",
                            0.8,
                        )
                    highlighted_transactions.extend(transaction_data)

                # Try to extract balance information
                balance_data = self._extract_balance_from_highlight(highlight_text)
                if balance_data:
                    enhanced_fields.update(balance_data)

        # Merge highlighted transactions with existing transactions
        if highlighted_transactions:
            existing_transactions = enhanced_fields.get("transactions", [])
            all_transactions = existing_transactions + highlighted_transactions

            # Remove duplicates based on date and amount
            unique_transactions = self._deduplicate_transactions(all_transactions)
            enhanced_fields["transactions"] = unique_transactions

            # Re-identify work expenses with highlighted transactions
            work_expenses = self._identify_work_expenses(unique_transactions)
            if work_expenses:
                enhanced_fields["work_expenses"] = work_expenses

        return enhanced_fields

    def _extract_balance_from_highlight(self, highlight_text: str) -> dict[str, Any]:
        """Extract balance information from highlighted text."""
        balance_data = {}

        # Check for opening balance
        opening_balance = self._extract_opening_balance(highlight_text)
        if opening_balance != 0.0:
            balance_data["opening_balance"] = opening_balance

        # Check for closing balance
        closing_balance = self._extract_closing_balance(highlight_text)
        if closing_balance != 0.0:
            balance_data["closing_balance"] = closing_balance

        return balance_data

    def _deduplicate_transactions(
        self,
        transactions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove duplicate transactions based on date and amount."""
        seen = set()
        unique_transactions = []

        for transaction in transactions:
            # Create key based on date and amount
            key = (transaction.get("date", ""), transaction.get("amount", 0))

            if key not in seen:
                seen.add(key)
                unique_transactions.append(transaction)
            # If duplicate, prefer highlighted version
            elif transaction.get("highlighted", False):
                # Replace existing with highlighted version
                for i, existing in enumerate(unique_transactions):
                    existing_key = (
                        existing.get("date", ""),
                        existing.get("amount", 0),
                    )
                    if existing_key == key:
                        unique_transactions[i] = transaction
                        break

        return unique_transactions
