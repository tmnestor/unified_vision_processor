"""Australian Tax Office Field Validators

This module provides comprehensive validation for Australian business
and tax-related fields including ABN, BSB, GST calculations, and dates.
"""

import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Generic validation result."""

    is_valid: bool
    errors: list[str]


@dataclass
class DateValidationResult(ValidationResult):
    """Date validation result."""

    parsed_date: datetime | None = None
    normalized_date: str | None = None


@dataclass
class AmountValidationResult(ValidationResult):
    """Amount validation result."""

    parsed_amount: float | None = None
    normalized_amount: str | None = None


class ABNValidator:
    """Australian Business Number (ABN) validator with checksum verification.

    Features:
    - Format validation (11 digits)
    - Checksum algorithm verification
    - Formatting and normalization
    """

    def __init__(self):
        # ABN checksum weights
        self.weights = [10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    def validate(self, abn: str) -> tuple[bool, str, list[str]]:
        """Validate ABN format and checksum.

        Args:
            abn: ABN string to validate

        Returns:
            Tuple of (is_valid, formatted_abn, validation_issues)

        """
        issues = []

        if not abn or not abn.strip():
            return False, "", ["ABN is required"]

        # Clean ABN (remove spaces, hyphens)
        clean_abn = re.sub(r"[^\d]", "", abn.strip())

        # Check length
        if len(clean_abn) != 11:
            issues.append(f"ABN must be 11 digits, got {len(clean_abn)}")
            return False, clean_abn, issues

        # Validate checksum
        if not self._validate_checksum(clean_abn):
            issues.append("ABN checksum validation failed")
            return False, clean_abn, issues

        # Format ABN (XX XXX XXX XXX)
        formatted = f"{clean_abn[:2]} {clean_abn[2:5]} {clean_abn[5:8]} {clean_abn[8:]}"

        return True, formatted, []

    def _normalize_abn(self, abn: str) -> str:
        """Normalize ABN format for testing."""
        if not abn:
            return ""
        # Remove all non-digits
        clean_abn = re.sub(r"[^\d]", "", abn.strip())
        return clean_abn

    def _validate_checksum(self, abn: str) -> bool:
        """Validate ABN using the official checksum algorithm."""
        try:
            # Ensure ABN is exactly 11 digits
            if len(abn) != 11 or not abn.isdigit():
                return False

            # Convert to list of integers
            digits = [int(d) for d in abn]

            # Apply ABN checksum algorithm:
            # 1. Subtract 1 from the first digit
            first_digit = digits[0] - 1

            # Handle case where first digit becomes -1 (for ABNs starting with 0)
            if first_digit < 0:
                return False

            # 2. Calculate weighted sum: first digit * 10 + remaining digits * weights
            weighted_sum = first_digit * 10

            # Add products of remaining 10 digits with their weights [1,3,5,7,9,11,13,15,17,19]
            remaining_weights = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
            for i in range(10):
                weighted_sum += digits[i + 1] * remaining_weights[i]

            # 3. Check if divisible by 89
            return weighted_sum % 89 == 0

        except (ValueError, IndexError):
            return False


class BSBValidator:
    """Bank State Branch (BSB) validator for Australian banks.

    Features:
    - Format validation (XXX-XXX)
    - Known bank range validation
    - Institution identification
    """

    def __init__(self):
        # Major Australian bank BSB ranges
        self.bank_ranges = {
            "ANZ": [(10, 19), (70, 79), (550, 559)],
            "Commonwealth Bank": [(60, 69), (730, 739)],
            "Westpac": [(30, 39), (340, 349), (730, 789)],
            "NAB": [(80, 89), (300, 319)],
            "Bendigo Bank": [(630, 639)],
            "Suncorp": [(484, 484), (640, 649)],
            "ING": [(923, 923)],
            "Macquarie": [(182, 182)],
            "Bank of Queensland": [(124, 124)],
        }

    def validate(self, bsb: str) -> tuple[bool, str, list[str], str | None]:
        """Validate BSB format and identify institution.

        Args:
            bsb: BSB string to validate

        Returns:
            Tuple of (is_valid, formatted_bsb, validation_issues, bank_name)

        """
        issues = []

        if not bsb or not bsb.strip():
            return False, "", ["BSB is required"], None

        # Clean BSB (remove spaces, hyphens)
        clean_bsb = re.sub(r"[^\d]", "", bsb.strip())

        # Check length
        if len(clean_bsb) != 6:
            issues.append(f"BSB must be 6 digits, got {len(clean_bsb)}")
            return False, clean_bsb, issues, None

        # Format BSB (XXX-XXX)
        formatted = f"{clean_bsb[:3]}-{clean_bsb[3:]}"

        # Identify bank
        bank_name = self._identify_bank(clean_bsb)

        if not bank_name:
            issues.append("BSB not recognized as valid Australian bank")

        return len(issues) == 0, formatted, issues, bank_name

    def _identify_bank(self, bsb: str) -> str | None:
        """Identify bank from BSB prefix."""
        try:
            prefix = int(bsb[:3])

            for bank, ranges in self.bank_ranges.items():
                for start, end in ranges:
                    if start <= prefix <= end:
                        return bank

            return None

        except ValueError:
            return None


class DateValidator:
    """Australian date format validator with business context.

    Features:
    - DD/MM/YYYY format validation
    - Business day validation
    - Financial year context
    - Reasonable date range checking
    """

    def __init__(self):
        self.supported_formats = [
            "%d/%m/%Y",  # 01/01/2023
            "%d-%m-%Y",  # 01-01-2023
            "%d %m %Y",  # 01 01 2023
            "%d.%m.%Y",  # 01.01.2023
            "%d/%m/%y",  # 01/01/23
            "%d-%m-%y",  # 01-01-23
            "%d %b %Y",  # 25 Mar 2024
            "%d %B %Y",  # 25 March 2024
        ]

    def validate(
        self,
        date_str: str,
    ) -> tuple[bool, datetime | None, str, list[str]]:
        """Validate Australian date format.

        Args:
            date_str: Date string to validate

        Returns:
            Tuple of (is_valid, parsed_date, formatted_date, validation_issues)

        """
        issues = []

        if not date_str or not date_str.strip():
            return False, None, "", ["Date is required"]

        # Try to parse with different formats
        parsed_date = None
        for date_format in self.supported_formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), date_format)
                break
            except ValueError:
                continue

        if not parsed_date:
            issues.append(f"Invalid date format. Expected DD/MM/YYYY, got: {date_str}")
            return False, None, date_str, issues

        # Check reasonable date range (1950 to current year + 2)
        current_year = datetime.now().year
        if parsed_date.year < 1950 or parsed_date.year > current_year + 2:
            issues.append(
                f"Date year {parsed_date.year} outside reasonable range (1950-{current_year + 2})",
            )

        # Format as Australian standard (DD/MM/YYYY)
        formatted = parsed_date.strftime("%d/%m/%Y")

        return len(issues) == 0, parsed_date, formatted, issues

    def get_financial_year(self, date: datetime) -> str:
        """Get Australian financial year (July 1 - June 30)."""
        if date.month >= 7:  # July-December
            return f"{date.year}-{date.year + 1}"
        # January-June
        return f"{date.year - 1}-{date.year}"

    def validate_format(self, date_str: str) -> DateValidationResult:
        """Simplified date format validation for tests."""
        is_valid, parsed_date, formatted, issues = self.validate(date_str)
        return DateValidationResult(
            is_valid=is_valid,
            errors=issues,
            parsed_date=parsed_date,
            normalized_date=formatted,
        )


class GSTValidator:
    """Goods and Services Tax (GST) validator for Australian tax calculations.

    Features:
    - 10% GST rate validation
    - GST-inclusive/exclusive calculations
    - Rounding verification
    - Business compliance checks
    """

    def __init__(self):
        self.gst_rate = 0.10  # 10% GST rate
        self.tolerance = 0.005  # 0.5 cent tolerance for rounding

    def validate_gst_calculation(
        self,
        subtotal: float,
        gst_amount: float,
        total: float,
    ) -> tuple[bool, dict[str, float], list[str]]:
        """Validate GST calculation correctness.

        Args:
            subtotal: Subtotal amount (excluding GST)
            gst_amount: GST amount
            total: Total amount (including GST)

        Returns:
            Tuple of (is_valid, calculated_values, validation_issues)

        """
        issues = []

        # Calculate expected values
        expected_gst = subtotal * self.gst_rate
        expected_total = subtotal + expected_gst

        calculated_values = {
            "expected_gst": round(expected_gst, 2),
            "expected_total": round(expected_total, 2),
            "provided_gst": gst_amount,
            "provided_total": total,
            "gst_difference": abs(gst_amount - expected_gst),
            "total_difference": abs(total - expected_total),
        }

        # Check if amounts are positive
        if subtotal < 0:
            issues.append("Subtotal cannot be negative")
        if gst_amount < 0:
            issues.append("GST amount cannot be negative")
        if total < 0:
            issues.append("Total amount cannot be negative")

        # Handle special case: No GST (GST-exempt or under threshold)
        if gst_amount == 0.0 and total == subtotal:
            # This is valid - business may be GST-exempt or under $75k threshold
            return True, calculated_values, issues

        # Validate GST amount (with tolerance for rounding)
        if abs(gst_amount - expected_gst) > self.tolerance:
            issues.append(
                f"GST amount {gst_amount:.2f} does not match expected "
                f"{expected_gst:.2f} (10% of {subtotal:.2f})",
            )

        # Validate total (with tolerance for rounding)
        if abs(total - expected_total) > self.tolerance:
            issues.append(
                f"Total amount {total:.2f} does not match expected {expected_total:.2f} (subtotal + GST)",
            )

        return len(issues) == 0, calculated_values, issues

    def extract_gst_from_total(self, total_including_gst: float) -> dict[str, float]:
        """Extract GST and subtotal from GST-inclusive amount.

        Args:
            total_including_gst: Total amount including GST

        Returns:
            Dictionary with subtotal, gst_amount, and calculations

        """
        # GST = Total × (GST rate / (1 + GST rate))
        gst_amount = total_including_gst * (self.gst_rate / (1 + self.gst_rate))
        subtotal = total_including_gst - gst_amount

        return {
            "total_including_gst": total_including_gst,
            "subtotal": round(subtotal, 2),
            "gst_amount": round(gst_amount, 2),
            "gst_rate": self.gst_rate,
            "calculation_method": "gst_inclusive",
        }

    def validate_business_gst_registration(
        self,
        annual_turnover: float | None,
    ) -> tuple[bool, list[str]]:
        """Validate if business should be GST registered based on turnover.

        Args:
            annual_turnover: Annual business turnover

        Returns:
            Tuple of (should_be_registered, compliance_notes)

        """
        notes = []

        if annual_turnover is None:
            notes.append(
                "Cannot determine GST registration requirement without turnover data",
            )
            return False, notes

        # GST registration threshold in Australia
        gst_threshold = 75000.0  # $75,000 for most businesses

        if annual_turnover >= gst_threshold:
            notes.append(
                f"Business must be GST registered (turnover ${annual_turnover:,.2f} ≥ ${gst_threshold:,.2f})",
            )
            return True, notes
        notes.append(
            f"GST registration optional (turnover ${annual_turnover:,.2f} < ${gst_threshold:,.2f})",
        )
        return False, notes

    def validate_calculation(self, subtotal: float, gst: float, total: float) -> bool:
        """Simplified GST calculation validation for tests."""
        # Handle special case: No GST (GST-exempt or under threshold)
        if gst == 0.0 and total == subtotal:
            return True

        # Calculate expected GST (10% of subtotal) and round to 2 decimal places
        expected_gst_raw = subtotal * self.gst_rate
        expected_gst = round(expected_gst_raw, 2)
        expected_total = subtotal + expected_gst

        # Check GST amount with tolerance (compare against rounded expected value)
        gst_valid = abs(gst - expected_gst) <= self.tolerance
        # Check total amount with tolerance
        total_valid = abs(total - expected_total) <= self.tolerance

        return gst_valid and total_valid


class AmountValidator:
    """Australian currency amount validator.

    Features:
    - Currency symbol handling ($)
    - Thousands separator support (,)
    - Decimal precision validation
    - Negative amount handling
    """

    def __init__(self):
        # Currency patterns - more restrictive for proper thousands separator placement
        # Matches: $1,234.56 or 1234.56 or $1234 but not $1,23.45
        self.currency_pattern = re.compile(r"^-?\$?(\d{1,3}(,\d{3})*|\d+)(\.\d{1,2})?$")

    def validate(self, amount_str: str) -> tuple[bool, float | None, str, list[str]]:
        """Validate and parse Australian currency amount.

        Args:
            amount_str: Amount string to validate

        Returns:
            Tuple of (is_valid, parsed_amount, formatted_amount, validation_issues)
        """
        issues = []

        if not amount_str or not amount_str.strip():
            return False, None, "", ["Amount is required"]

        # Clean the amount string
        clean_amount = amount_str.strip()

        # Check basic format
        if not self.currency_pattern.match(clean_amount):
            issues.append("Invalid amount format")
            return False, None, clean_amount, issues

        try:
            # Remove currency symbol and commas for parsing
            numeric_str = clean_amount.replace("$", "").replace(",", "")
            parsed_amount = float(numeric_str)

            # For ATO compliance, negative amounts are not allowed in general receipts
            if parsed_amount < 0:
                issues.append("Amount cannot be negative")
                # Return parsed but invalid
                formatted_amount = f"${abs(parsed_amount):,.2f}"
                return False, abs(parsed_amount), formatted_amount, issues

            # Format as currency
            formatted_amount = f"${parsed_amount:,.2f}"

            return True, parsed_amount, formatted_amount, issues

        except (ValueError, TypeError):
            issues.append("Cannot parse amount as number")
            return False, None, clean_amount, issues

    def validate_and_parse(self, amount_str: str) -> AmountValidationResult:
        """Validate and parse amount - simplified interface for tests."""
        is_valid, parsed_amount, formatted, issues = self.validate(amount_str)
        return AmountValidationResult(
            is_valid=is_valid,
            errors=issues,
            parsed_amount=parsed_amount,
            normalized_amount=formatted,
        )
