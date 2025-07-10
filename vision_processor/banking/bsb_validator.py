"""BSB Validator - Australian Bank State Branch Validation

This module provides comprehensive validation of Australian Bank State Branch (BSB)
numbers with format checking, bank identification, and range validation for
11 major Australian banking institutions.

Features:
- BSB format validation (XXX-XXX pattern)
- Bank identification from BSB ranges
- Institution categorization and metadata
- Validation error reporting and suggestions
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BSBValidationResult:
    """Result of BSB validation with detailed information."""

    is_valid: bool
    formatted_bsb: str
    validation_issues: list[str]
    bank_name: str | None
    bank_category: str | None
    suggestions: list[str]


class AustralianBSBValidator:
    """Comprehensive BSB validator for Australian banks.

    Validates Bank State Branch numbers according to Australian banking
    standards and identifies the associated financial institution.

    Features:
    - Format validation (XXX-XXX)
    - Known bank range validation
    - Institution identification
    - Validation error reporting
    - Format correction suggestions
    """

    def __init__(self):
        # Major Australian bank BSB ranges
        self.bank_ranges = {
            "ANZ": {
                "ranges": [(10, 19), (70, 79), (550, 559)],
                "category": "big_four",
                "official_name": "Australia and New Zealand Banking Group",
            },
            "Commonwealth Bank": {
                "ranges": [(60, 69), (730, 739)],
                "category": "big_four",
                "official_name": "Commonwealth Bank of Australia",
            },
            "Westpac": {
                "ranges": [(30, 39), (340, 349), (730, 789)],
                "category": "big_four",
                "official_name": "Westpac Banking Corporation",
            },
            "NAB": {
                "ranges": [(80, 89), (300, 319)],
                "category": "big_four",
                "official_name": "National Australia Bank",
            },
            "Bendigo Bank": {
                "ranges": [(630, 639)],
                "category": "regional",
                "official_name": "Bendigo and Adelaide Bank",
            },
            "Suncorp": {
                "ranges": [(484, 484), (640, 649)],
                "category": "regional",
                "official_name": "Suncorp Bank",
            },
            "ING": {
                "ranges": [(923, 923)],
                "category": "online",
                "official_name": "ING Australia",
            },
            "Macquarie": {
                "ranges": [(182, 182)],
                "category": "regional",
                "official_name": "Macquarie Bank",
            },
            "Bank of Queensland": {
                "ranges": [(124, 124)],
                "category": "regional",
                "official_name": "Bank of Queensland",
            },
            "HSBC": {
                "ranges": [(342, 342)],
                "category": "international",
                "official_name": "HSBC Bank Australia",
            },
            "Citibank": {
                "ranges": [(243, 243)],
                "category": "international",
                "official_name": "Citibank Australia",
            },
        }

        logger.info("AustralianBSBValidator initialized with 11 major banks")

    def validate(self, bsb: str) -> BSBValidationResult:
        """Validate BSB format and identify institution.

        Args:
            bsb: BSB string to validate

        Returns:
            BSBValidationResult with comprehensive validation information

        """
        issues = []
        suggestions = []

        if not bsb or not bsb.strip():
            return BSBValidationResult(
                is_valid=False,
                formatted_bsb="",
                validation_issues=["BSB is required"],
                bank_name=None,
                bank_category=None,
                suggestions=["Please provide a 6-digit BSB number"],
            )

        # Clean BSB (remove spaces, hyphens, and other non-digit characters)
        clean_bsb = re.sub(r"[^\d]", "", bsb.strip())

        # Check length
        if len(clean_bsb) == 0:
            return BSBValidationResult(
                is_valid=False,
                formatted_bsb="",
                validation_issues=["BSB contains no digits"],
                bank_name=None,
                bank_category=None,
                suggestions=["BSB should be 6 digits (e.g., 123-456)"],
            )
        if len(clean_bsb) < 6:
            issues.append(f"BSB too short: {len(clean_bsb)} digits (expected 6)")
            suggestions.append(f"Add {6 - len(clean_bsb)} more digits to complete BSB")
        elif len(clean_bsb) > 6:
            issues.append(f"BSB too long: {len(clean_bsb)} digits (expected 6)")
            suggestions.append("Remove extra digits - BSB should be exactly 6 digits")
            # Truncate to 6 digits for further processing
            clean_bsb = clean_bsb[:6]

        # Format BSB (XXX-XXX)
        if len(clean_bsb) == 6:
            formatted = f"{clean_bsb[:3]}-{clean_bsb[3:]}"
        else:
            formatted = clean_bsb

        # Identify bank and category
        bank_name = None
        bank_category = None

        if len(clean_bsb) >= 3:  # Need at least 3 digits for bank identification
            bank_info = self._identify_bank(clean_bsb)
            if bank_info:
                bank_name = bank_info["name"]
                bank_category = bank_info["category"]
            else:
                issues.append("BSB not recognized as valid Australian bank")
                suggestions.extend(
                    [
                        "Verify BSB with your bank",
                        "Check for typos in the BSB number",
                        "Ensure BSB belongs to an Australian bank",
                    ]
                )

        # Additional format validation
        if len(clean_bsb) == 6:
            # Check for obviously invalid patterns
            if clean_bsb == "000000":
                issues.append("Invalid BSB: 000-000 is not a valid bank code")
                suggestions.append("Use your bank's actual BSB number")
            elif clean_bsb == "123456":
                issues.append("BSB appears to be a placeholder (123-456)")
                suggestions.append("Replace with your bank's actual BSB number")

        is_valid = len(issues) == 0

        return BSBValidationResult(
            is_valid=is_valid,
            formatted_bsb=formatted,
            validation_issues=issues,
            bank_name=bank_name,
            bank_category=bank_category,
            suggestions=suggestions if not is_valid else [],
        )

    def _identify_bank(self, bsb: str) -> dict | None:
        """Identify bank from BSB prefix."""
        try:
            if len(bsb) < 3:
                return None

            prefix = int(bsb[:3])

            for bank, config in self.bank_ranges.items():
                for start, end in config["ranges"]:
                    if start <= prefix <= end:
                        return {
                            "name": bank,
                            "category": config["category"],
                            "official_name": config["official_name"],
                        }

            return None

        except ValueError:
            return None

    def format_bsb(self, bsb: str) -> str:
        """Format BSB into standard XXX-XXX pattern."""
        clean_bsb = re.sub(r"[^\d]", "", bsb.strip())

        if len(clean_bsb) == 6:
            return f"{clean_bsb[:3]}-{clean_bsb[3:]}"
        return clean_bsb

    def is_valid_format(self, bsb: str) -> bool:
        """Quick check if BSB has valid format."""
        clean_bsb = re.sub(r"[^\d]", "", bsb.strip())
        return len(clean_bsb) == 6 and clean_bsb.isdigit()

    def get_bank_info(self, bsb: str) -> dict | None:
        """Get detailed bank information from BSB."""
        clean_bsb = re.sub(r"[^\d]", "", bsb.strip())
        bank_info = self._identify_bank(clean_bsb)

        if bank_info:
            bank_name = bank_info["name"]
            config = self.bank_ranges[bank_name]

            return {
                "bank_name": bank_name,
                "official_name": config["official_name"],
                "category": config["category"],
                "bsb_ranges": config["ranges"],
                "formatted_bsb": self.format_bsb(bsb),
            }

        return None

    def get_supported_banks(self) -> list[str]:
        """Get list of all supported Australian banks."""
        return list(self.bank_ranges.keys())

    def get_bank_by_category(self, category: str) -> list[str]:
        """Get banks by category (big_four, regional, online, international)."""
        return [
            bank
            for bank, config in self.bank_ranges.items()
            if config["category"] == category
        ]

    def is_big_four_bsb(self, bsb: str) -> bool:
        """Check if BSB belongs to one of Australia's Big Four banks."""
        bank_info = self.get_bank_info(bsb)
        return bank_info and bank_info["category"] == "big_four"

    def validate_multiple(self, bsbs: list[str]) -> list[BSBValidationResult]:
        """Validate multiple BSBs at once."""
        return [self.validate(bsb) for bsb in bsbs]

    def suggest_corrections(self, bsb: str) -> list[str]:
        """Suggest corrections for invalid BSB."""
        suggestions = []
        clean_bsb = re.sub(r"[^\d]", "", bsb.strip())

        if not clean_bsb:
            suggestions.append("Enter a 6-digit BSB number")
            return suggestions

        # Length-based suggestions
        if len(clean_bsb) < 6:
            suggestions.append(f"Add {6 - len(clean_bsb)} more digits")
        elif len(clean_bsb) > 6:
            suggestions.append("Remove extra digits")
            clean_bsb = clean_bsb[:6]  # Use first 6 digits for further analysis

        # If we have 6 digits but no bank match, suggest checking with known banks
        if len(clean_bsb) == 6 and not self._identify_bank(clean_bsb):
            prefix = int(clean_bsb[:3])

            # Find closest bank ranges
            closest_banks = []
            for bank, config in self.bank_ranges.items():
                for start, end in config["ranges"]:
                    distance = min(abs(prefix - start), abs(prefix - end))
                    if distance <= 20:  # Within reasonable range
                        closest_banks.append((bank, distance))

            # Sort by distance and suggest closest banks
            closest_banks.sort(key=lambda x: x[1])
            if closest_banks:
                suggestions.append(
                    f"Check if this BSB belongs to {closest_banks[0][0]}"
                )

        return suggestions
