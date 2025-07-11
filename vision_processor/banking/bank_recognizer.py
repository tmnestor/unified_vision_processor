"""Australian Bank Recognition Module

This module provides comprehensive recognition and identification of Australian banks
from text content, supporting 11 major banking institutions with pattern matching
and confidence scoring.

Features:
- Big Four bank recognition (ANZ, Commonwealth, Westpac, NAB)
- Regional and online bank identification (ING, Macquarie, Bendigo, etc.)
- Pattern-based matching with confidence scoring
- Institution metadata and categorization
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BankMatch:
    """Represents a recognized Australian bank."""

    official_name: str
    common_names: list[str]
    confidence: float
    category: str  # "big_four", "regional", "online", "credit_union"
    bsb_ranges: list[tuple[int, int]]


class AustralianBankRecognizer:
    """Comprehensive Australian bank recognition system.

    Recognizes major Australian banking institutions from text content
    using pattern matching and provides confidence-scored results.
    """

    def __init__(self):
        self.initialized = False
        self.bank_patterns = {}
        self.bank_metadata = {}

    def initialize(self) -> None:
        """Initialize bank recognition patterns and metadata."""
        if self.initialized:
            return

        # Australian bank recognition patterns
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

        # Bank metadata and categorization
        self.bank_metadata = {
            "ANZ": {
                "official_name": "ANZ Bank",
                "common_names": ["ANZ", "Australia and New Zealand"],
                "category": "big_four",
                "bsb_ranges": [(10, 19), (70, 79), (550, 559)],
                "market_share": "large",
            },
            "Commonwealth Bank": {
                "official_name": "Commonwealth Bank of Australia",
                "common_names": ["CBA", "CommBank", "Commonwealth"],
                "category": "big_four",
                "bsb_ranges": [(60, 69), (730, 739)],
                "market_share": "large",
            },
            "Westpac": {
                "official_name": "Westpac Banking Corporation",
                "common_names": ["Westpac", "WBC"],
                "category": "big_four",
                "bsb_ranges": [(30, 39), (340, 349), (730, 789)],
                "market_share": "large",
            },
            "NAB": {
                "official_name": "NAB Bank",
                "common_names": ["NAB", "National Australia Bank"],
                "category": "big_four",
                "bsb_ranges": [(80, 89), (300, 319)],
                "market_share": "large",
            },
            "ING": {
                "official_name": "ING Australia",
                "common_names": ["ING", "ING Direct"],
                "category": "online",
                "bsb_ranges": [(923, 923)],
                "market_share": "medium",
            },
            "Macquarie": {
                "official_name": "Macquarie Bank",
                "common_names": ["Macquarie", "Macquarie Bank"],
                "category": "regional",
                "bsb_ranges": [(182, 182)],
                "market_share": "medium",
            },
            "Bendigo Bank": {
                "official_name": "Bendigo and Adelaide Bank",
                "common_names": ["Bendigo", "Bendigo Bank"],
                "category": "regional",
                "bsb_ranges": [(630, 639)],
                "market_share": "small",
            },
            "Suncorp": {
                "official_name": "Suncorp Bank",
                "common_names": ["Suncorp", "Suncorp Bank"],
                "category": "regional",
                "bsb_ranges": [(484, 484), (640, 649)],
                "market_share": "small",
            },
            "Bank of Queensland": {
                "official_name": "Bank of Queensland",
                "common_names": ["BOQ", "Bank of Queensland"],
                "category": "regional",
                "bsb_ranges": [(140, 149)],
                "market_share": "small",
            },
            "HSBC": {
                "official_name": "HSBC Bank Australia",
                "common_names": ["HSBC"],
                "category": "international",
                "bsb_ranges": [(342, 342)],
                "market_share": "small",
            },
            "Citibank": {
                "official_name": "Citibank Australia",
                "common_names": ["Citi", "Citibank"],
                "category": "international",
                "bsb_ranges": [(243, 243)],
                "market_share": "small",
            },
        }

        logger.info("AustralianBankRecognizer initialized with 11 major banks")
        self.initialized = True

    def recognize_banks(self, text: str) -> list[BankMatch]:
        """Recognize Australian banks from text content.

        Args:
            text: Text content to analyze

        Returns:
            List of BankMatch objects sorted by confidence

        """
        self.initialize()

        text_lower = text.lower()
        matches = []

        for bank_key, pattern in self.bank_patterns.items():
            # Find all matches for this bank
            pattern_matches = re.findall(pattern, text_lower, re.IGNORECASE)

            if pattern_matches:
                metadata = self.bank_metadata[bank_key]

                # Calculate confidence based on match quality and frequency
                confidence = self._calculate_confidence(
                    pattern_matches, pattern, text_lower
                )

                bank_match = BankMatch(
                    official_name=metadata["official_name"],
                    common_names=metadata["common_names"],
                    confidence=confidence,
                    category=metadata["category"],
                    bsb_ranges=metadata["bsb_ranges"],
                )

                matches.append(bank_match)

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x.confidence, reverse=True)

        return matches

    def identify_primary_bank(self, text: str) -> BankMatch | None:
        """Identify the primary bank from text content.

        Args:
            text: Text content to analyze

        Returns:
            BankMatch for primary bank, or None if no confident match

        """
        matches = self.recognize_banks(text)

        if matches and matches[0].confidence >= 0.7:
            return matches[0]

        return None

    def is_big_four_bank(self, bank_name: str) -> bool:
        """Check if a bank is one of Australia's Big Four."""
        self.initialize()

        bank_name_lower = bank_name.lower()
        for _bank_key, metadata in self.bank_metadata.items():
            if metadata["category"] == "big_four":
                # Check for exact matches
                if bank_name_lower == metadata["official_name"].lower() or any(
                    bank_name_lower == name.lower() for name in metadata["common_names"]
                ):
                    return True
                # Check if any common name is contained in the input (but not substring matches of input in names)
                if any(
                    name.lower() in bank_name_lower for name in metadata["common_names"]
                ):
                    return True
        return False

    def get_bank_category(self, bank_name: str) -> str | None:
        """Get the category of a bank (big_four, regional, online, etc.)."""
        self.initialize()

        bank_name_lower = bank_name.lower()
        for _bank_key, metadata in self.bank_metadata.items():
            # Check for exact matches
            if bank_name_lower == metadata["official_name"].lower() or any(
                bank_name_lower == name.lower() for name in metadata["common_names"]
            ):
                return metadata["category"]
            # Check if any common name is contained in the input (but not substring matches of input in names)
            if any(
                name.lower() in bank_name_lower for name in metadata["common_names"]
            ):
                return metadata["category"]
        return None

    def _calculate_confidence(
        self, matches: list[str], _pattern: str, text: str
    ) -> float:
        """Calculate confidence score for bank recognition."""
        if not matches:
            return 0.0

        # Start with higher base confidence for institutional matches
        base_confidence = 0.6

        # Bonus for exact institutional matches
        if any(len(match) > 10 for match in matches):  # Long institutional names
            base_confidence += 0.3

        # Bonus for multiple different name variations
        unique_matches = set(matches)
        if len(unique_matches) > 1:
            base_confidence += 0.1

        # Context bonus for banking-related text
        banking_indicators = [
            "statement",
            "account",
            "balance",
            "transaction",
            "bsb",
            "deposit",
            "withdrawal",
            "banking",
        ]

        context_score = sum(1 for indicator in banking_indicators if indicator in text)
        if context_score >= 3:
            base_confidence += 0.2
        elif context_score >= 1:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def get_supported_banks(self) -> list[str]:
        """Get list of all supported Australian banks."""
        self.initialize()
        return list(self.bank_metadata.keys())

    def get_bank_bsb_ranges(self, bank_name: str) -> list[tuple[int, int]]:
        """Get BSB ranges for a specific bank."""
        self.initialize()

        for _bank_key, metadata in self.bank_metadata.items():
            if bank_name.lower() in metadata["official_name"].lower() or any(
                name.lower() in bank_name.lower() for name in metadata["common_names"]
            ):
                return metadata["bsb_ranges"]
        return []
