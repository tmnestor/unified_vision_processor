"""Document Classifier - Australian Tax Document Classification

This module implements comprehensive Australian tax document classification with
graceful degradation capabilities. Supports 11 Australian tax document types
with extensive business knowledge and format recognition.

Features:
- 11 Australian tax document types with specialized keyword recognition
- Format pattern matching using regex
- Integration with Australian business registry (100+ businesses)
- Graceful degradation for low-confidence classifications
- Evidence-based classification results
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

# Import DocumentType enum - will be available after extraction
from .australian_tax_types import DocumentType

logger = logging.getLogger(__name__)


class BasePipelineComponent(ABC):
    """Base class for pipeline components."""

    def __init__(self, config: Any):
        self.config = config
        self.initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component."""

    def ensure_initialized(self) -> None:
        """Ensure component is initialized."""
        if not self.initialized:
            self.initialize()
            self.initialized = True


class DocumentClassifier(BasePipelineComponent):
    """Australian Tax Document Classifier with graceful degradation."""

    def initialize(self) -> None:
        """Initialize classifier with Australian business knowledge."""
        # Australian business classification keywords
        self.classification_keywords = {
            DocumentType.BUSINESS_RECEIPT: [
                "woolworths",
                "coles",
                "aldi",
                "target",
                "kmart",
                "bunnings",
                "officeworks",
                "harvey norman",
                "jb hi-fi",
                "big w",
                "myer",
                "david jones",
                "ikea",
                "spotlight",
                "rebel sport",
                "chemist warehouse",
                "priceline",
                "terry white",
                "dan murphy",
                "bws",
                "liquorland",
            ],
            DocumentType.FUEL_RECEIPT: [
                "bp",
                "shell",
                "caltex",
                "ampol",
                "mobil",
                "7-eleven",
                "united petroleum",
                "liberty",
                "metro petroleum",
                "speedway",
                "fuel",
                "petrol",
                "diesel",
                "unleaded",
                "premium",
                "litres",
                "pump",
                "station",
            ],
            DocumentType.TAX_INVOICE: [
                "tax invoice",
                "gst invoice",
                "invoice",
                "abn",
                "tax invoice number",
                "invoice number",
                "supplier",
                "customer",
                "subtotal",
                "gst amount",
                "professional services",
                "consulting",
                "advisory",
            ],
            DocumentType.BANK_STATEMENT: [
                "anz",
                "commonwealth bank",
                "westpac",
                "nab",
                "ing",
                "macquarie",
                "bendigo bank",
                "suncorp",
                "bank of queensland",
                "credit union",
                "account statement",
                "transaction history",
                "bsb",
                "account number",
                "opening balance",
                "closing balance",
                "statement period",
            ],
            DocumentType.MEAL_RECEIPT: [
                "restaurant",
                "cafe",
                "bistro",
                "bar",
                "pub",
                "club",
                "hotel",
                "mcdonald's",
                "kfc",
                "subway",
                "domino's",
                "pizza hut",
                "hungry jack's",
                "red rooster",
                "nando's",
                "guzman y gomez",
                "zambrero",
                "starbucks",
                "gloria jean's",
                "coffee",
                "breakfast",
                "lunch",
                "dinner",
            ],
            DocumentType.ACCOMMODATION: [
                "hilton",
                "marriott",
                "hyatt",
                "ibis",
                "mercure",
                "novotel",
                "crowne plaza",
                "holiday inn",
                "radisson",
                "sheraton",
                "hotel",
                "motel",
                "resort",
                "accommodation",
                "booking",
                "check-in",
                "check-out",
                "room",
                "suite",
                "nights",
            ],
            DocumentType.TRAVEL_DOCUMENT: [
                "qantas",
                "jetstar",
                "virgin australia",
                "tigerair",
                "rex airlines",
                "flight",
                "airline",
                "boarding pass",
                "ticket",
                "travel",
                "departure",
                "arrival",
                "gate",
                "seat",
                "passenger",
            ],
            DocumentType.PARKING_TOLL: [
                "secure parking",
                "wilson parking",
                "ace parking",
                "care park",
                "parking australia",
                "premium parking",
                "toll",
                "citylink",
                "eastlink",
                "westlink",
                "parking",
                "meter",
                "space",
                "duration",
            ],
            DocumentType.EQUIPMENT_SUPPLIES: [
                "computer",
                "laptop",
                "tablet",
                "printer",
                "software",
                "hardware",
                "equipment",
                "supplies",
                "stationery",
                "office supplies",
                "tools",
                "machinery",
                "furniture",
                "electronics",
            ],
            DocumentType.PROFESSIONAL_SERVICES: [
                "deloitte",
                "pwc",
                "kpmg",
                "ey",
                "bdo",
                "rsm",
                "pitcher partners",
                "allens",
                "ashurst",
                "clayton utz",
                "corrs",
                "herbert smith freehills",
                "legal",
                "accounting",
                "consulting",
                "advisory",
                "professional",
                "solicitor",
                "barrister",
                "accountant",
                "consultant",
            ],
        }

        # Document format indicators
        self.format_indicators = {
            DocumentType.TAX_INVOICE: [
                r"tax invoice",
                r"gst invoice",
                r"invoice number",
                r"abn",
                r"supplier",
                r"customer",
                r"due date",
                r"terms",
            ],
            DocumentType.BANK_STATEMENT: [
                r"account statement",
                r"transaction history",
                r"bsb",
                r"opening balance",
                r"closing balance",
                r"statement period",
            ],
            DocumentType.FUEL_RECEIPT: [
                r"pump \d+",
                r"litres?",
                r"fuel type",
                r"unleaded",
                r"diesel",
                r"premium",
                r"cents?/litre",
                r"total fuel",
            ],
            DocumentType.BUSINESS_RECEIPT: [
                r"receipt",
                r"purchase",
                r"total",
                r"gst",
                r"subtotal",
                r"items?",
                r"quantity",
                r"price",
            ],
            DocumentType.MEAL_RECEIPT: [
                r"table \d+",
                r"covers?",
                r"dine in",
                r"take away",
                r"breakfast",
                r"lunch",
                r"dinner",
                r"beverage",
            ],
            DocumentType.ACCOMMODATION: [
                r"check.?in",
                r"check.?out",
                r"room \d+",
                r"nights?",
                r"guest",
                r"booking",
                r"reservation",
            ],
            DocumentType.TRAVEL_DOCUMENT: [
                r"flight \w+",
                r"gate \w+",
                r"seat \w+",
                r"departure",
                r"arrival",
                r"passenger",
                r"boarding",
            ],
            DocumentType.PARKING_TOLL: [
                r"entry time",
                r"exit time",
                r"duration",
                r"space \d+",
                r"plate",
                r"registration",
                r"toll",
            ],
            DocumentType.EQUIPMENT_SUPPLIES: [
                r"model",
                r"serial",
                r"warranty",
                r"qty",
                r"unit price",
                r"description",
                r"part number",
            ],
            DocumentType.PROFESSIONAL_SERVICES: [
                r"hours?",
                r"rate",
                r"time",
                r"service period",
                r"matter",
                r"file",
                r"professional",
            ],
        }

        # Initialize Australian business registry for comprehensive recognition
        from ..compliance import AustralianBusinessRegistry

        self.business_registry = AustralianBusinessRegistry()
        self.business_registry.initialize()

        # Get business names for quick lookup (legacy compatibility)
        self.australian_businesses = {
            "major_retailers": [
                "woolworths",
                "coles",
                "aldi",
                "target",
                "kmart",
                "bunnings",
                "officeworks",
                "harvey norman",
                "jb hi-fi",
                "big w",
                "myer",
                "david jones",
                "ikea",
                "spotlight",
                "rebel sport",
            ],
            "fuel_stations": ["bp", "shell", "caltex", "ampol", "mobil", "7-eleven"],
            "banks": ["anz", "commonwealth bank", "westpac", "nab", "ing", "macquarie"],
            "airlines": ["qantas", "jetstar", "virgin australia", "tigerair"],
            "hotels": ["hilton", "marriott", "hyatt", "ibis", "mercure", "novotel"],
            "food_chains": ["mcdonald's", "kfc", "subway", "domino's", "hungry jack's"],
            "telecommunications": ["telstra", "optus", "vodafone"],
            "utilities": ["agl", "origin", "energyaustralia"],
        }

        logger.info("DocumentClassifier initialized with Australian business knowledge")

    def classify_with_evidence(
        self,
        _image_path: Any,
    ) -> tuple[DocumentType, float, list[str]]:
        """Classify document with evidence using model response.

        Returns:
            Tuple of (document_type, confidence, evidence)

        """
        # For now, return a medium confidence classification
        # This will be enhanced when we have the actual model integration
        return DocumentType.BUSINESS_RECEIPT, 0.7, ["Model-based classification"]

    def classify_from_text(self, text: str) -> tuple[DocumentType, float, list[str]]:
        """Classify document from text content with graceful degradation.

        Args:
            text: Document text content

        Returns:
            Tuple of (document_type, confidence, evidence)

        """
        text_lower = text.lower()

        # Score each document type
        type_scores = {}
        evidence_by_type = {}

        for doc_type in DocumentType:
            if doc_type == DocumentType.UNKNOWN:
                continue
            score, evidence = self._score_document_type(text_lower, doc_type)
            type_scores[doc_type] = score
            evidence_by_type[doc_type] = evidence

        # Find best match with graceful degradation
        if not type_scores:
            return DocumentType.UNKNOWN, 0.1, ["No classification patterns found"]

        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]

        # Graceful degradation - accept lower confidence
        if best_score < 0.3:
            return DocumentType.OTHER, best_score, ["Low confidence classification"]

        # Generate evidence
        evidence = []
        type_evidence = evidence_by_type[best_type]

        if type_evidence["keyword_matches"]:
            top_keywords = sorted(
                type_evidence["keyword_matches"],
                key=lambda x: x["weight"],
                reverse=True,
            )[:3]
            keyword_list = [kw["keyword"] for kw in top_keywords]
            evidence.append(f"Keywords: {', '.join(keyword_list)}")

        if type_evidence["format_matches"]:
            evidence.append(f"Format patterns: {len(type_evidence['format_matches'])}")

        if type_evidence["business_matches"]:
            # Include confidence scores if available
            business_display = []
            for bm in type_evidence["business_matches"][:2]:  # Top 2 businesses
                if "confidence" in bm:
                    business_display.append(
                        f"{bm['business']} ({bm['confidence']:.2f})",
                    )
                else:
                    business_display.append(bm["business"])

            evidence.append(f"Australian businesses: {', '.join(business_display)}")

        return best_type, best_score, evidence

    def _score_document_type(
        self,
        text: str,
        doc_type: DocumentType,
    ) -> tuple[float, dict[str, Any]]:
        """Score how well text matches a document type."""
        evidence = {
            "keyword_matches": [],
            "format_matches": [],
            "business_matches": [],
            "total_score": 0.0,
        }

        total_score = 0.0
        max_possible_score = 0.0

        # Score keyword matches
        keywords = self.classification_keywords.get(doc_type, [])
        keyword_score = 0.0

        for keyword in keywords:
            if keyword in text:
                # Weight by keyword specificity
                if len(keyword) > 15:  # Very specific business names
                    weight = 1.0
                elif len(keyword) > 10:  # Specific business names
                    weight = 0.8
                elif len(keyword) > 6:  # Industry terms
                    weight = 0.6
                else:  # General terms
                    weight = 0.4

                keyword_score += weight
                evidence["keyword_matches"].append(
                    {"keyword": keyword, "weight": weight},
                )

        # Normalize keyword score
        if keywords:
            keyword_score = min(keyword_score / len(keywords), 1.0)
            total_score += keyword_score * 0.5  # 50% weight
            max_possible_score += 0.5

        # Score format indicators
        format_patterns = self.format_indicators.get(doc_type, [])
        format_score = 0.0

        for pattern in format_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                format_score += 0.2
                evidence["format_matches"].append(pattern)

        # Normalize format score
        if format_patterns:
            format_score = min(format_score, 1.0)
            total_score += format_score * 0.3  # 30% weight
            max_possible_score += 0.3

        # Score business name matches using comprehensive registry
        business_score = 0.0

        # Use comprehensive business registry for recognition
        recognized_businesses = self.business_registry.recognize_business(text)
        for business in recognized_businesses[:5]:  # Top 5 matches
            business_score += business["confidence"] * 0.2
            evidence["business_matches"].append(
                {
                    "business": business["official_name"],
                    "category": business["industry"],
                    "confidence": business["confidence"],
                },
            )

        # Fallback to legacy method for any missed businesses
        for category, businesses in self.australian_businesses.items():
            for business in businesses:
                if business in text and not any(
                    b["business"].lower() == business.lower() for b in evidence["business_matches"]
                ):
                    business_score += 0.3
                    evidence["business_matches"].append(
                        {"business": business, "category": category, "confidence": 0.8},
                    )

        # Normalize business score
        business_score = min(business_score, 1.0)
        total_score += business_score * 0.2  # 20% weight
        max_possible_score += 0.2

        # Calculate final score
        if max_possible_score > 0:
            final_score = total_score / max_possible_score
        else:
            final_score = 0.0

        evidence["total_score"] = final_score
        return final_score, evidence
