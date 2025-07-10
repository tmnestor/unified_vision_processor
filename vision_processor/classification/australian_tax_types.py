"""Australian Tax Document Types and Classification Results

This module defines the unified taxonomy of Australian tax document types
and related classification data structures used throughout the vision processor.

Document Types:
- 11 specialized Australian tax document categories
- UNKNOWN and OTHER types for graceful degradation
- Unified naming convention across all processing modules
"""

from dataclasses import dataclass
from enum import Enum


class DocumentType(Enum):
    """Australian tax document types with unified taxonomy."""

    # Primary Australian tax document categories
    FUEL_RECEIPT = "fuel_receipt"  # Vehicle fuel expenses
    TAX_INVOICE = "tax_invoice"  # GST invoices with ABN
    BUSINESS_RECEIPT = "business_receipt"  # General business purchases
    BANK_STATEMENT = "bank_statement"  # Australian bank statements
    MEAL_RECEIPT = "meal_receipt"  # Food and entertainment expenses
    ACCOMMODATION = "accommodation"  # Travel accommodation
    TRAVEL_DOCUMENT = "travel_document"  # Flight tickets and travel
    PARKING_TOLL = "parking_toll"  # Parking and toll expenses
    PROFESSIONAL_SERVICES = "professional_services"  # Legal, accounting, consulting
    EQUIPMENT_SUPPLIES = "equipment_supplies"  # Office equipment and supplies

    # Fallback categories for graceful degradation
    OTHER = "other"  # Recognized document but unclear type
    UNKNOWN = "unknown"  # No clear classification patterns


@dataclass
class ClassificationResult:
    """Result from document classification with evidence."""

    document_type: DocumentType
    confidence: float
    evidence: list[str]

    def is_confident(self, threshold: float = 0.7) -> bool:
        """Check if classification meets confidence threshold."""
        return self.confidence >= threshold

    def is_production_ready(self, threshold: float = 0.8) -> bool:
        """Check if classification is suitable for production processing."""
        return self.confidence >= threshold and self.document_type not in [
            DocumentType.UNKNOWN,
            DocumentType.OTHER,
        ]

    def get_primary_evidence(self, max_items: int = 3) -> list[str]:
        """Get the most important evidence items for classification."""
        return self.evidence[:max_items]


# Australian tax document type metadata for reference
DOCUMENT_TYPE_METADATA = {
    DocumentType.FUEL_RECEIPT: {
        "description": "Vehicle fuel expenses and receipts",
        "ato_category": "Work-related car expenses",
        "common_fields": [
            "date",
            "fuel_type",
            "litres",
            "total_amount",
            "station_name",
        ],
        "requires_gst": True,
        "business_context": "Vehicle operation costs",
    },
    DocumentType.TAX_INVOICE: {
        "description": "GST tax invoices with ABN identification",
        "ato_category": "Business expenses",
        "common_fields": ["date", "abn", "supplier", "total_amount", "gst_amount"],
        "requires_gst": True,
        "business_context": "Professional services and supplies",
    },
    DocumentType.BUSINESS_RECEIPT: {
        "description": "General business purchase receipts",
        "ato_category": "Business expenses",
        "common_fields": ["date", "merchant", "items", "total_amount"],
        "requires_gst": False,
        "business_context": "Office supplies and general business purchases",
    },
    DocumentType.BANK_STATEMENT: {
        "description": "Australian bank account statements",
        "ato_category": "Financial records",
        "common_fields": ["account_number", "bsb", "transactions", "balance"],
        "requires_gst": False,
        "business_context": "Business account transaction records",
    },
    DocumentType.MEAL_RECEIPT: {
        "description": "Food and entertainment expense receipts",
        "ato_category": "Meal entertainment expenses",
        "common_fields": ["date", "restaurant", "amount", "meal_type"],
        "requires_gst": True,
        "business_context": "Business meals and client entertainment",
    },
    DocumentType.ACCOMMODATION: {
        "description": "Travel accommodation expenses",
        "ato_category": "Work-related travel expenses",
        "common_fields": ["date", "hotel", "nights", "total_amount"],
        "requires_gst": True,
        "business_context": "Business travel accommodation",
    },
    DocumentType.TRAVEL_DOCUMENT: {
        "description": "Flight tickets and travel documentation",
        "ato_category": "Work-related travel expenses",
        "common_fields": ["date", "airline", "destination", "ticket_price"],
        "requires_gst": True,
        "business_context": "Business travel transportation",
    },
    DocumentType.PARKING_TOLL: {
        "description": "Parking and toll road expenses",
        "ato_category": "Work-related car expenses",
        "common_fields": ["date", "location", "duration", "amount"],
        "requires_gst": False,
        "business_context": "Vehicle operation costs",
    },
    DocumentType.PROFESSIONAL_SERVICES: {
        "description": "Legal, accounting, and consulting services",
        "ato_category": "Professional fees",
        "common_fields": ["date", "service_provider", "hours", "rate", "total_amount"],
        "requires_gst": True,
        "business_context": "Professional advisory services",
    },
    DocumentType.EQUIPMENT_SUPPLIES: {
        "description": "Office equipment and supply purchases",
        "ato_category": "Business equipment",
        "common_fields": ["date", "supplier", "items", "quantities", "total_amount"],
        "requires_gst": True,
        "business_context": "Business equipment and office supplies",
    },
    DocumentType.OTHER: {
        "description": "Recognized business document of unclear type",
        "ato_category": "Unclassified business expense",
        "common_fields": ["date", "merchant", "amount"],
        "requires_gst": False,
        "business_context": "Requires manual classification",
    },
    DocumentType.UNKNOWN: {
        "description": "Document with no clear classification patterns",
        "ato_category": "Unknown",
        "common_fields": [],
        "requires_gst": False,
        "business_context": "Requires manual review",
    },
}


def get_document_type_info(doc_type: DocumentType) -> dict:
    """Get metadata information for a document type."""
    return DOCUMENT_TYPE_METADATA.get(doc_type, {})


def is_business_expense_type(doc_type: DocumentType) -> bool:
    """Check if document type represents a business expense."""
    return doc_type not in [DocumentType.UNKNOWN, DocumentType.BANK_STATEMENT]


def requires_gst_validation(doc_type: DocumentType) -> bool:
    """Check if document type requires GST validation."""
    metadata = get_document_type_info(doc_type)
    return metadata.get("requires_gst", False)


def get_expected_fields(doc_type: DocumentType) -> list[str]:
    """Get list of expected fields for a document type."""
    metadata = get_document_type_info(doc_type)
    return metadata.get("common_fields", [])
