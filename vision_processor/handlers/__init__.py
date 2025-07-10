"""Document Handlers Module

This module provides specialized document handlers for the 11 Australian tax document types,
following the Llama-3.2 7-step processing pipeline foundation with InternVL feature integration.
"""

from .accommodation_handler import AccommodationHandler
from .bank_statement_handler import BankStatementHandler
from .base_ato_handler import BaseATOHandler
from .business_receipt_handler import BusinessReceiptHandler
from .equipment_supplies_handler import EquipmentSuppliesHandler
from .fuel_receipt_handler import FuelReceiptHandler
from .meal_receipt_handler import MealReceiptHandler
from .other_document_handler import OtherDocumentHandler
from .parking_toll_handler import ParkingTollHandler
from .professional_services_handler import ProfessionalServicesHandler
from .tax_invoice_handler import TaxInvoiceHandler
from .travel_document_handler import TravelDocumentHandler

__all__ = [
    "AccommodationHandler",
    "BankStatementHandler",
    "BaseATOHandler",
    "BusinessReceiptHandler",
    "EquipmentSuppliesHandler",
    "FuelReceiptHandler",
    "MealReceiptHandler",
    "OtherDocumentHandler",
    "ParkingTollHandler",
    "ProfessionalServicesHandler",
    "TaxInvoiceHandler",
    "TravelDocumentHandler",
]
