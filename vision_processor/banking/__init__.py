"""Banking Module - Australian Banking Integration

This module provides comprehensive Australian banking functionality including
bank recognition, BSB validation, transaction categorization, and highlight
processing for business expense identification.

Components:
- AustralianBankRecognizer: Bank identification from text and patterns
- AustralianBSBValidator: BSB format validation and bank identification
- AustralianTransactionCategorizer: Work expense categorization
- BankStatementHighlightProcessor: Visual highlight processing
"""

from .bank_recognizer import (
    AustralianBankRecognizer,
    BankMatch,
)
from .bsb_validator import (
    AustralianBSBValidator,
    BSBValidationResult,
)
from .highlight_processor import (
    BankingHighlight,
    BankStatementHighlightProcessor,
    HighlightedTransaction,
)
from .transaction_categorizer import (
    AustralianTransactionCategorizer,
    Transaction,
    WorkExpense,
)

__all__ = [
    # Bank recognition
    "AustralianBankRecognizer",
    "BankMatch",
    # BSB validation
    "AustralianBSBValidator",
    "BSBValidationResult",
    # Transaction categorization
    "AustralianTransactionCategorizer",
    "Transaction",
    "WorkExpense",
    # Highlight processing
    "BankStatementHighlightProcessor",
    "BankingHighlight",
    "HighlightedTransaction",
]
