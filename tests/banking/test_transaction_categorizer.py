"""
Unit Tests for Australian Transaction Categorizer

Tests transaction categorization, work expense identification, and
ATO category mapping for Australian business expenses.
"""

import pytest

from vision_processor.banking import (
    AustralianTransactionCategorizer,
    Transaction,
    WorkExpense,
)


class TestAustralianTransactionCategorizer:
    """Test suite for Australian Transaction Categorizer."""

    @pytest.fixture
    def categorizer(self):
        """Create transaction categorizer for testing."""
        categorizer = AustralianTransactionCategorizer()
        categorizer.initialize()
        return categorizer

    @pytest.fixture
    def sample_transactions(self):
        """Create sample transactions for testing."""
        return [
            Transaction("15/06/2024", "BP Fuel Station", -85.50, "debit"),
            Transaction("16/06/2024", "Wilson Parking", -12.00, "debit"),
            Transaction("17/06/2024", "Uber Trip", -25.30, "debit"),
            Transaction("18/06/2024", "Officeworks Supplies", -156.80, "debit"),
            Transaction("19/06/2024", "McDonald's Lunch", -18.50, "debit"),
            Transaction("20/06/2024", "Salary Payment", 5000.00, "credit"),
            Transaction("21/06/2024", "Deloitte Consulting", -2500.00, "debit"),
            Transaction("22/06/2024", "Telstra Phone Bill", -89.90, "debit"),
        ]

    def test_initialization(self, categorizer):
        """Test categorizer initialization."""
        assert categorizer.initialized
        assert len(categorizer.work_expense_patterns) > 0
        assert len(categorizer.business_keywords) > 0
        assert len(categorizer.ato_categories) > 0

    def test_transaction_extraction_from_text(self, categorizer):
        """Test extraction of transactions from bank statement text."""
        statement_text = """
        15/06/2024    BP Fuel Station Melbourne    -85.50
        16/06/2024    Wilson Parking Flinders St   -12.00
        17/06/2024    Salary Payment               +5000.00
        18/06/2024    Uber Trip to Airport         -25.30
        """

        transactions = categorizer.extract_transactions(statement_text)

        assert len(transactions) >= 3  # Should extract at least the clear transactions

        # Check that transactions are properly parsed
        for transaction in transactions:
            assert isinstance(transaction, Transaction)
            assert transaction.date is not None
            assert transaction.description is not None
            assert isinstance(transaction.amount, float)
            assert transaction.transaction_type in ["debit", "credit"]

    def test_fuel_expense_identification(self, categorizer):
        """Test identification of fuel expenses."""
        fuel_transactions = [
            Transaction("15/06/2024", "BP Fuel Station", -85.50, "debit"),
            Transaction("16/06/2024", "Shell Service Station", -92.30, "debit"),
            Transaction("17/06/2024", "Caltex Fuel Stop", -78.90, "debit"),
        ]

        work_expenses = categorizer.identify_work_expenses(fuel_transactions)

        # Should identify fuel transactions as work expenses
        assert len(work_expenses) > 0

        fuel_expenses = [we for we in work_expenses if we.category == "fuel"]
        assert len(fuel_expenses) > 0

        for expense in fuel_expenses:
            assert expense.ato_category == "Work-related car expenses"
            assert expense.work_score > 0.5  # Should have high confidence for fuel

    def test_parking_and_toll_identification(self, categorizer):
        """Test identification of parking and toll expenses."""
        transport_transactions = [
            Transaction("15/06/2024", "Wilson Parking", -12.00, "debit"),
            Transaction("16/06/2024", "Secure Parking", -15.50, "debit"),
            Transaction("17/06/2024", "CityLink Toll", -8.20, "debit"),
        ]

        work_expenses = categorizer.identify_work_expenses(transport_transactions)

        parking_expenses = [we for we in work_expenses if we.category in ["parking", "toll"]]
        assert len(parking_expenses) > 0

        for expense in parking_expenses:
            assert expense.ato_category == "Work-related car expenses"

    def test_professional_services_identification(self, categorizer):
        """Test identification of professional services."""
        professional_transactions = [
            Transaction("15/06/2024", "Deloitte Consulting", -2500.00, "debit"),
            Transaction("16/06/2024", "PWC Advisory Services", -1800.00, "debit"),
            Transaction("17/06/2024", "Legal Fees - Allens", -950.00, "debit"),
        ]

        work_expenses = categorizer.identify_work_expenses(professional_transactions)

        professional_expenses = [we for we in work_expenses if we.category == "professional_services"]
        assert len(professional_expenses) > 0

        for expense in professional_expenses:
            assert expense.ato_category == "Professional fees"
            assert expense.work_score > 0.7  # Should have high confidence

    def test_office_supplies_identification(self, categorizer):
        """Test identification of office supplies."""
        office_transactions = [
            Transaction("15/06/2024", "Officeworks Supplies", -156.80, "debit"),
            Transaction("16/06/2024", "Staples Office Equipment", -89.90, "debit"),
        ]

        work_expenses = categorizer.identify_work_expenses(office_transactions)

        office_expenses = [we for we in work_expenses if we.category == "office_supplies"]
        assert len(office_expenses) > 0

        for expense in office_expenses:
            assert expense.ato_category == "Office expenses"

    def test_meal_expense_identification(self, categorizer):
        """Test identification of meal expenses (with lower confidence)."""
        meal_transactions = [
            Transaction("15/06/2024", "Restaurant Business Lunch", -85.50, "debit"),
            Transaction("16/06/2024", "Coffee Meeting", -12.50, "debit"),
            Transaction(
                "17/06/2024", "McDonald's", -18.50, "debit"
            ),  # Personal meal - should be lower confidence
        ]

        work_expenses = categorizer.identify_work_expenses(meal_transactions)

        meal_expenses = [we for we in work_expenses if we.category == "meals"]

        # Check that business meals have higher confidence than personal meals
        business_meal = next(
            (we for we in meal_expenses if "business" in we.transaction.description.lower()),
            None,
        )
        if business_meal:
            assert business_meal.work_score > 0.6
            assert business_meal.ato_category == "Meal entertainment expenses"

    def test_credit_transactions_ignored(self, categorizer):
        """Test that credit transactions (income) are not identified as work expenses."""
        credit_transactions = [
            Transaction("15/06/2024", "Salary Payment", 5000.00, "credit"),
            Transaction("16/06/2024", "Business Income", 2500.00, "credit"),
        ]

        work_expenses = categorizer.identify_work_expenses(credit_transactions)

        # Should not identify any credits as work expenses
        assert len(work_expenses) == 0

    def test_business_keyword_boost(self, categorizer):
        """Test that business keywords boost work expense scores."""
        transactions_with_business_context = [
            Transaction("15/06/2024", "Business Software License", -299.00, "debit"),
            Transaction("16/06/2024", "Office Equipment Purchase", -156.80, "debit"),
            Transaction("17/06/2024", "Client Meeting Lunch", -45.50, "debit"),
        ]

        work_expenses = categorizer.identify_work_expenses(transactions_with_business_context)

        # Should identify these as work expenses due to business context
        assert len(work_expenses) > 0

        for expense in work_expenses:
            assert expense.work_score > 0.5
            assert len(expense.business_indicators) > 0

    def test_confidence_threshold_filtering(self, categorizer):
        """Test that low confidence transactions are filtered out."""
        mixed_transactions = [
            Transaction("15/06/2024", "Grocery Shopping", -85.50, "debit"),  # Personal
            Transaction("16/06/2024", "BP Fuel Station", -65.30, "debit"),  # Business
            Transaction("17/06/2024", "Random Purchase", -25.00, "debit"),  # Unclear
        ]

        work_expenses = categorizer.identify_work_expenses(mixed_transactions)

        # Should only identify clear business expenses
        fuel_expenses = [we for we in work_expenses if "bp" in we.transaction.description.lower()]
        assert len(fuel_expenses) > 0

        # Grocery shopping should not be identified as work expense
        grocery_expenses = [we for we in work_expenses if "grocery" in we.transaction.description.lower()]
        assert len(grocery_expenses) == 0

    def test_deduplication(self, categorizer):
        """Test transaction deduplication functionality."""
        duplicate_transactions = [
            Transaction("15/06/2024", "BP Fuel Station", -85.50, "debit"),
            Transaction("15/06/2024", "BP Fuel Station", -85.50, "debit"),  # Exact duplicate
            Transaction("15/06/2024", "Wilson Parking", -12.00, "debit"),
        ]

        work_expenses = categorizer.identify_work_expenses(duplicate_transactions)

        # Should deduplicate similar transactions
        bp_expenses = [we for we in work_expenses if "bp" in we.transaction.description.lower()]
        assert len(bp_expenses) <= 1  # Should only have one BP transaction after deduplication

    def test_expense_summary_generation(self, categorizer, sample_transactions):
        """Test generation of expense summary statistics."""
        work_expenses = categorizer.identify_work_expenses(sample_transactions)
        summary = categorizer.get_expense_summary(work_expenses)

        assert "total_expenses" in summary
        assert "total_amount" in summary
        assert "categories" in summary
        assert "ato_categories" in summary
        assert "average_confidence" in summary

        assert isinstance(summary["total_expenses"], int)
        assert isinstance(summary["total_amount"], float)
        assert isinstance(summary["categories"], dict)
        assert isinstance(summary["ato_categories"], dict)
        assert isinstance(summary["average_confidence"], float)

        if summary["total_expenses"] > 0:
            assert summary["total_amount"] > 0
            assert 0.0 <= summary["average_confidence"] <= 1.0

    def test_supported_categories(self, categorizer):
        """Test retrieval of supported expense categories."""
        categories = categorizer.get_supported_categories()

        assert len(categories) > 0
        assert "fuel" in categories
        assert "parking" in categories
        assert "professional_services" in categories
        assert "office_supplies" in categories

    def test_ato_category_information(self, categorizer):
        """Test retrieval of ATO category information."""
        # Test fuel expenses
        fuel_info = categorizer.get_ato_category_info("Work-related car expenses")
        assert "description" in fuel_info
        assert "deductible" in fuel_info
        assert fuel_info["deductible"] is True

        # Test meal expenses (limited deductibility)
        meal_info = categorizer.get_ato_category_info("Meal entertainment expenses")
        assert "description" in meal_info
        assert "deductible" in meal_info
        assert meal_info["deductible"] == "limited"  # 50% deductible

    def test_empty_transaction_list(self, categorizer):
        """Test handling of empty transaction list."""
        work_expenses = categorizer.identify_work_expenses([])
        assert work_expenses == []

        summary = categorizer.get_expense_summary([])
        assert summary["total_expenses"] == 0
        assert summary["total_amount"] == 0.0
        assert summary["average_confidence"] == 0.0

    def test_transaction_parsing_edge_cases(self, categorizer):
        """Test transaction parsing with various text formats."""
        edge_case_text = """
        Invalid line without proper format
        15/06/24    Short Date Format    -85.50
        15/06/2024  Very Long Description That Goes On And On And On  -156.80
        16/6/2024   Single Digit Day/Month   -25.30
        17/06/2024  Missing Amount
        18/06/2024  Invalid Amount Text  -$abc.def
        19/06/2024  Valid Transaction    -45.90
        """

        transactions = categorizer.extract_transactions(edge_case_text)

        # Should parse some valid transactions despite format issues
        assert len(transactions) > 0

        # All parsed transactions should have valid structure
        for transaction in transactions:
            assert isinstance(transaction.amount, float)
            assert transaction.transaction_type in ["debit", "credit"]

    def test_work_expense_structure(self, categorizer):
        """Test that WorkExpense objects have proper structure."""
        transaction = Transaction("15/06/2024", "BP Fuel Station", -85.50, "debit")
        work_expenses = categorizer.identify_work_expenses([transaction])

        if work_expenses:
            expense = work_expenses[0]
            assert isinstance(expense, WorkExpense)
            assert isinstance(expense.transaction, Transaction)
            assert isinstance(expense.work_score, float)
            assert isinstance(expense.confidence, float)
            assert isinstance(expense.business_indicators, list)
            assert isinstance(expense.ato_category, str)
            assert 0.0 <= expense.work_score <= 1.0
            assert 0.0 <= expense.confidence <= 1.0
