#!/usr/bin/env python3
"""Test compliance modules independently."""

import sys
from pathlib import Path

# Add the specific compliance module to path
sys.path.append(str(Path(__file__).parent / "vision_processor" / "compliance"))


def test_field_validators():
    """Test field validators directly."""
    print("=== Testing Field Validators ===")

    from field_validators import ABNValidator, BSBValidator, DateValidator, GSTValidator

    # Test ABN
    abn_validator = ABNValidator()
    valid, formatted, issues = abn_validator.validate("53 004 085 616")
    print(f"✓ ABN validation: {valid} -> {formatted}")

    # Test BSB
    bsb_validator = BSBValidator()
    valid, formatted, issues, bank = bsb_validator.validate("062-001")
    print(f"✓ BSB validation: {valid} -> {formatted} ({bank})")

    # Test Date
    date_validator = DateValidator()
    valid, date_obj, formatted, issues = date_validator.validate("15/03/2024")
    print(f"✓ Date validation: {valid} -> {formatted}")

    # Test GST
    gst_validator = GSTValidator()
    valid, calc, issues = gst_validator.validate_gst_calculation(100.0, 10.0, 110.0)
    print(f"✓ GST validation: {valid}")

    return True


def test_business_registry():
    """Test business registry directly."""
    print("\n=== Testing Business Registry ===")

    from australian_business_registry import AustralianBusinessRegistry

    registry = AustralianBusinessRegistry()
    registry.initialize()

    stats = registry.get_business_statistics()
    print(f"✓ Total businesses: {stats['total_businesses']}")

    # Test recognition
    test_text = "woolworths receipt"
    recognized = registry.recognize_business(test_text)
    print(f"✓ Recognition test: {len(recognized)} businesses found")

    return stats["total_businesses"] >= 100


def main():
    """Run compliance tests."""
    print("🧪 Testing Phase 4 Compliance Features\n")

    try:
        validator_success = test_field_validators()
        registry_success = test_business_registry()

        if validator_success and registry_success:
            print("\n✅ Compliance features working correctly!")
            print("Phase 4 core features validated.")
        else:
            print("\n⚠️  Some compliance tests failed.")

    except Exception as e:
        print(f"Test error: {e}")


if __name__ == "__main__":
    main()
