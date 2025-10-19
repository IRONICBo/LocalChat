# -*- coding: utf-8 -*-
"""
Quick test script for all generators.

Run this to verify that all generators are working correctly.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_basic_pii_generator():
    """Test basic PII generator."""
    print("=" * 60)
    print("Testing Basic PII Generator")
    print("=" * 60)

    try:
        from utils.fake_pii_generator import FakePIIGenerator

        gen = FakePIIGenerator()

        tests = [
            ("person_name", "test_name"),
            ("email_address", "test_email"),
            ("phone_number", "test_phone"),
            ("credit_card_info", "test_card"),
            ("street_address", "test_addr"),
        ]

        for category, test_id in tests:
            value = gen.get_fake_value(category, test_id)
            print(f"✓ {category}: {value}")

        print("\n✅ Basic PII Generator: PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Basic PII Generator: FAILED")
        print(f"Error: {e}\n")
        return False


def test_credit_card_generator():
    """Test credit card generator."""
    print("=" * 60)
    print("Testing Credit Card Generator")
    print("=" * 60)

    try:
        from utils.credit_card_generator import CreditCardGenerator

        gen = CreditCardGenerator()

        # Test different formats
        formats = ["plain", "space_4", "dash_4", "last_4_only"]
        for fmt in formats:
            card = gen.generate_card(format_type=fmt)
            print(f"✓ Format '{fmt}': {card}")

        # Test different card types
        card_types = ["visa", "mastercard", "amex"]
        for card_type in card_types:
            card = gen.generate_card(card_type=card_type, format_type="space_4")
            valid = gen.verify_luhn(card)
            print(f"✓ {card_type.capitalize()}: {card} (Valid: {valid})")

        # Test bulk generation
        bulk = gen.generate_bulk(count=5)
        print(f"✓ Bulk generation: {len(bulk)} cards")

        print("\n✅ Credit Card Generator: PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Credit Card Generator: FAILED")
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_address_generator():
    """Test address generator."""
    print("=" * 60)
    print("Testing Address Generator")
    print("=" * 60)

    try:
        from utils.address_generator import AddressGenerator

        gen = AddressGenerator()

        # Test different countries
        countries = ["US", "GB", "DE", "FR"]
        for country in countries:
            address = gen.generate_address(country=country, format_type="single_line")
            print(f"✓ {country}: {address}")

        # Test bulk generation
        bulk = gen.generate_bulk_addresses(count=3, countries=["US", "GB", "DE"])
        print(f"✓ Bulk generation: {len(bulk)} addresses")

        print("\n✅ Address Generator: PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Address Generator: FAILED")
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_pii_generator():
    """Test enhanced PII generator with new generators."""
    print("=" * 60)
    print("Testing Enhanced PII Generator")
    print("=" * 60)

    try:
        from utils.fake_pii_generator import FakePIIGenerator

        gen = FakePIIGenerator(use_enhanced_generators=True)

        # Test credit cards (should use enhanced generator)
        print("Credit Cards:")
        for i in range(3):
            card = gen.get_fake_value("credit_card_info", f"card_{i}")
            print(f"  ✓ {card}")

        # Test addresses (should use enhanced generator)
        print("\nAddresses:")
        for i in range(3):
            address = gen.get_fake_value("street_address", f"addr_{i}")
            print(f"  ✓ {address}")

        print("\n✅ Enhanced PII Generator: PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Enhanced PII Generator: FAILED")
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_luhn_validation():
    """Test Luhn algorithm validation."""
    print("=" * 60)
    print("Testing Luhn Algorithm Validation")
    print("=" * 60)

    try:
        from utils.credit_card_generator import CreditCardGenerator

        gen = CreditCardGenerator()

        # Test valid cards
        valid_cards = [
            gen.generate_card(format_type="plain") for _ in range(5)
        ]

        for card in valid_cards:
            is_valid = gen.verify_luhn(card)
            if is_valid:
                print(f"✓ {card}: Valid")
            else:
                print(f"✗ {card}: Invalid (should be valid!)")
                return False

        # Test invalid card
        invalid_card = "1234567890123456"
        is_valid = gen.verify_luhn(invalid_card)
        if not is_valid:
            print(f"✓ {invalid_card}: Invalid (as expected)")
        else:
            print(f"✗ {invalid_card}: Valid (should be invalid!)")
            return False

        print("\n✅ Luhn Validation: PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Luhn Validation: FAILED")
        print(f"Error: {e}\n")
        return False


def test_consistency():
    """Test that same input produces same output."""
    print("=" * 60)
    print("Testing Consistency (Same Input → Same Output)")
    print("=" * 60)

    try:
        from utils.fake_pii_generator import FakePIIGenerator

        gen = FakePIIGenerator()

        # Test multiple times with same ID
        test_id = "consistent_test"
        values = []

        for _ in range(5):
            value = gen.get_fake_value("person_name", test_id)
            values.append(value)

        # All values should be the same
        if len(set(values)) == 1:
            print(f"✓ Same input produces same output: {values[0]}")
            print("\n✅ Consistency Test: PASSED\n")
            return True
        else:
            print(f"✗ Inconsistent outputs: {values}")
            print("\n❌ Consistency Test: FAILED\n")
            return False

    except Exception as e:
        print(f"\n❌ Consistency Test: FAILED")
        print(f"Error: {e}\n")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "LLM Adapter - Generator Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    tests = [
        ("Basic PII Generator", test_basic_pii_generator),
        ("Credit Card Generator", test_credit_card_generator),
        ("Address Generator", test_address_generator),
        ("Enhanced PII Generator", test_enhanced_pii_generator),
        ("Luhn Validation", test_luhn_validation),
        ("Consistency", test_consistency),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name}: FAILED with exception")
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All tests passed! The generators are working correctly.\n")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the output above.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
