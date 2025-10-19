"""
Complete example for generating PII datasets with enhanced generators.

This script demonstrates how to use the enhanced address and credit card
generators to create realistic synthetic PII datasets.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fake_pii_generator import FakePIIGenerator
from utils.address_generator import AddressGenerator
from utils.credit_card_generator import CreditCardGenerator
import json


def example_basic_generation():
    """Basic PII generation example."""
    print("=" * 60)
    print("Example 1: Basic PII Generation")
    print("=" * 60)

    generator = FakePIIGenerator(locale="en_US")

    print("\n1. Person Names:")
    for i in range(3):
        name = generator.get_fake_value("person_name", f"original_{i}")
        print(f"   {i+1}. {name}")

    print("\n2. Email Addresses:")
    for i in range(3):
        email = generator.get_fake_value("email_address", f"email_{i}")
        print(f"   {i+1}. {email}")

    print("\n3. Phone Numbers:")
    for i in range(3):
        phone = generator.get_fake_value("phone_number", f"phone_{i}")
        print(f"   {i+1}. {phone}")

    print("\n4. Credit Card Numbers:")
    for i in range(3):
        card = generator.get_fake_value("credit_card_info", f"card_{i}")
        print(f"   {i+1}. {card}")


def example_enhanced_generation():
    """Enhanced generation with new generators."""
    print("\n" + "=" * 60)
    print("Example 2: Enhanced PII Generation")
    print("=" * 60)

    # Initialize with enhanced generators
    generator = FakePIIGenerator(
        locale="en_US",
        use_enhanced_generators=True
    )

    print("\n1. Credit Cards (Multiple Formats):")
    for i in range(5):
        card = generator.get_fake_value("credit_card_info", f"card_{i}")
        print(f"   {i+1}. {card}")

    print("\n2. Addresses (Multi-Country):")
    for i in range(5):
        address = generator.get_fake_value("street_address", f"addr_{i}")
        print(f"   {i+1}. {address}")


def example_credit_card_generator():
    """Demonstrate credit card generator capabilities."""
    print("\n" + "=" * 60)
    print("Example 3: Credit Card Generator")
    print("=" * 60)

    generator = CreditCardGenerator()

    print("\n1. Different Card Types:")
    for card_type in ["visa", "mastercard", "amex", "discover"]:
        card = generator.generate_card(card_type=card_type, format_type="space_4")
        valid = generator.verify_luhn(card)
        print(f"   {card_type.capitalize()}: {card} (Valid: {valid})")

    print("\n2. Different Formats:")
    formats = ["plain", "space_4", "dash_4", "last_4_only", "first_6_last_4"]
    for fmt in formats:
        card = generator.generate_card(card_type="visa", format_type=fmt)
        print(f"   {fmt}: {card}")

    print("\n3. Complete Card Information:")
    metadata = generator.generate_with_metadata(card_type="visa", format_type="space_4")
    print(f"   Card Number: {metadata['card_number']}")
    print(f"   Card Type: {metadata['card_type']}")
    print(f"   CVV: {metadata['cvv']}")
    print(f"   Expiry: {metadata['expiry']}")
    print(f"   Last 4: {metadata['last_4']}")
    print(f"   Valid: {metadata['valid']}")


def example_address_generator():
    """Demonstrate address generator capabilities."""
    print("\n" + "=" * 60)
    print("Example 4: Address Generator")
    print("=" * 60)

    generator = AddressGenerator()

    print("\n1. Single-line Format:")
    for country in ["US", "GB", "DE", "FR"]:
        address = generator.generate_address(country=country, format_type="single_line")
        print(f"   {country}: {address}")

    print("\n2. Multi-line Format:")
    for country in ["US", "GB", "DE"]:
        print(f"\n   {country}:")
        address = generator.generate_address(country=country, format_type="multi_line")
        for line in address.split("\n"):
            print(f"      {line}")

    print("\n3. Bulk Generation (Mixed Countries):")
    addresses = generator.generate_bulk_addresses(
        count=5,
        countries=["US", "GB", "DE", "FR", "CA"]
    )
    for i, addr in enumerate(addresses, 1):
        print(f"   {i}. {addr}")


def example_dataset_generation():
    """Generate a complete synthetic dataset."""
    print("\n" + "=" * 60)
    print("Example 5: Complete Dataset Generation")
    print("=" * 60)

    generator = FakePIIGenerator(
        locale="en_US",
        use_enhanced_generators=True
    )
    card_gen = CreditCardGenerator()
    addr_gen = AddressGenerator()

    # Templates for synthetic records
    templates = [
        "Lost card {card}! Block it and send replacement to {address} for {name}",
        "Update billing address to {address} for card {card}",
        "{name} requested password reset - send code to {phone}",
        "Email statement for card {card} to {email}",
        "Transfer from {iban} to {address} for {name}"
    ]

    print("\nGenerating 5 synthetic records...\n")

    records = []
    for i in range(5):
        # Generate PII components
        name = generator.get_fake_value("person_name", f"name_{i}")
        card = card_gen.generate_card(card_type="random", format_type="space_4")
        address = addr_gen.generate_address(country="US", format_type="single_line")
        phone = generator.get_fake_value("phone_number", f"phone_{i}")
        email = generator.get_fake_value("email_address", f"email_{i}")
        iban = generator.get_fake_value("banking_number", f"iban_{i}")

        # Select random template
        import random
        template = random.choice(templates)

        # Fill template
        text = template.format(
            name=name,
            card=card,
            address=address,
            phone=phone,
            email=email,
            iban=iban
        )

        record = {
            "id": i + 1,
            "text": text,
            "entities": {
                "name": name,
                "card": card,
                "address": address,
                "phone": phone,
                "email": email
            }
        }

        records.append(record)

        print(f"Record {i+1}:")
        print(f"   Text: {text}")
        print(f"   Entities: {record['entities']}")
        print()

    # Save to JSONL
    output_file = "synthetic_dataset_example.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Dataset saved to: {output_file}")


def example_privacy_preservation():
    """Demonstrate privacy preservation features."""
    print("\n" + "=" * 60)
    print("Example 6: Privacy Preservation")
    print("=" * 60)

    generator = FakePIIGenerator(locale="en_US")

    print("\n1. Consistent Mapping:")
    print("   Same original value always maps to same fake value:")
    for _ in range(3):
        name = generator.get_fake_value("person_name", "john_doe")
        print(f"      john_doe -> {name}")

    print("\n2. Different Original Values:")
    for original in ["alice", "bob", "charlie"]:
        name = generator.get_fake_value("person_name", original)
        print(f"      {original} -> {name}")

    print("\n3. Memory Cache:")
    print(f"   Cached categories: {len(generator.memory)}")
    for category, cache in list(generator.memory.items())[:5]:
        if cache:
            print(f"      {category}: {len(cache)} cached values")


def example_realistic_scenarios():
    """Demonstrate realistic usage scenarios."""
    print("\n" + "=" * 60)
    print("Example 7: Realistic Usage Scenarios")
    print("=" * 60)

    generator = FakePIIGenerator(locale="en_US", use_enhanced_generators=True)

    print("\n1. Customer Support Ticket:")
    name = generator.get_fake_value("person_name", "customer_1")
    email = generator.get_fake_value("email_address", "customer_1")
    phone = generator.get_fake_value("phone_number", "customer_1")

    ticket = f"""
Support Ticket #12345
Customer: {name}
Email: {email}
Phone: {phone}
Issue: Unable to process payment
    """.strip()
    print(f"   {ticket}")

    print("\n2. Banking Transaction:")
    card_gen = CreditCardGenerator()
    card = card_gen.generate_card(card_type="visa", format_type="last_4_only")
    amount = "$1,234.56"

    transaction = f"""
Transaction Receipt
Card: {card}
Amount: {amount}
Date: 2025-01-15 14:30:22
Status: Approved
    """.strip()
    print(f"   {transaction}")

    print("\n3. Shipping Label:")
    addr_gen = AddressGenerator()
    address = addr_gen.generate_address(country="US", format_type="multi_line")

    label = f"""
Ship To:
{name}
{address}
    """.strip()
    print(f"   {label}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "     PII Dataset Generation - Complete Examples".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    try:
        example_basic_generation()
        example_enhanced_generation()
        example_credit_card_generator()
        example_address_generator()
        example_dataset_generation()
        example_privacy_preservation()
        example_realistic_scenarios()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
