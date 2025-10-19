"""
Enhanced credit card number generator with multiple format support.

This module generates valid synthetic credit card numbers in various formats
for testing and dataset generation purposes. All generated numbers pass the
Luhn algorithm but are completely synthetic and not linked to real accounts.
"""

import random
from typing import Optional, List


class CreditCardGenerator:
    """
    Generates synthetic credit card numbers in various formats.

    All generated numbers:
    - Pass Luhn algorithm validation
    - Follow real card number patterns
    - Are completely synthetic and safe for testing
    - Support multiple card types (Visa, Mastercard, Amex, etc.)
    """

    # Card type prefixes (IIN ranges)
    CARD_PREFIXES = {
        "visa": ["4"],
        "mastercard": ["51", "52", "53", "54", "55", "2221", "2720"],
        "amex": ["34", "37"],
        "discover": ["6011", "644", "645", "646", "647", "648", "649", "65"],
        "jcb": ["3528", "3589"],
        "diners": ["36", "38"],
        "unionpay": ["62"],
    }

    # Card lengths by type
    CARD_LENGTHS = {
        "visa": 16,
        "mastercard": 16,
        "amex": 15,
        "discover": 16,
        "jcb": 16,
        "diners": 14,
        "unionpay": 16,
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the credit card generator.

        Args:
            seed: Optional random seed for reproducible generation
        """
        if seed is not None:
            random.seed(seed)

    def generate_card(
        self,
        card_type: str = "random",
        format_type: str = "space_4",
        include_cvv: bool = False,
        include_expiry: bool = False
    ) -> str:
        """
        Generate a synthetic credit card number.

        Args:
            card_type: Type of card - "visa", "mastercard", "amex", "discover", "random"
            format_type: Format - "plain", "space_4", "dash_4", "last_4_only", "custom"
            include_cvv: Whether to include CVV code
            include_expiry: Whether to include expiry date

        Returns:
            Formatted credit card number (and optionally CVV/expiry)
        """
        # Generate base card number
        if card_type == "random":
            card_type = random.choice(list(self.CARD_PREFIXES.keys()))

        card_number = self._generate_valid_number(card_type)

        # Format the number
        formatted_number = self._format_number(card_number, format_type)

        # Add additional components if requested
        result = formatted_number

        if include_cvv:
            cvv = self._generate_cvv(card_type)
            result += f" CVV: {cvv}"

        if include_expiry:
            expiry = self._generate_expiry()
            result += f" Exp: {expiry}"

        return result

    def _generate_valid_number(self, card_type: str) -> str:
        """
        Generate a valid card number using Luhn algorithm.

        Args:
            card_type: Type of card to generate

        Returns:
            Valid card number as string
        """
        # Get random prefix for this card type
        prefix = random.choice(self.CARD_PREFIXES[card_type])

        # Get target length
        length = self.CARD_LENGTHS[card_type]

        # Generate random digits for the rest (except check digit)
        remaining_length = length - len(prefix) - 1
        middle_digits = "".join([str(random.randint(0, 9)) for _ in range(remaining_length)])

        # Combine prefix and middle digits
        partial_number = prefix + middle_digits

        # Calculate Luhn check digit
        check_digit = self._calculate_luhn_check_digit(partial_number)

        return partial_number + str(check_digit)

    def _calculate_luhn_check_digit(self, partial_number: str) -> int:
        """
        Calculate Luhn algorithm check digit.

        Args:
            partial_number: Card number without check digit

        Returns:
            Check digit (0-9)
        """
        digits = [int(d) for d in partial_number]

        # Double every second digit from right to left
        for i in range(len(digits) - 1, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9

        # Sum all digits
        total = sum(digits)

        # Check digit makes total divisible by 10
        check_digit = (10 - (total % 10)) % 10

        return check_digit

    def verify_luhn(self, card_number: str) -> bool:
        """
        Verify if a card number passes Luhn algorithm.

        Args:
            card_number: Card number to verify (digits only)

        Returns:
            True if valid, False otherwise
        """
        # Remove non-digit characters
        card_number = "".join(filter(str.isdigit, card_number))

        if not card_number:
            return False

        digits = [int(d) for d in card_number]

        # Double every second digit from right to left (excluding check digit)
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9

        return sum(digits) % 10 == 0

    def _format_number(self, card_number: str, format_type: str) -> str:
        """
        Format card number according to specified format.

        Args:
            card_number: Raw card number
            format_type: Desired format

        Returns:
            Formatted card number
        """
        if format_type == "plain":
            return card_number

        elif format_type == "space_4":
            # Most common: 4532 1111 2222 3333
            return " ".join([card_number[i:i+4] for i in range(0, len(card_number), 4)])

        elif format_type == "dash_4":
            # Alternative: 4532-1111-2222-3333
            return "-".join([card_number[i:i+4] for i in range(0, len(card_number), 4)])

        elif format_type == "last_4_only":
            # Masked: **** **** **** 3333
            groups = [card_number[i:i+4] for i in range(0, len(card_number), 4)]
            masked_groups = ["****"] * (len(groups) - 1) + [groups[-1]]
            return " ".join(masked_groups)

        elif format_type == "first_6_last_4":
            # Common secure format: 453211******3333
            if len(card_number) <= 10:
                return card_number
            return card_number[:6] + "*" * (len(card_number) - 10) + card_number[-4:]

        elif format_type == "amex_format":
            # Amex specific: 3782 822463 10005
            if len(card_number) == 15:
                return f"{card_number[:4]} {card_number[4:10]} {card_number[10:]}"
            else:
                return self._format_number(card_number, "space_4")

        elif format_type == "diners_format":
            # Diners specific: 3056 930902 5904
            if len(card_number) == 14:
                return f"{card_number[:4]} {card_number[4:10]} {card_number[10:]}"
            else:
                return self._format_number(card_number, "space_4")

        else:
            # Default to space_4
            return self._format_number(card_number, "space_4")

    def _generate_cvv(self, card_type: str) -> str:
        """
        Generate a CVV code.

        Args:
            card_type: Type of card (Amex uses 4 digits, others use 3)

        Returns:
            CVV code as string
        """
        if card_type == "amex":
            return "".join([str(random.randint(0, 9)) for _ in range(4)])
        else:
            return "".join([str(random.randint(0, 9)) for _ in range(3)])

    def _generate_expiry(self) -> str:
        """
        Generate a future expiry date.

        Returns:
            Expiry date in MM/YY format
        """
        # Generate date 1-5 years in the future
        import datetime
        today = datetime.date.today()
        years_ahead = random.randint(1, 5)
        month = random.randint(1, 12)
        year = (today.year + years_ahead) % 100  # Last 2 digits

        return f"{month:02d}/{year:02d}"

    def generate_bulk(
        self,
        count: int,
        card_types: Optional[List[str]] = None,
        format_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate multiple card numbers at once.

        Args:
            count: Number of cards to generate
            card_types: List of card types to randomly select from
            format_types: List of formats to randomly select from

        Returns:
            List of generated card numbers
        """
        if card_types is None:
            card_types = ["visa", "mastercard", "amex", "discover"]

        if format_types is None:
            format_types = ["space_4", "dash_4", "plain"]

        cards = []
        for _ in range(count):
            card_type = random.choice(card_types)
            format_type = random.choice(format_types)
            card = self.generate_card(card_type, format_type)
            cards.append(card)

        return cards

    def generate_with_metadata(
        self,
        card_type: str = "random",
        format_type: str = "space_4"
    ) -> dict:
        """
        Generate card with full metadata.

        Args:
            card_type: Type of card
            format_type: Format type

        Returns:
            Dictionary with card details
        """
        if card_type == "random":
            card_type = random.choice(list(self.CARD_PREFIXES.keys()))

        card_number = self._generate_valid_number(card_type)
        formatted_number = self._format_number(card_number, format_type)

        return {
            "card_number": formatted_number,
            "card_type": card_type,
            "cvv": self._generate_cvv(card_type),
            "expiry": self._generate_expiry(),
            "last_4": card_number[-4:],
            "first_6": card_number[:6],
            "length": len(card_number),
            "valid": self.verify_luhn(card_number)
        }


# Example usage and testing
if __name__ == "__main__":
    generator = CreditCardGenerator()

    print("=== Credit Card Generator Examples ===\n")

    # Example 1: Different card types
    print("1. Different Card Types:")
    for card_type in ["visa", "mastercard", "amex", "discover"]:
        card = generator.generate_card(card_type=card_type, format_type="space_4")
        print(f"   {card_type.capitalize()}: {card}")

    # Example 2: Different formats
    print("\n2. Different Formats (Visa):")
    formats = ["plain", "space_4", "dash_4", "last_4_only", "first_6_last_4"]
    for fmt in formats:
        card = generator.generate_card(card_type="visa", format_type=fmt)
        print(f"   {fmt}: {card}")

    # Example 3: With CVV and Expiry
    print("\n3. Complete Card Details:")
    card = generator.generate_card(
        card_type="visa",
        format_type="space_4",
        include_cvv=True,
        include_expiry=True
    )
    print(f"   {card}")

    # Example 4: Card-specific formats
    print("\n4. Card-Specific Formats:")
    amex = generator.generate_card(card_type="amex", format_type="amex_format")
    print(f"   Amex: {amex}")

    diners = generator.generate_card(card_type="diners", format_type="diners_format")
    print(f"   Diners: {diners}")

    # Example 5: Bulk generation
    print("\n5. Bulk Generation (5 random cards):")
    bulk_cards = generator.generate_bulk(
        count=5,
        card_types=["visa", "mastercard"],
        format_types=["space_4", "dash_4"]
    )
    for i, card in enumerate(bulk_cards, 1):
        print(f"   {i}. {card}")

    # Example 6: Full metadata
    print("\n6. Card with Full Metadata:")
    metadata = generator.generate_with_metadata(card_type="visa", format_type="space_4")
    for key, value in metadata.items():
        print(f"   {key}: {value}")

    # Example 7: Luhn verification
    print("\n7. Luhn Algorithm Verification:")
    test_cards = [
        generator.generate_card(card_type="visa", format_type="plain"),
        "4532111111111111",  # Valid test card
        "1234567890123456",  # Invalid
    ]
    for card in test_cards:
        valid = generator.verify_luhn(card)
        print(f"   {card}: {'Valid ✓' if valid else 'Invalid ✗'}")

    # Example 8: Realistic usage scenarios
    print("\n8. Realistic Usage Scenarios:")

    # Scenario 1: E-commerce checkout display
    card_metadata = generator.generate_with_metadata(card_type="visa")
    print(f"   Checkout: Card ending in {card_metadata['last_4']}")

    # Scenario 2: Secure display (first 6 + last 4)
    secure_card = generator.generate_card(card_type="mastercard", format_type="first_6_last_4")
    print(f"   Secure Display: {secure_card}")

    # Scenario 3: Full card input
    full_card = generator.generate_card(card_type="visa", format_type="space_4", include_cvv=True, include_expiry=True)
    print(f"   Full Input: {full_card}")
