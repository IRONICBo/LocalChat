from faker import Faker

from typing import Dict, List, Optional, Union
import random
import re


class FakePIIGenerator:
    """
    Generates synthetic data for various PII categories using the Faker library.
    Ensures consistency by storing previously generated values.
    """

    def __init__(
        self,
        locale: str = "en_US",
        use_enhanced_generators: bool = False,
        llm_base_url: Optional[str] = None
    ):
        """
        Initializes the fake data generator with a specified locale.

        Args:
            locale: Locale setting for generating fake data.
            use_enhanced_generators: Use enhanced address and credit card generators
            llm_base_url: Base URL for LLM API (optional, for enhanced address generation)
        """
        self.faker = Faker(locale)
        self.memory: Dict[str, Dict[str, str]] = {
            category: {} for category in self.get_supported_categories()
        }
        self.use_enhanced_generators = use_enhanced_generators

        # Initialize enhanced generators if enabled
        if use_enhanced_generators:
            try:
                from utils.address_generator import AddressGenerator
                from utils.credit_card_generator import CreditCardGenerator

                self.address_generator = AddressGenerator(
                    use_llm=bool(llm_base_url),
                    llm_base_url=llm_base_url
                )
                self.credit_card_generator = CreditCardGenerator()
            except ImportError:
                print("Warning: Enhanced generators not available, falling back to Faker")
                self.use_enhanced_generators = False

    @staticmethod
    def get_supported_categories() -> List[str]:
        """
        Provides a list of PII categories that can be generated.

        Returns:
            List of supported PII categories.
        """
        return [
            "age",
            "credit_card_info",
            "credit_card_number",  # NEW: Specific card number generation
            "credit_card_cvv",     # NEW: CVV generation
            "credit_card_expiry",  # NEW: Expiry date generation
            "nationality",
            "date",
            "date_of_birth",
            "domain_name",
            "email_address",
            "demographic_group",
            "gender",
            "gender_identity",     # NEW: Extended gender options
            "sex",                 # NEW: Biological sex
            "personal_id",
            "other_id",
            "banking_number",
            "iban",                # NEW: IBAN number
            "routing_number",      # NEW: Bank routing number
            "medical_condition",
            "organization_name",
            "person_name",
            "first_name",          # NEW: First name only
            "last_name",           # NEW: Last name only
            "phone_number",
            "phone_number_intl",   # NEW: International format
            "street_address",
            "password",
            "secure_credential",
            "religious_affiliation",
            "ip_address",          # NEW: IP address
            "ipv4_address",        # NEW: IPv4 specific
            "ipv6_address",        # NEW: IPv6 specific
            "mac_address",         # NEW: MAC address
            "numeric_id",          # NEW: Numeric identifier
            "numeric_value",       # NEW: Generic numeric value
            "amount",              # NEW: Monetary amount
            "percentage",          # NEW: Percentage value
            "zip_code",            # NEW: ZIP/Postal code
            "ssn",                 # NEW: Social Security Number
            "driver_license",      # NEW: Driver's license
            "passport_number",     # NEW: Passport number
            "url",                 # NEW: URL
            "username",            # NEW: Username
        ]

    def get_fake_value(self, category: str, original_value: str) -> str:
        """
        Retrieves or generates a fake value for a given PII category and original value.
        If the original value has been encountered before, returns the previously generated fake value.

        Args:
            category: The PII category for which to generate a fake value.
            original_value: The original PII value to be replaced.

        Returns:
            The generated fake value.
        """
        if original_value in self.memory[category]:
            return self.memory[category][original_value]

        fake_value = self._generate_fake_value(category, original_value)
        self.memory[category][original_value] = fake_value
        return fake_value

    def _generate_fake_value(self, category: str, original: str) -> str:
        """
        Generates a fake value based on the specified category.

        Args:
            category: The PII category for which to generate a fake value.
            original: The original PII value to be replaced.

        Returns:
            The generated fake value.
        """
        method_name = f"_generate_{category.lower()}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(original)
        return self._generate_generic(original)

    def _generate_age(self, original: str) -> str:
        """Generates a fake age value."""
        return str(self.faker.random_int(min=18, max=90))

    def _generate_credit_card_info(self, original: str) -> str:
        """Generates a fake credit card number with enhanced formatting."""
        if self.use_enhanced_generators and hasattr(self, 'credit_card_generator'):
            # Use enhanced generator with random format
            formats = ["plain", "space_4", "dash_4"]
            format_type = random.choice(formats)
            return self.credit_card_generator.generate_card(
                card_type="random",
                format_type=format_type
            )
        else:
            return self.faker.credit_card_number(card_type=None)

    def _generate_nationality(self, original: str) -> str:
        """Generates a fake nationality."""
        return self.faker.country()

    def _generate_date(self, original: str) -> str:
        """Generates a fake date."""
        return self.faker.date()

    def _generate_date_of_birth(self, original: str) -> str:
        """Generates a fake date of birth."""
        return self.faker.date_of_birth(minimum_age=18, maximum_age=90).strftime(
            "%Y-%m-%d"
        )

    def _generate_domain_name(self, original: str) -> str:
        """Generates a fake domain name."""
        return self.faker.domain_name()

    def _generate_email_address(self, original: str) -> str:
        """Generates a fake email address."""
        return self.faker.email()

    def _generate_demographic_group(self, original: str) -> str:
        """Generates a fake demographic group."""
        return self.faker.word()

    def _generate_gender(self, original: str) -> str:
        """Generates a fake gender."""
        return self.faker.random_element(
            elements=("Male", "Female", "Non-binary", "Prefer not to say")
        )

    def _generate_personal_id(self, original: str) -> str:
        """Generates a fake personal ID."""
        return self.faker.ssn()

    def _generate_other_id(self, original: str) -> str:
        """Generates a fake organization ID."""
        return f"ID-{self.faker.uuid4()[:8]}"

    def _generate_banking_number(self, original: str) -> str:
        """Generates a fake banking number."""
        return self.faker.bban()

    def _generate_medical_condition(self, original: str) -> str:
        """Generates a fake medical condition."""
        conditions = [
            "Common Cold",
            "Seasonal Allergies",
            "Migraine",
            "Minor Sprain",
            "General Checkup",
        ]
        return self.faker.random_element(elements=conditions)

    def _generate_organization_name(self, original: str) -> str:
        """Generates a fake organization name."""
        return self.faker.company()

    def _generate_person_name(self, original: str) -> str:
        """Generates a fake person name."""
        return self.faker.name()

    def _generate_phone_number(self, original: str) -> str:
        """Generates a fake phone number."""
        return self.faker.phone_number()

    def _generate_street_address(self, original: str) -> str:
        """Generates a fake street address with multi-country support."""
        if self.use_enhanced_generators and hasattr(self, 'address_generator'):
            # Use enhanced generator with random country
            countries = ["US", "GB", "DE", "FR", "CA", "AU"]
            country = random.choice(countries)
            return self.address_generator.generate_address(
                country=country,
                format_type="single_line"
            )
        else:
            return self.faker.street_address()

    def _generate_password(self, original: str) -> str:
        """Generates a fake password."""
        return self.faker.password(
            length=12, special_chars=True, digits=True, upper_case=True, lower_case=True
        )

    def _generate_secure_credential(self, original: str) -> str:
        """Generates a fake secure credential."""
        return f"FAKE_API_KEY_{self.faker.uuid4()}"

    def _generate_religious_affiliation(self, original: str) -> str:
        """Generates a fake religious affiliation."""
        affiliations = ["Religion A", "Faith B", "Belief System C", "Spiritual Group D"]
        return self.faker.random_element(elements=affiliations)

    # ==================== NEW: Enhanced Credit Card Methods ====================

    def _generate_credit_card_number(self, original: str) -> str:
        """
        Generates a fake credit card number with format matching.

        Attempts to match the format of the original card number (spaces, dashes, etc.).
        """
        if self.use_enhanced_generators and hasattr(self, 'credit_card_generator'):
            # Detect format from original
            if ' ' in original:
                format_type = "space_4"
            elif '-' in original:
                format_type = "dash_4"
            elif '*' in original:
                format_type = "last_4_only"
            else:
                format_type = "plain"

            return self.credit_card_generator.generate_card(
                card_type="random",
                format_type=format_type
            )
        else:
            return self.faker.credit_card_number(card_type=None)

    def _generate_credit_card_cvv(self, original: str) -> str:
        """Generates a fake CVV code (3 or 4 digits based on original)."""
        if len(original) == 4:
            # Amex-style 4-digit CVV
            return str(random.randint(1000, 9999))
        else:
            # Standard 3-digit CVV
            return str(random.randint(100, 999))

    def _generate_credit_card_expiry(self, original: str) -> str:
        """
        Generates a fake credit card expiry date.

        Matches format: MM/YY, MM/YYYY, MM-YY, etc.
        """
        import datetime
        today = datetime.date.today()
        future_year = today.year + random.randint(1, 5)
        month = random.randint(1, 12)

        # Detect format from original
        if '/' in original:
            separator = '/'
        elif '-' in original:
            separator = '-'
        else:
            separator = '/'

        # Detect year format
        year_match = re.search(r'\d{2,4}$', original)
        if year_match and len(year_match.group()) == 4:
            year_str = str(future_year)
        else:
            year_str = str(future_year % 100).zfill(2)

        return f"{month:02d}{separator}{year_str}"

    # ==================== NEW: Enhanced Gender Methods ====================

    def _generate_gender_identity(self, original: str) -> str:
        """Generates a fake gender identity with extended options."""
        identities = [
            "Male", "Female", "Non-binary", "Genderqueer", "Genderfluid",
            "Agender", "Two-Spirit", "Prefer not to say", "Other"
        ]
        return self.faker.random_element(elements=identities)

    def _generate_sex(self, original: str) -> str:
        """Generates a fake biological sex."""
        options = ["Male", "Female", "Intersex", "Prefer not to say"]
        return self.faker.random_element(elements=options)

    # ==================== NEW: Enhanced Name Methods ====================

    def _generate_first_name(self, original: str) -> str:
        """Generates a fake first name."""
        return self.faker.first_name()

    def _generate_last_name(self, original: str) -> str:
        """Generates a fake last name."""
        return self.faker.last_name()

    # ==================== NEW: Enhanced Banking Methods ====================

    def _generate_iban(self, original: str) -> str:
        """Generates a fake IBAN number."""
        return self.faker.iban()

    def _generate_routing_number(self, original: str) -> str:
        """Generates a fake bank routing number (9 digits)."""
        return ''.join([str(random.randint(0, 9)) for _ in range(9)])

    # ==================== NEW: Phone Number Formats ====================

    def _generate_phone_number_intl(self, original: str) -> str:
        """Generates a fake international phone number."""
        country_codes = ['+1', '+44', '+49', '+33', '+86', '+81', '+61', '+91']
        code = random.choice(country_codes)
        number = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        return f"{code} {number[:3]} {number[3:6]} {number[6:]}"

    # ==================== NEW: Network/Technical IDs ====================

    def _generate_ip_address(self, original: str) -> str:
        """Generates a fake IP address (auto-detects v4 or v6)."""
        if ':' in original:
            return self._generate_ipv6_address(original)
        else:
            return self._generate_ipv4_address(original)

    def _generate_ipv4_address(self, original: str) -> str:
        """Generates a fake IPv4 address."""
        return self.faker.ipv4()

    def _generate_ipv6_address(self, original: str) -> str:
        """Generates a fake IPv6 address."""
        return self.faker.ipv6()

    def _generate_mac_address(self, original: str) -> str:
        """Generates a fake MAC address."""
        return self.faker.mac_address()

    # ==================== NEW: Numeric Values ====================

    def _generate_numeric_id(self, original: str) -> str:
        """
        Generates a fake numeric ID matching the original length.

        Preserves leading zeros and digit count.
        """
        # Extract only digits from original
        digits_only = ''.join(filter(str.isdigit, original))
        length = len(digits_only) if digits_only else 8

        # Generate new numeric ID with same length
        if digits_only.startswith('0'):
            # Preserve leading zeros possibility
            return ''.join([str(random.randint(0, 9)) for _ in range(length)])
        else:
            # Generate without leading zeros
            first_digit = str(random.randint(1, 9))
            rest = ''.join([str(random.randint(0, 9)) for _ in range(length - 1)])
            return first_digit + rest

    def _generate_numeric_value(self, original: str) -> str:
        """
        Generates a fake numeric value matching the original format.

        Handles integers, decimals, and formatted numbers.
        """
        # Check for decimal point
        if '.' in original:
            integer_part, decimal_part = original.split('.')
            int_len = len(integer_part.replace(',', '').replace(' ', ''))
            dec_len = len(decimal_part)

            int_val = random.randint(10**(int_len-1), 10**int_len - 1) if int_len > 0 else 0
            dec_val = random.randint(0, 10**dec_len - 1)

            return f"{int_val}.{str(dec_val).zfill(dec_len)}"
        elif ',' in original:
            # Formatted number with thousand separators
            digits_only = original.replace(',', '')
            new_value = random.randint(10**(len(digits_only)-1), 10**len(digits_only) - 1)
            return f"{new_value:,}"
        else:
            # Plain integer
            length = len(original)
            return str(random.randint(10**(length-1), 10**length - 1))

    def _generate_amount(self, original: str) -> str:
        """
        Generates a fake monetary amount.

        Detects currency symbols and formats.
        """
        # Extract currency symbol if present
        currency_symbols = ['$', '€', '£', '¥', '₹', '₽']
        symbol = ''
        for s in currency_symbols:
            if s in original:
                symbol = s
                break

        # Extract numeric part
        numeric_part = re.sub(r'[^\d.]', '', original)

        # Generate new amount
        if '.' in numeric_part:
            parts = numeric_part.split('.')
            int_len = len(parts[0])
            max_val = 10**int_len - 1
            new_int = random.randint(1, max_val)
            new_dec = random.randint(0, 99)
            amount = f"{new_int}.{new_dec:02d}"
        else:
            length = len(numeric_part) if numeric_part else 3
            amount = str(random.randint(1, 10**length - 1))

        return f"{symbol}{amount}" if symbol else amount

    def _generate_percentage(self, original: str) -> str:
        """Generates a fake percentage value."""
        # Check for decimal percentages
        if '.' in original:
            return f"{random.uniform(0, 100):.2f}%"
        else:
            return f"{random.randint(0, 100)}%"

    # ==================== NEW: Document IDs ====================

    def _generate_zip_code(self, original: str) -> str:
        """Generates a fake ZIP/postal code."""
        return self.faker.postcode()

    def _generate_ssn(self, original: str) -> str:
        """Generates a fake Social Security Number."""
        return self.faker.ssn()

    def _generate_driver_license(self, original: str) -> str:
        """Generates a fake driver's license number."""
        # US-style: letter + 7 digits
        letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        digits = ''.join([str(random.randint(0, 9)) for _ in range(7)])
        return f"{letter}{digits}"

    def _generate_passport_number(self, original: str) -> str:
        """Generates a fake passport number."""
        # US-style: letter + 8 digits
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=1))
        digits = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        return f"{letters}{digits}"

    # ==================== NEW: Web/Username ====================

    def _generate_url(self, original: str) -> str:
        """Generates a fake URL."""
        return self.faker.url()

    def _generate_username(self, original: str) -> str:
        """Generates a fake username."""
        return self.faker.user_name()

    def _generate_generic(self, original: str) -> str:
        """Fallback generator for unsupported categories."""
        return f"REPLACED_DATA_{self.faker.uuid4()[:8]}"

    # ==================== Utility Methods ====================

    def generate_consistent_identity(self, identity_id: str) -> Dict[str, str]:
        """
        Generate a consistent fake identity with multiple related fields.

        All fields for the same identity_id will be consistently generated.

        Args:
            identity_id: Unique identifier for this identity

        Returns:
            Dictionary with all identity fields
        """
        return {
            "first_name": self.get_fake_value("first_name", f"{identity_id}_first"),
            "last_name": self.get_fake_value("last_name", f"{identity_id}_last"),
            "full_name": self.get_fake_value("person_name", identity_id),
            "email": self.get_fake_value("email_address", identity_id),
            "phone": self.get_fake_value("phone_number", identity_id),
            "phone_intl": self.get_fake_value("phone_number_intl", identity_id),
            "address": self.get_fake_value("street_address", identity_id),
            "dob": self.get_fake_value("date_of_birth", identity_id),
            "ssn": self.get_fake_value("ssn", identity_id),
            "gender": self.get_fake_value("gender", identity_id),
            "credit_card": self.get_fake_value("credit_card_number", identity_id),
        }

    def generate_batch(
        self,
        category: str,
        count: int,
        format_hint: Optional[str] = None
    ) -> List[str]:
        """
        Generate multiple fake values for a category.

        Args:
            category: PII category
            count: Number of values to generate
            format_hint: Optional format hint for generation

        Returns:
            List of generated values
        """
        values = []
        for i in range(count):
            original = format_hint if format_hint else f"batch_{i}"
            values.append(self.get_fake_value(category, f"{category}_{i}_{original}"))
        return values

    def clear_memory(self, category: Optional[str] = None):
        """
        Clear cached values.

        Args:
            category: Specific category to clear, or None to clear all
        """
        if category:
            if category in self.memory:
                self.memory[category] = {}
        else:
            for cat in self.memory:
                self.memory[cat] = {}
