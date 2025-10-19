from faker import Faker

from typing import Dict, List, Optional
import random


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
            "nationality",
            "date",
            "date_of_birth",
            "domain_name",
            "email_address",
            "demographic_group",
            "gender",
            "personal_id",
            "other_id",
            "banking_number",
            "medical_condition",
            "organization_name",
            "person_name",
            "phone_number",
            "street_address",
            "password",
            "secure_credential",
            "religious_affiliation",
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

    def _generate_generic(self, original: str) -> str:
        """Fallback generator for unsupported categories."""
        return f"REPLACED_DATA_{self.faker.uuid4()[:8]}"
