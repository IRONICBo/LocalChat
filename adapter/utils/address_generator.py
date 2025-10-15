"""
Multi-country address generator with optional LLM enhancement.

This module provides realistic address generation for multiple countries,
with optional LLM-based enhancement for more contextually appropriate addresses.
"""

from typing import Dict, Optional, List
from faker import Faker
import requests
import json


class AddressGenerator:
    """
    Generates realistic addresses for multiple countries.

    Supports traditional Faker-based generation and optional LLM-enhanced
    generation for more realistic and contextually appropriate addresses.
    """

    # Mapping of country codes to Faker locales
    COUNTRY_LOCALES = {
        "US": "en_US",
        "GB": "en_GB",
        "DE": "de_DE",
        "FR": "fr_FR",
        "CN": "zh_CN",
        "JP": "ja_JP",
        "IN": "en_IN",
        "CA": "en_CA",
        "AU": "en_AU",
        "BR": "pt_BR",
        "ES": "es_ES",
        "IT": "it_IT",
        "NL": "nl_NL",
        "SE": "sv_SE",
        "NO": "no_NO",
        "DK": "da_DK",
        "FI": "fi_FI",
        "PL": "pl_PL",
        "RU": "ru_RU",
        "KR": "ko_KR",
    }

    # Country-specific address formats
    ADDRESS_FORMATS = {
        "US": {
            "fields": ["street", "city", "state", "postal_code", "country"],
            "format": "{street}\n{city}, {state} {postal_code}\n{country}"
        },
        "GB": {
            "fields": ["street", "city", "postal_code", "country"],
            "format": "{street}\n{city}\n{postal_code}\n{country}"
        },
        "DE": {
            "fields": ["street", "postal_code", "city", "country"],
            "format": "{street}\n{postal_code} {city}\n{country}"
        },
        "FR": {
            "fields": ["street", "postal_code", "city", "country"],
            "format": "{street}\n{postal_code} {city}\n{country}"
        },
        "CN": {
            "fields": ["country", "province", "city", "district", "street"],
            "format": "{country}{province}{city}{district}\n{street}"
        },
        "JP": {
            "fields": ["postal_code", "prefecture", "city", "street", "country"],
            "format": "〒{postal_code}\n{prefecture}{city}\n{street}\n{country}"
        },
    }

    def __init__(
        self,
        use_llm: bool = False,
        llm_base_url: Optional[str] = None,
        llm_model: str = "qwen2-0.5b"
    ):
        """
        Initialize the address generator.

        Args:
            use_llm: Whether to use LLM for enhanced address generation
            llm_base_url: Base URL for LLM API (e.g., "http://localhost:11434" for Ollama)
            llm_model: Model name to use for LLM generation
        """
        self.use_llm = use_llm
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.fakers: Dict[str, Faker] = {}

        # Initialize Faker instances for each locale
        for country, locale in self.COUNTRY_LOCALES.items():
            try:
                self.fakers[country] = Faker(locale)
            except Exception:
                # Fallback to en_US if locale not available
                self.fakers[country] = Faker("en_US")

    def generate_address(
        self,
        country: str = "US",
        context: Optional[str] = None,
        format_type: str = "single_line"
    ) -> str:
        """
        Generate a realistic address for the specified country.

        Args:
            country: ISO country code (e.g., "US", "GB", "DE")
            context: Optional context for LLM generation (e.g., "business district")
            format_type: Output format - "single_line", "multi_line", or "dict"

        Returns:
            Generated address as string or dict
        """
        if self.use_llm and self.llm_base_url and context:
            return self._generate_with_llm(country, context, format_type)
        else:
            return self._generate_with_faker(country, format_type)

    def _generate_with_faker(self, country: str, format_type: str) -> str:
        """Generate address using Faker library."""
        faker = self.fakers.get(country, self.fakers["US"])

        if country == "US":
            components = {
                "street": faker.street_address(),
                "city": faker.city(),
                "state": faker.state_abbr(),
                "postal_code": faker.zipcode(),
                "country": "United States"
            }
        elif country == "GB":
            components = {
                "street": faker.street_address(),
                "city": faker.city(),
                "postal_code": faker.postcode(),
                "country": "United Kingdom"
            }
        elif country == "DE":
            components = {
                "street": faker.street_address(),
                "city": faker.city(),
                "postal_code": faker.postcode(),
                "country": "Germany"
            }
        elif country == "FR":
            components = {
                "street": faker.street_address(),
                "city": faker.city(),
                "postal_code": faker.postcode(),
                "country": "France"
            }
        elif country == "CN":
            components = {
                "country": "中国",
                "province": faker.province(),
                "city": faker.city(),
                "district": faker.district(),
                "street": faker.street_address()
            }
        elif country == "JP":
            components = {
                "postal_code": faker.postcode(),
                "prefecture": faker.prefecture(),
                "city": faker.city(),
                "street": faker.street_address(),
                "country": "日本"
            }
        elif country == "IN":
            components = {
                "street": faker.street_address(),
                "city": faker.city(),
                "state": faker.state(),
                "postal_code": faker.postcode(),
                "country": "India"
            }
        elif country == "CA":
            components = {
                "street": faker.street_address(),
                "city": faker.city(),
                "province": faker.province_abbr(),
                "postal_code": faker.postcode(),
                "country": "Canada"
            }
        elif country == "AU":
            components = {
                "street": faker.street_address(),
                "city": faker.city(),
                "state": faker.state_abbr(),
                "postal_code": faker.postcode(),
                "country": "Australia"
            }
        else:
            # Default format
            components = {
                "street": faker.street_address(),
                "city": faker.city(),
                "postal_code": faker.postcode(),
                "country": faker.country()
            }

        return self._format_address(components, format_type, country)

    def _generate_with_llm(
        self,
        country: str,
        context: str,
        format_type: str
    ) -> str:
        """
        Generate address using LLM with context awareness.

        This provides more realistic addresses that match the given context.
        """
        prompt = self._build_llm_prompt(country, context)

        try:
            response = requests.post(
                f"{self.llm_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                address_text = result.get("response", "").strip()

                # Parse LLM response into components
                components = self._parse_llm_response(address_text, country)
                return self._format_address(components, format_type, country)
            else:
                # Fallback to Faker on error
                return self._generate_with_faker(country, format_type)

        except Exception as e:
            print(f"LLM generation failed: {e}, falling back to Faker")
            return self._generate_with_faker(country, format_type)

    def _build_llm_prompt(self, country: str, context: str) -> str:
        """Build prompt for LLM address generation."""
        country_names = {
            "US": "United States",
            "GB": "United Kingdom",
            "DE": "Germany",
            "FR": "France",
            "CN": "China",
            "JP": "Japan",
            "IN": "India",
            "CA": "Canada",
            "AU": "Australia",
            "BR": "Brazil"
        }

        country_name = country_names.get(country, country)

        return f"""Generate a realistic fictional address in {country_name}.
Context: {context}

Requirements:
- The address must be completely fictional (not a real address)
- It should match the typical format used in {country_name}
- It should fit the context: {context}
- Provide only the address components, no additional text

Format for {country_name}:
"""

    def _parse_llm_response(self, response: str, country: str) -> Dict[str, str]:
        """Parse LLM response into address components."""
        # Simple parsing - in production, you'd use more robust parsing
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        if country == "US":
            return {
                "street": lines[0] if len(lines) > 0 else "",
                "city": lines[1].split(",")[0] if len(lines) > 1 else "",
                "state": lines[1].split(",")[1].strip().split()[0] if len(lines) > 1 and "," in lines[1] else "",
                "postal_code": lines[1].split()[-1] if len(lines) > 1 else "",
                "country": "United States"
            }
        else:
            # Generic parsing
            return {
                "street": lines[0] if len(lines) > 0 else "",
                "city": lines[1] if len(lines) > 1 else "",
                "postal_code": lines[2] if len(lines) > 2 else "",
                "country": country
            }

    def _format_address(
        self,
        components: Dict[str, str],
        format_type: str,
        country: str
    ) -> str:
        """Format address components according to specified format."""
        if format_type == "dict":
            return str(components)

        # Get country-specific format or use default
        address_format = self.ADDRESS_FORMATS.get(country)

        if format_type == "multi_line":
            if address_format:
                return address_format["format"].format(**components)
            else:
                # Default multi-line format
                return "\n".join([v for v in components.values() if v])

        else:  # single_line
            return ", ".join([v for v in components.values() if v])

    def generate_bulk_addresses(
        self,
        count: int,
        countries: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> List[str]:
        """
        Generate multiple addresses at once.

        Args:
            count: Number of addresses to generate
            countries: List of country codes to randomly select from
            context: Optional context for all addresses

        Returns:
            List of generated addresses
        """
        if countries is None:
            countries = ["US", "GB", "DE", "FR", "CA"]

        import random
        addresses = []

        for _ in range(count):
            country = random.choice(countries)
            address = self.generate_address(country, context)
            addresses.append(address)

        return addresses


# Example usage
if __name__ == "__main__":
    # Traditional Faker-based generation
    print("=== Traditional Address Generation ===")
    gen = AddressGenerator()

    for country in ["US", "GB", "DE", "FR", "CN", "JP"]:
        print(f"\n{country}:")
        print(gen.generate_address(country, format_type="multi_line"))

    # LLM-enhanced generation (requires Ollama or similar LLM service)
    print("\n\n=== LLM-Enhanced Address Generation ===")
    gen_llm = AddressGenerator(
        use_llm=True,
        llm_base_url="http://localhost:11434"
    )

    contexts = [
        "downtown business district",
        "suburban residential area",
        "university campus",
        "industrial zone"
    ]

    for context in contexts:
        print(f"\nContext: {context}")
        print(gen_llm.generate_address("US", context=context, format_type="multi_line"))

    # Bulk generation
    print("\n\n=== Bulk Address Generation ===")
    addresses = gen.generate_bulk_addresses(
        count=5,
        countries=["US", "GB", "DE"]
    )
    for i, addr in enumerate(addresses, 1):
        print(f"{i}. {addr}")
