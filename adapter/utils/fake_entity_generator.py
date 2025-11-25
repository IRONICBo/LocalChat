# -*- coding: utf-8 -*-
"""
Fake Entity Generator for PII Masking

Generates realistic fake entities (人名、邮箱、电话等) to replace real PII,
maintaining data utility while protecting privacy.

Uses Faker library with deterministic seeding for consistency.
"""

import hashlib
import re
from typing import Dict, Optional
from faker import Faker


class FakeEntityGenerator:
    """Generate fake entities with consistent mapping"""

    def __init__(self, locale: str = 'zh_CN', seed: Optional[int] = None):
        """
        Initialize generator with locale and optional seed

        Args:
            locale: Locale for fake data generation (zh_CN, en_US, etc.)
            seed: Optional seed for reproducible generation
        """
        self.locale = locale
        self.faker = Faker(locale)

        if seed is not None:
            Faker.seed(seed)

        # Cache for consistent mapping within same session
        self._cache = {}

    def generate_consistent(
        self,
        original_value: str,
        entity_type: str,
        session_id: str,
        **kwargs
    ) -> str:
        """
        Generate fake entity with consistent mapping

        The same original_value will always map to the same fake value
        within the same session.

        Args:
            original_value: Original PII value
            entity_type: Type of entity (PERSON, EMAIL, PHONE, etc.)
            session_id: Session ID for scoped consistency
            **kwargs: Additional parameters (gender, etc.)

        Returns:
            Fake entity value
        """
        # Create cache key
        cache_key = f"{session_id}:{entity_type}:{original_value}"

        # Return cached value if exists
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate deterministic seed from original value + session
        seed = self._generate_seed(original_value, session_id)

        # Create temporary faker with this seed
        temp_faker = Faker(self.locale)
        Faker.seed(seed)

        # Generate fake value based on entity type
        fake_value = self._generate_by_type(
            temp_faker,
            entity_type,
            original_value,
            **kwargs
        )

        # Cache and return
        self._cache[cache_key] = fake_value
        return fake_value

    def _generate_seed(self, original_value: str, session_id: str) -> int:
        """Generate deterministic seed from original value and session"""
        seed_string = f"{session_id}:{original_value}"
        hash_hex = hashlib.sha256(seed_string.encode()).hexdigest()
        # Use first 8 characters as seed (32 bits)
        return int(hash_hex[:8], 16)

    def _generate_by_type(
        self,
        faker: Faker,
        entity_type: str,
        original_value: str,
        **kwargs
    ) -> str:
        """
        Generate fake entity based on type

        Args:
            faker: Faker instance with seed set
            entity_type: Type of entity
            original_value: Original value (for format reference)
            **kwargs: Additional parameters

        Returns:
            Generated fake value
        """
        entity_type = entity_type.upper()

        if entity_type == 'PERSON':
            return self._generate_person_name(faker, **kwargs)

        elif entity_type == 'EMAIL' or entity_type == 'EMAIL_ADDRESS':
            return self._generate_email(faker, original_value, **kwargs)

        elif entity_type == 'PHONE' or entity_type == 'PHONE_NUMBER':
            return self._generate_phone(faker, original_value, **kwargs)

        elif entity_type == 'LOCATION' or entity_type == 'ADDRESS':
            return self._generate_address(faker, **kwargs)

        elif entity_type == 'CREDIT_CARD':
            return self._generate_credit_card(faker, original_value, **kwargs)

        elif entity_type == 'SSN' or entity_type == 'US_SSN':
            return self._generate_ssn(faker, **kwargs)

        elif entity_type == 'ORGANIZATION':
            return self._generate_organization(faker, **kwargs)

        elif entity_type == 'IP_ADDRESS':
            return self._generate_ip_address(faker, **kwargs)

        elif entity_type == 'URL':
            return self._generate_url(faker, **kwargs)

        elif entity_type == 'DATE' or entity_type == 'DATE_TIME':
            return self._generate_date(faker, **kwargs)

        else:
            # Default: return generic fake value
            return f"[FAKE_{entity_type}]"

    def _generate_person_name(self, faker: Faker, **kwargs) -> str:
        """Generate fake person name"""
        gender = kwargs.get('gender')

        if gender == 'male':
            return faker.name_male()
        elif gender == 'female':
            return faker.name_female()
        else:
            return faker.name()

    def _generate_email(
        self,
        faker: Faker,
        original_value: str,
        **kwargs
    ) -> str:
        """
        Generate fake email address

        Tries to maintain similar format to original (e.g., keep domain if corporate)
        """
        # Check if original has a specific domain we should preserve format
        if '@' in original_value:
            _, domain = original_value.split('@', 1)

            # For common personal domains, generate completely fake email
            personal_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', '163.com', 'qq.com']

            if any(pd in domain.lower() for pd in personal_domains):
                return faker.email()
            else:
                # For corporate domains, keep similar structure
                username = faker.user_name()
                return f"{username}@{domain}"
        else:
            return faker.email()

    def _generate_phone(
        self,
        faker: Faker,
        original_value: str,
        **kwargs
    ) -> str:
        """
        Generate fake phone number

        Tries to maintain similar format to original
        """
        phone = faker.phone_number()

        # Try to match format of original
        # If original has dashes, ensure fake also has dashes
        if '-' in original_value and '-' not in phone:
            # Reformat: 13800000000 -> 138-0000-0000
            if len(phone) == 11:
                phone = f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"
            elif len(phone) == 10:
                phone = f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"

        # If original has parentheses, try to match
        if '(' in original_value and '(' not in phone:
            if len(phone) >= 10:
                # (555) 123-4567 format
                phone = f"({phone[:3]}) {phone[3:6]}-{phone[6:10]}"

        return phone

    def _generate_address(self, faker: Faker, **kwargs) -> str:
        """Generate fake address"""
        return faker.address().replace('\n', ', ')

    def _generate_credit_card(
        self,
        faker: Faker,
        original_value: str,
        **kwargs
    ) -> str:
        """
        Generate fake credit card number

        Maintains format (dashes, spaces) from original
        """
        fake_cc = faker.credit_card_number()

        # Match format of original
        if '-' in original_value:
            # Format: 1234-5678-9012-3456
            if len(fake_cc) == 16:
                fake_cc = f"{fake_cc[:4]}-{fake_cc[4:8]}-{fake_cc[8:12]}-{fake_cc[12:]}"
        elif ' ' in original_value:
            # Format: 1234 5678 9012 3456
            if len(fake_cc) == 16:
                fake_cc = f"{fake_cc[:4]} {fake_cc[4:8]} {fake_cc[8:12]} {fake_cc[12:]}"

        return fake_cc

    def _generate_ssn(self, faker: Faker, **kwargs) -> str:
        """Generate fake SSN"""
        return faker.ssn()

    def _generate_organization(self, faker: Faker, **kwargs) -> str:
        """Generate fake organization name"""
        return faker.company()

    def _generate_ip_address(self, faker: Faker, **kwargs) -> str:
        """Generate fake IP address"""
        return faker.ipv4()

    def _generate_url(self, faker: Faker, **kwargs) -> str:
        """Generate fake URL"""
        return faker.url()

    def _generate_date(self, faker: Faker, **kwargs) -> str:
        """Generate fake date"""
        return faker.date()

    def clear_cache(self):
        """Clear the internal cache"""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Get current cache size"""
        return len(self._cache)


class MultiLocaleGenerator:
    """Manage multiple generators for different locales"""

    def __init__(self):
        self.generators: Dict[str, FakeEntityGenerator] = {}

    def get_generator(self, locale: str = 'zh_CN') -> FakeEntityGenerator:
        """
        Get or create generator for locale

        Args:
            locale: Locale code (zh_CN, en_US, ja_JP, etc.)

        Returns:
            FakeEntityGenerator instance
        """
        if locale not in self.generators:
            self.generators[locale] = FakeEntityGenerator(locale=locale)

        return self.generators[locale]

    def generate_consistent(
        self,
        original_value: str,
        entity_type: str,
        session_id: str,
        locale: str = 'zh_CN',
        **kwargs
    ) -> str:
        """
        Generate fake entity with specified locale

        Args:
            original_value: Original PII value
            entity_type: Type of entity
            session_id: Session ID
            locale: Locale for generation
            **kwargs: Additional parameters

        Returns:
            Generated fake value
        """
        generator = self.get_generator(locale)
        return generator.generate_consistent(
            original_value,
            entity_type,
            session_id,
            **kwargs
        )

    def detect_locale(self, text: str) -> str:
        """
        Detect locale from text

        Simple heuristic based on character ranges

        Args:
            text: Text to analyze

        Returns:
            Detected locale code
        """
        # Check for Chinese characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh_CN'

        # Check for Japanese characters
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja_JP'

        # Check for Korean characters
        if re.search(r'[\uac00-\ud7af]', text):
            return 'ko_KR'

        # Default to English
        return 'en_US'

    def clear_all_caches(self):
        """Clear caches for all generators"""
        for generator in self.generators.values():
            generator.clear_cache()


# Global instance for easy access
_global_multi_generator = MultiLocaleGenerator()


def generate_fake_entity(
    original_value: str,
    entity_type: str,
    session_id: str,
    locale: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to generate fake entity

    Args:
        original_value: Original PII value
        entity_type: Type of entity
        session_id: Session ID
        locale: Optional locale (auto-detect if not provided)
        **kwargs: Additional parameters

    Returns:
        Generated fake value

    Example:
        >>> fake_name = generate_fake_entity("张三", "PERSON", "session_123")
        >>> fake_email = generate_fake_entity("alice@example.com", "EMAIL", "session_123")
    """
    if locale is None:
        locale = _global_multi_generator.detect_locale(original_value)

    return _global_multi_generator.generate_consistent(
        original_value,
        entity_type,
        session_id,
        locale=locale,
        **kwargs
    )


if __name__ == "__main__":
    # Test the generator
    print("=== Testing Fake Entity Generator ===\n")

    session_id = "test_session_001"

    # Test Chinese locale
    print("--- Chinese (zh_CN) ---")
    gen_zh = FakeEntityGenerator(locale='zh_CN')

    name1 = gen_zh.generate_consistent("张三", "PERSON", session_id)
    name2 = gen_zh.generate_consistent("张三", "PERSON", session_id)  # Should be same
    print(f"Name 1: {name1}")
    print(f"Name 2: {name2}")
    print(f"Consistent: {name1 == name2}\n")

    email = gen_zh.generate_consistent("zhangsan@example.com", "EMAIL", session_id)
    print(f"Email: {email}\n")

    phone = gen_zh.generate_consistent("138-0000-0000", "PHONE", session_id)
    print(f"Phone: {phone}\n")

    address = gen_zh.generate_consistent("北京市朝阳区", "ADDRESS", session_id)
    print(f"Address: {address}\n")

    # Test English locale
    print("--- English (en_US) ---")
    gen_en = FakeEntityGenerator(locale='en_US')

    name_en = gen_en.generate_consistent("John Smith", "PERSON", session_id)
    print(f"Name: {name_en}\n")

    email_en = gen_en.generate_consistent("john.smith@company.com", "EMAIL", session_id)
    print(f"Email: {email_en}\n")

    # Test convenience function with auto-detect
    print("--- Auto-detect Locale ---")
    fake1 = generate_fake_entity("李四", "PERSON", session_id)
    fake2 = generate_fake_entity("John Doe", "PERSON", session_id)
    print(f"Chinese name: {fake1}")
    print(f"English name: {fake2}\n")

    print(f"Cache size (zh_CN): {gen_zh.get_cache_size()}")
    print(f"Cache size (en_US): {gen_en.get_cache_size()}")
