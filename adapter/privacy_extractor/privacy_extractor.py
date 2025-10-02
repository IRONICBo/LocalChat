from abc import ABC, abstractmethod
import re
from typing import Dict, List


class PrivacyExtractor(ABC):
    @abstractmethod
    def extract(self, content: str) -> Dict[str, List[str]]:
        pass


class RegexPrivacyExtractor(PrivacyExtractor):
    def extract(self, content: str) -> Dict[str, List[str]]:
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "id_card": r"\b\d{17}[\dXx]\b",
        }
        extracted_data = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                extracted_data[key] = matches
        return extracted_data


if __name__ == "__main__":
    # Example usage
    extractor = RegexPrivacyExtractor()
    sensitive_data = extractor.extract(
        "Contact me at john.doe@example.com or 123-456-7890."
    )
    print(sensitive_data)
    # {'email': ['john.doe@example.com'], 'phone': ['123-456-7890']}
