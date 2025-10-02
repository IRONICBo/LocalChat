from abc import ABC, abstractmethod
import hashlib
from typing import List, Tuple, Dict


class MaskingStrategy(ABC):
    @abstractmethod
    def mask(
        self, content: str, extracted_data: Dict[str, List[str]]
    ) -> Tuple[str, Dict]:
        pass


class HashMaskingStrategy(MaskingStrategy):
    def mask(
        self, content: str, extracted_data: Dict[str, List[str]]
    ) -> Tuple[str, Dict]:
        mappings = {}
        for category, items in extracted_data.items():
            for item in items:
                hash_str = hashlib.sha256(item.encode()).hexdigest()[:16]
                content = content.replace(item, f"{category.upper()}[HASH:{hash_str}]")
                mappings[hash_str] = item
        return content, mappings


if __name__ == "__main__":
    # Example usage
    sensitive_data = {"email": ["john.doe@example.com"], "phone": ["123-456-7890"]}
    masking_strategy = HashMaskingStrategy()
    masked_content, mappings = masking_strategy.mask(
        "Contact me at john.doe@example.com or 123-456-7890.", sensitive_data
    )
    print("Masked Content:", masked_content)
    print("Mappings:", mappings)
    # Masked Content: Contact me at EMAIL[HASH:836f82db99121b34] or PHONE[HASH:29ec0a06044bedff].
    # Mappings: {'836f82db99121b34': 'john.doe@example.com', '29ec0a06044bedff': '123-456-7890'}
