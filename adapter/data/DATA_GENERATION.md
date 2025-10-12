# Dataset Generation Guide

## Overview

This document describes the dataset generation pipeline for the LLM Adapter project, which creates synthetic datasets containing Personally Identifiable Information (PII) for training privacy-aware language models.

## Purpose

The generated datasets serve multiple purposes:

1. **Privacy Protection Training**: Train LLM adapters to recognize and handle sensitive information appropriately
2. **PII Detection Evaluation**: Benchmark PII detection and masking systems
3. **Synthetic Testing**: Provide realistic test data without exposing real user information
4. **Model Fine-tuning**: Fine-tune language models to better understand context with PII entities

## Privacy Considerations

### Synthetic Data Only

All generated data is **completely synthetic** and does not contain any real personal information:

- Names, addresses, phone numbers, credit card numbers, and other PII are generated using the Faker library
- No real user data is collected, stored, or processed
- Generated data follows realistic patterns but represents fictional entities
- Safe for research, development, and testing purposes without privacy concerns

### Ethical Use

This dataset is designed for:
- ✅ Training privacy-preserving ML models
- ✅ Testing PII detection systems
- ✅ Research in data anonymization
- ✅ Educational purposes
- ❌ NOT for generating fake identities for fraudulent purposes
- ❌ NOT for evading security systems

## Dataset Generation Pipeline

### 1. Template-Based Generation

**File**: `data/templates.txt`

The generation process starts with sentence templates containing PII placeholders:

```
I want to increase limit on my card # {{credit_card_number}} for certain duration of time. is it possible?
My credit card {{credit_card_number}} has been lost, Can I request you to block it.
Please update the billing address with {{address}} for this card: {{credit_card_number}}
```

**Supported Placeholder Types**:
- `{{person}}`, `{{first_name}}`, `{{last_name}}` - Person names
- `{{credit_card_number}}` - Credit card numbers
- `{{address}}`, `{{street_name}}`, `{{city}}`, `{{country}}` - Address components
- `{{phone_number}}` - Phone numbers
- `{{email}}` - Email addresses
- `{{date_of_birth}}`, `{{date_time}}` - Dates and times
- `{{iban}}` - International bank account numbers
- `{{organization}}` - Company names
- `{{us_driver_license}}` - Driver's license numbers
- `{{ip_address}}` - IP addresses
- `{{ssn}}` - Social security numbers (US format)
- And 20+ more categories

### 2. Entity Generation

**File**: `utils/fake_pii_generator.py`

The `FakePIIGenerator` class generates realistic synthetic PII using Faker:

```python
from utils.fake_pii_generator import FakePIIGenerator

generator = FakePIIGenerator(locale="en_US")
fake_name = generator.get_fake_value("person_name", "original_value")
fake_card = generator.get_fake_value("credit_card_info", "4532-1111-2222-3333")
```

**Key Features**:
- Consistent mapping: Same original value always maps to the same fake value
- Locale support: Generate data appropriate for different regions
- Memory caching: Maintains consistency across multiple generations
- 20+ PII categories supported

### 3. Advanced Address Generation

**File**: `utils/address_generator.py`

Supports multi-country realistic address generation with two modes:

#### Traditional Faker-based Generation

```python
from utils.address_generator import AddressGenerator

gen = AddressGenerator()
address = gen.generate_address(country="US")
# Output: {
#   "street": "123 Main Street",
#   "city": "New York",
#   "state": "NY",
#   "postal_code": "10001",
#   "country": "United States"
# }
```

#### LLM-Enhanced Generation (Optional)

For more realistic and contextually appropriate addresses:

```python
gen = AddressGenerator(use_llm=True, llm_base_url="http://localhost:11434")
address = gen.generate_address(
    country="US",
    context="Business address in downtown area"
)
```

**Supported Countries**:
- United States (US)
- United Kingdom (GB)
- Germany (DE)
- France (FR)
- China (CN)
- Japan (JP)
- India (IN)
- Canada (CA)
- Australia (AU)
- Brazil (BR)

### 4. Enhanced Credit Card Number Generation

**File**: `utils/credit_card_generator.py`

Generates credit card numbers in various realistic formats:

```python
from utils.credit_card_generator import CreditCardGenerator

gen = CreditCardGenerator()

# Various formats
card1 = gen.generate_card(format_type="plain")        # 4532111122223333
card2 = gen.generate_card(format_type="space_4")      # 4532 1111 2222 3333
card3 = gen.generate_card(format_type="dash_4")       # 4532-1111-2222-3333
card4 = gen.generate_card(format_type="last_4_only")  # **** **** **** 3333

# Specific card types
visa = gen.generate_card(card_type="visa")
mastercard = gen.generate_card(card_type="mastercard")
amex = gen.generate_card(card_type="amex")
```

**Supported Formats**:
- `plain`: 16 digits without separators
- `space_4`: 4-digit groups separated by spaces (most common)
- `dash_4`: 4-digit groups separated by dashes
- `last_4_only`: Masked with only last 4 digits visible
- `custom`: Custom separator and grouping

**Supported Card Types**:
- Visa (16 digits, starts with 4)
- Mastercard (16 digits, starts with 51-55 or 2221-2720)
- American Express (15 digits, starts with 34 or 37)
- Discover (16 digits, starts with 6011 or 65)
- Random (mixed types)

### 5. Dataset Generation Script

**File**: `data/dateset_generator.py`

Main script that combines templates and entity generation:

```bash
cd data
python dateset_generator.py
```

**Configuration**:
```python
number_of_samples = 10000      # Number of samples to generate
lower_case_ratio = 0.05        # Ratio of lowercase text
locale = "en"                  # Language locale
```

**Output Format** (JSONL):
```json
{
  "full_text": "My credit card 4532 1111 2222 3333 has been lost",
  "masked_text": "My credit card [CREDIT_CARD] has been lost",
  "spans": [
    {
      "entity_type": "CREDIT_CARD",
      "start": 15,
      "end": 34,
      "entity_value": "4532 1111 2222 3333"
    }
  ],
  "template_id": 2
}
```

### 6. Data Filtering and Validation

**File**: `data/filter_templates.py`

Filters out invalid or low-quality generated samples:

```python
python data/filter_templates.py
```

**Filtering Criteria**:
- Removes samples with malformed PII
- Validates entity format (e.g., credit card passes Luhn check)
- Filters out samples with insufficient context
- Ensures proper entity span annotations

### 7. Upload to Hugging Face

**File**: `data/hf_upload.py`

Uploads the generated dataset to Hugging Face Hub:

```python
python data/hf_upload.py \
    --dataset_path ./generated_size_10000_en.jsonl \
    --repo_name Asklv/llm-adapter-dataset \
    --private False
```

## Generation Workflow

### Quick Start

```bash
# 1. Navigate to data directory
cd /Users/asklv/Projects/AO.space/LocalLLM/LocalChat/adapter/data

# 2. Generate dataset
python dateset_generator.py

# 3. Filter invalid samples
python filter_templates.py

# 4. Upload to Hugging Face (optional)
python hf_upload.py
```

### Advanced Usage

#### Custom Templates

Add your own templates to `data/templates.txt`:

```
{{person}} ordered a product to {{address}} using card {{credit_card_number}}
Transfer ${{amount}} from {{iban}} to {{iban}} by {{date_time}}
```

#### Custom Locale

Generate data for different regions:

```python
# In dateset_generator.py
sentence_faker = PresidioSentenceFaker(
    "de_DE",  # German locale
    lower_case_ratio=0.05,
    sentence_templates=sentence_templates
)
```

#### Extend Entity Types

Add custom entity generators in `utils/fake_pii_generator.py`:

```python
def _generate_custom_id(self, original: str) -> str:
    """Generates a custom ID format."""
    return f"CUSTOM-{self.faker.uuid4()[:8]}"
```

## Dataset Statistics

Current dataset (v1.0):
- **Total samples**: 10,000
- **Unique templates**: 497
- **Average entities per sample**: 2.3
- **Entity type distribution**:
  - Person names: 28%
  - Addresses: 22%
  - Credit cards: 18%
  - Phone numbers: 12%
  - Email addresses: 8%
  - Other: 12%

## Data Quality

### Validation

All generated data passes through validation:
1. Entity format validation
2. Span alignment check
3. Template consistency verification
4. Luhn algorithm for credit cards
5. IBAN checksum validation
6. Email format RFC compliance

### Known Limitations

- Some entity combinations may be unrealistic (e.g., mixing countries)
- Address components may not always be geographically consistent
- Phone number formats follow general patterns but may not match specific carriers
- Synthetic data patterns may differ from real-world distributions

## Continuous Improvement

The generation pipeline is continuously improved:

1. **Template expansion**: Adding more diverse templates
2. **Entity quality**: Improving entity generation realism
3. **Multi-language**: Expanding to more languages
4. **Context awareness**: Using LLMs for more realistic combinations
5. **Format diversity**: Supporting more PII format variations

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{llm_adapter_dataset,
  title={LLM Adapter PII Dataset},
  author={AO.space Team},
  year={2025},
  url={https://huggingface.co/datasets/Asklv/llm-adapter-dataset},
  note={Synthetic PII dataset for privacy-aware LLM training}
}
```

## License

This dataset is released under the MIT License. The generated data is completely synthetic and free to use for any purpose, including commercial applications.

## Support

For issues, questions, or contributions:
- GitHub Issues: [LocalChat/adapter](https://github.com/AO-space/LocalLLM/LocalChat/adapter)
- Email: support@ao.space
- Documentation: See `README.md` in the project root
