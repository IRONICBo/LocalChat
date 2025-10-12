# Quick Start Guide

Get started with LLM Adapter dataset generation in 5 minutes.

## Installation

```bash
# Install required packages
pip install faker presidio-evaluator requests pandas numpy

# Optional: Install Presidio for PII detection
pip install presidio-analyzer presidio-anonymizer
```

## Quick Examples

### 1. Generate Basic PII

```python
from utils.fake_pii_generator import FakePIIGenerator

# Initialize
gen = FakePIIGenerator()

# Generate
name = gen.get_fake_value("person_name", "user_1")
email = gen.get_fake_value("email_address", "user_1")
card = gen.get_fake_value("credit_card_info", "card_1")

print(f"Name: {name}")
print(f"Email: {email}")
print(f"Card: {card}")
```

### 2. Generate Enhanced Credit Cards

```python
from utils.credit_card_generator import CreditCardGenerator

gen = CreditCardGenerator()

# Different formats
print(gen.generate_card(format_type="plain"))        # 4532111122223333
print(gen.generate_card(format_type="space_4"))      # 4532 1111 2222 3333
print(gen.generate_card(format_type="dash_4"))       # 4532-1111-2222-3333
print(gen.generate_card(format_type="last_4_only"))  # **** **** **** 3333

# With details
card = gen.generate_card(
    card_type="visa",
    format_type="space_4",
    include_cvv=True,
    include_expiry=True
)
print(card)  # 4532 1111 2222 3333 CVV: 123 Exp: 12/28
```

### 3. Generate Multi-Country Addresses

```python
from utils.address_generator import AddressGenerator

gen = AddressGenerator()

# Different countries
us_addr = gen.generate_address(country="US")
uk_addr = gen.generate_address(country="GB")
de_addr = gen.generate_address(country="DE")

print(f"US: {us_addr}")
print(f"UK: {uk_addr}")
print(f"DE: {de_addr}")

# Multi-line format
address = gen.generate_address(country="US", format_type="multi_line")
print(address)
# Output:
# 123 Main Street
# New York, NY 10001
# United States
```

### 4. Generate Complete Dataset

```python
from utils.fake_pii_generator import FakePIIGenerator
from utils.credit_card_generator import CreditCardGenerator
from utils.address_generator import AddressGenerator
import json

# Initialize generators
pii_gen = FakePIIGenerator(use_enhanced_generators=True)
card_gen = CreditCardGenerator()
addr_gen = AddressGenerator()

# Generate records
records = []
for i in range(10):
    record = {
        "id": i,
        "name": pii_gen.get_fake_value("person_name", f"user_{i}"),
        "email": pii_gen.get_fake_value("email_address", f"user_{i}"),
        "phone": pii_gen.get_fake_value("phone_number", f"user_{i}"),
        "card": card_gen.generate_card(format_type="space_4"),
        "address": addr_gen.generate_address(country="US")
    }
    records.append(record)

# Save to JSONL
with open("dataset.jsonl", "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

print(f"Generated {len(records)} records")
```

## Run Examples

```bash
# Run all examples
python examples/generate_dataset_example.py

# Test individual generators
python utils/address_generator.py
python utils/credit_card_generator.py
```

## Generate Full Dataset

```bash
# Navigate to data directory
cd data

# Generate dataset (10,000 samples by default)
python dateset_generator.py

# Output: generated_size_10000_en.jsonl
```

## Customize Generation

### Modify Templates

Edit `data/templates.txt` to add your own templates:

```
{{person}} purchased item with card {{credit_card_number}}
Ship to {{address}} for {{person}}
Contact {{phone_number}} or {{email}} for support
```

### Change Sample Size

Edit `data/dateset_generator.py`:

```python
number_of_samples = 50000  # Change from 10000 to 50000
lower_case_ratio = 0.1     # Adjust lowercase ratio
locale = "en"              # Change locale if needed
```

### Use Enhanced Generators

```python
from utils.fake_pii_generator import FakePIIGenerator

# Enable enhanced generators
gen = FakePIIGenerator(
    locale="en_US",
    use_enhanced_generators=True,
    llm_base_url="http://localhost:11434"  # Optional: for LLM-enhanced addresses
)
```

## Validation

### Verify Credit Card Numbers

```python
from utils.credit_card_generator import CreditCardGenerator

gen = CreditCardGenerator()
card = gen.generate_card()

# Verify Luhn algorithm
is_valid = gen.verify_luhn(card)
print(f"Card {card} is {'valid' if is_valid else 'invalid'}")
```

### Check Generated Data

```python
import json

# Load generated dataset
with open("data/generated_size_10000_en.jsonl", "r") as f:
    for line in f:
        record = json.loads(line)
        print(f"Text: {record['full_text']}")
        print(f"Entities: {len(record['spans'])}")
        break  # Print first record only
```

## Common Use Cases

### 1. Testing PII Detection

```python
from utils.fake_pii_generator import FakePIIGenerator

gen = FakePIIGenerator()

# Generate test data
test_data = [
    f"Contact {gen.get_fake_value('person_name', f'user_{i}')} at {gen.get_fake_value('email_address', f'user_{i}')}"
    for i in range(100)
]

# Test your PII detector
for text in test_data:
    detected_pii = your_pii_detector(text)
    print(f"Detected: {detected_pii}")
```

### 2. Training Data Generation

```python
# Generate training data for NER model
training_data = []

for i in range(1000):
    text = f"Card {card_gen.generate_card()} belongs to {pii_gen.get_fake_value('person_name', f'u{i}')}"

    training_data.append({
        "text": text,
        "entities": [
            {"start": 5, "end": 24, "label": "CREDIT_CARD"},
            {"start": 36, "end": 46, "label": "PERSON"}
        ]
    })
```

### 3. Realistic Test Scenarios

```python
# Generate customer support tickets
for i in range(10):
    name = pii_gen.get_fake_value("person_name", f"customer_{i}")
    email = pii_gen.get_fake_value("email_address", f"customer_{i}")
    card = card_gen.generate_card(format_type="last_4_only")

    ticket = f"""
Ticket #{1000 + i}
Customer: {name}
Email: {email}
Issue: Transaction declined for card {card}
Status: Pending
    """.strip()

    print(ticket)
    print("-" * 50)
```

## Performance Tips

1. **Batch Generation**: Generate in batches for better performance
2. **Memory Cache**: FakePIIGenerator caches values for consistency
3. **Parallel Processing**: Use multiprocessing for large datasets
4. **Disable LLM**: Turn off LLM enhancement for faster generation

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the adapter directory
cd /Users/asklv/Projects/AO.space/LocalLLM/LocalChat/adapter

# Add to Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Dependencies

```bash
pip install faker presidio-evaluator requests pandas numpy
```

### LLM Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama if needed
ollama serve
```

## Next Steps

1. Read [DATA_GENERATION.md](DATA_GENERATION.md) for detailed documentation
2. Check [examples/generate_dataset_example.py](examples/generate_dataset_example.py) for more examples
3. Explore [data/templates.txt](data/templates.txt) to understand template format
4. See [README.md](README.md) for full project documentation

## Support

- GitHub Issues: Report bugs or request features
- Documentation: See DATA_GENERATION.md
- Examples: Check examples/ directory
