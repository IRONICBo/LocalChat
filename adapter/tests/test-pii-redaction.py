from pii_redaction import tag_pii_in_documents, clean_dataset, PIIHandlingMode

# Process text documents
documents = [
    "My name is John Doe and my email is john.doe@example.com",
    "Call me at 555-123-4567 and ask for my SSN: 123-45-6789"
]

# Tag PII (default mode)
tagged_documents = tag_pii_in_documents(documents, mode=PIIHandlingMode.TAG)

# Redact PII completely
redacted_documents = tag_pii_in_documents(documents, mode=PIIHandlingMode.REDACT)

# Replace PII with fake data
anonymized_documents = tag_pii_in_documents(
    documents,
    mode=PIIHandlingMode.REPLACE,
    locale="en_US"
)

# Process a JSONL dataset
# Tag PII (default mode)
clean_dataset('input.jsonl', 'output.jsonl', mode=PIIHandlingMode.TAG)

# Redact PII in a JSONL dataset
clean_dataset('input.jsonl', 'redacted.jsonl', mode=PIIHandlingMode.REDACT)

# Replace PII with fake data in a JSONL dataset
clean_dataset(
    'input.jsonl',
    'anonymized.jsonl',
    mode=PIIHandlingMode.REPLACE,
    locale="en_US"
)