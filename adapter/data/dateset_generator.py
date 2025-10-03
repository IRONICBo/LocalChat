import datetime
import pprint
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

from presidio_evaluator import InputSample
from presidio_evaluator.data_generator import PresidioSentenceFaker

number_of_samples = 10000
lower_case_ratio = 0.05
locale = "en"
cur_time = datetime.date.today().strftime("%B_%d_%Y")

# Give new path to your templates file if needed
template_file_path = Path(__file__).parent / "templates.txt"
sentence_templates = [
    line.strip()
    for line in template_file_path.read_text().splitlines()
]
output_file = f"./generated_size_{number_of_samples}_en.jsonl"

sentence_faker = PresidioSentenceFaker("en_US", lower_case_ratio=lower_case_ratio)

import random
from faker.providers import BaseProvider

from presidio_evaluator.data_generator.faker_extensions.providers import *

IpAddressProvider  # Both Ipv4 and IPv6 IP addresses
NationalityProvider  # Read countries + nationalities from file
OrganizationProvider  # Read organization names from file
UsDriverLicenseProvider  # Read US driver license numbers from file
AgeProvider  # Age values (unavailable on Faker
AddressProviderNew  # Extend the default address formats
PhoneNumberProviderNew  # Extend the default phone number formats
ReligionProvider  # Read religions from file
HospitalProvider  # Read hospital names from file

sentence_faker.add_provider(IpAddressProvider)
sentence_faker.add_provider(NationalityProvider)
sentence_faker.add_provider(OrganizationProvider)
sentence_faker.add_provider(UsDriverLicenseProvider)
sentence_faker.add_provider(AgeProvider)
sentence_faker.add_provider(AddressProviderNew)
sentence_faker.add_provider(PhoneNumberProviderNew)
sentence_faker.add_provider(ReligionProvider)
sentence_faker.add_provider(HospitalProvider)

# Create entity aliases (e.g. if your provider supports "name" but templates contain "person").
provider_aliases = PresidioSentenceFaker.PROVIDER_ALIASES
provider_aliases

# To customize, call `PresidioSentenceFaker(locale="en_US",...,provider_aliases=provider_aliases)`

fake_records = sentence_faker.generate_new_fake_sentences(num_samples=number_of_samples)
pprint.pprint(fake_records[0])

count_per_template_id = Counter([sample.template_id for sample in fake_records])

print(f"Total: {sum(count_per_template_id.values())}")
print(f"Avg # of records per template: {np.mean(list(count_per_template_id.values()))}")
print(
    f"Median # of records per template: {np.median(list(count_per_template_id.values()))}"
)
print(f"Std: {np.std(list(count_per_template_id.values()))}")

count_per_entity = Counter()
for record in fake_records:
    count_per_entity.update(Counter([span.entity_type for span in record.spans]))

for record in fake_records[:10]:
    print(record)

InputSample.to_json(dataset=fake_records, output_file=output_file)
