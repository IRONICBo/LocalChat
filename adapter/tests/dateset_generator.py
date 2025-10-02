import datetime
import pprint
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

from presidio_evaluator import InputSample
from presidio_evaluator.data_generator import PresidioSentenceFaker

number_of_samples = 1500
lower_case_ratio = 0.05
locale = "en"
cur_time = datetime.date.today().strftime("%B_%d_%Y")

output_file = f"../data/generated_size_{number_of_samples}_date_{cur_time}.json"
output_conll = f"../data/generated_size_{number_of_samples}_date_{cur_time}.tsv"

sentence_faker = PresidioSentenceFaker("en_US", lower_case_ratio=0.05)

import random
from faker.providers import BaseProvider


class MarsIdProvider(BaseProvider):
    def mars_id(self):
        # Generate a random row number between 1 and 50
        row = random.randint(1, 50)
        # Generate a random letter for the seat location from A-K
        location = random.choice("ABCDEFGHIJK")
        # Return the seat in the format "row-letter" (e.g., "25A")
        return f"{row}{location}"


sentence_faker.add_provider(MarsIdProvider)
# Now a new `mars_id` entity can be generated if a template has `mars_id` in it.

from presidio_evaluator.data_generator.faker_extensions.providers import *

IpAddressProvider  # Both Ipv4 and IPv6 IP addresses
NationalityProvider  # Read countries + nationalities from file
OrganizationProvider  # Read organization names from file
UsDriverLicenseProvider  # Read US driver license numbers from file
AgeProvider  # Age values (unavailable on Faker
AddressProviderNew  # Extend the default address formats
PhoneNumberProviderNew  # Extend the default phone number formats
ReligionProvider  # Read religions from file

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


# conll = InputSample.create_conll_dataset(dataset=fake_records)
# conll.head(10)
# conll.to_csv(output_conll, sep="\t")
# print(f"CoNLL2003 dataset structure output location: {output_conll}")
