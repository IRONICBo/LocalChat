from typing import List
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine

# Initialize the analyzer and anonymizer engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
deanonymizer = DeanonymizeEngine()


def add_entity_id(results: List[RecognizerResult]) -> List[RecognizerResult]:
    results.sort(key=lambda x: x.start)

    entity_count = {}

    for result in results:
        entity_type = result.entity_type
        if entity_type not in entity_count:
            entity_count[entity_type] = 0
        else:
            entity_count[entity_type] += 1

        result.entity_type = f"{entity_type}_{entity_count[entity_type]}"

    return results


def anonymize_text(text):
    # Analyze the text to find PII entities
    results = analyzer.analyze(text=text, language="en")
    print(f"Analysis Results: {results}")
    results = add_entity_id(results)
    print(f"Analysis Results with IDs: {results}")
    # Anonymize the identified PII entities
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_text.text, anonymized_text.items


def test_cases():
    cases = [
        "Erik spent a year at B. Riley Financial Inc. 5.50% Senior Notes Due 2026 as the assistant to Erik Baader, and the following year at Adthink SA in Remniku, which later became Salona Cotspin Ltd in 1996."
    ]
    # {{PERSON}} spent a year at {{ORGANIZATION}} as the assistant to {{PERSON}}, and the following year at {{ORGANIZATION}} in {{GPE}}, which later became {{ORGANIZATION}} in {{DATE_TIME}}.
    for i, case in enumerate(cases, 1):
        anonymized, items = anonymize_text(case)
        print(f"Test Case {i} Output:\n{anonymized}\n")
        print(f"Test Case {i} Items:\n{items}\n")


if __name__ == "__main__":
    test_cases()
