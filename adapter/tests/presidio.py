from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

text = "我的手机号码是 212-555-5555"
# text="My phone number is 212-555-5555"

# Set up the engine, loads the NLP module (spaCy model by default)
# and other PII recognizers
analyzer = AnalyzerEngine()

# Call analyzer to get results
results = analyzer.analyze(
    text=text,
    #    Set to none to use default supported languages
    #    entities=["PHONE_NUMBER"],
    language="en",
)
print(results)

# Analyzer results are passed to the AnonymizerEngine for anonymization

anonymizer = AnonymizerEngine()

anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)

# Output:
# 我的手机号码是 <PHONE_NUMBER>
print(anonymized_text.text)
