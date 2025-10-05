import re
from pathlib import Path

def extract_entities(template_file_path):
    pattern = re.compile(r'\{\{\w+\}\}')
    entities = set()

    with open(template_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            matches = pattern.findall(line.strip())
            entities.update(matches)

    return entities

def filter_sentences(input_file_path, output_file_path, valid_entities):
    pattern = re.compile(r'\{\{\w+\}\}')

    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            matches = set(pattern.findall(line))

            if matches.issubset(valid_entities):
                outfile.write(line + '\n')

def main():
    template_file_path = Path('templates.txt')
    input_file_path = Path('input_sentences.txt')
    output_file_path = Path('filtered_sentences.txt')

    valid_entities = extract_entities(template_file_path)
    print(f"Find {len(valid_entities)} unique entities:")
    print(valid_entities)

    filter_sentences(input_file_path, output_file_path, valid_entities)
    print(f"Filtering complete, results saved to {output_file_path}")

if __name__ == '__main__':
    main()