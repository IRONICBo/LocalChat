import string


def calculate_special_char_ratio(text: str):
    """
    Calculates the ratio of special characters (whitespace and punctuation) in a string.
    """
    special_chars = set(string.whitespace + string.punctuation)
    special_char_count = sum(1 for char in text if char in special_chars)
    total_length = len(text)
    special_char_ratio = special_char_count / total_length if total_length > 0 else 0

    return special_char_ratio


if __name__ == "__main__":
    text = "Hello, world!  How are you?   "
    special_char_ratio = calculate_special_char_ratio(text)

    print(f"Special character ratio: {special_char_ratio:.2%}")
