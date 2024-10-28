import string


class Evaluator:
    def __init__(self, choices):
        self.choices = choices
        self.puncs = list(string.punctuation)

    def normalize_answer(self, s):
        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(self.puncs)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def exact_match(self, pred, target):
        return self.normalize_answer(pred) == self.normalize_answer(target)

    def contains_valid_choice(self, answer):
        """
        Check if the answer contains any of the valid choices
        """
        normalized_answer = self.normalize_answer(answer)
        for choice in self.choices:
            if choice.lower() in normalized_answer:
                return True
        return False
