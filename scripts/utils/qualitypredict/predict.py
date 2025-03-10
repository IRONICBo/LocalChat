import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    "question": [
        "What is the capital of France?",
        "How does a car engine work?",
        "What is machine learning?",
    ],
    "answer_1": [
        "Paris is the capital of France.",
        "A car engine works by converting fuel into motion.",
        "Machine learning is a field of AI that allows computers to learn from data.",
    ],
    "answer_2": [
        "The capital of France is Paris.",
        "The car engine operates by transforming chemical energy into mechanical work.",
        "Machine learning allows machines to perform tasks without explicit programming.",
    ],
    "label": [1, 0, 1],  # 1 = answer_1 is better, 0 = answer_2 is better
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(stop_words="english")

features_1 = vectorizer.fit_transform(df["question"] + " " + df["answer_1"])
features_2 = vectorizer.transform(df["question"] + " " + df["answer_2"])

X = np.hstack([features_1.toarray(), features_2.toarray()])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save model
import pickle

with open("svm_model.pkl", "wb") as model_file:
    pickle.dump(svm_model, model_file)
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
