import numpy as np
import pickle


with open("svm_model.pkl", "rb") as model_file:
    loaded_svm_model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

new_features_1 = loaded_vectorizer.transform(
    ["What is the capital of Italy? " + "Rome is the capital of Italy."]
)
new_features_2 = loaded_vectorizer.transform(
    ["What is the capital of Italy? " + "Milan is the capital of Italy."]
)
new_X = np.hstack([new_features_1.toarray(), new_features_2.toarray()])

new_prediction = loaded_svm_model.predict(new_X)
print("Predicted better answer:", "Answer 1" if new_prediction[0] == 1 else "Answer 2")
