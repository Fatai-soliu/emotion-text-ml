import pickle
import joblib

# Inspect the model

model = joblib.load("svm_model.pkl")
print(model)

print(type(model))       # tells you what it is
print(model)             # full details
print(model.classes_)    # emotion labels e.g. ['anger', 'joy', 'optimism', 'sadness']

# Inspect Vectorizer

vectorizer = joblib.load("tfidf_vectorizer.pkl")
print(vectorizer)

print(type(vectorizer))       # tells you what it is
print(vectorizer)             # full details

# Test a prediction manually
text        = ["I hope we make it one day"]
transformed = vectorizer.transform(text)      # text → numbers
prediction  = model.predict(transformed) # numbers → emotion label

print(prediction)        # e.g. ['sadness']
print(model.classes_)    # e.g. ['anger', 'joy', 'optimism', 'sadness']