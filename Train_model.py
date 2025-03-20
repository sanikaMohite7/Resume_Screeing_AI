import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Sample dataset (Replace this with your real dataset)
data = {
    "Resume": [
        "Experienced Python developer with Django skills",
        "Data analyst with expertise in SQL and Excel",
        "Cybersecurity expert with knowledge in network security",
        "Machine learning engineer skilled in TensorFlow and NLP",
    ],
    "Category": ["Software Developer", "Data Analyst", "Cybersecurity", "ML Engineer"],
}

df = pd.DataFrame(data)

# Encode job categories
le = LabelEncoder()
df["Category_Encoded"] = le.fit_transform(df["Category"])

# ✅ Fix the TF-IDF vectorizer to have a fixed feature count
tfidf = TfidfVectorizer(max_features=5000)  # Set a fixed number of features
X_tfidf = tfidf.fit_transform(df["Resume"])
y = df["Category_Encoded"]

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the model
svc_model = SVC()
svc_model.fit(X_train, y_train)

# ✅ Save the trained model and vectorizer
pickle.dump(tfidf, open("tfidf.pkl", "wb"))  # Save the SAME vectorizer
pickle.dump(svc_model, open("clf.pkl", "wb"))  # Save trained model
pickle.dump(le, open("encoder.pkl", "wb"))  # Save label encoder

print("🎉 Model trained and saved successfully!")
