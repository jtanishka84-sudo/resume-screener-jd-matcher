import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Resume.csv")

def clean_text(text):
    text = re.sub(r'http\S+|[^a-zA-Z\s]', ' ', text)
    return text.lower()

df['cleaned'] = df['Resume'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['Category'], test_size=0.2, random_state=42, stratify=df['Category']
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump(model, open("category_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Saved!")