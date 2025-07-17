import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ========== SETUP ==========
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ========== LOAD DATA ==========
df = pd.read_csv('data/Amazon_Unlocked_Mobile.csv')
df = df[['reviews.text', 'reviews.rating']].dropna()

# ========== LABEL SENTIMENT ==========
def label_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['reviews.rating'].apply(label_sentiment)

# ========== TEXT PREPROCESSING ==========
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['reviews.text'].apply(preprocess)
print(df[['cleaned_text', 'sentiment']].head())

# ========== FEATURE EXTRACTION ==========
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# ========== TRAIN / TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== TRAIN MODEL ==========
model = MultinomialNB()
model.fit(X_train, y_train)

# ========== PREDICT & EVALUATE ==========
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ========== VISUALIZATION ==========
labels = ['negative', 'neutral', 'positive']
report = classification_report(y_test, y_pred, output_dict=True)

# Metrics
precision = [report[label]['precision'] for label in labels]
recall = [report[label]['recall'] for label in labels]
f1 = [report[label]['f1-score'] for label in labels]

# Bar Chart
x = np.arange(len(labels))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1-Score')

plt.xticks(x, labels)
plt.xlabel('Sentiment Class')
plt.ylabel('Score')
plt.title('Classification Report Metrics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
