# amazon-review-sentiment-analysis
This project analyzes Amazon customer reviews using machine learning and NLP techniques to uncover sentiment trends and gain actionable insights from customer feedback.

## 📌 Problem Statement

Understanding customer sentiment helps businesses make data-driven decisions. This project leverages sentiment analysis to analyze text reviews and determine if they express positive, negative, or neutral opinions.

---

## 🧠 Technologies & Tools Used

- **Python 3**
- **Pandas** for data manipulation
- **NLTK** for text preprocessing (stopwords, lemmatization)
- **Scikit-learn** for model building and evaluation
- **TfidfVectorizer** for text feature extraction
- **Naive Bayes Classifier** for classification
- **Matplotlib & Seaborn** for visualizations

---

---

## 🔄 Workflow

1. **Load dataset** from CSV
2. **Clean and preprocess text**: lowercasing, removing stopwords, lemmatization
3. **Label sentiment** based on review rating:
    - `1–2` → Negative  
    - `3` → Neutral  
    - `4–5` → Positive  
4. **Extract features** using TF-IDF
5. **Train a Naive Bayes Classifier**
6. **Evaluate model** with precision, recall, F1-score, confusion matrix
7. **Visualize metrics** using bar plots and heatmaps

---

## 📊 Results

Model performance on test data:
- **Accuracy**: ~94% (on sample data)
- **Precision/Recall/F1-Score** per class displayed in bar chart
- **Confusion Matrix** visualizes predictions

---

## 🚀 Getting Started

### 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/sentiment-analysis-amazon.git
cd sentiment-analysis-amazon

**Install Dependencies**
pip install -r requirements.txt

Download NLTK resources:
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

▶️ Run Training Script

python src/train.py
