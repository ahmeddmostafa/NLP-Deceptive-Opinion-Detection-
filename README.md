# 🔍 Deceptive Opinion Detection — NLP Project

A Natural Language Processing project that detects **deceptive (fake) vs. truthful hotel reviews** using 5 different machine learning algorithms, with a graphical user interface (GUI) for real-time prediction.

---

## 📋 Project Overview

| Item | Details |
|------|---------|
| **Input** | A hotel review written in English |
| **Output** | Classification: `deceptive` or `truthful` |
| **Dataset** | `deceptive-opinion.csv` — 1,600 hotel reviews |
| **GUI** | Tkinter desktop application |

---

## 📁 Project Structure

```
NLP Project 2/
├── deceptive-opinion.csv        # Dataset (1,600 reviews)
├── cleaning.ipynb               # Data pre-processing notebook
├── SVM.ipynb                    # SVM model training & evaluation
├── gui.py                       # GUI application
├── svm_model.pkl                # Trained SVM model
├── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
├── README.md                    # This file
│
├── lr_model.pkl                 # (TODO) Logistic Regression model
├── nb_model.pkl                 # (TODO) Naive Bayes model
├── rf_model.pkl                 # (TODO) Random Forest model
└── knn_model.pkl                # (TODO) KNN model
```

---

## 🧹 Data Pre-processing

The following cleaning steps are applied to the `text` column before training and prediction:

1. **Remove commas** — all `,` characters are stripped
2. **Lowercase** — all text is converted to lowercase
3. **Remove stop words** — common English stop words (the, is, at, ...) are removed

```python
def clean_text(text):
    text = text.replace(',', '')
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)
```

---

## 🔢 Feature Extraction

| Method | Details |
|--------|---------|
| **TF-IDF Vectorizer** | Converts text into numerical vectors |
| `max_features` | 5,000 |
| Saved as | `tfidf_vectorizer.pkl` |

All 5 models use the **same TF-IDF vectorizer** to ensure consistency.

---

## 🤖 Machine Learning Models

### Algorithm 1: Support Vector Machine (SVM) ✅

| Parameter | Value |
|-----------|-------|
| **Kernel** | Linear |
| **C** | 1.0 |
| **Accuracy** | 86.87% |
| **Status** | ✅ Complete |
| **Saved as** | `svm_model.pkl` |

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| deceptive | 0.84 | 0.89 | 0.87 | 152 |
| truthful | 0.89 | 0.85 | 0.87 | 168 |

**Confusion Matrix:**

|  | Predicted Deceptive | Predicted Truthful |
|--|--------------------|--------------------|
| **Actual Deceptive** | 135 | 17 |
| **Actual Truthful** | 25 | 143 |

---

### Algorithm 2: Logistic Regression 🔲

| Parameter | Suggested Value |
|-----------|-----------------|
| **Solver** | lbfgs |
| **max_iter** | 1000 |
| **Status** | 🔲 Not implemented yet |
| **Save as** | `lr_model.pkl` |

**Training Template:**
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred = lr_model.predict(X_test_tfidf)

joblib.dump(lr_model, 'lr_model.pkl')
```

---

### Algorithm 3: Naive Bayes 🔲

| Parameter | Suggested Value |
|-----------|-----------------|
| **Type** | MultinomialNB |
| **alpha** | 1.0 (Laplace smoothing) |
| **Status** | 🔲 Not implemented yet |
| **Save as** | `nb_model.pkl` |

**Training Template:**
```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)
y_pred = nb_model.predict(X_test_tfidf)

joblib.dump(nb_model, 'nb_model.pkl')
```

---

### Algorithm 4: Random Forest 🔲

| Parameter | Suggested Value |
|-----------|-----------------|
| **n_estimators** | 200 |
| **random_state** | 42 |
| **Status** | 🔲 Not implemented yet |
| **Save as** | `rf_model.pkl` |

**Training Template:**
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred = rf_model.predict(X_test_tfidf)

joblib.dump(rf_model, 'rf_model.pkl')
```

---

### Algorithm 5: K-Nearest Neighbors (KNN) 🔲

| Parameter | Suggested Value |
|-----------|-----------------|
| **n_neighbors** | 5 |
| **metric** | cosine |
| **Status** | 🔲 Not implemented yet |
| **Save as** | `knn_model.pkl` |

**Training Template:**
```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_model.fit(X_train_tfidf, y_train)
y_pred = knn_model.predict(X_test_tfidf)

joblib.dump(knn_model, 'knn_model.pkl')
```

---

## 🖥️ GUI Application

The GUI (`gui.py`) provides a simple interface for real-time prediction:

- **Input**: A text box to enter a hotel review
- **Process Button**: Cleans the text, vectorizes it, and runs all loaded models
- **Output**: 5 result fields showing the prediction of each algorithm

### How to Run

```bash
python gui.py
```

### Adding a New Model to the GUI

1. Train and save your model as a `.pkl` file in the project folder
2. Open `gui.py`
3. Uncomment the `joblib.load(...)` line for your algorithm
4. Uncomment the `set_result(...)` line in the `process_text()` function
5. Run the GUI again

---

## 📊 Train/Test Split

| Set | Size | Percentage |
|-----|------|------------|
| Training | 1,280 | 80% |
| Testing | 320 | 20% |
| `random_state` | 42 | — |

---

## 🛠️ Requirements

```
pandas
scikit-learn
joblib
tkinter (built-in with Python)
nltk (for cleaning.ipynb only)
```

---

## 👥 Team Responsibilities

| Member | Algorithm | File |
|--------|-----------|------|
| Member 1 | SVM | `SVM.ipynb` |
| Member 2 | Logistic Regression | `LR.ipynb` (TODO) |
| Member 3 | Naive Bayes | `NB.ipynb` (TODO) |
| Member 4 | Random Forest | `RF.ipynb` (TODO) |
| Member 5 | KNN | `KNN.ipynb` (TODO) |

> **Note:** Each team member should create their own notebook, train their model using the same pre-processed data and TF-IDF vectorizer, then save the model as a `.pkl` file.

---

## 📌 Summary

| Step | Description | Status |
|------|-------------|--------|
| Data Cleaning | Remove commas, lowercase, remove stop words | ✅ |
| Feature Extraction | TF-IDF (max 5,000 features) | ✅ |
| SVM Model | Linear SVM, 86.87% accuracy | ✅ |
| Logistic Regression | — | 🔲 |
| Naive Bayes | — | 🔲 |
| Random Forest | — | 🔲 |
| KNN | — | 🔲 |
| GUI | Tkinter app with 5 algorithm slots | ✅ |
