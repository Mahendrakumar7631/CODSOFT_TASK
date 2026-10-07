# 🎬 Movie Genre Classification using NLP

**CodSoft Machine Learning Internship — Task 1**

An end-to-end **Machine Learning** project that predicts the genre of a movie based on its title and plot description. Built with **scikit-learn** for model training and **Streamlit** for an interactive web interface.

---

## 📌 Overview

Movie genre classification is a multi-class text classification problem. Given a movie's title and plot description, the model predicts which genre category the movie belongs to. This is an inherently challenging NLP task because genres overlap significantly (e.g., a movie can be both Action and Thriller).

---

## 🎯 Problem Statement

Given a movie's title and plot description, predict the genre of the movie using traditional Machine Learning and NLP techniques.

---

## 🎯 Objective

- Build a text classification model that maps movie descriptions to genres
- Compare multiple ML classifiers and select the best performer
- Deploy the model via an interactive Streamlit web application

---

## 📁 Dataset

This project uses the **Movie Genre Classification Dataset** which contains:

| Split | Samples | Format |
|-------|---------|--------|
| Training | 54,214 | ID ::: TITLE ::: GENRE ::: DESCRIPTION |
| Test | 54,200 | ID ::: TITLE ::: GENRE ::: DESCRIPTION |

- **27 original genre labels** are mapped to **10 clean categories** for better classification performance
- Source: IMDb (ftp://ftp.fu-berlin.de/pub/misc/movies/database/)

### Genre Mapping (27 → 10 Classes)

| Mapped Genre | Original Labels |
|-------------|----------------|
| Action/Adventure | action, adventure, thriller, crime, western, war |
| Comedy | comedy |
| Drama | drama, romance |
| Documentary | documentary, biography, history, news |
| Horror/Mystery | horror, mystery |
| Sci-Fi/Fantasy | sci-fi, fantasy |
| Family/Animation | family, animation |
| Entertainment | game-show, reality-tv, talk-show, music, musical, sport |
| Short Film | short |
| Adult | adult |

---

## ✨ Features

- **Multi-class text classification** across 10 movie genre categories
- **Multiple models compared** — Logistic Regression, LinearSVC, and Multinomial Naive Bayes
- **Dual TF-IDF** — word-level (80K features) + character-level (50K features) = 130K total features
- **Interactive Streamlit web app** with a premium dark-themed UI
- **No data leakage** — separate train/test sets used for evaluation

---

## 🛠️ Technologies

| Technology | Purpose |
|-----------|---------|
| Python 3.11 | Core language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| Scikit-learn | ML models, TF-IDF vectorization, metrics |
| SciPy | Sparse matrix operations |
| Joblib | Model serialization |
| Streamlit | Interactive web application |
| Jupyter Notebook | Model development and experimentation |

---

## 🔄 Workflow

### Text Preprocessing
1. Combine **Title + Description** into a single text field
2. Convert to lowercase
3. Remove special characters (keep only alphanumeric and spaces)
4. Normalize whitespace

### TF-IDF Feature Extraction

| Feature Type | Analyzer | N-gram Range | Max Features |
|-------------|----------|-------------|-------------|
| Word TF-IDF | word | (1, 2) — unigrams + bigrams | 80,000 |
| Char TF-IDF | char_wb | (3, 5) — character n-grams | 50,000 |

Both vectorizers use `sublinear_tf=True`, `min_df=3`, `max_df=0.7`.

### Models Trained
- **Logistic Regression** — C=1.0, LBFGS solver, max_iter=1000
- **LinearSVC** — C=1.0, max_iter=2000
- **Multinomial Naive Bayes** — alpha=0.1

### Evaluation
Models are compared on test set accuracy. The best model is automatically selected and saved.

### Final Model
**Logistic Regression** was selected as the best model based on test accuracy.

---

## 📊 Results

| Model | Test Accuracy |
|-------|--------------|
| **Logistic Regression** | **65.76%** |
| LinearSVC | 63.42% |
| MultinomialNB | 62.41% |

### Classification Report (Logistic Regression)

| Genre | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Action/Adventure | 0.60 | 0.52 | 0.56 | 5,348 |
| Adult | 0.69 | 0.31 | 0.43 | 590 |
| Comedy | 0.62 | 0.59 | 0.61 | 7,446 |
| Documentary | 0.75 | 0.86 | 0.80 | 13,784 |
| Drama | 0.62 | 0.77 | 0.69 | 14,284 |
| Entertainment | 0.66 | 0.51 | 0.57 | 2,905 |
| Family/Animation | 0.62 | 0.21 | 0.31 | 1,281 |
| Horror/Mystery | 0.71 | 0.57 | 0.63 | 2,522 |
| Sci-Fi/Fantasy | 0.57 | 0.25 | 0.35 | 968 |
| Short Film | 0.56 | 0.39 | 0.46 | 5,072 |
| **Weighted Avg** | **0.65** | **0.66** | **0.64** | **54,200** |

> **Note:** Movie genre classification is inherently difficult because genres overlap significantly. Research papers on this dataset report 58–62% as typical for 27-class classification. With 10 mapped classes, 65.76% is a strong result.

---

## 🖥️ Streamlit Application

The Streamlit app provides a clean, professional interface for genre prediction:

- **Left panel**: Project info, ML approach, dataset details, and statistics
- **Right panel**: Input fields for movie title and plot description with genre prediction result
- **Dark blue/purple gradient** theme with glassmorphism design

---

## 🏗️ Project Structure

```
CODSOFT_TASK1/
├── data/
│   ├── train_data.txt              # Training dataset (54,214 samples)
│   ├── test_data.txt               # Test dataset (without labels)
│   ├── test_data_solution.txt      # Test dataset (with labels, 54,200 samples)
│   └── description.txt             # Dataset description
├── notebooks/
│   └── model_development.ipynb     # Model development notebook
├── models/
│   └── movie_genre_model.pkl       # Saved model bundle (model + vectorizers)
├── src/
│   ├── preprocessing.py            # Data loading, text cleaning, genre mapping
│   ├── train_model.py              # Model training & evaluation pipeline
│   └── predict.py                  # Prediction module
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CODSOFT_TASK1.git
cd CODSOFT_TASK1

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train the Model

```bash
python src/train_model.py
```

This will:
- Load and preprocess the dataset
- Train 3 classifiers (Logistic Regression, LinearSVC, MultinomialNB)
- Compare accuracy and select the best model
- Save the model bundle to `models/movie_genre_model.pkl`

### Run the Web App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` where you can enter any movie title and plot description to predict its genre.

---

## 🔮 Future Improvements

- Experiment with ensemble methods (Voting Classifier, Stacking)
- Try advanced text features (word embeddings, Doc2Vec)
- Implement multi-label classification (movies can have multiple genres)
- Add hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- Handle class imbalance with SMOTE or class weighting
- Explore more advanced NLP preprocessing (lemmatization, Named Entity Recognition)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
