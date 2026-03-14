# Election Misinformation and Fake News Detection

## Overview
This project implements a Fake News Detection system that uses
Natural Language Processing and Machine Learning to automatically
classify news articles as Real or Fake.

The model predicts:
- Real News (1) — legitimate and verified news articles
- Fake News (0) — misinformation and fabricated news articles

All models are evaluated using Accuracy, Precision, Recall and F1 Score.

---

## Project Structure
```
election-misinformation-and-fake-news-detection/
├── Election misinformation and fake news detection.ipynb  # main notebook
├── cleaned_news.csv                                       # cleaned dataset
├── requirements.txt                                       # required libraries
└── README.md
```

---

## Dataset
| Dataset   | Source  | Description                              |
|-----------|---------|------------------------------------------|
| Fake.csv  | Kaggle  | 23,481 fake news articles                |
| True.csv  | Kaggle  | 21,417 real news articles                |
| Combined  | Merged  | 44,898 total articles with labels        |

Dataset Source:
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Only politics and election related news articles are included.
Both files are merged and labeled before training.
- Fake.csv → label = 0
- True.csv → label = 1

---

## Project Workflow
| Phase  | Description                                          |
|--------|------------------------------------------------------|
| 1      | Define goal, metric and success threshold            |
| 2      | Load dataset, check shape, columns and null values   |
| 3      | Exploratory Data Analysis                            |
| 4      | Text Cleaning and Preprocessing                      |
| 5      | TF-IDF Vectorization                                 |
| 6      | Train Test Split (80/20)                             |
| 7      | Model Training                                       |
| 8      | Model Evaluation                                     |
| 9      | Model Comparison                                     |
| 10     | Conclusion                                           |

---

## Text Preprocessing Steps
- Converted text to lowercase
- Removed URLs, HTML tags, punctuation and numbers
- Removed stopwords using NLTK (179 stopwords)
- Applied lemmatization using WordNetLemmatizer
- Combined title and text into single column

---

## Vectorization
| Setting       | Value                        |
|---------------|------------------------------|
| Method        | TF-IDF                       |
| Max Features  | 5000                         |
| Ngram Range   | (1,2) unigrams and bigrams   |
| Train Split   | 80% (35,918 articles)        |
| Test Split    | 20% (8,980 articles)         |

---

## Models Used
| Model               | Description                                      |
|---------------------|--------------------------------------------------|
| Logistic Regression | Simple linear model for text classification      |
| Naive Bayes         | Probability based model for text data            |
| Random Forest       | Ensemble of multiple decision trees              |
| Gradient Boosting   | Sequential ensemble model                        |

---

## Results
| Model               | Accuracy  | Precision | Recall | F1 Score | Rank |
|---------------------|-----------|-----------|--------|----------|------|
| Random Forest       | 99.89%    | 99.95%    | 99.88% | 99.92%   | 1    |
| Gradient Boosting   | 99.77%    | 99.88%    | 99.76% | 99.82%   | 2    |
| Logistic Regression | 99.56%    | 99.51%    | 99.81% | 99.66%   | 3    |
| Naive Bayes         | 97.35%    | 97.95%    | 97.93% | 97.94%   | 4    |

Success Threshold : 85% F1 Score
Status            : Achieved by all 4 models

Best Model        : Random Forest
Best Accuracy     : 99.89%
Best F1 Score     : 100%

---

## How to Run

### On Google Colab (Recommended)
```
1. Open the notebook in Google Colab
2. Mount Google Drive
3. Upload Fake.csv and True.csv to Google Drive
4. Run all cells from top to bottom
```

### Local Setup
```
pip install -r requirements.txt
jupyter notebook "Election misinformation and fake news detection.ipynb"
```

---

## Requirements
```
pandas
numpy
matplotlib
seaborn
wordcloud
nltk
scikit-learn
jupyter
```

---

## Evaluation Metrics
| Metric            | Description                                        |
|-------------------|----------------------------------------------------|
| Accuracy          | Percentage of total correct predictions            |
| Precision         | Out of predicted fake how many were actually fake  |
| Recall            | Out of actual fake how many were correctly found   |
| F1 Score          | Harmonic mean of precision and recall              |
| Confusion Matrix  | Visual breakdown of correct and wrong predictions  |
| ROC AUC Score     | How well model separates fake from real news       |

---

## Output Files
| File              | Description                                        |
|-------------------|----------------------------------------------------|
| cleaned_news.csv  | Preprocessed and cleaned dataset                  |
| confusion matrix  | Heatmap for each model                            |
| ROC curves        | ROC curve comparison for all models               |
| PR curves         | Precision recall curve for all models             |
| bar chart         | Model performance comparison chart                |

---

## Conclusion
Random Forest was selected as the best model achieving
99.89% accuracy and 100% F1 Score which far exceeded
the defined success threshold of 85%.

All 4 models performed exceptionally well confirming that
TF-IDF features combined with machine learning effectively
detect election misinformation and fake news articles.

---

## References
- Kaggle Fake News Detection Dataset
  https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
- Scikit-learn Documentation
  https://scikit-learn.org
- NLTK Documentation
  https://www.nltk.org
