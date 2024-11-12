🎬📊 Sentiment Analysis on IMDb Reviews

📄 Project Overview

This project focuses on building a Sentiment Analysis model to classify IMDb movie reviews as either positive or negative. Using Natural Language Processing (NLP) techniques, we analyze the textual data to predict the sentiment behind movie reviews. The project was completed as part of a guided learning course on Coursera, with enhancements and custom modifications for better understanding and visualization.

🚀 Objectives

Extract insights from IMDb movie reviews and determine if the sentiment is positive or negative.
Implement various text preprocessing techniques to clean and prepare the data.
Build and evaluate models using different feature extraction methods like Bag-of-Words and TF-IDF.
Visualize the data distribution, word frequencies, and model performance using matplotlib and seaborn.

📊 Key Features

Data Cleaning & Preprocessing: Removal of noise, punctuation, stopwords, and tokenization.
Feature Engineering: Implementation of Bag-of-Words and TF-IDF to transform text into numerical features.
Model Training: Trained a Logistic Regression classifier to predict the sentiment of IMDb reviews.
Data Visualization: Visualized class distribution, review length, word clouds, n-grams, and model performance (confusion matrix, precision-recall, and ROC curves).

🗃️ Dataset

The dataset used in this project is the IMDb Movie Reviews Dataset, which contains 50,000 reviews labeled as either positive or negative. The dataset can be downloaded from the following sources:

Kaggle

🛠️ Tools & Technologies

Python: Core programming language.
Jupyter Notebook: Development environment.
Libraries:
pandas, numpy: Data manipulation and analysis.
scikit-learn: Model building, feature extraction, and evaluation.
matplotlib, seaborn: Data visualization.
nltk: Natural Language Processing toolkit.
wordcloud: Visualization of frequent words.

📉 Exploratory Data Analysis (EDA)

We performed extensive EDA to understand the dataset:

Class Distribution: Visualized the balance between positive and negative reviews.
Word Clouds: Created separate word clouds for positive and negative reviews to highlight common words.
N-gram Analysis: Analyzed frequently occurring bigrams and trigrams.
Review Length Distribution: Explored the distribution of review lengths and their correlation with sentiment.
TF-IDF Analysis: Visualized the most significant words based on their TF-IDF scores.

🧑‍💻 Model Building & Evaluation

Preprocessed the data using Bag-of-Words and TF-IDF vectorizers.
Trained a Logistic Regression model to classify reviews as positive or negative.
Evaluated model performance using:
-Confusion Matrix
-Precision-Recall Curve
-ROC Curve

📈 Results

The Logistic Regression model achieved:

Accuracy: 88%
Precision: 87%
Recall: 89%
F1-Score: 88%
