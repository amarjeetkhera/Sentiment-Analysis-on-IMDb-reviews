# Sentiment Analysis on IMDb Reviews

## Project Overview

This project focuses on building a Sentiment Analysis model to classify IMDb movie reviews as either positive or negative. Using Natural Language Processing (NLP) techniques, we analyze the textual data to predict the sentiment behind movie reviews. The concepts covered in this project form the basic building-block for understanding more complex Generative AI models, whose implementation will be covered in other projects.

## Description

The dataset used in this project is the IMDb Movie Reviews Dataset, sourced from Kaggle. This dataset is widely used for natural language processing (NLP) tasks, particularly for sentiment analysis. It contains a large collection of movie reviews from IMDb, along with their associated sentiment labels. The dataset can be downloaded from the following source:

Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

This dataset was originally provided by Stanford University for use in sentiment classification tasks and was made available on Kaggle for broader usage in NLP research and practice.
This dataset is ideal for sentiment analysis projects because it provides a balanced set of positive and negative reviews, making it perfect for training machine learning models.
The reviews are varied in length and vocabulary, providing a realistic challenge for text processing and model training.

### Dataset Summary

Total Reviews: 50,000

Labels: Binary (positive or negative sentiment)

Data Split: 25,000 reviews for training and 25,000 reviews for testing

Review Format: Text data containing user-submitted reviews

Sentiment Labels:

1 (positive sentiment)

0 (negative sentiment)

## Results

The TF-IDF Logistic Regression model achieved:

Accuracy: 88.46 %

Weighted Average Precision: 88 %

Weighted Average Recall: 88 %

Wighted Average F1-Score: 88 %

## Deployment

The trained Logistic Regression model was deployed as a web application using Streamlit. This allows users to input their own IMDb movie reviews and receive real-time sentiment predictions.

**Streamlit App Link:** https://sentiment-analysis-on-imdb-reviews-7zkvymoemaji92rpqqokxk.streamlit.app

**How to Use:**

1.  Visit the Streamlit app link.
2.  Enter your IMDb movie review in the provided text area.
3.  Click the "Analyze Sentiment" button.
4.  View the predicted sentiment displayed on the screen.

## Real-World Use-Case: Product Performance Analysis

Beyond movie reviews, this sentiment analysis model can be effectively applied to understand customer feedback on e-commerce platforms like Amazon and Alibaba.

By analyzing product reviews, businesses can:

-   **Identify key product strengths and weaknesses:** Determine what customers like or dislike about a product.
-   **Monitor customer satisfaction:** Track changes in sentiment over time to gauge customer happiness.
-   **Gain insights into product quality:** Understand common issues and areas for improvement.
-   **Inform marketing and product development:** Tailor strategies based on customer feedback.

  This deployment showcases the versatile practical application of the sentiment analysis model, making it accessible for real-world use.
