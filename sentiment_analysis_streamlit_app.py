# -*- coding: utf-8 -*-
"""Sentiment Analysis Streamlit_app.ipynb"""

import streamlit as st
import joblib

# Load the saved model and vectorizer
model_file = joblib.load("logistic_regression_tfidf.pkl")
vec_file = joblib.load("tfidf_vectorizer.pkl")

# Streamlit app
st.title("Sentiment Analysis App")
st.write("Enter a movie review to analyze its sentiment.")

# Input text box
review = st.text_area("Review Text", "")

# Analyze button
if st.button("Analyze Sentiment"):
    if review:
        # Preprocess the input and make a prediction
        review_vectorized = vec_file.transform([review])
        prediction = model_file.predict(review_vectorized)

        # Display the result
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"Sentiment: **{sentiment}**")
    else:
        st.write("Please enter a review!")
# Footer
st.markdown("---")
st.markdown("""
<style>
    .footer {
        text-align: center;
        padding: 10px;
        color: gray;
    }
</style>
<div class="footer">
    Model: Logistic Regression | Made with Streamlit by Amarjeet Khera
</div>
""", unsafe_allow_html=True)
