import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download NLTK resources
nltk.download('vader_lexicon')

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('essay_scores.csv')
    return data

data = load_data()

# Separate features (essays) and target (scores)
X = data['essay']
y = data['score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train_tfidf, y_train)

# Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Define the Streamlit app
def main():
    st.title("Automated Essay Scoring App")

    # Display example essay input
    st.subheader("Enter your essay here:")
    example_essay = st.text_area("")

    # If example essay is provided, predict the score and judge factors
    if example_essay:
        # Sentiment analysis
        sentiment_score = sentiment_analyzer.polarity_scores(example_essay)
        
        # Calculate factor scores based on sentiment analysis
        grammar_score = sentiment_score['compound']  # Using compound score as a proxy for grammar
        
        # Coherence score using TextBlob
        coherence_score = TextBlob(example_essay).sentiment.polarity
        
        # Argument strength score using TextBlob
        argument_strength_score = TextBlob(example_essay).sentiment.subjectivity
        
        # Combine factor scores
        factor_scores = {
            'Grammar': grammar_score,
            'Argument Strength': argument_strength_score,
            'Coherence': coherence_score
        }
        # Predict the score
        example_essay_tfidf = vectorizer.transform([example_essay])
        predicted_score = model.predict(example_essay_tfidf)[0]
        
        # Display predicted score
        st.subheader("Predicted Score:")
        st.write(predicted_score)
        
        # Display factor scores
        st.subheader("Factors")
        for factor, score in factor_scores.items():
            st.text(f"{factor}: {score}")
        

if __name__ == "__main__":
    main()
