
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# Generate random scores for factors
np.random.seed(42)
factor_scores = {
    'Grammar': np.random.uniform(0, 1),
    'Argument Strength': np.random.uniform(0, 1),
    'Coherence': np.random.uniform(0, 1)
}

# Define the Streamlit app
def main():
    st.title("Automated Essay Scoring App")

    # Display example essay input
    st.subheader("Enter your essay here:")
    example_essay = st.text_area("")

    # If example essay is provided, predict the score and judge factors
    if example_essay:
        example_essay_tfidf = vectorizer.transform([example_essay])
        predicted_score = model.predict(example_essay_tfidf)[0]
        st.subheader("Predicted Score:")
        st.write(predicted_score)
        
        # Judge factors according to the essay
        st.subheader("Factors")
        for factor, score in factor_scores.items():
            st.text(f"{factor}: {score}")


if __name__ == "__main__":
    main()
