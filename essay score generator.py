import pandas as pd
import random

# Define essay topics
essay_topics = [
    "The importance of education in society.",
    "The benefits of regular exercise.",
    "Climate change and its impact.",
    "The role of social media in modern society.",
    "Technology and the future of work.",
    "Artificial intelligence and its applications.",
    "Mental health awareness.",
    "Ethical implications of genetic engineering.",
    "Renewable energy sources.",
    "Globalization and cultural diversity."
]

# Generate synthetic essay data
essays = []
scores = []

for _ in range(1000):  # Generate 1000 essays
    essay = random.choice(essay_topics)
    score = random.randint(60, 100)  # Random score between 60 and 100
    essays.append(essay)
    scores.append(score)

# Create a DataFrame
data = pd.DataFrame({'essay': essays, 'score': scores})

# Save the dataset to a CSV file
data.to_csv('essay_scores.csv', index=False)
