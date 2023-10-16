import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define a function to calculate sentiment score for a list of words using SentiWordNet
def calculate_sentiment_score(words):
    sentiment_score = 0
    word_count = 0
    lemmatizer = WordNetLemmatizer()

    for word in words:
        tokens = word_tokenize(word)
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if token not in stopwords.words('english')]

        for token in tokens:
            pos_score, neg_score = 0, 0
            synsets = list(swn.senti_synsets(token))
            
            if synsets:
                pos_score = synsets[0].pos_score()
                neg_score = synsets[0].neg_score()
                sentiment_score += pos_score - neg_score
                word_count += 1

    if word_count > 0:
        sentiment_score /= word_count

    return sentiment_score

# Function to classify a review's sentiment
def classify_sentiment(review, threshold=0):
    sentiment_score = calculate_sentiment_score(review)
    return "positive" if sentiment_score > threshold else "negative"

# Function to classify a batch of reviews and calculate accuracy
def classify_batch_reviews(reviews, labels, threshold=0):
    predictions = [classify_sentiment(review, threshold) for review in reviews]
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels)
    return accuracy, predictions

# Streamlit app
st.title("Movie Review Sentiment Classification")

# Input for entering a movie review
review_input = st.text_area("Enter a movie review:")

# Button for classifying the review
if st.button("Classify"):
    if review_input:
        sentiment = classify_sentiment(review_input.split())
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to classify.")