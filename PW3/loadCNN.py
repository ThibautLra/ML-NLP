import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
import json

def loadCNN():
    articles_url = "https://github.com/ThibautLra/ML-NLP/blob/main/PW3/CNNArticles"
    abstracts_url = "https://github.com/ThibautLra/ML-NLP/blob/main/PW3/CNNGold"

    articles_response = requests.get(articles_url)
    abstracts_response = requests.get(abstracts_url)

    if articles_response.status_code == 200 and abstracts_response.status_code == 200:
                articles = pickle.loads(articles_response.content)
                abstracts = pickle.loads(abstracts_response.content)


    #file = open("./CNNArticles", 'rb')
    #articles = pickle.load(file)
    #file = open("./CNNGold", 'rb')
    #abstracts = pickle.load(file)

    articlesCl = []
    for article in articles:
        articlesCl.append(article.replace("”", "").rstrip("\n"))
    articles = articlesCl

    articlesCl = []
    for article in abstracts:
        articlesCl.append(article.replace("”", "").rstrip("\n"))
    abstracts = articlesCl

    return articles, abstracts

articles, abstracts = loadCNN()

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(articles)

# Streamlit app
st.title("Information Retrieval System")

# User input for summary
summary_input = st.text_input("Enter a summary:")

if st.button("Retrieve Documents"):
    # Preprocess the summary input
    summary_tfidf = tfidf_vectorizer.transform([summary_input])

    # Calculate cosine similarities
    cosine_similarities = linear_kernel(summary_tfidf, tfidf_matrix).flatten()

    # Create a list of (document_index, cosine_similarity_score) tuples
    document_scores = [(i, score) for i, score in enumerate(cosine_similarities)]

    # Sort documents by cosine similarity in descending order
    ranked_documents = sorted(document_scores, key=lambda x: x[1], reverse=True)

    st.header("Ranked Documents:")
    for i, (document_index, score) in enumerate(ranked_documents[:100]):
        st.write(f"Rank {i + 1}: Document {document_index}, Similarity Score: {score:.4f}")

    # Get the index of the document with the highest score
    highest_score_index = ranked_documents[0][0]

    # Display the content of the highest-ranked document
    st.header("Content of the Highest-Ranked Document:")
    st.text(articles[highest_score_index])
    document_text = articles[highest_score_index]
    
#import spacy

# Load the spaCy language model
#nlp = spacy.load("en_core_web_sm")

# Function to extract and highlight keywords
#def extract_and_highlight_keywords(document_text):
    # Process the document text with spaCy
#    doc = nlp(document_text)

    # Extract keywords (in this example, we're using nouns as keywords)
#    keywords = [token.text for token in doc if token.pos_ == "NOUN"]

    # Highlight keywords with CSS
#    highlighted_text = document_text
#    for keyword in keywords:
#        highlighted_text = highlighted_text.replace(keyword, f'<span style="text-decoration: underline;">{keyword}</span>')

#    return keywords, highlighted_text

# Extract and highlight keywords
#extracted_keywords, highlighted_document_text = extract_and_highlight_keywords(document_text)

#st.title("Keyword Extraction")
#st.text(extracted_keywords)
#st.header("Highlighted Document Text:")
#st.markdown(highlighted_document_text, unsafe_allow_html=True)
