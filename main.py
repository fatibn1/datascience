import nltk
from pip._internal.resolution.resolvelib.factory import C

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st

# Load the text file and preprocess the data
file_path = "ww2.txt"  # Update with your file path if needed

try:
    with open('ww2.txt', 'r', encoding='utf-8') as f:
        data = f.read().replace('\n', ' ')
except FileNotFoundError:
    st.error(f"File '{file_path}' not found. Please check the file path.")

# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    query = preprocess(query)

    max_similarity = 0
    most_relevant_sentence = "Sorry, I couldn't find a relevant answer."

    for sentence in corpus:
        if len(set(query).union(sentence)) > 0:  # Prevent division by zero
            similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
            if similarity > max_similarity:
                max_similarity = similarity
                most_relevant_sentence = " ".join(sentence)

    return most_relevant_sentence

# Chatbot function
def chatbot(question):
    return get_most_relevant_sentence(question)

# Streamlit App
def main():
    st.title("Chatbot 🤖")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

    # Get user input
    question = st.text_input("You:")

    if st.button("Submit"):
        response = chatbot(question)
        st.write("Chatbot: " + response)

if __name__ == "__main__":
    main()
