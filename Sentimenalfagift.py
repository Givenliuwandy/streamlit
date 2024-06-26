import streamlit as st
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import os

# Specify NLTK data directory
nltk_data_dir = "/home/appuser/nltk_data"

# Check if NLTK data directory exists, if not, create it
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set NLTK data directory
nltk.data.path.append(nltk_data_dir)

# Download NLTK resources
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

# Set Streamlit options
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load positive and negative lexicons
positive_lexicon = pd.read_csv("positive.csv")
negative_lexicon = pd.read_csv("negative.csv")
positive_lexicon_dict = dict(zip(positive_lexicon['word'], positive_lexicon['weight']))
negative_lexicon_dict = dict(zip(negative_lexicon['word'], negative_lexicon['weight']))

# Define functions for text processing and sentiment labeling
def remove_stopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    custom_stop_words = {'yg', 'nya', 'aja'}
    stop_words.update(custom_stop_words)
    return ' '.join([word for word in text.split() if word not in stop_words])

def clean_text(text):
    removelist = "yg"
    text = re.sub('<.*?>', '', text)          # Remove HTML tags
    text = re.sub('https://.*', '', text)     # Remove URLs
    text = re.sub(r'[^a-zA-Z'+removelist+']', ' ', text)    # Remove non-alphanumeric characters
    text = text.lower()
    return text

def labeling_sentimen(sentence):
    words = sentence.split()
    sentence_score = sum(positive_lexicon_dict.get(word, 0) for word in words) + sum(negative_lexicon_dict.get(word, 0) for word in words)

    if sentence_score > 0:
        return "positif"
    elif sentence_score <= 0:
        return "negatif"
    else:
        return "netral"

# Define generate_word_cloud function
def generate_word_cloud(text_data, title=None):
    wordcloud = WordCloud(width=300, height=200, background_color='white').generate(text_data)
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title)
    st.pyplot()

# Define generate_bar_chart function
def generate_bar_chart(text_data, title=None):
    words = nltk.word_tokenize(text_data)
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word.lower() not in stop_words]
    word_freq = Counter(words)
    top_words = dict(word_freq.most_common(15))
    plt.figure(figsize=(8, 5))
    plt.bar(top_words.keys(), top_words.values(), color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    if title:
        plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

# Title
st.markdown("""
    <h1 style='text-align: center;'>Sentimen Analisis Pengiriman Alfagift</h1>
    """, unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])

if uploaded_file is not None:
    # Read uploaded CSV file
    if uploaded_file.type == 'text/csv':
        df = pd.read_csv(uploaded_file)

        # Rename columns and drop unnecessary columns
        df = df.rename(columns={'tb_rating': 'score', 'tb_review': 'content'})
        df = df.drop(columns=['tb_id', 'tb_created_date'])

        # Apply stop words removal and data cleaning to DataFrame column
        df['content'] = df['content'].apply(remove_stopwords)
        df['content'] = df['content'].apply(clean_text)
        
        # Label the sentiment of each review
        df['label'] = df['content'].apply(labeling_sentimen)

        # Filter out 'netral' and 'kosong' labels
        df = df[(df['label'] != 'netral') & (df['label'] != 'kosong')]

        # Calculate average length of each review
        avg_length = df['content'].apply(lambda x: len(x.split())).mean()

        # Calculate percentage of positive and negative sentiments
        pos_percentage = (df['label'] == 'positif').mean() * 100
        neg_percentage = (df['label'] == 'negatif').mean() * 100

        st.write("Average length of each review : ", avg_length)
        st.write("Data processing completed.")
        
        # Separate positive and negative sentiment data
        df_pos = df[df['label'] == 'positif']
        df_neg = df[df['label'] == 'negatif']

        # Concatenate all words in positive and negative sentiment data
        words_pos = ' '.join(df_pos['content'])
        words_neg = ' '.join(df_neg['content'])

        # Pie chart and bar chart side by side
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart for sentiment distribution
            st.write("Sentiment Distribution:")
            fig, ax = plt.subplots(figsize=(6, 6))
            labels = ['Positive', 'Negative']
            sizes = [pos_percentage, neg_percentage]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
            ax.axis('equal')
            st.pyplot(fig)

        with col2:
            # Bar chart for the number of each score
            st.write("Number of each score:")
            score_counts = df['score'].value_counts().sort_index()
            st.bar_chart(score_counts, width=400)

        # Display word clouds and bar charts for positive and negative sentiments
        if st.button("Generate Word Cloud and Top words"):
            st.session_state.wordcloud_generated = True

        if st.session_state.get('wordcloud_generated', False):
            # Word cloud and top 10 words for positive sentiment
            st.write("<h2 style='text-align: center;'>Word Cloud:</h2>", unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                generate_word_cloud(words_pos, 'Word Cloud Positive Sentiment')
            with col4:
                generate_word_cloud(words_neg, 'Word Cloud Negative Sentiment')

            # Word cloud and top 10 words for negative sentiment
            st.write("<h2 style='text-align: center;'>Top 15 Words:</h2>", unsafe_allow_html=True)
            col5, col6 = st.columns(2)
            with col5:
                generate_bar_chart(words_pos, 'Top 15 Words Positive Sentiment')
            with col6:
                generate_bar_chart(words_neg, 'Top 15 Words Negative Sentiment')

        # Train the LSTM model
        st.header("Training LSTM Model")
        # Hyperparameters of the model
        vocab_size = 3000
        oov_tok = ''
        embedding_dim = 100
        max_length = 200
        padding_type='post'
        trunc_type='post'

        # Tokenize sentences
        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(df['content'])
        word_index = tokenizer.word_index

        # Convert input sentence to sequence and pad sequences
        sequences = tokenizer.texts_to_sequences(df['content'])
        padded = pad_sequences(sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length)

        # Load the LSTM model
        modellstm = keras.Sequential([
            keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            keras.layers.Bidirectional(keras.layers.LSTM(64)),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        modellstm.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
        
        # Train the model
        labels = (df['label'] == 'positif').astype(int)  # Convert labels to binary
        num_epochs = 1
        history = modellstm.fit(padded, labels, epochs=num_epochs, verbose=1)
        
        # LSTM Model and Input Box
        st.header("Sentiment Prediction with LSTM")
        sentence_input = st.text_input("Enter a sentence:")
        if st.button("Predict Sentiment"):
            st.session_state.wordcloud_generated = False  # Reset the wordcloud state
            # Tokenize input sentence and pad sequence
            sequences = tokenizer.texts_to_sequences([sentence_input])
            padded_input = pad_sequences(sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length)

            # Predict sentiment
            prediction = modellstm.predict(padded_input)

            # Output the sentiment prediction
            if prediction >= 0.5:
                st.write("Predicted sentiment: Positive😁👍")
            else:
                st.write("Predicted sentiment: Negative😭👎")
