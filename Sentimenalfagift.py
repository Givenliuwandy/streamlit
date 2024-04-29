import streamlit as st
import pandas as pd
import re
import numpy as np
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

st.set_option('deprecation.showPyplotGlobalUse', False)

# Define generate_word_cloud function
def generate_word_cloud(text_data, title=None):
    # Generate word cloud
    wordcloud = WordCloud(width=300, height=200, background_color='white').generate(text_data)

    # Plot word cloud
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title)
    st.pyplot()

# Define generate_bar_chart function
def generate_bar_chart(text_data, title=None):
    # Tokenize words
    words = nltk.word_tokenize(text_data)

    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word.lower() not in stop_words]

    # Count word frequencies
    word_freq = Counter(words)

    # Get top 10 words and their frequencies
    top_words = dict(word_freq.most_common(10))

    # Plot bar chart
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

        # Rename columns
        df = df.rename(columns={'tbtdr_rating': 'score', 'tbtdr_review': 'content'})
        df = df.drop(columns=['tbtdr_id', 'tbtdr_created_date'])

        # Labeling data based on rating
        label = []
        for row in df['score']:
            if row < 3:
                label.append('negatif')
            elif row == 3:
                label.append('netral')
            elif row > 3:
                label.append('positif')
            else:
                label.append('kosong')
        df['label'] = label

        # Filter out 'netral' and 'kosong' labels
        df = df[df.label != 'netral']
        df = df[df.label != 'kosong']

        # Calculate average length of each review
        avg_length = df['content'].apply(lambda x: len(x.split())).mean()

        # Calculate percentage of positive and negative sentiments
        pos_percentage = (df['label'] == 'positif').mean() * 100
        neg_percentage = (df['label'] == 'negatif').mean() * 100

        st.write("Average length of each review : ", avg_length)

        # Display data processing completion message
        st.write("Data processing completed.")

        # Define stopwords and custom stop words
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('stopwords')

        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('indonesian'))
        custom_stop_words = {'yg', 'nya', 'aja'}
        stop_words.update(custom_stop_words)

        # Function to remove HTML tags, URLs, non-alphanumeric characters, and stopwords
        def remove_tags(string):
            removelist = "yg"
            result = re.sub('<.*?>', '', string)          # Remove HTML tags
            result = re.sub('https://.*', '', result)     # Remove URLs
            result = re.sub(r'[^a-zA-Z'+removelist+']', ' ', result)    # Remove non-alphanumeric characters
            result = result.lower()
            return result

        # Apply stop words removal and stemming to DataFrame column
        df['content'] = df['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        df['content'] = df['content'].apply(lambda cw: remove_tags(cw))

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
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

        with col2:
            # Bar chart for the number of each score
            st.write("Number of each score:")
            score_counts = df['score'].value_counts().sort_index()
            st.bar_chart(score_counts, width=400)

        # Display word clouds and bar charts for positive and negative sentiments
        if st.button("Generate Word Cloud and Top words"):
            # Word cloud and top 10 words for positive sentiment
            st.write("<h2 style='text-align: center;'>Word Cloud:</h2>", unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                generate_word_cloud(words_pos, 'Word Cloud Positive Sentiment')
            with col4:
                generate_word_cloud(words_neg, 'Word Cloud Negative Sentiment')

            # Word cloud and top 10 words for negative sentiment
            st.write("<h2 style='text-align: center;'>Top 10 Words:</h2>", unsafe_allow_html=True)
            col5, col6 = st.columns(2)
            with col5:
                generate_bar_chart(words_pos, 'Top 10 Words Positive Sentiment')
            with col6:
                generate_bar_chart(words_neg, 'Top 10 Words Negative Sentiment')

    # LSTM Model and Input Box
    st.header("Sentiment Prediction with LSTM")
    sentence_input = st.text_input("Enter a sentence:")
    if st.button("Predict Sentiment"):
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
        sequences = tokenizer.texts_to_sequences([sentence_input])
        padded = pad_sequences(sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length)

        # Load the LSTM model
        # Hyperparameters of the model
        vocab_size = 3000
        oov_tok = ''
        embedding_dim = 100
        max_length = 200
        padding_type='post'
        trunc_type='post'

        modellstm = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
        ])

        # Predict sentiment
        prediction = modellstm.predict(padded)

        # Output the sentiment prediction
        if prediction >= 0.5:
            st.write("Predicted sentiment: Positive😁👍")
        else:
            st.write("Predicted sentiment: Negative😭👎")
