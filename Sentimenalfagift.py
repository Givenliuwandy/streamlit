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

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

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
        df = df.rename(columns={'tb_rating': 'score', 'tb_review': 'content_indo'})
        df = df.drop(columns=['tb_id', 'tb_created_date'])

        # Label the sentiment of each review
        df['label'] = df['content_indo'].apply(labeling_sentimen)

        # Filter out 'netral' and 'kosong' labels
        df = df[(df['label'] != 'netral') & (df['label'] != 'kosong')]

        # Calculate average length of each review
        avg_length = df['content_indo'].apply(lambda x: len(x.split())).mean()

        # Calculate percentage of positive and negative sentiments
        pos_percentage = (df['label'] == 'positif').mean() * 100
        neg_percentage = (df['label'] == 'negatif').mean() * 100

        st.write("Average length of each review : ", avg_length)
        st.write("Data processing completed.")

        # Apply stop words removal and data cleaning to DataFrame column
        df['content'] = df['content_indo'].apply(remove_stopwords)
        df['content'] = df['content'].apply(clean_text)

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
            st.write("<h2 style='text-align: center;'>Top 15 Words:</h2>", unsafe_allow_html=True)
            col5, col6 = st.columns(2)
            with col5:
                generate_bar_chart(words_pos, 'Top 15 Words Positive Sentiment')
            with col6:
                generate_bar_chart(words_neg, 'Top 15 Words Negative Sentiment')

        # LSTM Model and Input Box
        st.header("Sentiment Prediction with LSTM")
        sentence_input = st.text_input("Enter a sentence:")
        if st.button("Predict Sentiment"):
            # Tokenization and padding
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(df['content'])
            vocab_size = len(tokenizer.word_index) + 1
            max_length = max([len(s.split()) for s in df['content']])
            padded_sequences = pad_sequences(tokenizer.texts_to_sequences([sentence_input]), maxlen=max_length)

            # Model training
            model = keras.Sequential([
                keras.layers.Embedding(vocab_size, 100, input_length=max_length),
                keras.layers.Bidirectional(keras.layers.LSTM(64)),
                keras.layers.Dense(24, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(padded_sequences, np.array([0]), epochs=1, verbose=0)

            # Sentiment prediction
            prediction = model.predict(padded_sequences)

            # Output the sentiment prediction
            if prediction >= 0.5:
                st.write("Predicted sentiment: Positive 😁👍")
            else:
                st.write("Predicted sentiment: Negative 😭👎")
