import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb  # IMBD dataset
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model

# Load the imdb dataset word index

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# load the model
model = load_model(r'E:\folder E\Udemy\IMBD_Review_Simple_RNN\SimpleRNN_project\simple_rnn_imdb.h5')

# step 2 helper function
# function to decode reviews
def decode_reviews(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review



import streamlit as st
## streamlit app

st.title('IMBD Movies Review Sentiment Analysis')
st.write('Enter a moview review to classify it as positive or negative')

# user input
user_input = st.text_area('Moview review')

if st.button('Classify'):
    
    preprocessed_input = preprocess_text(user_input)
    
    ## make prediction
    prediction = model.predict(preprocessed_input)
    sentiments = 'Positive' if prediction[0][0] > 0.4 else 'Negative'
    
    # Display the result
    st.write(f'Sentiment: {sentiments}')
    st.write(f'Score: {prediction[0][0]}')
    
else:
    st.write('please enter the movie review')