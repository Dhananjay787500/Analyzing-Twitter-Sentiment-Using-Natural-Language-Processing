import streamlit as st
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Set background image
background_image = "https://miro.medium.com/max/1000/1*vp1M37AGMOFwCvLxVm62IA.jpeg"  # Replace with your image URL
background_style = f'''
    <style>
        .stApp {{
            background: url("{background_image}") no-repeat center center fixed;
            background-size: cover;
        }}
    </style>
'''
st.markdown(background_style, unsafe_allow_html=True)

# Streamlit UI
st.title("Tweet Sentiment Analysis")
st.subheader("Predict whether a tweet is Positive, Negative, or Neutral")

# User input
user_input = st.text_area("Enter a tweet:")

if st.button("Predict Sentiment"):
    if user_input:
        # Transform input using TF-IDF vectorizer
        input_tfidf = tfidf.transform([user_input])

        # Predict sentiment
        prediction = model.predict(input_tfidf)[0]

        # Map prediction to sentiment label
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map[prediction]

        # Display result
        st.success(f"Tweet Sentiment: {sentiment}")
    else:
        st.warning("Please enter a tweet for prediction.")
