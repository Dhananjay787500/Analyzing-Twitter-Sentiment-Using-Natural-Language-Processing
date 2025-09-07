# Analyzing Twitter Sentiment Using Natural Language Processing

A simple yet complete workflow to classify tweets as **Positive**, **Negative**, or **Neutral** using machine learning and NLP.  
The project includes:

- Data preprocessing & model training in a Jupyter Notebook  
- A trained **scikit-learn** model with **TF-IDF** features  
- A lightweight **Streamlit** UI for real-time prediction  

---

## 📂 Project Structure
twitter-sentiment-nlp/
│
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── Sentiment_analysis_of_tweets.ipynb # Model training & evaluation
├── Tweets.csv # Dataset (sample tweets)
├── UI.py # Streamlit prediction interface
├── sentiment_model.pkl # Saved classifier
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
└── .gitignore # Ignored files/folders

---

## 🚀 Features
- Clean & preprocess tweet text
- Convert text to TF-IDF numerical features
- Train a **scikit-learn** classifier (Logistic Regression / Naive Bayes / etc.)
- Save & load the trained pipeline
- Simple Streamlit web app to test any tweet instantly

---

## 📊 Dataset
- **File:** `Tweets.csv`
- Contains tweets labeled as **Positive**, **Negative**, or **Neutral**.
- If you share the dataset publicly, make sure you respect licensing & privacy.

---

## ⚙️ Installation & Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-nlp.git
   cd twitter-sentiment-nlp

2. **Install dependencies**

    pip install -r requirements.txt


3. **Run the Streamlit UI**

    streamlit run UI.py


4. Open the URL shown in the terminal (usually http://localhost:8501) to access the app.

## 📈 Model Training

Training is fully documented in Sentiment_analysis_of_tweets.ipynb:

Text cleaning & tokenization

Stopword removal

TF-IDF vectorization

Model fitting & evaluation

Saving model + vectorizer (joblib.dump)


## 🖥️ Usage

Launch the Streamlit app.

Enter any tweet or sentence.

Click Predict Sentiment to see if it’s Positive, Negative, or Neutral.


## 🛠️ Tech Stack

Python 3.x

scikit-learn – Machine Learning

pandas / numpy – Data manipulation

Streamlit – Web interface

joblib – Model persistence

## 📄 License

MIT License (or whichever you choose).

## ✍️ Author

Dhananjay – GitHub Profile

Feel free to open issues or submit pull requests for improvements.
