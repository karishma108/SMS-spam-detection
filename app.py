import os
port = int(os.environ.get("PORT", 8501))
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Page configuration
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for glassmorphic effect, dark overlay, and green button
st.markdown(
    """
    <style>
    /* Full-page background with Unsplash image fallback color */
    body {
        margin: 0;
        padding: 0;
        background-color: #000814;
        background-image: url('https://source.unsplash.com/1920x1080/?network,abstract');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Glassmorphic container */
    .stApp {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Title and text: force white color */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp p,
    .stApp div {
        color: #ffffff !important;
    }

    /* Text area styling: translucent white bg, black input text */
    .stTextArea textarea,
    .stTextArea div[role="textbox"] {
        background-color: rgba(255,255,255,0.85) !important;
        color: #000000 !important;
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.3) !important;
        padding: 0.5rem;
    }
    .stTextArea textarea::placeholder,
    .stTextArea div[role="textbox"]::placeholder {
        color: rgba(0, 0, 0, 0.5) !important;
    }

    /* Button styling: green gradient */
    .stButton>button {
        background: linear-gradient(90deg, #28a745, #218838) !important;
        color: #ffffff !important;
        padding: .6rem 1.2rem;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
        transition: transform 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize stemmer and NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

# Text transformation function
@st.cache_data
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = [t for t in tokens if t.isalnum()]
    no_stop = [t for t in filtered if t not in stopwords.words('english')]
    stemmed = [ps.stem(t) for t in no_stop]
    return " ".join(stemmed)

# Load model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title and instruction
st.title("📧 SMS Spam Classifier")
st.write("Enter a message below to check if it's spam or not.")

# User input area
input_sms = st.text_area(
    label="Your message",
    height=150,
    placeholder="Type or paste your message here..."
)

# Prediction button
if st.button('Check now'):
    if not input_sms:
        st.warning("Please enter a message to analyze.")
    else:
        with st.spinner('Analyzing...'):
            transformed = transform_text(input_sms)
            vector_input = tfidf.transform([transformed])
            result = model.predict(vector_input)[0]
        if result == 1:
            st.error("🚩 This message is Spam!")
        else:
            st.success("✅ This message is Not Spam.")
