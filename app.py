import streamlit as st
import pandas as pd
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF
from docx import Document
import PyPDF2

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model once
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_model = load_model()

# Keyword extraction
def extract_keywords(text):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words]
    return words

# Simple explanation
def explain_sentiment(text, sentiment_label):
    sentiment_label = sentiment_label.upper()
    keywords = extract_keywords(text)
    if sentiment_label == "POSITIVE":
        return f"Positive sentiment, driven by: {', '.join(keywords[:5])}"
    elif sentiment_label == "NEGATIVE":
        return f"Negative sentiment, driven by: {', '.join(keywords[:5])}"
    else:
        return f"Neutral sentiment, main words: {', '.join(keywords[:5])}"

# Extract texts from uploaded files
def extract_texts_from_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    texts = []

    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column!")
        else:
            texts = df['text'].dropna().astype(str).tolist()

    elif file_type == "txt":
        content = uploaded_file.read().decode("utf-8")
        texts = [line.strip() for line in content.split("\n") if line.strip()]

    elif file_type == "docx":
        doc = Document(uploaded_file)
        texts = [para.text.strip() for para in doc.paragraphs if para.text.strip()]

    elif file_type == "pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                for line in page_text.split("\n"):
                    if line.strip():
                        texts.append(line.strip())

    else:
        st.error("Unsupported file type!")
    
    return texts

# Streamlit UI
st.title("📊 Sentiment Analysis Dashboard")

text_input = st.text_area("Enter text for analysis:")
uploaded_file = st.file_uploader("Or upload a file (CSV, TXT, DOCX, PDF)", type=["csv", "txt", "docx", "pdf"])

texts = []
if uploaded_file:
    texts = extract_texts_from_file(uploaded_file)
elif text_input:
    texts = [text_input]

# Batch process
BATCH_SIZE = 32
results = []
if texts:
    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_results = sentiment_model(batch)
            results.extend(batch_results)
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        st.stop()

    # Map model labels to friendly names
    label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
    sentiments = [label_map.get(r['label'], r['label']).upper() for r in results]

    # Create results DataFrame
    df_results = pd.DataFrame({
        "text": texts,
        "sentiment": sentiments,
        "confidence": [r['score'] for r in results],
        "keywords": [extract_keywords(t) for t in texts],
    })
    df_results["explanation"] = [
        explain_sentiment(text, sentiment) for text, sentiment in zip(df_results["text"], df_results["sentiment"])
    ]

    # Show table
    st.subheader("Analysis Results")
    st.dataframe(df_results)

    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df_results['sentiment'].value_counts().reindex(["NEGATIVE", "NEUTRAL", "POSITIVE"], fill_value=0)
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette="Set2")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Sentiments")
    st.pyplot(fig)

    # Downloads
    st.subheader("Download Results")

    # CSV
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv, file_name="sentiment_results.csv", mime="text/csv")

    # JSON
    json_data = df_results.to_json(orient='records')
    st.download_button("⬇️ Download JSON", data=json_data, file_name="sentiment_results.json", mime="application/json")

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for i, row in df_results.iterrows():
        text = (
            f"Text: {row['text']}\n"
            f"Sentiment: {row['sentiment']}\n"
            f"Confidence: {row['confidence']:.2f}\n"
            f"Keywords: {', '.join(row['keywords'])}\n"
            f"Explanation: {row['explanation']}\n\n"
        )
        pdf.multi_cell(0, 10, text)

    pdf_buffer = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_buffer.write(pdf_bytes)
    pdf_buffer.seek(0)

    st.download_button(
        label="⬇️ Download PDF",
        data=pdf_buffer,
        file_name="sentiment_results.pdf",
        mime="application/pdf"
    )
