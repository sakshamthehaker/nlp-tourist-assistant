"""
preprocessing.py
Core NLP Text Preprocessing Pipeline
Provides tokenization, text normalization, stopword filtering, and stemming.
"""

import re
import string
from nltk.stem import PorterStemmer

# Tourist-specific stopwords set (preserving interrogatives like 'how', 'where', 'what' for intent relevance)
STANDARD_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'a', 'an', 'the', 'and', 'but',
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
    'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
    'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
}

# Stemmer instance
_stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """
    Cleans and normalizes text:
    - Lowercases text
    - Removes URLs and special punctuation
    - Normalizes extra whitespaces
    """
    if not text:
        return ""
    text = text.lower().strip()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric except basic punctuation for sentence boundaries
    text = re.sub(r'[^a-zA-Z0-9\s\?\!\.]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list[str]:
    """
    Tokenizes string into a list of word tokens.
    """
    cleaned = clean_text(text)
    # Tokenize by word boundary
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', cleaned)
    return tokens

def preprocess(text: str, remove_stopwords: bool = True, apply_stemming: bool = True) -> str:
    """
    Full NLP preprocessing pipeline:
    Cleaning -> Tokenization -> Stopword filtering -> Stemming -> Rejoined string.
    """
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STANDARD_STOPWORDS]
    if apply_stemming:
        tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

if __name__ == "__main__":
    sample = "What time does Qutub Minar open and how much is the entry ticket?"
    print("Original:", sample)
    print("Tokens:", tokenize(sample))
    print("Preprocessed:", preprocess(sample))
