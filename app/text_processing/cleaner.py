"""
Text cleaning utilities for Portuguese legal texts optimized for TF-IDF.
"""

import re
import unicodedata
from typing import List

import nltk
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Initialize NLP pipeline and stopwords once
nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])

# Basic Portuguese stopwords
stop_words = set(stopwords.words("portuguese"))


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, keep only a–z and spaces, collapse whitespace."""
    text = text.lower()
    # decompose accents then drop combining marks
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # keep only ascii letters & spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # collapse runs of whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_docs(
    doc: spacy.tokens.Doc, min_word_length: int, max_word_length: int
) -> str:
    """Process a spaCy document and return a cleaned string."""
    tokens = []
    for tok in doc:
        # skip if not alpha or spaCy stopword or NLTK stopword
        if not tok.is_alpha or tok.is_stop or tok.lemma_ in stop_words:
            continue
        lemma = tok.lemma_
        L = len(lemma)
        if L < min_word_length or L > max_word_length:
            continue
        tokens.append(lemma)
    return " ".join(tokens)


def clean_text(text: str, min_word_length: int = 3, max_word_length: int = 30) -> str:
    """Clean and normalize Portuguese text for TF-IDF."""
    norm_text = normalize_text(text)
    doc = nlp(norm_text)
    processed_text = process_docs(doc, min_word_length, max_word_length)
    return processed_text
