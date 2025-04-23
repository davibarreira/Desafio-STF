import multiprocessing
from typing import List, Optional

import pandas as pd
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm

from .cleaner import normalize_text, process_docs

nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
stop_words = set(stopwords.words("portuguese"))


# --- 2) Batch cleaning via spaCy ---
def batch_clean_texts(
    texts: List[str],
    min_word_length: int = 3,
    max_word_length: int = 30,
    batch_size: int = 20,
    n_process: Optional[int] = None,
) -> List[str]:
    """
    Clean & lemmatize a list of Portuguese texts using spaCy in batches + parallel.

    Returns a list of cleaned strings.
    """
    if n_process is None:
        # leave one core free
        n_process = max(1, multiprocessing.cpu_count() - 1)

    # 1) normalize up‐front (cheap)
    norm_texts = (normalize_text(txt or "") for txt in texts)

    # 2) pass through spaCy in batch
    docs = nlp.pipe(
        norm_texts,
        batch_size=batch_size,
        n_process=n_process,
        disable=["parser", "ner"],  # we only need tokenizer/tagger for lemma
    )

    cleaned_texts = []
    for doc in tqdm(docs, total=len(texts), desc="Batch-cleaning"):
        cleaned_texts.append(process_docs(doc, min_word_length, max_word_length))

    return cleaned_texts


# --- 3) DataFrame helper ---
def clean_df_column(
    df: pd.DataFrame,
    src_col: str = "texto_bruto",
    dest_col: str = "texto_limpo",
    **batch_kwargs,
) -> pd.DataFrame:
    """
    Add a new column `dest_col` to `df` with cleaned text from `src_col`.
    Pass any of batch_clean_texts' kwargs (e.g. batch_size, n_process).
    """
    # ensure all values are strings
    texts = df[src_col].astype(str).tolist()
    df[dest_col] = batch_clean_texts(texts, **batch_kwargs)
    return df
