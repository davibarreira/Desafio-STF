"""
Text cleaning utilities for Portuguese legal texts.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean and normalize Portuguese legal text for machine learning processing.
    
    This function:
    - Removes excessive whitespace
    - Converts to lowercase
    - Removes special characters
    - Normalizes accents
    - Removes line breaks and tabs
    
    Args:
        text: The input legal text in Portuguese
        
    Returns:
        Cleaned and normalized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize unicode characters (handle accents)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # Replace line breaks, tabs, and multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra punctuation and other symbols while keeping basic punctuation
    text = re.sub(r'[^\w\s\.,;:!?]', '', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text