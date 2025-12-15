"""
Physician Notetaker: Medical Transcription → NER → Summarization → Sentiment & SOAP

A complete NLP pipeline for converting physician-patient transcripts into 
structured medical outputs including:
- Named Entity Recognition (NER) for medical entities
- Structured medical report generation (JSON)
- Keyword extraction with relevance scores
- Sentiment and intent classification
- SOAP note generation

Author: Your Name
Version: 1.0.0
"""

__version__ = "1.0.0"

from .preprocess import MedicalPreprocessor
from .ner import MedicalNER
from .summarizer import MedicalSummarizer
from .sentiment import SentimentIntentClassifier
from .soap_generator import SOAPGenerator
from .keywords import KeywordExtractor

__all__ = [
    "MedicalPreprocessor",
    "MedicalNER",
    "MedicalSummarizer",
    "SentimentIntentClassifier",
    "SOAPGenerator",
    "KeywordExtractor",
]
