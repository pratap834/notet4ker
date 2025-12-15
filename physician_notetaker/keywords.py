"""
Keyword extraction module for medical transcripts.

Extracts and ranks important medical phrases using TF-IDF, 
RAKE, YAKE, and semantic embeddings.
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import yake

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .utils import get_logger

logger = get_logger(__name__)


class KeywordExtractor:
    """Extract and rank medical keywords and phrases."""
    
    def __init__(self, use_embeddings: bool = True):
        """
        Initialize keyword extractor.
        
        Args:
            use_embeddings: Whether to use semantic embeddings for ranking
        """
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        
        # Initialize YAKE extractor
        self.yake_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # Max n-gram size
            dedupLim=0.7,
            top=20,
            features=None
        )
        
        # Initialize sentence transformer if available
        if self.use_embeddings:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer for semantic ranking")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.use_embeddings = False
        
        # Medical domain keywords that should be boosted
        self.medical_boost_terms = {
            'injury', 'diagnosis', 'treatment', 'therapy', 'pain', 
            'accident', 'recovery', 'prognosis', 'examination', 'symptoms',
            'medication', 'surgery', 'chronic', 'acute', 'condition'
        }
    
    def extract_keywords(
        self, 
        text: str, 
        entities: Optional[List[Dict]] = None,
        top_k: int = 10
    ) -> List[Dict[str, any]]:
        """
        Extract and rank keywords from text.
        
        Args:
            text: Input text
            entities: Optional list of NER entities to boost
            top_k: Number of top keywords to return
        
        Returns:
            List of keyword dicts with 'term' and 'score'
        """
        # Method 1: YAKE keyword extraction
        yake_keywords = self._extract_yake_keywords(text, top_k * 2)
        
        # Method 2: TF-IDF on medical phrases
        tfidf_keywords = self._extract_tfidf_keywords(text, top_k * 2)
        
        # Method 3: Entity-based keywords
        entity_keywords = []
        if entities:
            entity_keywords = self._extract_entity_keywords(entities)
        
        # Combine and rank
        combined_keywords = self._combine_and_rank(
            yake_keywords,
            tfidf_keywords,
            entity_keywords,
            text,
            top_k
        )
        
        return combined_keywords
    
    def _extract_yake_keywords(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Extract keywords using YAKE algorithm.
        
        Args:
            text: Input text
            top_k: Number of keywords
        
        Returns:
            List of (keyword, score) tuples
        """
        try:
            keywords = self.yake_extractor.extract_keywords(text)
            # YAKE returns (keyword, score) where lower score is better
            # Invert score to make higher = better
            keywords = [(kw, 1.0 / (1.0 + score)) for kw, score in keywords[:top_k]]
            return keywords
        except Exception as e:
            logger.warning(f"YAKE extraction failed: {e}")
            return []
    
    def _extract_tfidf_keywords(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF on n-grams.
        
        Args:
            text: Input text
            top_k: Number of keywords
        
        Returns:
            List of (keyword, score) tuples
        """
        try:
            # Use n-grams to capture medical phrases
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=top_k * 3,
                stop_words='english'
            )
            
            # TF-IDF needs multiple documents, so split by sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            if len(sentences) < 2:
                # If only one sentence, duplicate it
                sentences = sentences * 2
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores across all sentences
            tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            
            # Get top keywords
            top_indices = tfidf_scores.argsort()[-top_k:][::-1]
            keywords = [(feature_names[i], float(tfidf_scores[i])) for i in top_indices]
            
            return keywords
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}")
            return []
    
    def _extract_entity_keywords(self, entities: List[Dict]) -> List[Tuple[str, float]]:
        """
        Convert entities to keywords with relevance scores.
        
        Args:
            entities: List of entity dicts from NER
        
        Returns:
            List of (keyword, score) tuples
        """
        keywords = []
        
        # Priority weights for entity types
        type_weights = {
            'DIAGNOSIS': 1.0,
            'TREATMENT': 0.95,
            'SYMPTOM': 0.85,
            'PROGNOSIS': 0.9,
            'INCIDENT': 0.8,
            'PROCEDURE': 0.9,
            'MEDICATION': 0.95,
        }
        
        for entity in entities:
            entity_type = entity.get('type', '')
            confidence = entity.get('confidence', 0.5)
            text = entity.get('normalized', entity.get('text', ''))
            
            # Calculate score based on entity type weight and confidence
            base_weight = type_weights.get(entity_type, 0.5)
            score = base_weight * confidence
            
            keywords.append((text, score))
        
        return keywords
    
    def _combine_and_rank(
        self,
        yake_keywords: List[Tuple[str, float]],
        tfidf_keywords: List[Tuple[str, float]],
        entity_keywords: List[Tuple[str, float]],
        text: str,
        top_k: int
    ) -> List[Dict[str, any]]:
        """
        Combine keywords from different sources and rank them.
        
        Args:
            yake_keywords: Keywords from YAKE
            tfidf_keywords: Keywords from TF-IDF
            entity_keywords: Keywords from entities
            text: Original text for context
            top_k: Number of top keywords to return
        
        Returns:
            Ranked list of keyword dicts
        """
        # Aggregate scores for each keyword
        keyword_scores = {}
        
        # Add YAKE keywords (weight: 0.4)
        for kw, score in yake_keywords:
            kw_clean = self._clean_keyword(kw)
            if kw_clean:
                keyword_scores[kw_clean] = keyword_scores.get(kw_clean, 0) + score * 0.4
        
        # Add TF-IDF keywords (weight: 0.3)
        for kw, score in tfidf_keywords:
            kw_clean = self._clean_keyword(kw)
            if kw_clean:
                # Normalize TF-IDF scores to 0-1 range
                max_tfidf = max([s for _, s in tfidf_keywords]) if tfidf_keywords else 1.0
                normalized_score = score / max_tfidf if max_tfidf > 0 else 0
                keyword_scores[kw_clean] = keyword_scores.get(kw_clean, 0) + normalized_score * 0.3
        
        # Add entity keywords (weight: 0.3)
        for kw, score in entity_keywords:
            kw_clean = self._clean_keyword(kw)
            if kw_clean:
                keyword_scores[kw_clean] = keyword_scores.get(kw_clean, 0) + score * 0.3
        
        # Boost medical domain terms
        for kw in keyword_scores:
            kw_lower = kw.lower()
            if any(term in kw_lower for term in self.medical_boost_terms):
                keyword_scores[kw] *= 1.2
        
        # Boost longer phrases (more specific)
        for kw in keyword_scores:
            word_count = len(kw.split())
            if word_count > 2:
                keyword_scores[kw] *= 1.1
        
        # Sort by score
        ranked_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format output
        result = []
        for kw, score in ranked_keywords[:top_k]:
            # Normalize score to 0-1 range
            max_score = ranked_keywords[0][1] if ranked_keywords else 1.0
            normalized_score = score / max_score if max_score > 0 else 0
            
            result.append({
                'term': kw,
                'score': round(min(normalized_score, 1.0), 2)
            })
        
        return result
    
    def _clean_keyword(self, keyword: str) -> Optional[str]:
        """
        Clean and validate keyword.
        
        Args:
            keyword: Raw keyword
        
        Returns:
            Cleaned keyword or None if invalid
        """
        # Remove excessive whitespace
        keyword = re.sub(r'\s+', ' ', keyword.strip())
        
        # Remove leading/trailing punctuation
        keyword = keyword.strip('.,;:!?-')
        
        # Filter out very short keywords
        if len(keyword) < 3:
            return None
        
        # Filter out keywords that are just numbers
        if keyword.isdigit():
            return None
        
        # Filter out single common words
        stopwords = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'has', 'had'}
        if keyword.lower() in stopwords:
            return None
        
        return keyword
    
    def extract_medical_phrases(self, text: str, min_words: int = 2, max_words: int = 5) -> List[str]:
        """
        Extract medical phrases using pattern matching.
        
        Args:
            text: Input text
            min_words: Minimum words in phrase
            max_words: Maximum words in phrase
        
        Returns:
            List of medical phrases
        """
        # Medical phrase patterns
        patterns = [
            r'\b(?:severe|mild|moderate|acute|chronic)\s+\w+(?:\s+\w+){0,2}\b',
            r'\b\d+\s+(?:sessions?|weeks?|months?|days?)\s+of\s+\w+(?:\s+\w+)?\b',
            r'\b(?:full|complete|partial)\s+recovery\b',
            r'\b\w+\s+(?:pain|injury|treatment|therapy|examination)\b',
        ]
        
        phrases = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrase = match.group().strip()
                word_count = len(phrase.split())
                if min_words <= word_count <= max_words:
                    phrases.append(phrase)
        
        return list(set(phrases))
