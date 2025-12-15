"""
Sentiment and intent classification module.

Classifies patient sentiment (Anxious, Neutral, Reassured) and intent
(Seeking reassurance, Reporting symptoms, Expressing concern, Requesting follow-up, Other).
"""

import re
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from .utils import get_logger

logger = get_logger(__name__)


class SentimentIntentClassifier:
    """Classify sentiment and intent of patient utterances."""
    
    # Sentiment and intent labels
    SENTIMENT_LABELS = ["Anxious", "Neutral", "Reassured"]
    INTENT_LABELS = [
        "Seeking reassurance",
        "Reporting symptoms",
        "Expressing concern",
        "Requesting follow-up",
        "Other"
    ]
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize sentiment and intent classifier.
        
        Args:
            model_name: Transformer model name
                       For production: use fine-tuned ClinicalBERT or BioBERT
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # In production, load fine-tuned medical sentiment model
            # For now, use general sentiment model as baseline
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Note: This would be replaced with fine-tuned model
            logger.info(f"Loaded tokenizer: {model_name}")
            logger.warning("Using rule-based sentiment/intent. For production, fine-tune a medical model.")
            self.use_model = False
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using rule-based approach.")
            self.use_model = False
    
    def classify_utterance(self, utterance: str, speaker: str = "Patient") -> Dict[str, any]:
        """
        Classify a single utterance for sentiment and intent.
        
        Args:
            utterance: Text to classify
            speaker: Speaker identifier (Patient or Physician)
        
        Returns:
            Dict with sentiment, sentiment_score, intent, and confidence
        """
        if speaker != "Patient":
            # Only classify patient utterances
            return {
                "Sentiment": "N/A",
                "Sentiment_Score": 0.0,
                "Intent": "N/A",
                "confidence": 1.0
            }
        
        # Use rule-based approach (would be replaced with fine-tuned model)
        sentiment, sentiment_score = self._rule_based_sentiment(utterance)
        intent = self._rule_based_intent(utterance)
        
        # Confidence based on keyword matches
        confidence = self._calculate_classification_confidence(utterance, sentiment, intent)
        
        return {
            "Sentiment": sentiment,
            "Sentiment_Score": sentiment_score,
            "Intent": intent,
            "confidence": round(confidence, 2)
        }
    
    def classify_transcript(
        self,
        text: str,
        speaker_turns: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Classify sentiment and intent for entire transcript.
        
        Args:
            text: Full transcript text
            speaker_turns: List of speaker turn dicts
        
        Returns:
            Dict with overall and per-turn classifications
        """
        if speaker_turns:
            # Classify each patient turn
            turn_classifications = []
            for turn in speaker_turns:
                if turn['speaker'] == 'Patient':
                    classification = self.classify_utterance(turn['text'], 'Patient')
                    classification['text'] = turn['text'][:100] + "..." if len(turn['text']) > 100 else turn['text']
                    turn_classifications.append(classification)
            
            # Aggregate overall sentiment
            if turn_classifications:
                sentiments = [t['Sentiment'] for t in turn_classifications]
                sentiment_scores = [t['Sentiment_Score'] for t in turn_classifications]
                
                overall_sentiment = self._aggregate_sentiment(sentiments, sentiment_scores)
                overall_score = np.mean(sentiment_scores)
                
                # Most common intent
                intents = [t['Intent'] for t in turn_classifications]
                overall_intent = max(set(intents), key=intents.count)
                
                confidences = [t['confidence'] for t in turn_classifications]
                overall_confidence = np.mean(confidences)
            else:
                overall_sentiment = "Neutral"
                overall_score = 0.0
                overall_intent = "Other"
                overall_confidence = 0.5
                turn_classifications = []
        else:
            # Classify entire text as one utterance
            overall_classification = self.classify_utterance(text, "Patient")
            overall_sentiment = overall_classification['Sentiment']
            overall_score = overall_classification['Sentiment_Score']
            overall_intent = overall_classification['Intent']
            overall_confidence = overall_classification['confidence']
            turn_classifications = [overall_classification]
        
        return {
            "overall_sentiment": overall_sentiment,
            "overall_sentiment_score": round(overall_score, 2),
            "overall_intent": overall_intent,
            "overall_confidence": round(overall_confidence, 2),
            "turn_classifications": turn_classifications
        }
    
    def _rule_based_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Rule-based sentiment classification.
        
        Returns:
            (sentiment_label, sentiment_score) where score is -1 to 1
        """
        text_lower = text.lower()
        
        # Anxious indicators
        anxious_keywords = [
            'worried', 'concerned', 'anxious', 'scared', 'afraid', 'nervous',
            'serious', 'bad', 'worse', 'terrible', 'severe', 'really hurts',
            'is that serious', 'should i be worried', 'how long'
        ]
        
        # Reassured indicators
        reassured_keywords = [
            'better', 'improved', 'reassuring', 'good', 'thank you',
            'that helps', 'feel better', 'glad', 'relieved', 'okay'
        ]
        
        # Neutral indicators
        neutral_keywords = [
            'yes', 'no', 'okay', 'i see', 'understand', 'fine'
        ]
        
        # Count keyword matches
        anxious_score = sum(1 for kw in anxious_keywords if kw in text_lower)
        reassured_score = sum(1 for kw in reassured_keywords if kw in text_lower)
        
        # Check for question marks (often indicate concern)
        if '?' in text:
            anxious_score += 1
        
        # Determine sentiment
        if anxious_score > reassured_score:
            sentiment = "Anxious"
            score = -min(anxious_score * 0.3, 1.0)  # Negative score
        elif reassured_score > anxious_score:
            sentiment = "Reassured"
            score = min(reassured_score * 0.3, 1.0)  # Positive score
        else:
            sentiment = "Neutral"
            score = 0.0
        
        return sentiment, score
    
    def _rule_based_intent(self, text: str) -> str:
        """
        Rule-based intent classification.
        
        Returns:
            Intent label
        """
        text_lower = text.lower()
        
        # Intent patterns
        intent_patterns = {
            "Seeking reassurance": [
                r'will i\s+\w+',
                r'do you think',
                r'is (?:it|that) (?:serious|bad|normal)',
                r'should i (?:be )?worried',
                r'how long',
                r'when (?:can|will)',
            ],
            "Reporting symptoms": [
                r'i have',
                r'i\'ve been (?:having|experiencing)',
                r'my \w+ (?:hurts|is|feels)',
                r'(?:pain|ache|swelling|stiff)',
                r'it (?:hurts|aches)',
            ],
            "Expressing concern": [
                r'worried about',
                r'concerned that',
                r'afraid',
                r'is (?:it|that) serious',
                r'what if',
            ],
            "Requesting follow-up": [
                r'what should i do',
                r'when should i',
                r'do i need to',
                r'should i (?:come back|return|see you)',
                r'follow(?:-| )up',
            ],
        }
        
        # Check each intent
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        
        return "Other"
    
    def _aggregate_sentiment(
        self,
        sentiments: List[str],
        scores: List[float]
    ) -> str:
        """
        Aggregate multiple sentiment classifications.
        
        Args:
            sentiments: List of sentiment labels
            scores: List of sentiment scores
        
        Returns:
            Overall sentiment label
        """
        # Weight by absolute score magnitude
        weighted_sentiments = {}
        for sentiment, score in zip(sentiments, scores):
            weight = abs(score)
            weighted_sentiments[sentiment] = weighted_sentiments.get(sentiment, 0) + weight
        
        if not weighted_sentiments:
            return "Neutral"
        
        # Return sentiment with highest weighted count
        return max(weighted_sentiments.items(), key=lambda x: x[1])[0]
    
    def _calculate_classification_confidence(
        self,
        text: str,
        sentiment: str,
        intent: str
    ) -> float:
        """
        Calculate confidence in classification.
        
        Args:
            text: Input text
            sentiment: Predicted sentiment
            intent: Predicted intent
        
        Returns:
            Confidence score
        """
        # Base confidence
        confidence = 0.6
        
        # Boost confidence if text is longer (more context)
        word_count = len(text.split())
        if word_count > 20:
            confidence += 0.15
        elif word_count > 10:
            confidence += 0.1
        
        # Boost confidence if strong keyword matches
        text_lower = text.lower()
        strong_keywords = {
            "Anxious": ['worried', 'concerned', 'serious'],
            "Reassured": ['better', 'thank you', 'reassuring'],
        }
        
        if sentiment in strong_keywords:
            matches = sum(1 for kw in strong_keywords[sentiment] if kw in text_lower)
            if matches > 0:
                confidence += 0.1
        
        return min(confidence, 0.95)
    
    def fine_tune_instructions(self) -> str:
        """
        Return instructions for fine-tuning on medical data.
        
        Returns:
            Fine-tuning instructions
        """
        instructions = """
        Fine-tuning Instructions for Medical Sentiment & Intent Classification:
        
        1. Dataset Preparation:
           - Collect 500-2000 annotated physician-patient transcripts
           - Label each patient utterance with:
             * Sentiment: Anxious, Neutral, Reassured
             * Intent: Seeking reassurance, Reporting symptoms, Expressing concern, 
                      Requesting follow-up, Other
           - Use BIO or multi-label format
        
        2. Model Selection:
           - Base model: ClinicalBERT, BioBERT, or BioClinicalBERT
           - These models are pre-trained on medical text
        
        3. Fine-tuning:
           - Task: Multi-task classification (sentiment + intent)
           - Loss: CrossEntropyLoss for each task
           - Optimizer: AdamW with learning rate 2e-5
           - Epochs: 3-5
           - Batch size: 16-32
           - Validation split: 20%
        
        4. Evaluation Metrics:
           - Accuracy, Precision, Recall, F1 per class
           - Macro-averaged F1 for overall performance
           - Confusion matrix to identify misclassifications
        
        5. Data Augmentation (optional):
           - Use paraphrasing to increase dataset size
           - Back-translation
           - Synonym replacement with medical terminology preserved
        
        6. Sample code structure:
           ```python
           from transformers import AutoModelForSequenceClassification, Trainer
           
           model = AutoModelForSequenceClassification.from_pretrained(
               "emilyalsentzer/Bio_ClinicalBERT",
               num_labels=len(LABELS)
           )
           
           trainer = Trainer(
               model=model,
               args=training_args,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               compute_metrics=compute_metrics
           )
           
           trainer.train()
           ```
        
        7. Deployment:
           - Save fine-tuned model to models/ directory
           - Update __init__ to load fine-tuned weights
           - Test on held-out transcripts
        """
        return instructions
