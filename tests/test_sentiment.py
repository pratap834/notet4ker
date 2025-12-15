"""
Tests for sentiment and intent classifier.
"""

import pytest
from physician_notetaker.sentiment import SentimentIntentClassifier


@pytest.fixture
def classifier():
    """Create classifier instance."""
    return SentimentIntentClassifier()


def test_sentiment_anxious(classifier):
    """Test detection of anxious sentiment."""
    text = "I'm really worried about this. Is it serious? Should I be concerned?"
    result = classifier.classify_utterance(text, speaker="Patient")
    
    assert result['Sentiment'] in ['Anxious', 'Neutral', 'Reassured']
    assert -1 <= result['Sentiment_Score'] <= 1
    assert 0 <= result['confidence'] <= 1


def test_sentiment_reassured(classifier):
    """Test detection of reassured sentiment."""
    text = "Thank you, doctor. That's really reassuring. I feel much better now."
    result = classifier.classify_utterance(text, speaker="Patient")
    
    assert result['Sentiment'] in ['Anxious', 'Neutral', 'Reassured']
    assert 0 <= result['confidence'] <= 1


def test_sentiment_neutral(classifier):
    """Test detection of neutral sentiment."""
    text = "I see. Okay. Yes, I understand."
    result = classifier.classify_utterance(text, speaker="Patient")
    
    assert result['Sentiment'] in ['Anxious', 'Neutral', 'Reassured']


def test_intent_seeking_reassurance(classifier):
    """Test intent classification for seeking reassurance."""
    text = "Will I recover? How long will it take?"
    result = classifier.classify_utterance(text, speaker="Patient")
    
    assert result['Intent'] in classifier.INTENT_LABELS


def test_intent_reporting_symptoms(classifier):
    """Test intent classification for reporting symptoms."""
    text = "I have severe neck pain and headaches."
    result = classifier.classify_utterance(text, speaker="Patient")
    
    assert result['Intent'] in classifier.INTENT_LABELS


def test_intent_expressing_concern(classifier):
    """Test intent classification for expressing concern."""
    text = "I'm worried that it might be something serious."
    result = classifier.classify_utterance(text, speaker="Patient")
    
    assert result['Intent'] in classifier.INTENT_LABELS


def test_physician_utterance(classifier):
    """Test that physician utterances return N/A."""
    text = "Your diagnosis is whiplash injury."
    result = classifier.classify_utterance(text, speaker="Physician")
    
    assert result['Sentiment'] == 'N/A'
    assert result['Intent'] == 'N/A'


def test_transcript_classification(classifier):
    """Test classification of full transcript."""
    speaker_turns = [
        {'speaker': 'Patient', 'text': 'I have severe pain.'},
        {'speaker': 'Physician', 'text': 'Let me examine you.'},
        {'speaker': 'Patient', 'text': 'Will I recover?'},
        {'speaker': 'Physician', 'text': 'Yes, full recovery expected.'},
        {'speaker': 'Patient', 'text': 'Thank you, that is reassuring.'}
    ]
    
    result = classifier.classify_transcript("", speaker_turns=speaker_turns)
    
    assert 'overall_sentiment' in result
    assert 'overall_intent' in result
    assert 'overall_confidence' in result
    assert 'turn_classifications' in result
    
    assert result['overall_sentiment'] in ['Anxious', 'Neutral', 'Reassured']
    assert 0 <= result['overall_confidence'] <= 1
    
    # Should have classifications for patient turns only
    patient_turns = [t for t in speaker_turns if t['speaker'] == 'Patient']
    assert len(result['turn_classifications']) == len(patient_turns)


def test_confidence_calculation(classifier):
    """Test confidence score calculation."""
    # Longer, more explicit text should have higher confidence
    text = "I am very worried about this severe pain. It's been bothering me for days."
    result = classifier.classify_utterance(text, speaker="Patient")
    
    assert result['confidence'] > 0.5


def test_empty_text(classifier):
    """Test handling of empty text."""
    result = classifier.classify_utterance("", speaker="Patient")
    
    assert 'Sentiment' in result
    assert 'Intent' in result


def test_sentiment_score_range(classifier):
    """Test that sentiment scores are in valid range."""
    texts = [
        "I'm terrified and very worried.",
        "I feel okay about this.",
        "Thank you so much, I feel great!"
    ]
    
    for text in texts:
        result = classifier.classify_utterance(text, speaker="Patient")
        assert -1 <= result['Sentiment_Score'] <= 1
