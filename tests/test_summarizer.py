"""
Tests for medical summarizer module.
"""

import pytest
import json
from physician_notetaker.summarizer import MedicalSummarizer
from physician_notetaker.ner import MedicalNER
from physician_notetaker.preprocess import MedicalPreprocessor
from physician_notetaker.keywords import KeywordExtractor


@pytest.fixture
def components():
    """Create component instances."""
    try:
        ner = MedicalNER(model_name="en_core_sci_sm")
    except:
        ner = MedicalNER(model_name="en_core_web_sm")
    
    return {
        'preprocessor': MedicalPreprocessor(),
        'ner': ner,
        'summarizer': MedicalSummarizer(),
        'keyword_extractor': KeywordExtractor()
    }


@pytest.fixture
def example_transcript():
    """Load example transcript."""
    transcript_path = "data/examples/transcript_01.txt"
    if not os.path.exists(transcript_path):
        pytest.skip("Example transcript not found")
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def expected_output():
    """Load expected output."""
    output_path = "data/examples/expected_output_01.json"
    if not os.path.exists(output_path):
        pytest.skip("Expected output not found")
    
    with open(output_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_summarizer_basic(components):
    """Test basic summarizer functionality."""
    text = "Patient has whiplash injury from car accident. Prescribed physiotherapy."
    
    preprocessed = components['preprocessor'].preprocess(text)
    ner_result = components['ner'].extract_entities(text, preprocessed)
    keywords = components['keyword_extractor'].extract_keywords(text, ner_result['entities'])
    
    summary = components['summarizer'].summarize(
        text,
        entities=ner_result['entities'],
        entities_by_type=ner_result['entities_by_type'],
        preprocessed_data=preprocessed,
        keywords=keywords
    )
    
    # Check required fields
    assert 'Patient_Name' in summary
    assert 'Incident' in summary
    assert 'Symptoms' in summary
    assert 'Diagnosis' in summary
    assert 'Treatment' in summary
    assert 'confidence' in summary


def test_summarizer_confidence(components):
    """Test that confidence scores are in valid range."""
    text = "Patient diagnosed with whiplash. Treatment includes physiotherapy."
    
    preprocessed = components['preprocessor'].preprocess(text)
    ner_result = components['ner'].extract_entities(text, preprocessed)
    keywords = components['keyword_extractor'].extract_keywords(text, ner_result['entities'])
    
    summary = components['summarizer'].summarize(
        text,
        entities=ner_result['entities'],
        entities_by_type=ner_result['entities_by_type'],
        preprocessed_data=preprocessed,
        keywords=keywords
    )
    
    assert 0 <= summary['confidence'] <= 1


def test_summarizer_null_handling(components):
    """Test handling of missing information."""
    text = "Patient has pain."  # Minimal information
    
    preprocessed = components['preprocessor'].preprocess(text)
    ner_result = components['ner'].extract_entities(text, preprocessed)
    keywords = components['keyword_extractor'].extract_keywords(text, ner_result['entities'])
    
    summary = components['summarizer'].summarize(
        text,
        entities=ner_result['entities'],
        entities_by_type=ner_result['entities_by_type'],
        preprocessed_data=preprocessed,
        keywords=keywords
    )
    
    # Should have nulls for missing information
    assert summary['Patient_Name'] is None or isinstance(summary['Patient_Name'], str)


def test_summarizer_incident_extraction(components):
    """Test incident information extraction."""
    text = "Patient had a car accident on September 1st, 2024 in Manchester."
    
    preprocessed = components['preprocessor'].preprocess(text)
    ner_result = components['ner'].extract_entities(text, preprocessed)
    keywords = components['keyword_extractor'].extract_keywords(text, ner_result['entities'])
    
    summary = components['summarizer'].summarize(
        text,
        entities=ner_result['entities'],
        entities_by_type=ner_result['entities_by_type'],
        preprocessed_data=preprocessed,
        keywords=keywords
    )
    
    assert isinstance(summary['Incident'], dict)
    assert 'type' in summary['Incident']
    assert 'date' in summary['Incident']
    assert 'location' in summary['Incident']


def test_summarizer_symptoms(components):
    """Test symptoms extraction."""
    text = "Patient reports neck pain, headache, and back pain."
    
    preprocessed = components['preprocessor'].preprocess(text)
    ner_result = components['ner'].extract_entities(text, preprocessed)
    keywords = components['keyword_extractor'].extract_keywords(text, ner_result['entities'])
    
    summary = components['summarizer'].summarize(
        text,
        entities=ner_result['entities'],
        entities_by_type=ner_result['entities_by_type'],
        preprocessed_data=preprocessed,
        keywords=keywords
    )
    
    assert isinstance(summary['Symptoms'], list)


def test_summarizer_diagnosis(components):
    """Test diagnosis extraction."""
    text = "Patient diagnosed with whiplash injury."
    
    preprocessed = components['preprocessor'].preprocess(text)
    ner_result = components['ner'].extract_entities(text, preprocessed)
    keywords = components['keyword_extractor'].extract_keywords(text, ner_result['entities'])
    
    summary = components['summarizer'].summarize(
        text,
        entities=ner_result['entities'],
        entities_by_type=ner_result['entities_by_type'],
        preprocessed_data=preprocessed,
        keywords=keywords
    )
    
    assert isinstance(summary['Diagnosis'], list)


def test_summarizer_treatment(components):
    """Test treatment extraction."""
    text = "Prescribed painkillers and physiotherapy sessions."
    
    preprocessed = components['preprocessor'].preprocess(text)
    ner_result = components['ner'].extract_entities(text, preprocessed)
    keywords = components['keyword_extractor'].extract_keywords(text, ner_result['entities'])
    
    summary = components['summarizer'].summarize(
        text,
        entities=ner_result['entities'],
        entities_by_type=ner_result['entities_by_type'],
        preprocessed_data=preprocessed,
        keywords=keywords
    )
    
    assert isinstance(summary['Treatment'], list)


def test_summarizer_prognosis(components):
    """Test prognosis extraction."""
    text = "Full recovery expected within six months."
    
    preprocessed = components['preprocessor'].preprocess(text)
    ner_result = components['ner'].extract_entities(text, preprocessed)
    keywords = components['keyword_extractor'].extract_keywords(text, ner_result['entities'])
    
    summary = components['summarizer'].summarize(
        text,
        entities=ner_result['entities'],
        entities_by_type=ner_result['entities_by_type'],
        preprocessed_data=preprocessed,
        keywords=keywords
    )
    
    # Prognosis can be string or None
    assert summary['Prognosis'] is None or isinstance(summary['Prognosis'], str)


def test_summarizer_full_transcript(components, example_transcript):
    """Test summarizer on full example transcript."""
    preprocessed = components['preprocessor'].preprocess(example_transcript)
    ner_result = components['ner'].extract_entities(example_transcript, preprocessed)
    keywords = components['keyword_extractor'].extract_keywords(
        example_transcript, 
        ner_result['entities']
    )
    
    summary = components['summarizer'].summarize(
        example_transcript,
        entities=ner_result['entities'],
        entities_by_type=ner_result['entities_by_type'],
        preprocessed_data=preprocessed,
        keywords=keywords
    )
    
    # Check major fields are populated
    assert summary['confidence'] > 0.5
    assert len(summary['Symptoms']) > 0 or len(summary['Diagnosis']) > 0


import os
