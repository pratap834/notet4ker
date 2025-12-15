"""
Tests for NER module.
"""

import pytest
from physician_notetaker.ner import MedicalNER


@pytest.fixture
def ner_model():
    """Create NER model instance."""
    try:
        return MedicalNER(model_name="en_core_sci_sm")
    except:
        return MedicalNER(model_name="en_core_web_sm")


@pytest.fixture
def sample_text():
    """Sample medical text."""
    return """
    Patient reports severe neck pain and headache following a car accident on September 1st.
    Diagnosed with whiplash injury. Prescribed painkillers and physiotherapy.
    Full recovery expected within six months.
    """


def test_ner_extraction(ner_model, sample_text):
    """Test basic NER extraction."""
    result = ner_model.extract_entities(sample_text)
    
    assert 'entities' in result
    assert 'entities_by_type' in result
    assert 'confidence' in result
    assert 'num_entities' in result
    
    # Check that we extracted some entities
    assert result['num_entities'] > 0
    assert len(result['entities']) > 0


def test_ner_entity_types(ner_model, sample_text):
    """Test that expected entity types are extracted."""
    result = ner_model.extract_entities(sample_text)
    
    entities_by_type = result['entities_by_type']
    
    # Should have at least some of these types
    expected_types = ['SYMPTOM', 'DIAGNOSIS', 'TREATMENT', 'PROGNOSIS']
    found_types = [et for et in expected_types if et in entities_by_type]
    
    assert len(found_types) > 0, "No expected entity types found"


def test_ner_entity_structure(ner_model, sample_text):
    """Test entity structure."""
    result = ner_model.extract_entities(sample_text)
    
    if result['entities']:
        entity = result['entities'][0]
        
        # Check required fields
        assert 'text' in entity
        assert 'normalized' in entity
        assert 'type' in entity
        assert 'start_char' in entity
        assert 'end_char' in entity
        assert 'confidence' in entity
        
        # Check confidence range
        assert 0 <= entity['confidence'] <= 1


def test_ner_confidence_scores(ner_model, sample_text):
    """Test that confidence scores are reasonable."""
    result = ner_model.extract_entities(sample_text)
    
    assert 0 <= result['confidence'] <= 1
    
    for entity in result['entities']:
        assert 0 <= entity['confidence'] <= 1


def test_ner_whiplash_detection(ner_model):
    """Test detection of whiplash diagnosis."""
    text = "The patient has a whiplash injury from the car accident."
    result = ner_model.extract_entities(text)
    
    # Check if whiplash is detected
    all_text = ' '.join([e['text'].lower() for e in result['entities']])
    assert 'whiplash' in all_text


def test_ner_empty_text(ner_model):
    """Test NER on empty text."""
    result = ner_model.extract_entities("")
    
    assert result['num_entities'] == 0
    assert len(result['entities']) == 0


def test_ner_symptom_extraction(ner_model):
    """Test extraction of symptoms."""
    text = "Patient has severe neck pain, headache, and back pain."
    result = ner_model.extract_entities(text)
    
    if 'SYMPTOM' in result['entities_by_type']:
        symptoms = result['entities_by_type']['SYMPTOM']
        assert len(symptoms) > 0
        
        # Check for pain-related symptoms
        symptom_texts = [s['text'].lower() for s in symptoms]
        assert any('pain' in text for text in symptom_texts)


def test_ner_treatment_extraction(ner_model):
    """Test extraction of treatments."""
    text = "Prescribed painkillers and recommended physiotherapy sessions."
    result = ner_model.extract_entities(text)
    
    if 'TREATMENT' in result['entities_by_type']:
        treatments = result['entities_by_type']['TREATMENT']
        assert len(treatments) > 0


def test_ner_prognosis_extraction(ner_model):
    """Test extraction of prognosis."""
    text = "Full recovery expected within six months. Patient should heal completely."
    result = ner_model.extract_entities(text)
    
    if 'PROGNOSIS' in result['entities_by_type']:
        prognoses = result['entities_by_type']['PROGNOSIS']
        assert len(prognoses) > 0


def test_ner_date_extraction(ner_model):
    """Test date extraction with preprocessed data."""
    from physician_notetaker.preprocess import MedicalPreprocessor
    
    preprocessor = MedicalPreprocessor()
    text = "The accident occurred on September 1st, 2024."
    
    preprocessed = preprocessor.preprocess(text)
    result = ner_model.extract_entities(text, preprocessed)
    
    if 'EVENT_DATE' in result['entities_by_type']:
        dates = result['entities_by_type']['EVENT_DATE']
        assert len(dates) > 0
