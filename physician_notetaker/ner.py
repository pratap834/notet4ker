"""
Named Entity Recognition (NER) module for medical entities.

Extracts medical entities including symptoms, diagnoses, treatments, prognosis, etc.
Uses spaCy and scispaCy models with custom rules and pattern matching.
"""

import spacy
from typing import Dict, List, Optional, Tuple, Any
import re
from collections import defaultdict
from .utils import get_logger, merge_overlapping_spans

logger = get_logger(__name__)


class MedicalNER:
    """Medical Named Entity Recognition system."""
    
    # Entity type definitions
    ENTITY_TYPES = [
        'SYMPTOM',
        'DIAGNOSIS',
        'TREATMENT',
        'PROGNOSIS',
        'EVENT_DATE',
        'INCIDENT',
        'DURATION',
        'SEVERITY',
        'BODY_PART',
        'MEDICATION',
        'PROCEDURE'
    ]
    
    def __init__(self, model_name: str = "en_core_sci_sm"):
        """
        Initialize NER system.
        
        Args:
            model_name: spaCy/scispaCy model name
                       Options: "en_core_sci_sm", "en_ner_bc5cdr_md", etc.
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"Model {model_name} not found. Using en_core_web_sm as fallback.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("No spaCy model available. Please install: python -m spacy download en_core_web_sm")
                raise
        
        # Add custom patterns
        self._add_custom_patterns()
    
    def _add_custom_patterns(self):
        """Add custom entity patterns using spaCy's EntityRuler."""
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.get_pipe("entity_ruler")
        
        # Define patterns for medical entities
        patterns = [
            # Symptoms
            {"label": "SYMPTOM", "pattern": [{"LOWER": "pain"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "headache"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "neck"}, {"LOWER": "pain"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "back"}, {"LOWER": "pain"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "chest"}, {"LOWER": "pain"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "stiff"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "stiffness"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "swelling"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "swollen"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "trouble"}, {"LOWER": "sleeping"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "shortness"}, {"LOWER": "of"}, {"LOWER": "breath"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "sweating"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "nausea"}]},
            
            # Diagnoses
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "whiplash"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "whiplash"}, {"LOWER": "injury"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "pneumonia"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "pleurisy"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "sprain"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "ankle"}, {"LOWER": "sprain"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "fracture"}]},
            
            # Treatments
            {"label": "TREATMENT", "pattern": [{"LOWER": "physiotherapy"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "physical"}, {"LOWER": "therapy"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "painkillers"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "antibiotics"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "rest"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "ice"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "compression"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "elevation"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "rice"}, {"LOWER": "protocol"}]},
            
            # Incidents
            {"label": "INCIDENT", "pattern": [{"LOWER": "car"}, {"LOWER": "accident"}]},
            {"label": "INCIDENT", "pattern": [{"LOWER": "accident"}]},
            {"label": "INCIDENT", "pattern": [{"LOWER": "rear-ended"}]},
            {"label": "INCIDENT", "pattern": [{"LOWER": "rear"}, {"LOWER": "ended"}]},
            {"label": "INCIDENT", "pattern": [{"LOWER": "collision"}]},
            
            # Severity
            {"label": "SEVERITY", "pattern": [{"LOWER": "severe"}]},
            {"label": "SEVERITY", "pattern": [{"LOWER": "mild"}]},
            {"label": "SEVERITY", "pattern": [{"LOWER": "moderate"}]},
            {"label": "SEVERITY", "pattern": [{"LOWER": "acute"}]},
            {"label": "SEVERITY", "pattern": [{"LOWER": "chronic"}]},
        ]
        
        ruler.add_patterns(patterns)
    
    def extract_entities(self, text: str, preprocessed_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract medical entities from text.
        
        Args:
            text: Input text
            preprocessed_data: Optional preprocessed data from MedicalPreprocessor
        
        Returns:
            Dict containing:
                - entities: List of entity dicts
                - entities_by_type: Dict of entities grouped by type
                - confidence: Overall extraction confidence
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        entities = []
        
        # Extract entities from spaCy
        for ent in doc.ents:
            entity_type = self._normalize_entity_type(ent.label_)
            
            if entity_type:
                entities.append({
                    'text': ent.text,
                    'normalized': self._normalize_entity_text(ent.text, entity_type),
                    'type': entity_type,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'confidence': self._calculate_entity_confidence(ent, doc)
                })
        
        # Add custom rule-based extraction
        rule_based_entities = self._extract_rule_based_entities(text)
        entities.extend(rule_based_entities)
        
        # Add entities from preprocessed data if available
        if preprocessed_data:
            if 'dates' in preprocessed_data:
                for date in preprocessed_data['dates']:
                    entities.append({
                        'text': date['raw'],
                        'normalized': date['normalized'],
                        'type': 'EVENT_DATE',
                        'start_char': date['start_char'],
                        'end_char': date['end_char'],
                        'confidence': date['confidence']
                    })
        
        # Merge overlapping entities
        entity_spans = [(e['start_char'], e['end_char'], e['type'], e) for e in entities]
        merged_spans = merge_overlapping_spans(entity_spans)
        entities = [span[3] for span in merged_spans]
        
        # Sort by position
        entities.sort(key=lambda x: x['start_char'])
        
        # Group by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity['type']].append(entity)
        
        # Calculate overall confidence
        confidences = [e['confidence'] for e in entities if e['confidence'] > 0]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return {
            'entities': entities,
            'entities_by_type': dict(entities_by_type),
            'confidence': overall_confidence,
            'num_entities': len(entities)
        }
    
    def _normalize_entity_type(self, label: str) -> Optional[str]:
        """
        Normalize entity labels to our standard types.
        
        Args:
            label: spaCy entity label
        
        Returns:
            Normalized entity type or None
        """
        # Map spaCy/scispaCy labels to our types
        label_mapping = {
            'DISEASE': 'DIAGNOSIS',
            'CHEMICAL': 'MEDICATION',
            'DRUG': 'MEDICATION',
            'SYMPTOM': 'SYMPTOM',
            'TREATMENT': 'TREATMENT',
            'PROCEDURE': 'PROCEDURE',
            'DATE': 'EVENT_DATE',
            'TIME': 'DURATION',
        }
        
        # Check if it's already one of our types
        if label in self.ENTITY_TYPES:
            return label
        
        # Try to map it
        return label_mapping.get(label)
    
    def _normalize_entity_text(self, text: str, entity_type: str) -> str:
        """
        Normalize entity text for consistency.
        
        Args:
            text: Entity text
            entity_type: Entity type
        
        Returns:
            Normalized text
        """
        # Convert to lowercase and strip
        normalized = text.lower().strip()
        
        # Remove articles
        normalized = re.sub(r'\b(a|an|the)\b\s*', '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _calculate_entity_confidence(self, ent, doc) -> float:
        """
        Calculate confidence score for an entity.
        
        Args:
            ent: spaCy entity
            doc: spaCy doc
        
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from model (if available)
        base_confidence = 0.7
        
        # Boost confidence for multi-word entities
        if len(ent.text.split()) > 1:
            base_confidence += 0.1
        
        # Boost confidence if entity has specific medical context
        # Check surrounding tokens
        start_idx = max(0, ent.start - 3)
        end_idx = min(len(doc), ent.end + 3)
        context = doc[start_idx:end_idx].text.lower()
        
        medical_context_keywords = ['patient', 'diagnosed', 'suffering', 'experiencing', 
                                    'treatment', 'prescribed', 'pain', 'injury']
        
        if any(kw in context for kw in medical_context_keywords):
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _extract_rule_based_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using rule-based patterns.
        
        Args:
            text: Input text
        
        Returns:
            List of entity dicts
        """
        entities = []
        
        # Prognosis patterns
        prognosis_patterns = [
            (r'full recovery(?:\s+expected)?(?:\s+within\s+[\w\s]+)?', 'PROGNOSIS'),
            (r'complete recovery', 'PROGNOSIS'),
            (r'should\s+(?:be\s+)?(?:recover|heal)(?:\s+(?:completely|fully|well))?', 'PROGNOSIS'),
            (r'(?:good|excellent|positive)\s+prognosis', 'PROGNOSIS'),
            (r'recovering\s+well', 'PROGNOSIS'),
        ]
        
        for pattern, entity_type in prognosis_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'normalized': match.group().lower(),
                    'type': entity_type,
                    'start_char': match.start(),
                    'end_char': match.end(),
                    'confidence': 0.85
                })
        
        # Duration patterns (enhanced)
        duration_pattern = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty)\s+(day|days|week|weeks|month|months|year|years|session|sessions)'
        matches = re.finditer(duration_pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append({
                'text': match.group(),
                'normalized': match.group().lower(),
                'type': 'DURATION',
                'start_char': match.start(),
                'end_char': match.end(),
                'confidence': 0.9
            })
        
        # Treatment location patterns (A&E, hospital, etc.)
        location_pattern = r'(?:A&E|Emergency Room|ER|hospital)(?:\s+at\s+[\w\s]+)?'
        matches = re.finditer(location_pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append({
                'text': match.group(),
                'normalized': match.group().lower(),
                'type': 'TREATMENT',
                'start_char': match.start(),
                'end_char': match.end(),
                'confidence': 0.8
            })
        
        return entities
    
    def get_entity_context(self, text: str, entity: Dict[str, Any], window: int = 50) -> str:
        """
        Get surrounding context for an entity.
        
        Args:
            text: Full text
            entity: Entity dict
            window: Number of characters before/after
        
        Returns:
            Context string
        """
        start = max(0, entity['start_char'] - window)
        end = min(len(text), entity['end_char'] + window)
        
        context = text[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context
