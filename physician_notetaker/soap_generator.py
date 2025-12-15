"""
SOAP note generator module.

Generates structured SOAP (Subjective, Objective, Assessment, Plan) notes
from medical transcripts using hybrid rule-based and ML approach.
"""

import re
from typing import Dict, List, Optional, Any
from .utils import get_logger, calculate_confidence

logger = get_logger(__name__)


class SOAPGenerator:
    """Generate SOAP notes from medical transcripts."""
    
    def __init__(self):
        """Initialize SOAP generator."""
        # Keywords for each SOAP section
        self.section_keywords = {
            'subjective': [
                'patient', 'complaint', 'reports', 'states', 'describes',
                'history', 'symptoms', 'pain', 'discomfort', 'feels',
                'experiencing', 'started', 'ago', 'since'
            ],
            'objective': [
                'examination', 'exam', 'vital signs', 'physical', 'inspection',
                'palpation', 'auscultation', 'observed', 'noted', 'findings',
                'blood pressure', 'heart rate', 'temperature', 'range of motion',
                'tenderness', 'swelling'
            ],
            'assessment': [
                'diagnosis', 'diagnosed', 'assessment', 'impression',
                'condition', 'appears', 'consistent with', 'likely',
                'differential', 'prognosis'
            ],
            'plan': [
                'treatment', 'plan', 'prescribe', 'recommend', 'advise',
                'follow-up', 'return', 'continue', 'start', 'medication',
                'therapy', 'referral', 'imaging', 'tests', 'instructions'
            ]
        }
    
    def generate_soap(
        self,
        text: str,
        speaker_turns: Optional[List[Dict[str, str]]] = None,
        entities_by_type: Optional[Dict[str, List[Dict]]] = None,
        medical_summary: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate SOAP note from transcript.
        
        Args:
            text: Full transcript text
            speaker_turns: List of speaker turns
            entities_by_type: Entities grouped by type
            medical_summary: Medical summary dict
        
        Returns:
            SOAP note dict with S, O, A, P sections and confidence
        """
        # Method 1: Sentence classification
        sentences = self._split_into_sentences(text)
        sentence_classifications = self._classify_sentences(sentences)
        
        # Method 2: Speaker-based classification (patient = S, doctor observations = O, etc.)
        if speaker_turns:
            speaker_based_sections = self._classify_by_speaker(speaker_turns)
        else:
            speaker_based_sections = None
        
        # Method 3: Use medical summary if available
        summary_based_sections = None
        if medical_summary:
            summary_based_sections = self._extract_from_summary(medical_summary)
        
        # Combine approaches
        soap_note = self._combine_soap_sections(
            sentence_classifications,
            speaker_based_sections,
            summary_based_sections
        )
        
        # Calculate confidence
        confidence = self._calculate_soap_confidence(soap_note, text)
        soap_note['confidence'] = confidence
        
        return soap_note
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _classify_sentences(self, sentences: List[str]) -> Dict[str, List[str]]:
        """
        Classify sentences into SOAP sections.
        
        Args:
            sentences: List of sentences
        
        Returns:
            Dict mapping section to sentences
        """
        classifications = {
            'Subjective': [],
            'Objective': [],
            'Assessment': [],
            'Plan': []
        }
        
        for sentence in sentences:
            section = self._classify_sentence(sentence)
            classifications[section].append(sentence)
        
        return classifications
    
    def _classify_sentence(self, sentence: str) -> str:
        """
        Classify a single sentence into SOAP section.
        
        Args:
            sentence: Input sentence
        
        Returns:
            SOAP section name
        """
        sentence_lower = sentence.lower()
        
        # Score each section
        scores = {}
        for section, keywords in self.section_keywords.items():
            score = sum(1 for kw in keywords if kw in sentence_lower)
            scores[section] = score
        
        # Additional heuristics
        
        # First-person indicators suggest Subjective
        if re.search(r'\b(I|my|me)\b', sentence):
            scores['subjective'] += 2
        
        # Past tense medical actions suggest Subjective (history)
        if re.search(r'(was|had|went|started|began)', sentence_lower):
            scores['subjective'] += 1
        
        # Examination phrases suggest Objective
        if re.search(r'(let me|can you|turn|move|press|listen)', sentence_lower):
            scores['objective'] += 2
        
        # Modal verbs + medical terms suggest Plan
        if re.search(r'(should|will|would|can|need to|going to)', sentence_lower):
            scores['plan'] += 1
        
        # Diagnostic statements suggest Assessment
        if re.search(r'(diagnosis|diagnosed|you have|appears to be|consistent with)', sentence_lower):
            scores['assessment'] += 2
        
        # Find section with highest score
        if max(scores.values()) > 0:
            # Map to proper case
            section_map = {
                'subjective': 'Subjective',
                'objective': 'Objective',
                'assessment': 'Assessment',
                'plan': 'Plan'
            }
            best_section = max(scores.items(), key=lambda x: x[1])[0]
            return section_map[best_section]
        
        # Default to Subjective if unsure
        return 'Subjective'
    
    def _classify_by_speaker(self, speaker_turns: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Classify content by speaker.
        
        Patient turns → Subjective
        Physician questions/observations → Objective
        Physician diagnoses → Assessment
        Physician advice → Plan
        
        Args:
            speaker_turns: List of speaker turn dicts
        
        Returns:
            Dict mapping section to content
        """
        sections = {
            'Subjective': [],
            'Objective': [],
            'Assessment': [],
            'Plan': []
        }
        
        for turn in speaker_turns:
            speaker = turn['speaker']
            text = turn['text']
            text_lower = text.lower()
            
            if speaker == 'Patient':
                # Patient statements are Subjective
                sections['Subjective'].append(text)
            else:
                # Physician - classify based on content
                if any(kw in text_lower for kw in ['examination', 'exam', 'let me', 'can you', 'listen', 'press']):
                    sections['Objective'].append(text)
                elif any(kw in text_lower for kw in ['diagnosis', 'you have', 'appears', 'consistent']):
                    sections['Assessment'].append(text)
                elif any(kw in text_lower for kw in ['recommend', 'prescribe', 'should', 'continue', 'follow-up']):
                    sections['Plan'].append(text)
                else:
                    # Default physician statements to Objective
                    sections['Objective'].append(text)
        
        return sections
    
    def _extract_from_summary(self, medical_summary: Dict) -> Dict[str, List[str]]:
        """
        Extract SOAP sections from medical summary.
        
        Args:
            medical_summary: Medical summary dict
        
        Returns:
            Dict mapping section to content
        """
        sections = {
            'Subjective': [],
            'Objective': [],
            'Assessment': [],
            'Plan': []
        }
        
        # Subjective: Symptoms, Incident
        if medical_summary.get('Symptoms'):
            sections['Subjective'].append(f"Patient reports: {', '.join(medical_summary['Symptoms'])}")
        
        if medical_summary.get('Incident'):
            incident = medical_summary['Incident']
            if incident.get('type'):
                incident_str = f"Incident: {incident['type']}"
                if incident.get('date'):
                    incident_str += f" on {incident['date']}"
                if incident.get('location'):
                    incident_str += f" at {incident['location']}"
                sections['Subjective'].append(incident_str)
        
        # Objective: Current Status, Examination findings
        if medical_summary.get('Current_Status'):
            sections['Objective'].append(f"Examination: {medical_summary['Current_Status']}")
        
        # Assessment: Diagnosis, Prognosis
        if medical_summary.get('Diagnosis'):
            sections['Assessment'].append(f"Diagnosis: {', '.join(medical_summary['Diagnosis'])}")
        
        if medical_summary.get('Prognosis'):
            sections['Assessment'].append(f"Prognosis: {medical_summary['Prognosis']}")
        
        # Plan: Treatment, Follow-up
        if medical_summary.get('Treatment'):
            sections['Plan'].append(f"Treatment: {', '.join(medical_summary['Treatment'])}")
        
        if medical_summary.get('Follow_Up_Advice'):
            sections['Plan'].append(f"Follow-up: {medical_summary['Follow_Up_Advice']}")
        
        return sections
    
    def _combine_soap_sections(
        self,
        sentence_based: Dict[str, List[str]],
        speaker_based: Optional[Dict[str, List[str]]],
        summary_based: Optional[Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Combine SOAP sections from different approaches.
        
        Args:
            sentence_based: Sentence classification results
            speaker_based: Speaker-based classification results
            summary_based: Summary-based extraction results
        
        Returns:
            Combined SOAP note dict
        """
        soap_note = {}
        
        for section in ['Subjective', 'Objective', 'Assessment', 'Plan']:
            content_parts = []
            
            # Prioritize summary-based (most structured)
            if summary_based and summary_based.get(section):
                content_parts.extend(summary_based[section])
            
            # Add speaker-based
            if speaker_based and speaker_based.get(section):
                for item in speaker_based[section]:
                    if item not in content_parts:
                        content_parts.append(item)
            
            # Add sentence-based (fill gaps)
            if sentence_based and sentence_based.get(section):
                for item in sentence_based[section]:
                    if item not in content_parts and len(content_parts) < 5:
                        content_parts.append(item)
            
            # Combine content
            if content_parts:
                combined_content = " ".join(content_parts)
            else:
                combined_content = "Not documented"
            
            soap_note[section] = {
                'content': combined_content,
                'details': content_parts if content_parts else []
            }
        
        return soap_note
    
    def _calculate_soap_confidence(self, soap_note: Dict, original_text: str) -> float:
        """
        Calculate confidence in SOAP note generation.
        
        Args:
            soap_note: Generated SOAP note
            original_text: Original transcript text
        
        Returns:
            Confidence score
        """
        confidences = []
        
        for section, data in soap_note.items():
            if section == 'confidence':
                continue
            
            content = data.get('content', '')
            
            # Check if section has meaningful content
            if content and content != "Not documented":
                # Base confidence
                section_confidence = 0.6
                
                # Boost if content is substantial
                word_count = len(content.split())
                if word_count > 20:
                    section_confidence += 0.2
                elif word_count > 10:
                    section_confidence += 0.1
                
                # Boost if content contains expected keywords
                expected_keywords = {
                    'Subjective': ['patient', 'reports', 'pain', 'symptoms'],
                    'Objective': ['examination', 'exam', 'findings', 'observed'],
                    'Assessment': ['diagnosis', 'prognosis', 'condition'],
                    'Plan': ['treatment', 'recommend', 'follow-up', 'prescribe']
                }
                
                content_lower = content.lower()
                keyword_matches = sum(1 for kw in expected_keywords.get(section, []) if kw in content_lower)
                if keyword_matches > 0:
                    section_confidence += min(keyword_matches * 0.05, 0.15)
                
                confidences.append(min(section_confidence, 0.95))
            else:
                confidences.append(0.3)
        
        return round(calculate_confidence(confidences), 2) if confidences else 0.5
    
    def format_soap_note(self, soap_note: Dict[str, Any]) -> str:
        """
        Format SOAP note as readable text.
        
        Args:
            soap_note: SOAP note dict
        
        Returns:
            Formatted SOAP note string
        """
        formatted = []
        
        for section in ['Subjective', 'Objective', 'Assessment', 'Plan']:
            if section in soap_note:
                formatted.append(f"{section.upper()}:")
                content = soap_note[section].get('content', 'Not documented')
                formatted.append(f"  {content}")
                formatted.append("")
        
        if 'confidence' in soap_note:
            formatted.append(f"Confidence: {soap_note['confidence']}")
        
        return "\n".join(formatted)
