"""
Text preprocessing module for medical transcripts.

Handles cleaning, normalization, date parsing, and text structure extraction.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import dateparser
from .utils import get_logger, normalize_whitespace, extract_speaker_turns

logger = get_logger(__name__)


class MedicalPreprocessor:
    """Preprocessor for medical transcripts."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}\b',
            r'\bSeptember\s+1st\b',
        ]
        
    def preprocess(self, text: str) -> Dict:
        """
        Main preprocessing pipeline.
        
        Args:
            text: Raw transcript text
        
        Returns:
            Dict containing:
                - cleaned_text: Cleaned full text
                - speaker_turns: List of speaker turns
                - dates: Extracted dates
                - metadata: Additional metadata
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Extract speaker turns
        speaker_turns = extract_speaker_turns(cleaned_text)
        
        # Extract dates
        dates = self._extract_dates(cleaned_text)
        
        # Extract patient vs physician text
        patient_text = " ".join([turn['text'] for turn in speaker_turns if turn['speaker'] == 'Patient'])
        physician_text = " ".join([turn['text'] for turn in speaker_turns if turn['speaker'] == 'Physician'])
        
        return {
            'cleaned_text': cleaned_text,
            'speaker_turns': speaker_turns,
            'dates': dates,
            'patient_text': patient_text,
            'physician_text': physician_text,
            'metadata': {
                'num_turns': len(speaker_turns),
                'num_patient_turns': sum(1 for t in speaker_turns if t['speaker'] == 'Patient'),
                'num_physician_turns': sum(1 for t in speaker_turns if t['speaker'] == 'Physician'),
            }
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = normalize_whitespace(text)
        
        # Remove special characters that might interfere with NER
        # But preserve medical notation (e.g., "10mg", "Grade 2")
        
        # Normalize common medical abbreviations
        text = self._normalize_medical_abbreviations(text)
        
        return text
    
    def _normalize_medical_abbreviations(self, text: str) -> str:
        """
        Normalize common medical abbreviations.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        # Common medical abbreviations
        replacements = {
            r'\bA&E\b': 'A&E',  # Keep as is
            r'\bER\b': 'Emergency Room',
            r'\bBP\b': 'blood pressure',
            r'\bHR\b': 'heart rate',
            r'\bPt\.\s': 'Patient ',
            r'\bDr\.\s': 'Doctor ',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_dates(self, text: str) -> List[Dict[str, any]]:
        """
        Extract dates from text.
        
        Args:
            text: Input text
        
        Returns:
            List of date dictionaries with 'raw', 'normalized', 'confidence'
        """
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                raw_date = match.group()
                
                # Parse date using dateparser
                parsed_date = dateparser.parse(raw_date, settings={
                    'STRICT_PARSING': False,
                    'PREFER_DATES_FROM': 'past'
                })
                
                if parsed_date:
                    dates.append({
                        'raw': raw_date,
                        'normalized': parsed_date.strftime('%Y-%m-%d'),
                        'confidence': 0.9 if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', raw_date) else 0.7,
                        'start_char': match.start(),
                        'end_char': match.end()
                    })
        
        # Remove duplicates (same normalized date)
        seen = set()
        unique_dates = []
        for date in dates:
            if date['normalized'] not in seen:
                seen.add(date['normalized'])
                unique_dates.append(date)
        
        return unique_dates
    
    def extract_duration_mentions(self, text: str) -> List[Dict[str, any]]:
        """
        Extract time duration mentions (e.g., "4 weeks", "six months").
        
        Args:
            text: Input text
        
        Returns:
            List of duration dictionaries
        """
        duration_patterns = [
            r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months|year|years)\b',
            r'\b(a|an)\s+(day|week|month|year)\b',
        ]
        
        durations = []
        
        for pattern in duration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                durations.append({
                    'raw': match.group(),
                    'start_char': match.start(),
                    'end_char': match.end(),
                    'confidence': 0.95
                })
        
        return durations
    
    def extract_severity_mentions(self, text: str) -> List[Dict[str, any]]:
        """
        Extract severity mentions (e.g., "severe", "mild", "moderate").
        
        Args:
            text: Input text
        
        Returns:
            List of severity dictionaries
        """
        severity_terms = [
            'severe', 'serious', 'bad', 'terrible', 'intense',
            'mild', 'minor', 'slight', 'small',
            'moderate', 'medium',
            'acute', 'chronic',
            'Grade 1', 'Grade 2', 'Grade 3', 'Grade I', 'Grade II', 'Grade III'
        ]
        
        severities = []
        
        for term in severity_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                severities.append({
                    'term': match.group(),
                    'normalized': term.lower(),
                    'start_char': match.start(),
                    'end_char': match.end(),
                    'confidence': 0.9
                })
        
        return severities
    
    def segment_by_medical_section(self, text: str) -> Dict[str, str]:
        """
        Attempt to segment text by medical sections (for SOAP-like structure).
        
        Args:
            text: Input text
        
        Returns:
            Dict with section names as keys
        """
        sections = {
            'chief_complaint': '',
            'history': '',
            'examination': '',
            'assessment': '',
            'plan': ''
        }
        
        # Simple keyword-based segmentation
        # In production, this would be more sophisticated
        
        lines = text.split('.')
        
        for line in lines:
            line_lower = line.lower()
            
            if any(kw in line_lower for kw in ['what brings you', 'here about', 'what happened']):
                sections['chief_complaint'] += line + '. '
            elif any(kw in line_lower for kw in ['examination', 'examine', 'let me check', 'listen to']):
                sections['examination'] += line + '. '
            elif any(kw in line_lower for kw in ['diagnosis', 'based on', 'i believe', 'you have']):
                sections['assessment'] += line + '. '
            elif any(kw in line_lower for kw in ['treatment', 'prescribe', 'i recommend', 'you should']):
                sections['plan'] += line + '. '
            else:
                sections['history'] += line + '. '
        
        # Clean up
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
