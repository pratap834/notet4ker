"""
Simple medical summarizer module - generates clean, minimal output.
"""

import re
from typing import Dict, List, Optional, Any
from .utils import get_logger

logger = get_logger(__name__)


class SimpleSummarizer:
    """Generate clean, simplified medical summaries."""
    
    def __init__(self):
        """Initialize the simple summarizer."""
        pass
    
    def summarize(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        entities_by_type: Dict[str, List[Dict[str, Any]]],
        preprocessed_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate simplified medical report with only essential fields.
        
        Args:
            text: Original transcript text
            entities: List of all entities
            entities_by_type: Entities grouped by type
            preprocessed_data: Preprocessed data from MedicalPreprocessor
        
        Returns:
            Simplified medical report dict with essential fields only
        """
        # Extract patient name
        patient_name = self._extract_patient_name(text, preprocessed_data)
        
        # Extract symptoms (cleaned list)
        symptoms = self._extract_symptoms_clean(entities_by_type, text)
        
        # Extract diagnosis (single string or list if multiple)
        diagnosis = self._extract_diagnosis_clean(entities_by_type, text)
        
        # Extract treatments (cleaned list)
        treatments = self._extract_treatments_clean(entities_by_type, text)
        
        # Extract current status
        current_status = self._extract_current_status(text)
        
        # Extract prognosis
        prognosis = self._extract_prognosis(entities_by_type, text)
        
        # Build simplified report
        report = {
            "Patient_Name": patient_name,
            "Symptoms": symptoms,
            "Diagnosis": diagnosis,
            "Treatment": treatments,
            "Current_Status": current_status,
            "Prognosis": prognosis
        }
        
        # Remove None values
        report = {k: v for k, v in report.items() if v is not None}
        
        return report
    
    def _extract_patient_name(
        self, 
        text: str, 
        preprocessed_data: Optional[Dict]
    ) -> Optional[str]:
        """Extract patient name from text."""
        # Look for explicit name patterns (case-sensitive for names)
        patterns = [
            r"(?:it's|name is|I'm|I am)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",  # Full name like "Janet Jones"
            r"(?:Ms\.|Mr\.|Mrs\.|Miss|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"patient(?:'s)?\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)  # Case-sensitive
            if match:
                name = match.group(1).strip()
                # Validate it looks like a name (starts with capitals)
                if name and name[0].isupper():
                    return name
        
        return None
    
    def _extract_symptoms_clean(
        self,
        entities_by_type: Dict[str, List[Dict[str, Any]]],
        text: str
    ) -> List[str]:
        """Extract symptoms as clean list."""
        symptoms = []
        seen = set()
        
        if 'SYMPTOM' in entities_by_type:
            for entity in entities_by_type['SYMPTOM']:
                symptom_text = entity['text'].strip()
                # Capitalize first letter
                symptom_text = symptom_text[0].upper() + symptom_text[1:] if symptom_text else symptom_text
                
                # Clean up duplicates and partial matches
                symptom_lower = symptom_text.lower()
                # Check if this is a substring or duplicate
                is_duplicate = False
                for existing in list(seen):
                    if symptom_lower in existing or existing in symptom_lower:
                        # Keep the longer/more complete version
                        if len(symptom_lower) > len(existing):
                            symptoms = [s for s in symptoms if s.lower() != existing]
                            seen.remove(existing)
                        else:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    symptoms.append(symptom_text)
                    seen.add(symptom_lower)
        
        # Additional pattern matching for common missed symptoms
        additional_patterns = {
            r'trouble sleeping': 'Trouble sleeping',
            r'head (?:impact|jerked)': 'Head impact',
        }
        
        for pattern, symptom_name in additional_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if symptom_name.lower() not in seen:
                    symptoms.append(symptom_name)
                    seen.add(symptom_name.lower())
        
        return symptoms if symptoms else []
    
    def _extract_diagnosis_clean(
        self,
        entities_by_type: Dict[str, List[Dict[str, Any]]],
        text: str
    ) -> Optional[str]:
        """Extract diagnosis as single string or None."""
        diagnoses = []
        
        if 'DIAGNOSIS' in entities_by_type:
            for entity in entities_by_type['DIAGNOSIS']:
                diagnosis_text = entity['text'].strip()
                # Capitalize first letter
                diagnosis_text = diagnosis_text[0].upper() + diagnosis_text[1:] if diagnosis_text else diagnosis_text
                diagnoses.append(diagnosis_text)
        
        if not diagnoses:
            return None
        
        # If multiple diagnoses, return the most specific one or join them
        if len(diagnoses) == 1:
            return diagnoses[0]
        
        # Remove duplicates and substrings
        unique_diagnoses = []
        for diag in diagnoses:
            if not any(diag.lower() in other.lower() and diag.lower() != other.lower() 
                      for other in diagnoses):
                unique_diagnoses.append(diag)
        
        if len(unique_diagnoses) == 1:
            return unique_diagnoses[0]
        
        # Return most specific (usually the longest)
        return max(unique_diagnoses, key=len)
    
    def _extract_treatments_clean(
        self,
        entities_by_type: Dict[str, List[Dict[str, Any]]],
        text: str
    ) -> List[str]:
        """Extract treatments as clean list."""
        treatments = []
        seen = set()
        
        if 'TREATMENT' in entities_by_type:
            for entity in entities_by_type['TREATMENT']:
                treatment_text = entity['text'].strip()
                
                # Skip very short fragments like "er"
                if len(treatment_text) <= 2:
                    continue
                
                # Capitalize first letter
                treatment_text = treatment_text[0].upper() + treatment_text[1:] if treatment_text else treatment_text
                
                treatment_lower = treatment_text.lower()
                if treatment_lower not in seen:
                    treatments.append(treatment_text)
                    seen.add(treatment_lower)
        
        # Look for specific treatment patterns (more specific patterns first)
        treatment_patterns = {
            r'about\s+(\d+)\s+(?:physiotherapy\s+)?sessions?': lambda m: f"{m.group(1)} physiotherapy sessions",
            r'(\d+)\s+physiotherapy\s+sessions?': lambda m: f"{m.group(1)} physiotherapy sessions",
            r'rest,?\s*ice,?\s*compression,?\s*(?:and\s+)?elevation': 'RICE protocol',
            r'RICE protocol': 'RICE protocol',
            r'painkillers?': 'Painkillers',
            r'antibiotics?': 'Antibiotics',
            r'physiotherapy': 'Physiotherapy',
        }
        
        for pattern, replacement in treatment_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if callable(replacement):
                    treatment = replacement(match)
                else:
                    treatment = replacement
                
                if treatment.lower() not in seen:
                    treatments.append(treatment)
                    seen.add(treatment.lower())
        
        # Remove generic duplicates (e.g., "physiotherapy" if "10 physiotherapy sessions" exists)
        cleaned_treatments = []
        for treatment in treatments:
            is_duplicate = False
            for other in treatments:
                if treatment != other and treatment.lower() in other.lower():
                    is_duplicate = True
                    break
            if not is_duplicate:
                cleaned_treatments.append(treatment)
        
        return cleaned_treatments if cleaned_treatments else []
    
    def _extract_current_status(self, text: str) -> Optional[str]:
        """Extract current status."""
        status_patterns = [
            r"(?:still get|getting)\s+(?:the\s+)?(occasional\s+[a-z]+ache|occasional\s+[a-z]+\s+pain)",
            r"(?:occasional\s+)([a-z]+ache|[a-z]+\s+pain)",
        ]
        
        for pattern in status_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                status = match.group().strip() if '(' not in pattern else match.group(1).strip()
                # Clean up and capitalize
                if len(status) >= 5:
                    # Remove trailing unnecessary words
                    status = re.sub(r'\s+(though\s+.+)$', '', status, flags=re.IGNORECASE)
                    status = status.strip().rstrip('.,;')
                    if len(status) >= 5:  # Ensure still valid after cleanup
                        # Make sure "occasional" is included
                        if 'occasional' not in status.lower():
                            status = "Occasional " + status
                        return status[0].upper() + status[1:] if status else None
        
        return None
    
    def _extract_prognosis(
        self,
        entities_by_type: Dict[str, List[Dict[str, Any]]],
        text: str
    ) -> Optional[str]:
        """Extract prognosis."""
        # Pattern matching for prognosis (most specific first)
        prognosis_patterns = [
            r"(full\s+recovery\s+(?:expected\s+)?within\s+(?:six\s+months|[\d\w\s]+))",
            r"(recover\s+completely\s+within\s+(?:six\s+months|[\d\w\s]+))",
            r"(should\s+recover\s+well)",
            r"(recovering\s+well)",
            r"(prognosis\s+is\s+[^.]+)",
        ]
        
        for pattern in prognosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                prognosis = match.group(1).strip()
                # Capitalize and clean
                prognosis = prognosis[0].upper() + prognosis[1:] if prognosis else None
                # Clean up endings and extra text
                if prognosis:
                    prognosis = prognosis.rstrip('.,;')
                    # Remove trailing clauses after comma
                    if ',' in prognosis:
                        prognosis = prognosis.split(',')[0].strip()
                return prognosis
        
        # Try entities as fallback
        if 'PROGNOSIS' in entities_by_type and entities_by_type['PROGNOSIS']:
            prognosis_entities = entities_by_type['PROGNOSIS']
            # Take the most confident or longest one
            prognosis_text = max(prognosis_entities, key=lambda x: len(x['text']))['text']
            return prognosis_text.strip()
        
        return None
