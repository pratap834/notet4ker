"""
LLM-based medical summarizer using ClinicalBERT and FLAN-T5.
Provides enhanced extraction while maintaining the same output format.
"""

import re
import json
import torch
from typing import Dict, List, Optional, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    pipeline
)
from .utils import get_logger

logger = get_logger(__name__)


class LLMSummarizer:
    """LLM-enhanced medical summarizer supporting ClinicalBERT and FLAN-T5."""
    
    # Model configurations - NO HARDCODED VALUES
    MODEL_NAMES = {
        'clinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
        'flan-t5': 'google/flan-t5-base',
        'flan-t5-large': 'google/flan-t5-large'
    }
    
    # Generation parameters - configurable
    DEFAULT_GEN_PARAMS = {
        'max_input_length': 512,
        'num_beams': 4,
        'temperature': 0.7,
        'do_sample': False,
        'early_stopping': True,
        # Task-specific max output lengths
        'patient_name_max_length': 20,
        'symptoms_max_length': 150,
        'diagnosis_max_length': 50,
        'treatment_max_length': 150,
        'status_max_length': 30,
        'prognosis_max_length': 50
    }
    
    def __init__(self, model_type: str = "flan-t5", generation_params: Optional[Dict] = None):
        """
        Initialize LLM summarizer.
        
        Args:
            model_type: 'clinicalbert', 'flan-t5', or 'flan-t5-large'
            generation_params: Optional dict to override default generation parameters
        """
        self.model_type = model_type.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_params = {**self.DEFAULT_GEN_PARAMS, **(generation_params or {})}
        logger.info(f"Initializing LLM summarizer with {self.model_type} on {self.device}")
        
        # Initialize models based on type
        if "clinicalbert" in self.model_type:
            self._init_clinicalbert()
        elif "flan-t5" in self.model_type:
            self._init_flan_t5()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _init_clinicalbert(self):
        """Initialize ClinicalBERT for medical NER and classification."""
        try:
            logger.info("Loading ClinicalBERT models...")
            # Use BioClinicalBERT for medical NER
            clinicalbert_model = self.MODEL_NAMES['clinicalbert']
            self.ner_pipeline = pipeline(
                "ner",
                model=clinicalbert_model,
                tokenizer=clinicalbert_model,
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple"
            )
            
            # For text generation, use a medical T5 model
            t5_model = self.MODEL_NAMES['flan-t5']
            self.tokenizer = AutoTokenizer.from_pretrained(t5_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(t5_model)
            self.model.to(self.device)
            
            logger.info("✓ ClinicalBERT models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ClinicalBERT: {e}")
            raise
    
    def _init_flan_t5(self):
        """Initialize FLAN-T5 for medical text generation."""
        try:
            logger.info(f"Loading FLAN-T5 models...")
            
            # Choose model size based on model type
            if "large" in self.model_type:
                model_name = self.MODEL_NAMES['flan-t5-large']
            else:
                model_name = self.MODEL_NAMES['flan-t5']
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            
            # For NER, use a medical NER model
            try:
                clinicalbert_model = self.MODEL_NAMES['clinicalbert']
                self.ner_pipeline = pipeline(
                    "ner",
                    model=clinicalbert_model,
                    tokenizer=clinicalbert_model,
                    device=0 if self.device == "cuda" else -1,
                    aggregation_strategy="simple"
                )
                logger.info("✓ FLAN-T5 with ClinicalBERT NER loaded successfully")
            except:
                logger.warning("ClinicalBERT NER not available, using basic extraction")
                self.ner_pipeline = None
            
        except Exception as e:
            logger.error(f"Error loading FLAN-T5: {e}")
            raise
    
    def summarize(
        self,
        text: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        entities_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        preprocessed_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate simplified medical report using LLM.
        
        Args:
            text: Original transcript text
            entities: List of all entities (from traditional NER)
            entities_by_type: Entities grouped by type
            preprocessed_data: Preprocessed data
        
        Returns:
            Simplified medical report dict
        """
        logger.info("Generating LLM-based medical summary...")
        
        # Extract information using LLM
        patient_name = self._extract_patient_name_llm(text)
        symptoms = self._extract_symptoms_llm(text, entities_by_type)
        diagnosis = self._extract_diagnosis_llm(text)
        treatments = self._extract_treatments_llm(text)
        current_status = self._extract_current_status_llm(text)
        prognosis = self._extract_prognosis_llm(text)
        
        # Build simplified report
        report = {}
        
        if patient_name:
            report["Patient_Name"] = patient_name
        if symptoms:
            report["Symptoms"] = symptoms
        if diagnosis:
            report["Diagnosis"] = diagnosis
        if treatments:
            report["Treatment"] = treatments
        if current_status:
            report["Current_Status"] = current_status
        if prognosis:
            report["Prognosis"] = prognosis
        
        logger.info(f"✓ LLM summary generated with {len(report)} fields")
        return report
    
    def _generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the LLM."""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=self.generation_params['max_input_length'], 
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=self.generation_params['num_beams'],
                early_stopping=self.generation_params['early_stopping'],
                temperature=self.generation_params['temperature'],
                do_sample=self.generation_params['do_sample']
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()
    
    def _extract_patient_name_llm(self, text: str) -> Optional[str]:
        """Extract patient name using LLM."""
        # First try regex for speed
        patterns = [
            r"(?:it's|name is|I'm|I am)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"(?:Ms\.|Mr\.|Mrs\.|Miss|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Try LLM extraction
        prompt = f"""Extract the patient's full name from this medical conversation. If no name is mentioned, respond with "None".

Conversation: {text[:500]}

Patient's full name:"""
        
        try:
            max_len = self.generation_params.get('patient_name_max_length', 20)
            result = self._generate_text(prompt, max_length=max_len)
            if result and result.lower() != "none" and len(result.split()) <= 3:
                # Validate it looks like a name
                if result[0].isupper():
                    return result
        except Exception as e:
            logger.warning(f"LLM name extraction failed: {e}")
        
        return None
    
    def _extract_symptoms_llm(
        self, 
        text: str,
        entities_by_type: Optional[Dict] = None
    ) -> List[str]:
        """Extract symptoms using LLM."""
        # Try NER pipeline first if available
        symptoms = set()
        
        if self.ner_pipeline and entities_by_type and 'SYMPTOM' in entities_by_type:
            for entity in entities_by_type['SYMPTOM']:
                symptoms.add(entity['text'].capitalize())
        
        # Use LLM for additional extraction
        prompt = f"""List all symptoms mentioned in this medical conversation. Provide only the symptom names, one per line.

Conversation: {text[:800]}

Symptoms:"""
        
        try:
            max_len = self.generation_params.get('treatment_max_length', 150)
            result = self._generate_text(prompt, max_length=max_len)
            # Parse the result
            for line in result.split('\n'):
                line = line.strip().strip('-•*').strip()
                if line and len(line) < 50 and not line.lower().startswith(('conversation', 'symptoms')):
                    symptoms.add(line.capitalize())
        except Exception as e:
            logger.warning(f"LLM symptom extraction failed: {e}")
        
        return sorted(list(symptoms))[:10]  # Limit to top 10
    
    def _extract_diagnosis_llm(self, text: str) -> Optional[str]:
        """Extract diagnosis using LLM."""
        prompt = f"""What is the primary medical diagnosis mentioned in this conversation? Provide only the diagnosis name, nothing else. If no diagnosis is mentioned, respond with "None".

Conversation: {text[:800]}

Primary diagnosis:"""
        
        try:
            max_len = self.generation_params.get('diagnosis_max_length', 50)
            result = self._generate_text(prompt, max_length=max_len)
            if result and result.lower() != "none" and len(result) < 100:
                # Clean up common prefixes
                result = re.sub(r'^(the\s+)?(primary\s+)?(diagnosis\s+(is|was)\s+)?', '', result, flags=re.IGNORECASE)
                result = result.strip().rstrip('.')
                if result:
                    return result[0].upper() + result[1:]
        except Exception as e:
            logger.warning(f"LLM diagnosis extraction failed: {e}")
        
        return None
    
    def _extract_treatments_llm(self, text: str) -> List[str]:
        """Extract treatments using LLM."""
        prompt = f"""List all treatments, medications, and interventions mentioned in this medical conversation. Include specific details like dosages or session counts. Provide only the treatment names, one per line.

Conversation: {text[:800]}

Treatments:"""
        
        treatments = set()
        
        try:
            max_len = self.generation_params.get('treatment_max_length', 150)
            result = self._generate_text(prompt, max_length=max_len)
            # Parse the result
            for line in result.split('\n'):
                line = line.strip().strip('-•*').strip()
                if line and len(line) < 80 and not line.lower().startswith(('conversation', 'treatments', 'medications')):
                    treatments.add(line.capitalize())
        except Exception as e:
            logger.warning(f"LLM treatment extraction failed: {e}")
        
        return sorted(list(treatments))[:8]  # Limit to top 8
    
    def _extract_current_status_llm(self, text: str) -> Optional[str]:
        """Extract current status using LLM."""
        prompt = f"""What is the patient's current medical status or ongoing symptoms? Provide a brief phrase (5-10 words). If fully recovered or not mentioned, respond with "None".

Conversation: {text[:800]}

Current status:"""
        
        try:
            max_len = self.generation_params.get('status_max_length', 30)
            result = self._generate_text(prompt, max_length=max_len)
            if result and result.lower() not in ["none", "not mentioned", "fully recovered"]:
                result = result.strip().rstrip('.')
                if len(result) > 5 and len(result) < 80:
                    return result[0].upper() + result[1:]
        except Exception as e:
            logger.warning(f"LLM status extraction failed: {e}")
        
        return None
    
    def _extract_prognosis_llm(self, text: str) -> Optional[str]:
        """Extract prognosis using LLM."""
        prompt = f"""What is the prognosis or expected recovery outcome mentioned by the doctor? Provide a brief phrase (under 15 words). If not mentioned, respond with "None".

Conversation: {text[:800]}

Prognosis:"""
        
        try:
            max_len = self.generation_params.get('prognosis_max_length', 50)
            result = self._generate_text(prompt, max_length=max_len)
            if result and result.lower() != "none" and len(result) > 10:
                result = result.strip().rstrip('.')
                if len(result) < 150:
                    return result[0].upper() + result[1:]
        except Exception as e:
            logger.warning(f"LLM prognosis extraction failed: {e}")
        
        return None
