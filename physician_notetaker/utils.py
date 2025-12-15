"""
Utility functions for the physician notetaker system.
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()


def calculate_confidence(scores: List[float], method: str = "mean") -> float:
    """
    Calculate overall confidence from multiple scores.
    
    Args:
        scores: List of confidence scores (0-1)
        method: "mean", "min", or "harmonic"
    
    Returns:
        Overall confidence score
    """
    if not scores:
        return 0.0
    
    if method == "mean":
        return sum(scores) / len(scores)
    elif method == "min":
        return min(scores)
    elif method == "harmonic":
        return len(scores) / sum(1/(s + 1e-10) for s in scores)
    else:
        return sum(scores) / len(scores)


def redact_phi(text: str, placeholder: str = "[REDACTED]") -> str:
    """
    Basic PHI redaction for demo purposes.
    In production, use a proper de-identification system.
    
    Args:
        text: Input text
        placeholder: Replacement text for PHI
    
    Returns:
        Redacted text
    """
    # Redact phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', placeholder, text)
    
    # Redact email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', placeholder, text)
    
    # Redact SSN-like patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', placeholder, text)
    
    return text


def extract_speaker_turns(text: str) -> List[Dict[str, str]]:
    """
    Extract speaker turns from transcript.
    
    Expected format: "SPEAKER: text"
    
    Returns:
        List of dicts with 'speaker' and 'text' keys
    """
    turns = []
    pattern = r'^(Doctor|DR\.|Physician|Patient|PT\.|Ms\.|Mr\.|Mrs\.)[\s:]+(.+?)(?=(?:^(?:Doctor|DR\.|Physician|Patient|PT\.|Ms\.|Mr\.|Mrs\.)[\s:])|$)'
    
    matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        speaker = match.group(1).strip()
        turn_text = match.group(2).strip()
        
        # Normalize speaker labels
        if speaker.lower() in ['doctor', 'dr.', 'physician']:
            speaker = "Physician"
        else:
            speaker = "Patient"
        
        turns.append({
            "speaker": speaker,
            "text": turn_text
        })
    
    return turns


def merge_overlapping_spans(spans: List[Tuple[int, int, str, Any]]) -> List[Tuple[int, int, str, Any]]:
    """
    Merge overlapping entity spans, keeping the one with highest confidence.
    
    Args:
        spans: List of (start, end, label, data) tuples
    
    Returns:
        List of non-overlapping spans
    """
    if not spans:
        return []
    
    # Sort by start position
    sorted_spans = sorted(spans, key=lambda x: (x[0], -x[1]))
    
    merged = [sorted_spans[0]]
    
    for current in sorted_spans[1:]:
        last = merged[-1]
        
        # Check for overlap
        if current[0] < last[1]:
            # Keep the span with higher confidence
            current_conf = current[3].get('confidence', 0) if isinstance(current[3], dict) else 0
            last_conf = last[3].get('confidence', 0) if isinstance(last[3], dict) else 0
            
            if current_conf > last_conf:
                merged[-1] = current
        else:
            merged.append(current)
    
    return merged


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data as JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def truncate_text(text: str, max_length: int = 512, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def requires_clarification(confidence: float, threshold: float = 0.5) -> bool:
    """Check if a field requires clarification based on confidence."""
    return confidence < threshold


def generate_clarification_question(field_name: str, context: str = "") -> str:
    """
    Generate a clarification question for ambiguous fields.
    
    Args:
        field_name: Name of the ambiguous field
        context: Additional context if available
    
    Returns:
        Clarification question
    """
    questions = {
        "Patient_Name": "Could you please confirm the patient's full name?",
        "Incident.date": "Could you please confirm the exact date of the incident?",
        "Incident.location": "Could you please provide the location of the incident?",
        "Diagnosis": "Could you please clarify the diagnosis?",
        "Treatment": "Could you please specify the treatment plan?",
        "Prognosis": "Could you please clarify the expected prognosis?",
    }
    
    return questions.get(field_name, f"Could you please provide more information about {field_name}?")
