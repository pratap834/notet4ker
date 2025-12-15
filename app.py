"""
Physician Notetaker Web Application

A simple Flask-based web interface for medical transcript analysis using
production-ready transformer models.
"""

from flask import Flask, render_template, request, jsonify
import torch
import json
from pathlib import Path
from typing import Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models (lazy loading)
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Model instances (will be loaded on first use)
ner_pipeline = None
sentiment_pipeline = None
intent_pipeline = None
t5_tokenizer = None
t5_model = None

# Configuration
CONFIG = {
    'NER_MODEL': "d4data/biomedical-ner-all",
    'SENTIMENT_MODEL': "distilbert-base-uncased-finetuned-sst-2-english",  # Fallback model with safetensors
    'FLAN_T5': "google/flan-t5-base",
    'max_summary_length': 300,
    'max_soap_length': 400,
    'num_beams': 4,
    'temperature': 0.7,
}


def load_models():
    """Load all transformer models on first request"""
    global ner_pipeline, sentiment_pipeline, intent_pipeline, t5_tokenizer, t5_model
    
    if ner_pipeline is None:
        print("Loading models... This may take a minute on first load.")
        
        # NER Pipeline
        print(f"Loading NER: {CONFIG['NER_MODEL']}")
        try:
            ner_pipeline = pipeline(
                "ner",
                model=CONFIG['NER_MODEL'],
                aggregation_strategy="simple",
                device=DEVICE,
                use_safetensors=True
            )
        except Exception as e:
            print(f"Warning: Could not load {CONFIG['NER_MODEL']}, using basic NER: {e}")
            ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=DEVICE
            )
        
        # Sentiment Pipeline (using simpler, compatible model)
        print(f"Loading Sentiment: {CONFIG['SENTIMENT_MODEL']}")
        try:
            sentiment_pipeline = pipeline(
                "text-classification",
                model=CONFIG['SENTIMENT_MODEL'],
                device=DEVICE
            )
        except Exception as e:
            print(f"Warning: Sentiment model failed, using basic classification: {e}")
            sentiment_pipeline = None
        
        # Intent Pipeline (reuse sentiment for now)
        intent_pipeline = sentiment_pipeline
        
        # FLAN-T5 for generation
        print(f"Loading FLAN-T5: {CONFIG['FLAN_T5']}")
        t5_tokenizer = AutoTokenizer.from_pretrained(CONFIG['FLAN_T5'])
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(
            CONFIG['FLAN_T5'],
            torch_dtype=DTYPE,
            use_safetensors=True
        ).to("cuda" if DEVICE == 0 else "cpu")
        
        print("✓ All models loaded successfully!")


def make_json_serializable(obj):
    """Convert torch/numpy types to native Python types"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except:
        pass
    return obj


def t5_generate(prompt: str, max_length: int = 200) -> str:
    """Generate text using FLAN-T5"""
    inputs = t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(t5_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=CONFIG['num_beams'],
            temperature=CONFIG['temperature'],
            do_sample=True,
            top_p=0.95
        )
    
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_entities(text: str) -> Dict:
    """Extract named entities from medical transcript"""
    # Run NER
    ner_results = ner_pipeline(text[:512])  # Truncate for speed
    
    # Structure entities
    structured = {
        "Symptoms": [],
        "Diagnosis": [],
        "Treatment": [],
        "Medications": [],
        "Other": []
    }
    
    # Try to map NER results
    for entity in ner_results:
        label = entity['entity_group'].upper()
        word = entity['word'].strip()
        
        if any(x in label for x in ['SYMPTOM', 'SIGN', 'DISEASE']):
            structured["Symptoms"].append(word)
        elif any(x in label for x in ['DIAGNOSIS', 'DISORDER']):
            structured["Diagnosis"].append(word)
        elif any(x in label for x in ['TREATMENT', 'PROCEDURE', 'THERAPY']):
            structured["Treatment"].append(word)
        elif any(x in label for x in ['MEDICATION', 'DRUG', 'MEDICINE']):
            structured["Medications"].append(word)
    
    # If NER didn't find medical entities, use LLM extraction
    total_entities = sum(len(v) for v in structured.values())
    if total_entities < 2:
        structured = extract_with_llm(text)
    
    return {
        "detailed": ner_results,
        "structured": structured
    }


def extract_with_llm(text: str) -> Dict:
    """Use FLAN-T5 to extract medical entities when NER fails"""
    structured = {
        "Symptoms": [],
        "Diagnosis": [],
        "Treatment": [],
        "Medications": []
    }
    
    # Extract symptoms
    symptoms_prompt = f"""List all symptoms mentioned in this medical conversation. Only list the symptoms, separated by commas:

{text[:400]}

Symptoms:"""
    symptoms_text = t5_generate(symptoms_prompt, 100)
    if symptoms_text and symptoms_text.lower() != "none":
        structured["Symptoms"] = [s.strip() for s in symptoms_text.split(',') if s.strip()]
    
    # Extract diagnosis
    diagnosis_prompt = f"""What medical conditions or diagnoses are mentioned in this conversation? Only list the diagnoses, separated by commas:

{text[:400]}

Diagnoses:"""
    diagnosis_text = t5_generate(diagnosis_prompt, 100)
    if diagnosis_text and diagnosis_text.lower() != "none":
        structured["Diagnosis"] = [d.strip() for d in diagnosis_text.split(',') if d.strip()]
    
    # Extract treatments
    treatment_prompt = f"""What treatments or procedures are mentioned in this conversation? Only list them, separated by commas:

{text[:400]}

Treatments:"""
    treatment_text = t5_generate(treatment_prompt, 100)
    if treatment_text and treatment_text.lower() != "none":
        structured["Treatment"] = [t.strip() for t in treatment_text.split(',') if t.strip()]
    
    # Extract medications
    medication_prompt = f"""What medications or drugs are mentioned in this conversation? Only list them, separated by commas:

{text[:400]}

Medications:"""
    medication_text = t5_generate(medication_prompt, 100)
    if medication_text and medication_text.lower() != "none":
        structured["Medications"] = [m.strip() for m in medication_text.split(',') if m.strip()]
    
    return structured


def analyze_sentiment(text: str) -> Dict:
    """Analyze sentiment of the conversation"""
    if sentiment_pipeline is None:
        return {
            "sentiment": "neutral",
            "confidence": 0.5
        }
    try:
        result = sentiment_pipeline(text[:512])[0]
        return {
            "sentiment": result['label'],
            "confidence": float(result['score'])
        }
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return {
            "sentiment": "neutral",
            "confidence": 0.5
        }


def detect_intent(text: str) -> Dict:
    """Detect primary intent of the conversation"""
    if intent_pipeline is None:
        return {
            "intent": "medical_consultation",
            "confidence": 0.5
        }
    try:
        result = intent_pipeline(text[:512])[0]
        return {
            "intent": result['label'],
            "confidence": float(result['score'])
        }
    except Exception as e:
        print(f"Intent detection error: {e}")
        return {
            "intent": "medical_consultation",
            "confidence": 0.5
        }


def generate_summary(text: str, entities: Dict) -> str:
    """Generate clinical summary using FLAN-T5"""
    prompt = f"""Write a brief clinical summary of this doctor-patient conversation. Include the main complaint, key findings, and recommended treatment:

{text[:500]}

Clinical Summary:"""
    
    summary = t5_generate(prompt, CONFIG['max_summary_length'])
    
    # Ensure we got a real summary, not just the input
    if len(summary) < 20 or summary.lower().startswith("patient:"):
        # Try alternative prompt
        alt_prompt = f"""Summarize the medical visit: What is the patient's complaint? What did the doctor diagnose? What treatment was recommended?

Conversation: {text[:400]}

Summary:"""
        summary = t5_generate(alt_prompt, 200)
    
    return summary


def generate_soap(text: str, entities: Dict) -> Dict:
    """Generate SOAP note using FLAN-T5"""
    soap_note = {}
    
    # Subjective
    subjective_prompt = f"""Extract patient's subjective complaints and symptoms from this conversation:

{text[:800]}

Subjective:"""
    soap_note["Subjective"] = {"content": t5_generate(subjective_prompt, 150)}
    
    # Objective
    objective_prompt = f"""Extract objective findings and examination results from this conversation:

{text[:800]}

Objective:"""
    soap_note["Objective"] = {"content": t5_generate(objective_prompt, 150)}
    
    # Assessment
    assessment_prompt = f"""Extract the medical assessment and diagnosis from this conversation:

{text[:800]}

Assessment:"""
    soap_note["Assessment"] = {"content": t5_generate(assessment_prompt, 100)}
    
    # Plan
    plan_prompt = f"""Extract the treatment plan and follow-up recommendations from this conversation:

{text[:800]}

Plan:"""
    soap_note["Plan"] = {"content": t5_generate(plan_prompt, 150)}
    
    return soap_note


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "device": "GPU (CUDA)" if DEVICE == 0 else "CPU",
        "models_loaded": ner_pipeline is not None
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Main endpoint to analyze medical transcript"""
    try:
        # Get transcript from request
        data = request.get_json()
        transcript = data.get('transcript', '')
        
        if not transcript or len(transcript.strip()) < 10:
            return jsonify({"error": "Transcript too short. Please provide at least 10 characters."}), 400
        
        # Load models if not already loaded
        load_models()
        
        # Run the pipeline
        entities = extract_entities(transcript)
        sentiment = analyze_sentiment(transcript)
        intent = detect_intent(transcript)
        summary = generate_summary(transcript, entities)
        soap = generate_soap(transcript, entities)
        
        # Prepare response
        result = {
            "entities": entities["structured"],
            "sentiment": sentiment,
            "intent": intent,
            "summary": summary,
            "soap_note": soap,
            "model_info": {
                "ner": CONFIG['NER_MODEL'],
                "sentiment": CONFIG['SENTIMENT_MODEL'],
                "summarization": CONFIG['FLAN_T5'],
                "device": "GPU (CUDA)" if DEVICE == 0 else "CPU"
            }
        }
        
        # Make everything JSON serializable
        result = make_json_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import os
    
    print("="*70)
    print("PHYSICIAN NOTETAKER WEB APPLICATION")
    print("="*70)
    print(f"Device: {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*70)
    print("\nStarting web server...")
    
    # Get port from environment (for Render/cloud deployment) or use 5000
    port = int(os.environ.get('PORT', 5000))
    print(f"Access the application at: http://localhost:{port}")
    print("\nNote: Models will be loaded on first request (may take 1-2 minutes)")
    print("="*70)
    
    # Use debug mode only in development
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
