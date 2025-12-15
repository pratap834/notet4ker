# Physician Notetaker Web Application 🏥

A modern, AI-powered web application for analyzing medical transcripts using production-grade transformer models. Built with Flask and state-of-the-art NLP models for medical text processing.

## 🌟 Features

- **Named Entity Recognition (NER)**: Automatically extracts symptoms, diagnoses, treatments, and medications from medical conversations
- **SOAP Note Generation**: Creates structured SOAP (Subjective, Objective, Assessment, Plan) notes using AI
- **Clinical Summarization**: Generates concise, accurate clinical summaries
- **Sentiment Analysis**: Analyzes the emotional tone of medical conversations
- **Intent Detection**: Identifies the primary purpose of the medical interaction
- **Modern Web UI**: Clean, responsive interface that works on desktop and mobile
- **GPU Acceleration**: Automatically uses CUDA if available for faster processing
- **Export Functionality**: Download results as JSON for integration with other systems

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Installation

1. **Clone or navigate to the repository**:
   ```bash
   cd d:\Matrix\emittr
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install spaCy medical models** (optional but recommended):
   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
   ```

### Running the Application

1. **Start the web server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **First-time load**: The application will download transformer models on the first request (may take 1-2 minutes)

## 📖 Usage

### Web Interface

1. **Enter or paste** a medical transcript in the text area
2. Click **"Load Sample"** to see an example transcript
3. Click **"Analyze Transcript"** to process the text
4. View the results:
   - SOAP Note (Subjective, Objective, Assessment, Plan)
   - Clinical Summary
   - Extracted Entities (Symptoms, Diagnosis, Treatment, Medications)
   - Analysis Metadata (Sentiment, Intent, Model Information)
5. Click **"Export Results (JSON)"** to download the analysis

### Keyboard Shortcuts

- `Ctrl + Enter` in the text area: Analyze transcript
- Standard text editing shortcuts work in the input area

## 🏗️ Architecture

### Tech Stack

- **Backend**: Flask web framework
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **NLP Models**:
  - `d4data/biomedical-ner-all` - Biomedical Named Entity Recognition
  - `emilyalsentzer/Bio_ClinicalBERT` - Clinical text classification
  - `google/flan-t5-base` - Text generation and summarization
- **ML Framework**: PyTorch with Transformers (Hugging Face)

### Project Structure

```
emittr/
├── app.py                    # Flask application (main entry point)
├── templates/
│   └── index.html           # Web UI template
├── static/
│   ├── style.css           # Styling
│   └── script.js           # Frontend logic
├── physician_notetaker/     # Core NLP modules
│   ├── __init__.py
│   ├── ner.py              # Named Entity Recognition
│   ├── sentiment.py        # Sentiment analysis
│   ├── summarizer.py       # Text summarization
│   ├── soap_generator.py   # SOAP note generation
│   └── ...
├── notebooks/              # Jupyter notebooks for development
├── data/                   # Sample data and examples
├── requirements.txt        # Python dependencies
└── README_WEB.md          # This file
```

## 🔧 Configuration

### Model Selection

Edit the `CONFIG` dictionary in [app.py](app.py#L33-L42) to change models:

```python
CONFIG = {
    'NER_MODEL': "d4data/biomedical-ner-all",
    'CLINICAL_BERT': "emilyalsentzer/Bio_ClinicalBERT",
    'FLAN_T5': "google/flan-t5-base",  # Can use "google/flan-t5-large" for better quality
    'max_summary_length': 300,
    'max_soap_length': 400,
    'num_beams': 4,
    'temperature': 0.7,
}
```

### Server Configuration

Change host/port in [app.py](app.py#L304):

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 🎨 Customization

The UI is designed to be easily upgradable:

### Styling

- Edit [static/style.css](static/style.css)
- CSS variables defined at the top for easy theming:
  ```css
  :root {
      --primary-color: #2563eb;
      --card-bg: #ffffff;
      /* ... more variables */
  }
  ```

### Functionality

- Modify [static/script.js](static/script.js) for frontend behavior
- Add new API endpoints in [app.py](app.py)

### Layout

- Update [templates/index.html](templates/index.html) for structure changes

## 🔌 API Endpoints

The application exposes the following REST API endpoints:

### `GET /`
Returns the web interface

### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "device": "GPU (CUDA)" | "CPU",
  "models_loaded": true
}
```

### `POST /analyze`
Analyze medical transcript

**Request Body**:
```json
{
  "transcript": "Doctor: How are you feeling?..."
}
```

**Response**:
```json
{
  "entities": {
    "Symptoms": ["headache", "nausea"],
    "Diagnosis": ["migraine"],
    "Treatment": ["sumatriptan"],
    "Medications": []
  },
  "sentiment": {
    "sentiment": "positive",
    "confidence": 0.95
  },
  "intent": {
    "intent": "medical_consultation",
    "confidence": 0.89
  },
  "summary": "Patient presents with...",
  "soap_note": {
    "Subjective": {"content": "..."},
    "Objective": {"content": "..."},
    "Assessment": {"content": "..."},
    "Plan": {"content": "..."}
  },
  "model_info": {...}
}
```

## 🚦 Performance

### First Request
- **Model Loading**: 1-2 minutes (one-time download)
- **Processing**: 10-30 seconds depending on hardware

### Subsequent Requests
- **CPU**: 5-15 seconds per transcript
- **GPU**: 2-5 seconds per transcript

### Optimization Tips
- Use GPU if available (automatic detection)
- For production, consider:
  - Using `google/flan-t5-small` for faster inference
  - Implementing request queuing
  - Adding caching for repeated analyses
  - Deploying with Gunicorn/uWSGI

## 🛡️ Important Notes

⚠️ **Disclaimer**: This application is for **demonstration and research purposes only**. It is NOT intended for clinical use or medical decision-making. Always consult qualified healthcare professionals for medical advice.

### Data Privacy
- All processing happens locally on your machine
- No data is sent to external servers (except initial model downloads)
- Implement proper PHI (Protected Health Information) handling if using with real patient data

### Model Limitations
- AI models may produce errors or hallucinations
- Results should be reviewed by qualified medical professionals
- Not validated for clinical accuracy or completeness

## 🐛 Troubleshooting

### Models Not Loading
```bash
# Clear cache and reinstall transformers
pip install --upgrade --force-reinstall transformers torch
```

### CUDA Out of Memory
- Reduce batch size or use smaller models
- Or disable GPU: Set `DEVICE = -1` in [app.py](app.py#L20)

### Port Already in Use
Change the port in [app.py](app.py#L304):
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

## 📚 Additional Resources

- **Main Project README**: See root directory for full project documentation
- **Notebook Demos**: Check `notebooks/` folder for interactive examples
- **Model Documentation**:
  - [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
  - [FLAN-T5](https://huggingface.co/google/flan-t5-base)
  - [biomedical-ner-all](https://huggingface.co/d4data/biomedical-ner-all)

## 🤝 Contributing

This is a production-ready implementation based on the notebook pipeline. To contribute:

1. Test changes with the notebook first (`notebooks/production_pipeline.ipynb`)
2. Update the main modules in `physician_notetaker/`
3. Update the web app (`app.py`) to expose new features
4. Test the complete workflow

## 📝 License

Same as the main Physician Notetaker project.

## 🎯 Future Enhancements

The current UI is designed to be easily upgraded. Potential enhancements:

- [ ] User authentication and session management
- [ ] Database integration for storing analyses
- [ ] Batch processing for multiple transcripts
- [ ] Real-time streaming analysis
- [ ] FHIR export functionality
- [ ] Integration with EHR systems
- [ ] Multi-language support
- [ ] Voice-to-text integration
- [ ] Advanced visualization of medical entities
- [ ] Confidence threshold settings
- [ ] Custom model fine-tuning interface

---

**Built with ❤️ using production-grade AI models for medical NLP**
