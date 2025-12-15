# Quick Reference - How to Run 🚀

## First Time Setup (One-Time Only)

```powershell
# 1. Navigate to project
cd D:\Matrix\emittr

# 2. Create virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\activate

# 4. Install everything
pip install -r requirements.txt

# 5. (Optional) Install medical models
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

---

## Run the Web Application

### Method 1: Simple Script (Easiest)
```powershell
start_web.bat
```
Then open: **http://localhost:5000**

### Method 2: Manual
```powershell
.venv\Scripts\activate
python app.py
```
Then open: **http://localhost:5000**

---

## Run Jupyter Notebook

```powershell
.venv\Scripts\activate
jupyter notebook notebooks/production_pipeline.ipynb
```

---

## Run Demo Script

```powershell
.venv\Scripts\activate
python run_demo.py
```

---

## Common Commands

```powershell
# Activate virtual environment
.venv\Scripts\activate

# Check if everything is installed
pip list

# Upgrade PyTorch if needed (for security)
pip install --upgrade torch>=2.6.0

# Check GPU status
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Clear model cache
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface"
```

---

## Keyboard Shortcuts (Web App)

- **Ctrl + Enter** in textarea: Analyze transcript
- **Load Sample**: Loads demo transcript
- **Export**: Download results as JSON

---

## What Happens on First Run?

1. **Models Download** (~2GB, 2-5 minutes)
   - d4data/biomedical-ner-all (~500MB)
   - distilbert-base-uncased (~270MB)
   - google/flan-t5-base (~990MB)

2. **Models Load into Memory** (~30 seconds)
   - NER pipeline
   - Sentiment classifier
   - Text generator

3. **Ready to Use!** 🎉

Subsequent runs are much faster (models are cached).

---

## Troubleshooting Quick Fixes

### PyTorch Version Error
```powershell
pip install --upgrade torch>=2.6.0 safetensors>=0.4.0
```

### Port Already in Use
```powershell
# Change port in app.py to 5001 or kill existing process:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Model Loading Issues
```powershell
# Clear cache and retry
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface"
python app.py
```

### Out of Memory (GPU)
Edit `app.py` line 20:
```python
DEVICE = -1  # Force CPU usage
```

---

## File Structure Quick Reference

```
📁 Main Files
├── app.py              ← Flask web server
├── start_web.bat       ← Easy startup script
├── requirements.txt    ← All dependencies
└── README.md          ← Full documentation

📁 Code (physician_notetaker/)
├── ner.py             ← Entity extraction
├── summarizer.py      ← Text summarization
├── soap_generator.py  ← SOAP note creation
└── sentiment.py       ← Sentiment analysis

📁 Web Interface
├── templates/
│   └── index.html     ← Web UI
└── static/
    ├── style.css      ← Styling
    └── script.js      ← Frontend logic

📁 Examples
├── data/examples/     ← Sample transcripts
├── notebooks/         ← Jupyter demos
└── final_output/      ← Processing results
```

---

## Resources

- **Full Documentation**: [README.md](README.md)
- **Web App Guide**: [README_WEB.md](README_WEB.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **LLM Guide**: [LLM_GUIDE.md](LLM_GUIDE.md)

---

## Need Help?

1. ✅ Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. ✅ Read error messages in terminal
3. ✅ Verify virtual environment is activated
4. ✅ Ensure PyTorch 2.6+ is installed

---

**That's it! You're ready to use Physician Notetaker! 🏥**
