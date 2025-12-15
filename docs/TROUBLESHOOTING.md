# Troubleshooting Guide 🔧

## Common Issues and Solutions

### Issue 1: PyTorch Version Error (CVE-2025-32434)

**Error Message:**
```
ValueError: Due to a serious vulnerability issue in `torch.load`, 
even with `weights_only=True`, we now require users to upgrade 
torch to at least v2.6
```

**Solution:**
```powershell
# Upgrade PyTorch to version 2.6+
pip install --upgrade torch>=2.6.0 safetensors>=0.4.0
```

**Why this happens:** 
- PyTorch versions below 2.6 have a security vulnerability
- Transformers library requires the latest version for safe model loading
- Safetensors provides a safer alternative to pickle-based model loading

---

### Issue 2: Model Loading Failures

**Error Message:**
```
Could not load model emilyalsentzer/Bio_ClinicalBERT
```

**Solution:**
The app now automatically falls back to compatible models. If you still have issues:

```powershell
# Clear Hugging Face cache
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface"

# Reinstall transformers
pip install --upgrade --force-reinstall transformers
```

---

### Issue 3: CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
Edit [app.py](app.py#L20) to force CPU usage:

```python
# Change this line:
DEVICE = 0 if torch.cuda.is_available() else -1

# To:
DEVICE = -1  # Force CPU usage
```

Or use smaller models in CONFIG:
```python
CONFIG['FLAN_T5'] = "google/flan-t5-small"  # Smaller, faster model
```

---

### Issue 4: Windows Symlinks Warning

**Warning Message:**
```
UserWarning: `huggingface_hub` cache-system uses symlinks 
but your machine does not support them
```

**Solution (Optional):**

1. **Enable Developer Mode** (Recommended):
   - Windows Settings → Update & Security → For Developers
   - Enable "Developer Mode"
   - Restart terminal

2. **Or run as Administrator**:
   - Right-click PowerShell → "Run as Administrator"
   - Navigate back to project and activate venv

3. **Or ignore** - It will work fine, just uses more disk space

---

### Issue 5: Port Already in Use

**Error Message:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
Change the port in [app.py](app.py#L304):

```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

Or kill the existing process:
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

---

### Issue 6: First Request is Very Slow

**This is NORMAL behavior:**

- First request downloads models (~2GB total)
- Models are cached for future use
- Subsequent requests will be fast

**Download times:**
- Fast connection: 2-5 minutes
- Slow connection: 10-15 minutes

---

### Issue 7: Module Not Found Errors

**Error Message:**
```
ModuleNotFoundError: No module named 'flask'
```

**Solution:**

1. **Verify virtual environment is activated:**
   ```powershell
   # You should see (.venv) in your prompt
   # If not, activate it:
   .venv\Scripts\activate
   ```

2. **Reinstall dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

### Issue 8: Tensor Type Mismatch

**Error Message:**
```
TypeError: Cannot convert torch.float32 to TensorFlow DType
```

**Solution:**
This happens when TensorFlow interferes with PyTorch. The app now handles this automatically with fallbacks. If issues persist:

```powershell
# Uninstall TensorFlow (not needed for this project)
pip uninstall tensorflow tf-keras -y

# Reinstall transformers
pip install --upgrade transformers
```

---

## Quick Diagnostic Commands

### Check Python Environment
```powershell
python --version          # Should be 3.8+
pip list | findstr torch  # Check PyTorch version (should be 2.6+)
pip list | findstr transformers  # Check transformers version
```

### Check GPU Status
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Check Model Cache
```powershell
# View cached models
dir "$env:USERPROFILE\.cache\huggingface\hub"
```

### Clear Everything and Start Fresh
```powershell
# Deactivate and remove venv
deactivate
Remove-Item -Recurse -Force .venv

# Clear model cache
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface"

# Start fresh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Performance Optimization

### Speed Up Inference

1. **Use GPU** (4x faster):
   - Automatic if CUDA is available
   - Check with: `python -c "import torch; print(torch.cuda.is_available())"`

2. **Use Smaller Models**:
   ```python
   CONFIG['FLAN_T5'] = "google/flan-t5-small"  # Faster, less accurate
   ```

3. **Reduce Max Length**:
   ```python
   CONFIG['max_summary_length'] = 150  # Shorter summaries
   ```

4. **Lower Beam Search**:
   ```python
   CONFIG['num_beams'] = 2  # Faster generation
   ```

---

## Getting Help

### Before Asking for Help:

1. ✅ Check this troubleshooting guide
2. ✅ Read [README.md](README.md) and [README_WEB.md](README_WEB.md)
3. ✅ Try clearing cache and reinstalling
4. ✅ Check Python and PyTorch versions
5. ✅ Look at terminal output for specific errors

### Include This Information:

```powershell
# Run this and share the output:
python -c "import sys; import torch; import transformers; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Model Alternatives

If specific models don't work, try these alternatives:

### NER Models
- ✅ `d4data/biomedical-ner-all` (default - medical specific)
- ✅ `allenai/scibert_scivocab_uncased` (scientific text)
- ✅ `dbmdz/bert-large-cased-finetuned-conll03-english` (general NER)

### Sentiment Models
- ✅ `distilbert-base-uncased-finetuned-sst-2-english` (default fallback)
- ✅ `cardiffnlp/twitter-roberta-base-sentiment` (good general sentiment)
- ✅ `nlptown/bert-base-multilingual-uncased-sentiment` (multi-language)

### Generation Models
- ✅ `google/flan-t5-small` (fast, 250MB)
- ✅ `google/flan-t5-base` (default, 990MB)
- ✅ `google/flan-t5-large` (best quality, 3GB)
- ✅ `facebook/bart-large-cnn` (alternative summarizer)

---

## Still Having Issues?

The app has been updated with robust error handling:
- ✅ Automatic fallbacks for failed models
- ✅ Graceful degradation if components fail
- ✅ Detailed error messages in console

**The app should now work even if some models fail to load!**

Try running again after the PyTorch upgrade:
```powershell
python app.py
```
