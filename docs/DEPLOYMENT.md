# Physician Notetaker - Deployment Guide

## 🚀 Deploy to Render

### Prerequisites
- GitHub account
- Render account (free tier available at [render.com](https://render.com))

### Quick Deploy Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Physician Notetaker"
   git remote add origin https://github.com/YOUR_USERNAME/physician-notetaker.git
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to [dashboard.render.com](https://dashboard.render.com)
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml` configuration

3. **Configuration**
   The following will be configured automatically via `render.yaml`:
   - **Name**: physician-notetaker
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free
   - **Disk**: 10GB for model caching

4. **Wait for Deployment**
   - First deployment takes 10-15 minutes (downloads ML models)
   - Subsequent deploys are faster (cached models)
   - Watch the build logs for progress

5. **Access Your App**
   - Your app will be available at: `https://physician-notetaker.onrender.com`
   - Models load on first request (may take 2-3 minutes)

### 📋 Deployment Files

- **render.yaml** - Render service configuration
- **build.sh** - Build script for dependencies
- **gunicorn.conf.py** - Production server configuration
- **Procfile** - Process definition
- **runtime.txt** - Python version specification
- **requirements.txt** - Python dependencies

### ⚙️ Environment Variables

Automatically set by `render.yaml`:
- `PYTORCH_DEVICE=cpu` - Use CPU (GPU not available on free tier)
- `MODEL_CACHE_DIR=/opt/render/project/.cache`
- `TRANSFORMERS_CACHE=/opt/render/project/.cache/transformers`

### 📊 Resource Requirements

**Free Tier Limits**:
- **RAM**: 512MB (sufficient for base models)
- **CPU**: Shared
- **Storage**: 10GB persistent disk for models
- **Bandwidth**: Limited

**Recommended Plan for Production**:
- **Starter Plan**: $7/month
- **RAM**: 2GB (better performance)
- **Storage**: 10GB disk included

### 🔧 Optimization Tips

1. **Model Selection**
   - Current setup uses lightweight models (FLAN-T5-base, DistilBERT)
   - For better accuracy, upgrade to larger models on paid plans

2. **Cold Start**
   - Free tier apps sleep after 15 minutes of inactivity
   - First request after sleep takes 30-60 seconds
   - Paid plans prevent sleep

3. **Performance**
   - CPU inference is slower than GPU
   - Expect 5-10 seconds per analysis on free tier
   - Consider caching frequent analyses

### 🐛 Troubleshooting

**Build Fails**
- Check build logs in Render dashboard
- Ensure `build.sh` has execute permissions
- Verify all dependencies in `requirements.txt`

**Out of Memory**
- Reduce model size (use smaller FLAN-T5)
- Increase worker timeout in `gunicorn.conf.py`
- Upgrade to paid plan with more RAM

**Slow Response**
- Normal on first request (model loading)
- Subsequent requests should be faster
- Check logs for bottlenecks

**Models Not Loading**
- Verify disk mount in render.yaml
- Check TRANSFORMERS_CACHE path
- Review logs for download errors

### 📚 Alternative Deployment Options

#### Heroku
```bash
# Use Procfile and runtime.txt
heroku create physician-notetaker
git push heroku main
```

#### Railway
- Connect GitHub repo
- Railway auto-detects Flask apps
- Set environment variables manually

#### AWS/GCP/Azure
- Use Docker container (see Dockerfile example below)
- Deploy to Elastic Beanstalk, App Engine, or App Service

### 🐳 Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--config", "gunicorn.conf.py"]
```

Build and run:
```bash
docker build -t physician-notetaker .
docker run -p 5000:5000 physician-notetaker
```

### 🔐 Security Considerations

1. **HTTPS**: Render provides free SSL certificates
2. **CORS**: Configure if building separate frontend
3. **Rate Limiting**: Add Flask-Limiter for production
4. **Input Validation**: Already implemented in app.py
5. **Error Handling**: Comprehensive error handling included

### 📈 Monitoring

**Render Dashboard Provides**:
- Request logs
- Error tracking
- Resource usage
- Deploy history

**Add Application Monitoring**:
- Sentry for error tracking
- LogDNA/Datadog for logs
- New Relic for performance

### 💡 Cost Optimization

**Free Tier Strategy**:
- Use for demos/testing
- Accept cold start delays
- Limited to lightweight models

**Paid Plan Benefits**:
- No sleep (always responsive)
- More RAM (larger models)
- Better CPU allocation
- Custom domains

### 🚀 Next Steps

1. Deploy to Render using steps above
2. Test all features thoroughly
3. Monitor logs and performance
4. Optimize based on usage patterns
5. Consider upgrading plan for production use

### 📞 Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **GitHub Issues**: Report bugs in repository
- **Community**: Render community forum

---

**Ready to Deploy?** Follow the Quick Deploy Steps above! 🎉
