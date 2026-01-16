# Docker Deployment Guide

This guide covers deploying the Physician Notetaker application using Docker on DigitalOcean.

## Quick Start

### Local Testing

1. **Build the Docker image:**
   ```bash
   docker build -t physician-notetaker .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8080:8080 physician-notetaker
   ```

3. **Access the application:**
   Open http://localhost:8080 in your browser

### Using Docker Compose (Recommended for local dev)

```bash
docker-compose up -d
```

To stop:
```bash
docker-compose down
```

## DigitalOcean Deployment Options

### Option 1: DigitalOcean App Platform (Easiest)

1. **Push your code to GitHub**

2. **Create a new App on DigitalOcean:**
   - Go to https://cloud.digitalocean.com/apps
   - Click "Create App"
   - Select your GitHub repository
   - Choose "Dockerfile" as the source
   - Set the HTTP port to `8080`
   - Choose instance size: **Basic - $12/mo minimum** (2GB RAM recommended for ML models)

3. **Environment Variables (Optional):**
   - `PORT=8080` (auto-set by DO)
   - `PYTORCH_DEVICE=cpu`
   - `MODEL_CACHE_DIR=/app/.cache`

4. **Deploy:** Click "Create Resources"

### Option 2: DigitalOcean Container Registry + Droplet

#### Step 1: Setup Container Registry

1. **Create a Container Registry:**
   ```bash
   doctl registry create physician-notetaker
   ```

2. **Login to registry:**
   ```bash
   doctl registry login
   ```

3. **Build and push image:**
   ```bash
   docker build -t registry.digitalocean.com/<your-registry>/physician-notetaker:latest .
   docker push registry.digitalocean.com/<your-registry>/physician-notetaker:latest
   ```

#### Step 2: Deploy to Droplet

1. **Create a Droplet:**
   - Size: At least **2GB RAM** (Basic $12/mo)
   - Image: Ubuntu 22.04 with Docker pre-installed

2. **SSH into your droplet:**
   ```bash
   ssh root@your-droplet-ip
   ```

3. **Login to registry on the droplet:**
   ```bash
   doctl registry login
   ```

4. **Pull and run the container:**
   ```bash
   docker pull registry.digitalocean.com/<your-registry>/physician-notetaker:latest
   
   docker run -d \
     --name physician-notetaker \
     -p 80:8080 \
     --restart unless-stopped \
     registry.digitalocean.com/<your-registry>/physician-notetaker:latest
   ```

5. **Access your app:**
   Open http://your-droplet-ip in your browser

### Option 3: DigitalOcean Kubernetes (For Production Scale)

1. **Create a Kubernetes cluster**

2. **Create deployment.yaml:**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: physician-notetaker
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: physician-notetaker
     template:
       metadata:
         labels:
           app: physician-notetaker
       spec:
         containers:
         - name: physician-notetaker
           image: registry.digitalocean.com/<your-registry>/physician-notetaker:latest
           ports:
           - containerPort: 8080
           resources:
             requests:
               memory: "2Gi"
               cpu: "500m"
             limits:
               memory: "4Gi"
               cpu: "2000m"
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: physician-notetaker
   spec:
     type: LoadBalancer
     ports:
     - port: 80
       targetPort: 8080
     selector:
       app: physician-notetaker
   ```

3. **Deploy:**
   ```bash
   kubectl apply -f deployment.yaml
   ```

## Resource Requirements

- **Minimum:** 2GB RAM, 1 vCPU
- **Recommended:** 4GB RAM, 2 vCPU
- **Storage:** 10GB minimum (for model cache)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8080 | Port the application listens on |
| PYTORCH_DEVICE | cpu | Device for PyTorch (cpu/cuda) |
| MODEL_CACHE_DIR | /app/.cache | Directory for model cache |
| TRANSFORMERS_CACHE | /app/.cache/transformers | HuggingFace transformers cache |

## Monitoring & Logs

### View logs:
```bash
docker logs -f physician-notetaker
```

### Check health:
```bash
curl http://localhost:8080/
```

### Container stats:
```bash
docker stats physician-notetaker
```

## Troubleshooting

### Container won't start:
```bash
docker logs physician-notetaker
```

### Out of memory:
Upgrade to a larger droplet (4GB+ RAM recommended)

### Models not loading:
Ensure persistent volume is mounted for model cache:
```bash
docker run -v model_cache:/app/.cache ...
```

### Port already in use:
Change the host port:
```bash
docker run -p 8081:8080 physician-notetaker
```

## Updating the Application

1. **Rebuild the image:**
   ```bash
   docker build -t physician-notetaker:latest .
   ```

2. **Stop old container:**
   ```bash
   docker stop physician-notetaker
   docker rm physician-notetaker
   ```

3. **Run new container:**
   ```bash
   docker run -d --name physician-notetaker -p 8080:8080 physician-notetaker:latest
   ```

Or with Docker Compose:
```bash
docker-compose up -d --build
```

## Cost Estimate (DigitalOcean)

- **App Platform:** $12-25/mo (Basic to Professional)
- **Droplet (2GB):** $12/mo
- **Droplet (4GB):** $24/mo
- **Container Registry:** $20/mo (first 500GB free)
- **Kubernetes:** Starts at $12/mo per node

## Security Best Practices

1. Use non-root user (included in Dockerfile)
2. Keep base image updated
3. Use DigitalOcean's private networking
4. Enable firewall rules
5. Use HTTPS with Let's Encrypt
6. Set up automated backups

## Next Steps

- Set up SSL/TLS with Let's Encrypt
- Configure domain name
- Set up monitoring with DigitalOcean Monitoring
- Configure automated backups
- Set up CI/CD pipeline with GitHub Actions
