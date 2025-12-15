"""Gunicorn configuration for production deployment"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = 1  # Use 1 worker due to heavy ML models
worker_class = 'sync'
worker_connections = 1000
timeout = 300  # 5 minutes for model loading
keepalive = 2

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'physician-notetaker'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Memory management
max_requests = 100  # Restart worker after N requests to prevent memory leaks
max_requests_jitter = 10
