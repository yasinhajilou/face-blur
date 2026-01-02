# Face Blur API - Backend Deployment Guide

## Oracle Cloud ARM Deployment

This guide covers deploying the Face Blur API backend to Oracle Cloud Infrastructure (OCI) using an ARM-based VM.

### Why Oracle Cloud?

- **Always Free Tier**: 4 ARM OCPUs + 24GB RAM for free
- **ARM Performance**: Excellent for compute-intensive tasks
- **Global Availability**: Good accessibility from various regions
- **No Credit Card Requirement**: For free tier resources

---

## Prerequisites

- Oracle Cloud account (sign up at [cloud.oracle.com](https://cloud.oracle.com))
- SSH client (Terminal on Mac/Linux, PuTTY on Windows)
- Basic Linux command line knowledge

---

## Step 1: Create Oracle Cloud VM

### 1.1 Create Compute Instance

1. Log in to Oracle Cloud Console
2. Go to **Compute** → **Instances** → **Create Instance**
3. Configure:
   - **Name**: `face-blur-api`
   - **Image**: Ubuntu 22.04 (aarch64)
   - **Shape**: VM.Standard.A1.Flex
     - OCPUs: 4
     - Memory: 24 GB
   - **Networking**: Create new VCN or use existing
   - **Add SSH keys**: Upload your public key

4. Click **Create**

### 1.2 Configure Security List

1. Go to **Networking** → **Virtual Cloud Networks**
2. Select your VCN → **Security Lists** → **Default Security List**
3. Add **Ingress Rule**:
   - Source Type: CIDR
   - Source CIDR: `0.0.0.0/0`
   - IP Protocol: TCP
   - Destination Port Range: `8000`
4. Save changes

---

## Step 2: Connect to VM

```bash
# Get your VM's public IP from Oracle Console
ssh -i ~/.ssh/your_private_key ubuntu@YOUR_VM_PUBLIC_IP
```

---

## Step 3: Deploy Application

### 3.1 Upload Files

From your local machine:

```bash
# Create directory on VM
ssh ubuntu@YOUR_VM_IP "mkdir -p /home/ubuntu/face-blur-app"

# Upload application files
scp main.py processor.py requirements.txt setup.sh ubuntu@YOUR_VM_IP:/home/ubuntu/face-blur-app/
```

### 3.2 Run Setup Script

On the VM:

```bash
cd /home/ubuntu/face-blur-app
chmod +x setup.sh
./setup.sh
```

This script will:
- Update system packages
- Install Python 3.11, FFmpeg, and dependencies
- Create Python virtual environment
- Install all Python packages
- Create systemd service
- Configure firewall

### 3.3 Start the Service

```bash
sudo systemctl start faceblur
sudo systemctl status faceblur
```

---

## Step 4: Verify Deployment

### Test Health Endpoint

```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "status": "healthy",
  "service": "Face Blur API",
  "version": "1.0.0"
}
```

### Test from External

```bash
curl http://YOUR_VM_PUBLIC_IP:8000/
```

---

## Step 5: Production Setup (Optional)

### 5.1 Setup Nginx Reverse Proxy

```bash
sudo apt install nginx -y

sudo tee /etc/nginx/sites-available/faceblur > /dev/null <<EOF
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;
    
    client_max_body_size 500M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/faceblur /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### 5.2 Setup SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d YOUR_DOMAIN
```

### 5.3 Update Security List for HTTPS

Add ingress rules for ports 80 and 443.

---

## Monitoring & Maintenance

### View Logs

```bash
# Real-time logs
sudo journalctl -u faceblur -f

# Last 100 lines
sudo journalctl -u faceblur -n 100
```

### Restart Service

```bash
sudo systemctl restart faceblur
```

### Update Application

```bash
# Stop service
sudo systemctl stop faceblur

# Upload new files
# ... scp new files ...

# Start service
sudo systemctl start faceblur
```

### Check Disk Space

```bash
df -h /tmp
# Clean old files if needed
sudo rm -rf /tmp/face-blur/*
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check detailed status
sudo systemctl status faceblur -l

# Check journal for errors
sudo journalctl -u faceblur --no-pager -n 50
```

### Memory Issues

```bash
# Check memory usage
free -h

# Check process memory
ps aux | grep uvicorn
```

### FFmpeg Errors

```bash
# Verify FFmpeg installation
ffmpeg -version
ffprobe -version

# Test with sample video
ffprobe -v error -show_format test_video.mp4
```

### MediaPipe Issues on ARM

If MediaPipe fails to install, try:

```bash
pip install --upgrade pip
pip install mediapipe --no-cache-dir
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/health` | API health check |
| POST | `/api/upload` | Upload video for processing |
| GET | `/api/status/{job_id}` | Get processing status |
| GET | `/api/download/{job_id}` | Download processed video |
| POST | `/api/cleanup/{job_id}` | Clean up job files |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_SIZE` | 500MB | Maximum upload file size |
| `MAX_DURATION_SECONDS` | 300 | Maximum video duration (seconds) |
| `CLEANUP_AFTER_HOURS` | 1 | Auto-cleanup interval |
| `MAX_REQUESTS_PER_HOUR` | 10 | Rate limit per IP |

---

## Cost Estimation

### Oracle Cloud Always Free Tier

- **Compute**: 4 ARM OCPUs + 24GB RAM = **$0/month**
- **Storage**: 200GB block volume = **$0/month**
- **Network**: 10TB egress = **$0/month**

Total: **FREE** (within free tier limits)

---

## Security Recommendations

1. **Never expose port 8000 directly** - Use Nginx reverse proxy
2. **Enable HTTPS** - Use Let's Encrypt certificates
3. **Restrict CORS** - Update `ALLOWED_ORIGINS` in main.py
4. **Monitor logs** - Set up log rotation
5. **Regular updates** - Keep system packages updated

```bash
# Weekly update script
sudo apt update && sudo apt upgrade -y
pip install --upgrade -r requirements.txt
sudo systemctl restart faceblur
```

---

## Support

For issues and questions:
- Create an issue on GitHub
- Check Oracle Cloud documentation
- Review FastAPI documentation
