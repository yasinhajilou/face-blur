#!/bin/bash
# ============================================
# Face Blur API - Oracle Cloud ARM Setup Script
# ============================================
# This script sets up the Face Blur API on an Oracle Cloud ARM VM
# Tested on: Ubuntu 22.04 (aarch64)
# VM Spec: VM.Standard.A1.Flex, 4 OCPUs, 24GB RAM

set -e  # Exit on error

echo "============================================"
echo "Face Blur API - Setup Script"
echo "Oracle Cloud ARM (aarch64) Deployment"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/home/ubuntu/face-blur-app"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="faceblur"

# Step 1: Update system
echo -e "${YELLOW}Step 1: Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Step 2: Install required system packages
echo -e "${YELLOW}Step 2: Installing system dependencies...${NC}"
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    git

# Step 3: Create application directory
echo -e "${YELLOW}Step 3: Creating application directory...${NC}"
mkdir -p $APP_DIR
cd $APP_DIR

# Step 4: Create Python virtual environment
echo -e "${YELLOW}Step 4: Creating Python virtual environment...${NC}"
python3.11 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Step 5: Upgrade pip
echo -e "${YELLOW}Step 5: Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools

# Step 6: Install Python dependencies
echo -e "${YELLOW}Step 6: Installing Python dependencies...${NC}"
pip install \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    aiofiles==23.2.1 \
    python-multipart==0.0.6 \
    mediapipe==0.10.9 \
    opencv-python-headless==4.9.0.80 \
    numpy>=1.24.0

# Step 7: Create temp directory
echo -e "${YELLOW}Step 7: Creating temp directory...${NC}"
sudo mkdir -p /tmp/face-blur
sudo chown ubuntu:ubuntu /tmp/face-blur

# Step 8: Create systemd service
echo -e "${YELLOW}Step 8: Creating systemd service...${NC}"
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Face Blur API
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=$VENV_DIR/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=false

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

# Step 9: Reload systemd and enable service
echo -e "${YELLOW}Step 9: Enabling systemd service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# Step 10: Configure firewall (if ufw is active)
echo -e "${YELLOW}Step 10: Configuring firewall...${NC}"
if command -v ufw &> /dev/null; then
    sudo ufw allow 8000/tcp
    sudo ufw allow 22/tcp
    echo "UFW rules added for port 8000 and 22"
fi

# Step 11: Create log directory
echo -e "${YELLOW}Step 11: Setting up logging...${NC}"
sudo mkdir -p /var/log/faceblur
sudo chown ubuntu:ubuntu /var/log/faceblur

# Step 12: Verify installation
echo -e "${YELLOW}Step 12: Verifying installation...${NC}"
echo ""
echo "Python version:"
python3.11 --version

echo ""
echo "FFmpeg version:"
ffmpeg -version | head -n 1

echo ""
echo "Installed Python packages:"
pip list | grep -E "fastapi|uvicorn|mediapipe|opencv"

# Print completion message
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy your application files to: $APP_DIR"
echo "   - main.py"
echo "   - processor.py"
echo ""
echo "2. Start the service:"
echo "   sudo systemctl start $SERVICE_NAME"
echo ""
echo "3. Check service status:"
echo "   sudo systemctl status $SERVICE_NAME"
echo ""
echo "4. View logs:"
echo "   sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "5. Test the API:"
echo "   curl http://localhost:8000/"
echo ""
echo "6. Don't forget to configure Oracle Cloud Security List:"
echo "   - Allow ingress TCP traffic on port 8000"
echo ""
echo -e "${YELLOW}Your public IP: $(curl -s ifconfig.me)${NC}"
echo ""
