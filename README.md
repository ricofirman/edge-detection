# edge-detection

# 🔍 Edge Detection with YOLO + OpenVINO

Real-time object detection and edge detection using YOLO model (OpenVINO optimized) with IP camera (DroidCam) integration.

## 📋 Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## ✨ Features
- Real-time object detection using YOLO with OpenVINO acceleration
- Edge detection (Sobel operator) on detected ROIs
- Support for multiple OpenVINO models
- IP camera support (DroidCam)
- FPS counter display
- Adjustable detection parameters

## 🖥️ System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 - 3.11
- **RAM**: 4GB minimum (8GB recommended)
- **CPU**: Intel Core i5 or equivalent (OpenVINO optimized for Intel)

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/edge-detection-yolo.git
cd edge-detection-yolo

# Create Virtual Environment (Recommended)
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

#Install Dependencies
pip install -r requirements.txt

#jika dengan manual
pip install ultralytics==8.0.200
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install openvino==2023.3.0

# Pip Requirements File 
ultralytics==8.0.200
opencv-python==4.8.1.78
numpy==1.24.3
openvino==2023.3.0
torch>=2.0.0
torchvision>=0.15.0



📱 IP Camera Setup (DroidCam)
import cv2
cap = cv2.VideoCapture("http://YOUR_PHONE_IP:4747/video")

