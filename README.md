# 🤖 ML-Powered Web Scraper - Advanced Intelligence Platform

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-red.svg)
![AWS](https://img.shields.io/badge/AWS-Cloud-yellow.svg)
![Ollama](https://img.shields.io/badge/Ollama-AI-purple.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

A comprehensive, production-ready web scraping system that combines intelligent multi-engine scraping with advanced ML analysis, local AI processing using Ollama, and enterprise-grade cloud storage integration. Built with PyTorch neural networks, PostgreSQL database, and modern FastAPI architecture to deliver end-to-end web content intelligence and automated insights generation.

---

## Video Demo

### 🎥 Here you can find a video of the updated project.

https://github.com/user-attachments/assets/9aeff8b8-297d-4da7-a3f2-3d08ef0a7d61

---

## ✨ Core Features

### 🕷️ **Intelligent Web Scraping Engine**
- **Multi-Engine Architecture**: Seamlessly handles static HTML, JavaScript-heavy sites, and dynamic content
- **Smart Auto-Detection**: Automatically selects optimal scraping approach based on site analysis
- **Concurrent Processing**: Scrape multiple URLs simultaneously with intelligent throttling
- **Real-Time Monitoring**: Live progress updates with detailed status tracking and error reporting
- **Adaptive Retry Logic**: Intelligent failure recovery with exponential backoff strategies

### 🧠 **Advanced ML & AI Analysis**
- **Local AI Processing**: Ollama integration (llama2:latest, llama3.1:8b) - completely cost-free inference
- **Neural Network Pipeline**: Custom PyTorch models for content classification and quality assessment
- **NLP Intelligence**: Advanced sentiment analysis, entity extraction, and topic modeling
- **Content Categorization**: Automated classification across Technology, Business, Science, and more
- **Duplicate Detection**: Smart content deduplication using semantic similarity analysis

### 🛡️ **Enterprise-Grade Safety & Quality**
- **Content Validation**: Multi-layer quality assessment with readability scoring
- **Smart Filtering**: Automatic removal of low-quality or irrelevant content
- **SEO Analysis**: Comprehensive SEO metrics and optimization recommendations
- **Data Integrity**: Robust error handling with comprehensive logging and recovery
- **Rate Limiting**: Intelligent request throttling to respect target site policies

### ☁️ **Cloud-Native Architecture**
- **AWS RDS Integration**: Scalable PostgreSQL database with automated backups
- **S3 Storage**: Secure cloud storage for exports and file management
- **Free Tier Optimized**: Designed to work within AWS free tier limitations
- **Multi-Region Support**: Global deployment capabilities with regional optimization
- **Auto-Scaling**: Dynamic resource allocation based on workload demands

### 🎨 **Modern User Experience**
- **Interactive Dashboard**: Clean, responsive Gradio interface with real-time visualizations
- **AI-Powered Chat**: Intelligent assistant for data insights and actionable recommendations
- **Dynamic Charts**: Interactive Plotly visualizations for comprehensive data analysis
- **Export Flexibility**: Multiple formats (JSON, CSV, TXT) with detailed metadata
- **Session Management**: Persistent user sessions with automatic state recovery

### 📊 **Advanced Analytics & Insights**
- **Performance Metrics**: Real-time system monitoring and resource utilization
- **Content Intelligence**: Automated trend analysis and pattern recognition
- **Quality Scoring**: Multi-factor content quality assessment algorithms
- **Competitive Analysis**: Comparative insights across scraped content sources
- **Predictive Analytics**: ML-driven forecasting for content trends and patterns

---

## 📋 Prerequisites

- **Python 3.8+** (3.10+ recommended for optimal performance)
- **16GB RAM** minimum (32GB recommended for large-scale operations)
- **20GB Storage** for models, database, and content cache
- **PostgreSQL** compatible database (AWS RDS recommended)
- **Ollama** for local LLM processing and content enhancement
- **AWS Account** (free tier) for cloud storage and database
- **Chrome/Chromium** browser for JavaScript-heavy site scraping

---


## 🚀 Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ml-powered-web-scraper.git
cd ml-powered-web-scraper
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### 3. Local AI Setup (Ollama)
```bash
# Install Ollama from https://ollama.ai
# Pull required models
ollama pull llama2:latest
ollama pull llama3.1:8b

# Verify installation
ollama list
ollama serve
```

### 4. AWS Cloud Infrastructure
```bash
# 1. Create AWS RDS PostgreSQL instance
# 2. Set up S3 bucket for file storage
# 3. Configure IAM roles and permissions
# 4. Note endpoints and credentials
```

### 5. Database Initialization
```bash
# Create database schema
python -c "from database.models import create_tables; create_tables()"

# Run initial migrations
python database/migrations/init.py
```

### 6. Environment Configuration
```bash
# Copy template and configure
cp .env.template .env

# Edit .env with your credentials
nano .env
```

### 7. Launch Application
```bash
# Run the enhanced startup script
python main.py

# Application will be available at:
# Main UI: http://localhost:7860
# API Docs: http://localhost:7861/docs
```

---

## 🚀 Performance Optimization

### Hardware Recommendations
| Component | Minimum | Recommended | Enterprise |
|-----------|---------|-------------|------------|
| **CPU** | 4 cores | 8+ cores | 16+ cores |
| **RAM** | 8GB | 16GB | 32GB+ |
| **Storage** | 20GB HDD | 50GB SSD | 100GB+ NVMe |
| **Network** | 10 Mbps | 50 Mbps | 100+ Mbps |

### Scraping Performance
| Configuration | Speed | Throughput | Resource Usage |
|---------------|-------|------------|----------------|
| **Single Thread** | 30-60 sec/page | ~60 pages/hour | Low CPU/Memory |
| **5 Threads** | 6-12 sec/page | ~300 pages/hour | Moderate resources |
| **10 Threads** | 3-6 sec/page | ~600 pages/hour | High CPU/Memory |

### Optimization Strategies
- **Connection Pooling**: Reuse HTTP connections for better performance
- **Caching**: Redis integration for frequently accessed data
- **Database Indexing**: Optimized queries for large datasets
- **Batch Processing**: Group operations for improved efficiency

---

## 🚀 Deployment Options

### Local Development
```bash
# Quick start for development
python main.py --dev
```

### Docker Deployment
```bash
# Build and run with Docker
docker-compose up -d
```

### AWS Deployment
```bash
# Deploy to AWS ECS/EC2
aws ecs create-cluster --cluster-name ml-scraper
```

### Kubernetes
```bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/
```

---

## 🔭 Project Outlook

<img width="1919" height="969" alt="Image" src="https://github.com/user-attachments/assets/a776bead-dd0e-4530-b65d-fd7b4f5f9a8d" />
<img width="1919" height="971" alt="Image" src="https://github.com/user-attachments/assets/91ac3d80-83bd-46ca-878c-af4f37583f94" />
<img width="1919" height="968" alt="Image" src="https://github.com/user-attachments/assets/89ea772f-3b91-4b26-bd03-7f294c0a4c8a" />
<img width="1919" height="966" alt="Image" src="https://github.com/user-attachments/assets/9c8fe4a0-40ae-405a-a934-04e25b240452" />
<img width="1919" height="970" alt="Image" src="https://github.com/user-attachments/assets/e9d6db90-6862-48f1-933d-919fc84f58dd" />
<img width="1919" height="972" alt="Image" src="https://github.com/user-attachments/assets/52d9bde5-8dbb-44c5-ab7e-71be3bec762c" />
<img width="1919" height="973" alt="Image" src="https://github.com/user-attachments/assets/43ab85b2-c4fb-43de-803d-9e01f464bf60" />
<img width="1919" height="885" alt="Image" src="https://github.com/user-attachments/assets/a68a5b11-dd17-4401-b0ba-026724b7478e" />
<img width="1560" height="875" alt="Image" src="https://github.com/user-attachments/assets/99e76588-7a72-4a3b-a308-06722e303b03" />
<img width="1160" height="822" alt="Image" src="https://github.com/user-attachments/assets/0794468b-23d0-4518-ac91-b78694611bea" />
<img width="1549" height="868" alt="Image" src="https://github.com/user-attachments/assets/4e911d09-2177-4d47-acb9-dc8f70c82e9a" />
<img width="1919" height="968" alt="Image" src="https://github.com/user-attachments/assets/feede49f-abb7-4554-83f4-adaa3cb34489" />
<img width="1919" height="971" alt="Image" src="https://github.com/user-attachments/assets/87915f0b-27cf-41c2-a8a3-3a4c00d0c582" />
<img width="1919" height="645" alt="Image" src="https://github.com/user-attachments/assets/87c9763b-cde8-4c1e-9186-442eb170c3f6" />
<img width="1919" height="553" alt="Image" src="https://github.com/user-attachments/assets/0d179b00-7d4f-4176-a5b5-3f300a1d2c0a" />
