# 🚀 ML-Powered Web Scraper - Setup Guide

## 📋 Prerequisites

Before setting up this project, ensure you have:

- **Python 3.8+** installed
- **Ollama** installed and running
- **Chrome/Chromium** browser (for JavaScript scraping)
- **AWS Account** with S3 and RDS access (optional but recommended for Database)
- **Git** for cloning the repository

## 🛠️ Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ml-web-scraper.git
cd ml-web-scraper
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Additional Dependencies

#### Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

#### Download NLTK Data
```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
print('NLTK data downloaded successfully')
"
```

#### Install ChromeDriver
- Download ChromeDriver from https://chromedriver.chromium.org/
- Add to your system PATH or place in project directory

### 5. Setup Ollama

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Pull required models
ollama pull llama2:latest
ollama pull llama3.1:8b

# Verify installation
ollama list
```

### 6. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration (Optional - for AWS RDS)
DB_HOST=your-rds-endpoint.amazonaws.com
DB_PORT=5432
DB_NAME=scraper_production
DB_USER=scraper_admin
DB_PASSWORD=your-secure-password

# AWS Configuration (Optional - for S3 exports)
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-s3-bucket-name

# Ollama Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
DEFAULT_MODEL=llama3.1:8b
BACKUP_MODEL=llama2:latest

# Application Settings
APP_HOST=localhost
APP_PORT=7860
DEBUG=True
MAX_CONCURRENT_SCRAPES=5
MAX_URLS_PER_BATCH=100

# Security (Generate a secure secret key)
SECRET_KEY=your-super-secret-key-change-this
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 7. AWS Setup (Optional)

#### Create AWS RDS PostgreSQL Database

1. **Go to AWS RDS Console**
2. **Create Database:**
   - Choose "Standard create"
   - Engine: PostgreSQL 15.4-R2
   - Template: Free tier
   - DB instance identifier: `ml-scraper-db`
   - Master username: `scraper_admin`
   - Master password: Set your secure password
   - DB instance class: `db.t3.micro`
   - Storage: 20 GiB (General Purpose SSD)
   - Public access: Yes
   - Initial database name: `scraper_production`

#### Create AWS S3 Bucket

1. **Go to AWS S3 Console**
2. **Create Bucket:**
   - Bucket name: `ml-scraper-exports-yourname` (globally unique)
   - Region: US East (N. Virginia) us-east-1
   - Block public access: Keep default settings
   - Versioning: Disable

## 🏃 Running the Application

### 1. Start Ollama (if not running)
```bash
ollama serve
```

### 2. Initialize Database (if using AWS RDS)
```bash
python -c "
import asyncio
from database.connection import init_database
asyncio.run(init_database())
print('Database initialized successfully!')
"
```

### 3. Start the Application
```bash
python main.py
```

### 4. Access the Interface
- **Main Interface**: http://localhost:7860
- **API Documentation**: http://localhost:7861/docs

## 🧪 Testing the Setup

### 1. Test Web Scraping
1. Go to "🔍 Web Scraping" tab
2. Enter test URL: `https://en.wikipedia.org/wiki/Artificial_intelligence`
3. Click "🚀 Start Scraping"
4. Wait for completion and click "🔄 Auto-Refresh Latest"

### 2. Test Content Analysis
1. Go to "🧠 Content Analysis" tab
2. Paste sample content
3. Click "🔍 Analyze Content"
4. Review analysis results

### 3. Test AI Chat
1. Go to "💬 AI Chat" tab
2. Ask: "Analyze the latest scraping results"
3. Verify AI responds with actual data insights

## 🔧 Troubleshooting

### Common Issues

#### Ollama Not Available
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Pull models if missing
ollama pull llama3.1:8b
```

#### ChromeDriver Issues
- Ensure Chrome browser is installed
- Download correct ChromeDriver version
- Add ChromeDriver to system PATH

#### Import Errors
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Install specific packages
pip install textblob nltk spacy
```

#### Database Connection (if using AWS)
- Verify RDS endpoint in `.env`
- Check security group allows your IP
- Ensure database is running and accessible

### Performance Optimization

#### Memory Management
- Reduce `MAX_CONCURRENT_SCRAPES` if experiencing memory issues
- Adjust `MAX_URLS_PER_BATCH` for large scraping jobs
- Monitor system resources during operation

#### Storage Management
- Regularly clean up `exports/` folder
- Monitor S3 usage if using AWS storage
- Use the built-in cleanup functionality

## 📈 Advanced Configuration

### Custom ML Models
- Modify `ml/neural_networks.py` for custom models
- Train on your specific content types
- Adjust classification categories in `database/models.py`

### Scaling Considerations
- Use Docker for containerized deployment
- Implement load balancing for multiple instances
- Consider using managed services for production

## 🔒 Security Notes

### For Production Deployment
- Use strong passwords and secret keys
- Enable HTTPS/SSL
- Implement proper authentication
- Regular security updates
- Use environment-specific configurations

### API Security
- Rate limiting for endpoints
- Input validation and sanitization
- Monitor for suspicious activity

## 📚 Additional Resources

- **Gradio Documentation**: https://gradio.app/docs/
- **Ollama Documentation**: https://ollama.ai/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **spaCy Documentation**: https://spacy.io/
- **NLTK Documentation**: https://www.nltk.org/

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs for detailed error messages
3. Verify all prerequisites are properly installed
4. Check GitHub issues for similar problems
5. Create a new issue with detailed error information

---

**Congrats on your implemetations !** 🎉