#!/usr/bin/env python3
"""
ML-Powered Web Scraper - Main Application Entry Point
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from database.connection import init_database
from ui.gradio_app import create_gradio_interface
from api.routes import app as fastapi_app
import uvicorn
import threading
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager"""
    print("🚀 Starting ML-Powered Web Scraper...")
    
    # Initialize database
    try:
        await init_database()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise
    
    # Download required NLTK data
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  NLTK download warning: {e}")
    
    # Verify Ollama connection
    try:
        from ml.ollama_client import OllamaClient
        ollama_client = OllamaClient()
        models = await ollama_client.list_models()
        print(f"✅ Ollama connected. Available models: {models}")
    except Exception as e:
        print(f"⚠️  Ollama connection warning: {e}")
    
    yield
    
    print("🛑 Shutting down ML-Powered Web Scraper...")


def run_fastapi():
    """Run FastAPI server in a separate thread"""
    config = uvicorn.Config(
        fastapi_app,
        host=settings.APP_HOST,
        port=settings.APP_PORT + 1,  # Use port 7861 for API
        log_level="info"
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


def main():
    """Main application entry point"""
    print("🌟 ML-Powered Web Scraper - Starting Application")
    print("=" * 50)
    
    # Start FastAPI server in background thread
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    print(f"🔧 FastAPI server starting on http://{settings.APP_HOST}:{settings.APP_PORT + 1}")
    
    # Create and launch Gradio interface
    try:
        interface = create_gradio_interface()
        print(f"🎨 Gradio interface starting on http://{settings.APP_HOST}:{settings.APP_PORT}")
        
        # Launch Gradio with custom settings
        interface.launch(
            server_name=settings.APP_HOST,
            server_port=settings.APP_PORT,
            share=False,
            debug=settings.DEBUG,
            show_error=True,
            inbrowser=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start Gradio interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()