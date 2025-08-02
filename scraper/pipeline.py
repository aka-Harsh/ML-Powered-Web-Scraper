"""
ETL Pipeline for web scraping and ML processing
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
import json

from scraper.engines import MultiEngine
from ml.nlp_processor import NLPProcessor
from ml.neural_networks import ContentClassifier
from ml.ollama_client import OllamaClient
from database.models import ScrapingJob, ScrapedContent
from database.connection import get_db_session
from storage.s3_client import S3Client
from utils.validators import validate_url, validate_content
from config.settings import settings

logger = logging.getLogger(__name__)


class ScrapingPipeline:
    """Main ETL pipeline for web scraping and ML processing"""
    
    def __init__(self):
        self.scraper = MultiEngine()
        self.nlp_processor = NLPProcessor()
        self.classifier = ContentClassifier()
        self.ollama_client = OllamaClient()
        self.s3_client = S3Client()
        self.active_jobs = {}
    
    async def process_urls(
        self,
        urls: List[str],
        job_id: Optional[str] = None,
        engine_type: str = 'auto',
        ml_analysis: bool = True,
        ollama_model: str = None,
        callback=None
    ) -> Dict[str, Any]:
        """
        Process multiple URLs through the complete pipeline
        
        Args:
            urls: List of URLs to process
            job_id: Optional job identifier
            engine_type: Scraping engine type
            ml_analysis: Whether to perform ML analysis
            ollama_model: Ollama model to use
            callback: Callback function for progress updates
        
        Returns:
            Dictionary containing job results
        """
        
        if not job_id:
            job_id = str(uuid.uuid4())
        
        logger.info(f"Starting pipeline job {job_id} with {len(urls)} URLs")
        
        # Initialize job tracking
        job_data = {
            'job_id': job_id,
            'status': 'running',
            'total_urls': len(urls),
            'processed_urls': 0,
            'failed_urls': 0,
            'results': [],
            'start_time': datetime.now(),
            'errors': []
        }
        
        self.active_jobs[job_id] = job_data
        
        try:
            # Step 1: Validate URLs
            valid_urls = []
            for url in urls:
                if validate_url(url):
                    valid_urls.append(url)
                else:
                    job_data['errors'].append(f"Invalid URL: {url}")
                    job_data['failed_urls'] += 1
            
            if callback:
                await callback(job_data)
            
            # Step 2: Scrape content
            scraped_results = []
            batch_size = min(len(valid_urls), settings.MAX_CONCURRENT_SCRAPES)
            
            for i in range(0, len(valid_urls), batch_size):
                batch_urls = valid_urls[i:i + batch_size]
                batch_results = await self.scraper.batch_scrape(batch_urls, engine_type)
                
                for result in batch_results:
                    if 'error' not in result:
                        scraped_results.append(result)
                        job_data['processed_urls'] += 1
                    else:
                        job_data['errors'].append(f"Scraping failed for {result['url']}: {result['error']}")
                        job_data['failed_urls'] += 1
                
                if callback:
                    await callback(job_data)
                
                # Small delay between batches
                await asyncio.sleep(0.5)
            
            # Step 3: ML Processing
            if ml_analysis and scraped_results:
                for i, result in enumerate(scraped_results):
                    try:
                        # NLP Processing
                        nlp_analysis = await self.nlp_processor.analyze_content(
                            result['text_content'],
                            result['title']
                        )
                        
                        # Neural Network Classification
                        classification = await self.classifier.classify_content(
                            result['text_content']
                        )
                        
                        # Ollama Analysis
                        ollama_analysis = {}
                        if ollama_model:
                            ollama_analysis = await self.ollama_client.analyze_content(
                                result['text_content'],
                                model=ollama_model
                            )
                        
                        # Combine all analysis
                        enhanced_result = {
                            **result,
                            'nlp_analysis': nlp_analysis,
                            'classification': classification,
                            'ollama_analysis': ollama_analysis,
                            'analysis_timestamp': datetime.now().isoformat()
                        }
                        
                        job_data['results'].append(enhanced_result)
                        
                        if callback:
                            await callback(job_data)
                            
                    except Exception as e:
                        logger.error(f"ML analysis failed for {result['url']}: {e}")
                        job_data['errors'].append(f"ML analysis failed for {result['url']}: {str(e)}")
                        # Still add the basic result without ML analysis
                        job_data['results'].append(result)
            else:
                job_data['results'] = scraped_results
            
            # Step 4: Store results in database
            try:
                await self._store_results(job_data)
            except Exception as e:
                logger.error(f"Database storage failed: {e}")
                job_data['errors'].append(f"Database storage failed: {str(e)}")
            
            # Step 5: Export to S3 (optional)
            try:
                export_url = await self._export_results(job_data)
                job_data['export_url'] = export_url
            except Exception as e:
                logger.warning(f"S3 export failed: {e}")
                job_data['errors'].append(f"S3 export failed: {str(e)}")
            
            # Complete job
            job_data['status'] = 'completed'
            job_data['end_time'] = datetime.now()
            job_data['duration'] = (job_data['end_time'] - job_data['start_time']).total_seconds()
            
            logger.info(f"Pipeline job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline job {job_id} failed: {e}")
            job_data['status'] = 'failed'
            job_data['end_time'] = datetime.now()
            job_data['errors'].append(f"Pipeline failure: {str(e)}")
        
        finally:
            if callback:
                await callback(job_data)
        
        return job_data
    
    async def _store_results(self, job_data: Dict[str, Any]) -> None:
        """Store results in database"""
        async with get_db_session() as session:
            try:
                # Create scraping job record
                job_record = ScrapingJob(
                    job_id=job_data['job_id'],
                    status=job_data['status'],
                    total_urls=job_data['total_urls'],
                    processed_urls=job_data['processed_urls'],
                    failed_urls=job_data['failed_urls'],
                    start_time=job_data['start_time'],
                    end_time=job_data.get('end_time'),
                    errors=json.dumps(job_data['errors']),
                    job_metadata=json.dumps({
                        'duration': job_data.get('duration'),
                        'export_url': job_data.get('export_url')
                    })
                )
                
                session.add(job_record)
                
                # Store individual results
                for result in job_data['results']:
                    content_record = ScrapedContent(
                        job_id=job_data['job_id'],
                        url=result['url'],
                        title=result.get('title', ''),
                        content=result.get('text_content', ''),
                        word_count=result.get('word_count', 0),
                        scraping_engine=result.get('engine_used', 'unknown'),
                        scraped_at=datetime.fromtimestamp(result.get('scraped_at', 0)),
                        nlp_analysis=json.dumps(result.get('nlp_analysis', {})),
                        classification=json.dumps(result.get('classification', {})),
                        ollama_analysis=json.dumps(result.get('ollama_analysis', {})),
                        content_metadata=json.dumps({
                            'links_count': len(result.get('links', [])),
                            'images_count': len(result.get('images', [])),
                            'headers_count': len(result.get('headers', []))
                        })
                    )
                    session.add(content_record)
                
                await session.commit()
                logger.info(f"Successfully stored {len(job_data['results'])} results in database")
                
            except Exception as e:
                await session.rollback()
                raise e
    
    async def _export_results(self, job_data: Dict[str, Any]) -> str:
        """Export results to S3"""
        try:
            # Prepare export data
            export_data = {
                'job_summary': {
                    'job_id': job_data['job_id'],
                    'status': job_data['status'],
                    'total_urls': job_data['total_urls'],
                    'processed_urls': job_data['processed_urls'],
                    'failed_urls': job_data['failed_urls'],
                    'start_time': job_data['start_time'].isoformat() if job_data.get('start_time') else None,
                    'end_time': job_data['end_time'].isoformat() if job_data.get('end_time') else None,
                    'duration': job_data.get('duration'),
                    'errors': job_data['errors']
                },
                'results': job_data['results']
            }
            
            # Upload to S3
            filename = f"scraping-results/{job_data['job_id']}.json"
            s3_url = await self.s3_client.upload_json(export_data, filename)
            
            logger.info(f"Successfully exported results to S3: {s3_url}")
            return s3_url
            
        except Exception as e:
            logger.error(f"S3 export failed: {e}")
            raise e
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active job"""
        return self.active_jobs.get(job_id)
    
    def get_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all active jobs"""
        return self.active_jobs.copy()
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active job"""
        if job_id in self.active_jobs:
            self.active_jobs[job_id]['status'] = 'cancelled'
            return True
        return False


class ContentProcessor:
    """Specialized processor for individual content items"""
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.classifier = ContentClassifier()
        self.ollama_client = OllamaClient()
    
    async def process_content(
        self,
        content: str,
        title: str = "",
        url: str = "",
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process individual content through ML pipeline
        
        Args:
            content: Text content to process
            title: Optional title
            url: Optional source URL
            options: Processing options
        
        Returns:
            Dictionary containing analysis results
        """
        
        if not options:
            options = {}
        
        # Validate content
        if not validate_content(content):
            raise ValueError("Invalid content provided")
        
        results = {
            'content_length': len(content),
            'word_count': len(content.split()),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        try:
            # NLP Analysis
            if options.get('enable_nlp', True):
                nlp_results = await self.nlp_processor.analyze_content(content, title)
                results['nlp_analysis'] = nlp_results
            
            # Neural Network Classification
            if options.get('enable_classification', True):
                classification_results = await self.classifier.classify_content(content)
                results['classification'] = classification_results
            
            # Ollama Analysis
            if options.get('enable_ollama', True) and options.get('ollama_model'):
                ollama_results = await self.ollama_client.analyze_content(
                    content,
                    model=options['ollama_model']
                )
                results['ollama_analysis'] = ollama_results
            
            # Quality Assessment
            if options.get('enable_quality_check', True):
                quality_score = await self._assess_content_quality(content, results)
                results['quality_score'] = quality_score
            
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            results['processing_error'] = str(e)
        
        return results
    
    async def _assess_content_quality(self, content: str, analysis_results: Dict) -> float:
        """Assess content quality based on various metrics"""
        quality_factors = []
        
        # Length factor
        length_score = min(len(content) / 1000, 1.0)  # Normalized to 1000 chars
        quality_factors.append(length_score * 0.2)
        
        # Readability factor (simplified)
        sentences = content.count('.') + content.count('!') + content.count('?')
        words = len(content.split())
        if sentences > 0:
            avg_sentence_length = words / sentences
            readability_score = max(0, 1 - abs(avg_sentence_length - 15) / 15)  # Optimal ~15 words
            quality_factors.append(readability_score * 0.3)
        
        # Sentiment factor (if available)
        if 'nlp_analysis' in analysis_results and 'sentiment' in analysis_results['nlp_analysis']:
            sentiment_score = abs(analysis_results['nlp_analysis']['sentiment'].get('compound', 0))
            quality_factors.append(sentiment_score * 0.2)
        
        # Classification confidence (if available)
        if 'classification' in analysis_results and 'confidence' in analysis_results['classification']:
            conf_score = analysis_results['classification']['confidence']
            quality_factors.append(conf_score * 0.3)
        
        # Calculate final quality score
        if quality_factors:
            return sum(quality_factors) / len(quality_factors)
        else:
            return 0.5  # Default neutral score


# Global pipeline instance
pipeline = ScrapingPipeline()