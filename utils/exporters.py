"""
Data export utilities for various formats
"""

import asyncio
import json
import csv
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io

from config.settings import settings
from storage.s3_client import S3Client
from utils.validators import sanitize_filename

logger = logging.getLogger(__name__)


class DataExporter:
    """Data export utility for multiple formats"""
    
    def __init__(self):
        self.s3_client = S3Client()
        self.export_dir = Path(settings.EXPORT_DIR)
        self.export_dir.mkdir(exist_ok=True)
    
    async def export_job_data(
        self,
        job_id: str,
        format: str,
        export_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export job data in specified format
        
        Args:
            job_id: Job identifier
            format: Export format (json, csv, pdf, xlsx)
            export_id: Export identifier
            filters: Optional filters to apply
            
        Returns:
            Export result dictionary
        """
        
        try:
            # Get job data (this would typically come from database)
            job_data = await self._get_job_data(job_id, filters)
            
            if not job_data:
                raise ValueError(f"Job {job_id} not found")
            
            # Apply filters if specified
            if filters:
                job_data = self._apply_filters(job_data, filters)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = sanitize_filename(f"{job_id}_{timestamp}.{format}")
            file_path = self.export_dir / filename
            
            # Export based on format
            if format == 'json':
                result = await self._export_json(job_data, file_path)
            elif format == 'csv':
                result = await self._export_csv(job_data, file_path)
            elif format == 'pdf':
                result = await self._export_pdf(job_data, file_path)
            elif format == 'xlsx':
                result = await self._export_xlsx(job_data, file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Upload to S3
            s3_key = f"exports/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
            s3_url = await self.s3_client.upload_file(file_path, s3_key)
            
            # Clean up local file
            try:
                file_path.unlink()
            except:
                pass
            
            return {
                'export_id': export_id,
                'job_id': job_id,
                'format': format,
                'filename': filename,
                'file_size': result.get('file_size', 0),
                'record_count': result.get('record_count', 0),
                's3_url': s3_url,
                'status': 'completed',
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Export failed for job {job_id}: {e}")
            return {
                'export_id': export_id,
                'job_id': job_id,
                'format': format,
                'status': 'failed',
                'error': str(e),
                'created_at': datetime.now().isoformat()
            }
    
    async def _get_job_data(self, job_id: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get job data from database or pipeline"""
        # This would typically query the database
        # For now, get from pipeline if active
        from scraper.pipeline import pipeline
        
        job_data = pipeline.get_job_status(job_id)
        
        if not job_data:
            # Try to get from database
            # This would be implemented with actual database queries
            return None
        
        return job_data
    
    def _apply_filters(self, job_data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filters to job data"""
        try:
            filtered_data = job_data.copy()
            results = job_data.get('results', [])
            
            if not results:
                return filtered_data
            
            filtered_results = []
            
            for result in results:
                include_result = True
                
                # Apply filters
                if 'min_word_count' in filters:
                    if result.get('word_count', 0) < filters['min_word_count']:
                        include_result = False
                
                if 'max_word_count' in filters:
                    if result.get('word_count', 0) > filters['max_word_count']:
                        include_result = False
                
                if 'category' in filters:
                    classification = result.get('classification', {})
                    if classification.get('predicted_category') != filters['category']:
                        include_result = False
                
                if 'min_quality_score' in filters:
                    quality_score = result.get('quality_score', 0)
                    if quality_score < filters['min_quality_score']:
                        include_result = False
                
                if 'has_errors' in filters:
                    has_error = 'error' in result
                    if has_error != filters['has_errors']:
                        include_result = False
                
                if include_result:
                    filtered_results.append(result)
            
            filtered_data['results'] = filtered_results
            filtered_data['filtered_count'] = len(filtered_results)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            return job_data
    
    async def _export_json(self, job_data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Export data as JSON"""
        try:
            # Prepare export data
            export_data = {
                'export_metadata': {
                    'export_time': datetime.now().isoformat(),
                    'format': 'json',
                    'version': '1.0'
                },
                'job_summary': {
                    'job_id': job_data.get('job_id'),
                    'status': job_data.get('status'),
                    'total_urls': job_data.get('total_urls', 0),
                    'processed_urls': job_data.get('processed_urls', 0),
                    'failed_urls': job_data.get('failed_urls', 0),
                    'start_time': job_data.get('start_time'),
                    'end_time': job_data.get('end_time'),
                    'duration': job_data.get('duration')
                },
                'results': job_data.get('results', []),
                'errors': job_data.get('errors', [])
            }
            
            # Write JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            
            file_size = file_path.stat().st_size
            record_count = len(export_data['results'])
            
            logger.info(f"JSON export completed: {file_path}")
            
            return {
                'file_size': file_size,
                'record_count': record_count
            }
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            raise
    
    async def _export_csv(self, job_data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Export data as CSV"""
        try:
            results = job_data.get('results', [])
            
            if not results:
                # Create empty CSV
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['No data available'])
                return {'file_size': file_path.stat().st_size, 'record_count': 0}
            
            # Flatten data for CSV
            flattened_data = []
            
            for result in results:
                flat_row = {
                    'url': result.get('url', ''),
                    'title': result.get('title', ''),
                    'word_count': result.get('word_count', 0),
                    'scraping_engine': result.get('engine_used', ''),
                    'scraped_at': result.get('scraped_at', ''),
                    'content_preview': (result.get('text_content', '')[:200] + '...') if result.get('text_content') else ''
                }
                
                # Add NLP analysis data
                nlp_analysis = result.get('nlp_analysis', {})
                if nlp_analysis:
                    flat_row.update({
                        'sentiment': nlp_analysis.get('sentiment', {}).get('overall', {}).get('sentiment', ''),
                        'sentiment_score': nlp_analysis.get('sentiment', {}).get('overall', {}).get('score', ''),
                        'language': nlp_analysis.get('language', {}).get('language', ''),
                        'readability_score': nlp_analysis.get('readability', {}).get('flesch_reading_ease', ''),
                        'category': nlp_analysis.get('category', {}).get('predicted_category', '')
                    })
                
                # Add classification data
                classification = result.get('classification', {})
                if classification:
                    flat_row.update({
                        'predicted_category': classification.get('predicted_category', ''),
                        'classification_confidence': classification.get('confidence', '')
                    })
                
                # Add quality score
                flat_row['quality_score'] = result.get('quality_score', '')
                
                flattened_data.append(flat_row)
            
            # Write CSV file
            if flattened_data:
                df = pd.DataFrame(flattened_data)
                df.to_csv(file_path, index=False, encoding='utf-8')
            
            file_size = file_path.stat().st_size
            record_count = len(flattened_data)
            
            logger.info(f"CSV export completed: {file_path}")
            
            return {
                'file_size': file_size,
                'record_count': record_count
            }
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise
    
    async def _export_xlsx(self, job_data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Export data as Excel file"""
        try:
            results = job_data.get('results', [])
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Job summary sheet
                job_summary = {
                    'Job ID': [job_data.get('job_id', '')],
                    'Status': [job_data.get('status', '')],
                    'Total URLs': [job_data.get('total_urls', 0)],
                    'Processed URLs': [job_data.get('processed_urls', 0)],
                    'Failed URLs': [job_data.get('failed_urls', 0)],
                    'Start Time': [str(job_data.get('start_time', ''))],
                    'End Time': [str(job_data.get('end_time', ''))],
                    'Duration (seconds)': [job_data.get('duration', '')]
                }
                
                summary_df = pd.DataFrame(job_summary)
                summary_df.to_excel(writer, sheet_name='Job Summary', index=False)
                
                if results:
                    # Main results sheet
                    main_data = []
                    
                    for result in results:
                        row = {
                            'URL': result.get('url', ''),
                            'Title': result.get('title', ''),
                            'Word Count': result.get('word_count', 0),
                            'Engine Used': result.get('engine_used', ''),
                            'Scraped At': str(result.get('scraped_at', '')),
                            'Quality Score': result.get('quality_score', '')
                        }
                        
                        # Add analysis results
                        if 'nlp_analysis' in result:
                            nlp = result['nlp_analysis']
                            row.update({
                                'Sentiment': nlp.get('sentiment', {}).get('overall', {}).get('sentiment', ''),
                                'Language': nlp.get('language', {}).get('language', ''),
                                'Readability': nlp.get('readability', {}).get('flesch_reading_ease', '')
                            })
                        
                        if 'classification' in result:
                            classification = result['classification']
                            row.update({
                                'Category': classification.get('predicted_category', ''),
                                'Confidence': classification.get('confidence', '')
                            })
                        
                        main_data.append(row)
                    
                    main_df = pd.DataFrame(main_data)
                    main_df.to_excel(writer, sheet_name='Results', index=False)
                    
                    # Detailed analysis sheet (if available)
                    detailed_data = []
                    for i, result in enumerate(results):
                        if 'ollama_analysis' in result:
                            ollama = result['ollama_analysis']
                            detailed_data.append({
                                'Row': i + 1,
                                'URL': result.get('url', ''),
                                'Ollama Summary': str(ollama.get('comprehensive', {}).get('raw_response', ''))[:500]
                            })
                    
                    if detailed_data:
                        detailed_df = pd.DataFrame(detailed_data)
                        detailed_df.to_excel(writer, sheet_name='Detailed Analysis', index=False)
                
                # Errors sheet
                errors = job_data.get('errors', [])
                if errors:
                    error_df = pd.DataFrame({'Errors': errors})
                    error_df.to_excel(writer, sheet_name='Errors', index=False)
            
            file_size = file_path.stat().st_size
            record_count = len(results)
            
            logger.info(f"Excel export completed: {file_path}")
            
            return {
                'file_size': file_size,
                'record_count': record_count
            }
            
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            raise
    
    async def _export_pdf(self, job_data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Export data as PDF report"""
        try:
            # Create PDF document
            doc = SimpleDocTemplate(str(file_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center
            )
            
            story.append(Paragraph("Web Scraping Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Job Summary
            story.append(Paragraph("Job Summary", styles['Heading2']))
            
            summary_data = [
                ['Job ID', job_data.get('job_id', '')],
                ['Status', job_data.get('status', '')],
                ['Total URLs', str(job_data.get('total_urls', 0))],
                ['Processed URLs', str(job_data.get('processed_urls', 0))],
                ['Failed URLs', str(job_data.get('failed_urls', 0))],
                ['Start Time', str(job_data.get('start_time', ''))[:19]],
                ['End Time', str(job_data.get('end_time', ''))[:19]],
                ['Duration', f"{job_data.get('duration', 0):.2f} seconds"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Results Overview
            results = job_data.get('results', [])
            if results:
                story.append(Paragraph("Results Overview", styles['Heading2']))
                
                # Statistics
                total_words = sum(r.get('word_count', 0) for r in results)
                avg_words = total_words / len(results) if results else 0
                
                # Category distribution
                categories = {}
                for result in results:
                    cat = result.get('classification', {}).get('predicted_category', 'Unknown')
                    categories[cat] = categories.get(cat, 0) + 1
                
                stats_data = [
                    ['Metric', 'Value'],
                    ['Total Results', str(len(results))],
                    ['Total Words', f"{total_words:,}"],
                    ['Average Words per Page', f"{avg_words:.1f}"],
                    ['Most Common Category', max(categories.items(), key=lambda x: x[1])[0] if categories else 'N/A']
                ]
                
                stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(stats_table)
                story.append(Spacer(1, 20))
                
                # Individual Results (first 10)
                story.append(Paragraph("Individual Results (First 10)", styles['Heading2']))
                
                for i, result in enumerate(results[:10]):
                    story.append(Paragraph(f"Result {i+1}", styles['Heading3']))
                    
                    result_text = f"""
                    <b>URL:</b> {result.get('url', '')}<br/>
                    <b>Title:</b> {result.get('title', 'No title')}<br/>
                    <b>Word Count:</b> {result.get('word_count', 0)}<br/>
                    <b>Engine:</b> {result.get('engine_used', 'Unknown')}<br/>
                    """
                    
                    # Add analysis results if available
                    if 'classification' in result:
                        classification = result['classification']
                        result_text += f"<b>Category:</b> {classification.get('predicted_category', 'Unknown')}<br/>"
                        result_text += f"<b>Confidence:</b> {classification.get('confidence', 0):.2f}<br/>"
                    
                    if 'nlp_analysis' in result and 'sentiment' in result['nlp_analysis']:
                        sentiment = result['nlp_analysis']['sentiment'].get('overall', {})
                        result_text += f"<b>Sentiment:</b> {sentiment.get('sentiment', 'Unknown')}<br/>"
                    
                    if 'quality_score' in result:
                        result_text += f"<b>Quality Score:</b> {result['quality_score']:.2f}<br/>"
                    
                    # Content preview
                    content_preview = result.get('text_content', '')[:200]
                    if content_preview:
                        result_text += f"<b>Content Preview:</b> {content_preview}...<br/>"
                    
                    story.append(Paragraph(result_text, styles['Normal']))
                    story.append(Spacer(1, 15))
            
            # Errors section
            errors = job_data.get('errors', [])
            if errors:
                story.append(Paragraph("Errors", styles['Heading2']))
                for i, error in enumerate(errors[:5]):  # Show first 5 errors
                    story.append(Paragraph(f"{i+1}. {error}", styles['Normal']))
                    story.append(Spacer(1, 10))
            
            # Footer
            story.append(Spacer(1, 30))
            footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(footer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            file_size = file_path.stat().st_size
            record_count = len(results)
            
            logger.info(f"PDF export completed: {file_path}")
            
            return {
                'file_size': file_size,
                'record_count': record_count
            }
            
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            raise
    
    async def export_custom_data(
        self,
        data: List[Dict[str, Any]],
        format: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export custom data in specified format
        
        Args:
            data: List of dictionaries to export
            format: Export format
            filename: Output filename
            metadata: Optional metadata to include
            
        Returns:
            Export result dictionary
        """
        
        try:
            # Sanitize filename
            clean_filename = sanitize_filename(filename)
            if not clean_filename.endswith(f'.{format}'):
                clean_filename += f'.{format}'
            
            file_path = self.export_dir / clean_filename
            
            # Prepare export data
            export_data = {
                'metadata': metadata or {},
                'export_time': datetime.now().isoformat(),
                'record_count': len(data),
                'data': data
            }
            
            # Export based on format
            if format == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            
            elif format == 'csv':
                if data:
                    df = pd.DataFrame(data)
                    df.to_csv(file_path, index=False, encoding='utf-8')
                else:
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['No data available'])
            
            elif format == 'xlsx':
                if data:
                    df = pd.DataFrame(data)
                    df.to_excel(file_path, index=False, engine='openpyxl')
                else:
                    # Create empty Excel file
                    pd.DataFrame({'Message': ['No data available']}).to_excel(file_path, index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            file_size = file_path.stat().st_size
            
            # Upload to S3
            s3_key = f"custom-exports/{datetime.now().strftime('%Y/%m/%d')}/{clean_filename}"
            s3_url = await self.s3_client.upload_file(file_path, s3_key)
            
            # Clean up local file
            try:
                file_path.unlink()
            except:
                pass
            
            return {
                'filename': clean_filename,
                'format': format,
                'file_size': file_size,
                'record_count': len(data),
                's3_url': s3_url,
                'status': 'completed',
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Custom export failed: {e}")
            return {
                'filename': filename,
                'format': format,
                'status': 'failed',
                'error': str(e),
                'created_at': datetime.now().isoformat()
            }
    
    def get_export_formats(self) -> List[Dict[str, str]]:
        """Get list of supported export formats"""
        return [
            {
                'format': 'json',
                'name': 'JSON',
                'description': 'JavaScript Object Notation - structured data format',
                'extension': '.json',
                'mime_type': 'application/json'
            },
            {
                'format': 'csv',
                'name': 'CSV',
                'description': 'Comma Separated Values - spreadsheet format',
                'extension': '.csv',
                'mime_type': 'text/csv'
            },
            {
                'format': 'xlsx',
                'name': 'Excel',
                'description': 'Microsoft Excel spreadsheet format',
                'extension': '.xlsx',
                'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            },
            {
                'format': 'pdf',
                'name': 'PDF',
                'description': 'Portable Document Format - formatted report',
                'extension': '.pdf',
                'mime_type': 'application/pdf'
            }
        ]
    
    async def cleanup_old_exports(self, days: int = 7) -> int:
        """
        Clean up old export files
        
        Args:
            days: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        
        try:
            deleted_count = 0
            
            # Clean up local files
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(days=days)
            
            for file_path in self.export_dir.glob('*'):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete {file_path}: {e}")
            
            # Clean up S3 files
            s3_deleted = await self.s3_client.cleanup_old_exports(days)
            deleted_count += s3_deleted
            
            logger.info(f"Cleaned up {deleted_count} old export files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Export cleanup failed: {e}")
            return 0
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics"""
        try:
            local_files = list(self.export_dir.glob('*'))
            local_count = len([f for f in local_files if f.is_file()])
            local_size = sum(f.stat().st_size for f in local_files if f.is_file())
            
            return {
                'local_files': local_count,
                'local_size_bytes': local_size,
                'local_size_mb': round(local_size / (1024 * 1024), 2),
                'export_directory': str(self.export_dir),
                'supported_formats': len(self.get_export_formats())
            }
            
        except Exception as e:
            logger.error(f"Failed to get export stats: {e}")
            return {
                'error': str(e),
                'local_files': 0,
                'local_size_bytes': 0,
                'local_size_mb': 0
            }


# Utility functions for specific export tasks
async def export_analysis_summary(analysis_results: List[Dict[str, Any]], format: str = 'json') -> str:
    """
    Export ML analysis summary
    
    Args:
        analysis_results: List of analysis results
        format: Export format
        
    Returns:
        S3 URL of exported file
    """
    
    try:
        exporter = DataExporter()
        
        # Prepare summary data
        summary_data = []
        
        for result in analysis_results:
            summary_item = {
                'url': result.get('url', ''),
                'title': result.get('title', ''),
                'word_count': result.get('word_count', 0),
                'sentiment': result.get('nlp_analysis', {}).get('sentiment', {}).get('overall', {}).get('sentiment', ''),
                'category': result.get('classification', {}).get('predicted_category', ''),
                'quality_score': result.get('quality_score', 0),
                'analysis_timestamp': result.get('analysis_timestamp', '')
            }
            summary_data.append(summary_item)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_summary_{timestamp}"
        
        # Export data
        export_result = await exporter.export_custom_data(
            summary_data,
            format,
            filename,
            metadata={
                'export_type': 'analysis_summary',
                'total_items': len(summary_data),
                'created_by': 'ML-Web-Scraper'
            }
        )
        
        return export_result.get('s3_url', '')
        
    except Exception as e:
        logger.error(f"Analysis summary export failed: {e}")
        return ''


async def export_performance_metrics(metrics: Dict[str, Any], format: str = 'json') -> str:
    """
    Export system performance metrics
    
    Args:
        metrics: Performance metrics dictionary
        format: Export format
        
    Returns:
        S3 URL of exported file
    """
    
    try:
        exporter = DataExporter()
        
        # Convert metrics to exportable format
        metrics_data = []
        
        for category, data in metrics.items():
            if isinstance(data, dict):
                for metric, value in data.items():
                    metrics_data.append({
                        'category': category,
                        'metric': metric,
                        'value': value,
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_metrics_{timestamp}"
        
        # Export data
        export_result = await exporter.export_custom_data(
            metrics_data,
            format,
            filename,
            metadata={
                'export_type': 'performance_metrics',
                'total_metrics': len(metrics_data),
                'system': 'ML-Web-Scraper'
            }
        )
        
        return export_result.get('s3_url', '')
        
    except Exception as e:
        logger.error(f"Performance metrics export failed: {e}")
        return ''