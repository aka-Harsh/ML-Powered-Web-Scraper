"""
Validation utilities for URLs, content, and data
"""

import re
import validators
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def validate_url(url: str) -> bool:
    """
    Validate URL format and accessibility
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        # Basic URL validation
        if not url or not isinstance(url, str):
            return False
        
        # Remove whitespace
        url = url.strip()
        
        # Check minimum length
        if len(url) < 10:
            return False
        
        # Use validators library for basic validation
        if not validators.url(url):
            return False
        
        # Parse URL for additional checks
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Check domain
        if not parsed.netloc:
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'localhost',
            r'127\.0\.0\.1',
            r'192\.168\.',
            r'10\.',
            r'172\.(1[6-9]|2[0-9]|3[0-1])\.',
            r'file://',
            r'ftp://',
            r'javascript:',
            r'data:'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                logger.warning(f"Suspicious URL pattern detected: {url}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"URL validation error for {url}: {e}")
        return False


def validate_content(content: str) -> bool:
    """
    Validate content for ML analysis
    
    Args:
        content: Text content to validate
        
    Returns:
        True if content is valid, False otherwise
    """
    try:
        # Basic checks
        if not content or not isinstance(content, str):
            return False
        
        # Remove whitespace for length check
        clean_content = content.strip()
        
        # Check minimum length
        if len(clean_content) < settings.MIN_TEXT_LENGTH:
            return False
        
        # Check maximum length
        if len(clean_content) > settings.MAX_TEXT_LENGTH:
            return False
        
        # Check for minimum word count
        words = clean_content.split()
        if len(words) < 3:
            return False
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in clean_content if not c.isalnum() and not c.isspace()) / len(clean_content)
        if special_char_ratio > 0.5:  # More than 50% special characters
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',
            r'data:image',
            r'base64,'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                logger.warning("Suspicious content pattern detected")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Content validation error: {e}")
        return False


def validate_job_data(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate job data structure
    
    Args:
        job_data: Job data dictionary
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Required fields
        required_fields = ['job_id', 'status', 'total_urls']
        
        for field in required_fields:
            if field not in job_data:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False
        
        # Validate job_id
        if 'job_id' in job_data:
            job_id = job_data['job_id']
            if not isinstance(job_id, str) or len(job_id) < 5:
                validation_result['errors'].append("Invalid job_id format")
                validation_result['is_valid'] = False
        
        # Validate status
        if 'status' in job_data:
            valid_statuses = ['pending', 'running', 'completed', 'failed', 'cancelled']
            if job_data['status'] not in valid_statuses:
                validation_result['errors'].append(f"Invalid status: {job_data['status']}")
                validation_result['is_valid'] = False
        
        # Validate URL counts
        for count_field in ['total_urls', 'processed_urls', 'failed_urls']:
            if count_field in job_data:
                count = job_data[count_field]
                if not isinstance(count, int) or count < 0:
                    validation_result['errors'].append(f"Invalid {count_field}: must be non-negative integer")
                    validation_result['is_valid'] = False
        
        # Logical validation
        if all(field in job_data for field in ['total_urls', 'processed_urls', 'failed_urls']):
            total = job_data['total_urls']
            processed = job_data['processed_urls']
            failed = job_data['failed_urls']
            
            if processed + failed > total:
                validation_result['warnings'].append("Processed + failed URLs exceed total URLs")
        
        # Validate results if present
        if 'results' in job_data:
            results = job_data['results']
            if not isinstance(results, list):
                validation_result['errors'].append("Results must be a list")
                validation_result['is_valid'] = False
            else:
                for i, result in enumerate(results):
                    if not isinstance(result, dict):
                        validation_result['errors'].append(f"Result {i} is not a dictionary")
                        validation_result['is_valid'] = False
                    elif 'url' not in result:
                        validation_result['warnings'].append(f"Result {i} missing URL")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Job data validation error: {e}")
        return {
            'is_valid': False,
            'errors': [f"Validation exception: {str(e)}"],
            'warnings': []
        }


def validate_export_request(export_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate export request parameters
    
    Args:
        export_request: Export request dictionary
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Required fields
        if 'job_id' not in export_request:
            validation_result['errors'].append("Missing job_id")
            validation_result['is_valid'] = False
        
        # Validate format
        if 'format' in export_request:
            valid_formats = ['json', 'csv', 'pdf', 'xlsx']
            if export_request['format'] not in valid_formats:
                validation_result['errors'].append(f"Invalid format: {export_request['format']}")
                validation_result['is_valid'] = False
        
        # Validate filters if present
        if 'filters' in export_request:
            filters = export_request['filters']
            if filters is not None and not isinstance(filters, dict):
                validation_result['errors'].append("Filters must be a dictionary")
                validation_result['is_valid'] = False
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Export request validation error: {e}")
        return {
            'is_valid': False,
            'errors': [f"Validation exception: {str(e)}"],
            'warnings': []
        }


def validate_ml_analysis_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate ML analysis request
    
    Args:
        request: ML analysis request dictionary
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Check content
        if 'content' not in request:
            validation_result['errors'].append("Missing content")
            validation_result['is_valid'] = False
        else:
            if not validate_content(request['content']):
                validation_result['errors'].append("Invalid content")
                validation_result['is_valid'] = False
        
        # Validate model if specified
        if 'ollama_model' in request:
            model = request['ollama_model']
            if model is not None:
                valid_models = settings.AVAILABLE_MODELS
                if model not in valid_models:
                    validation_result['warnings'].append(f"Unknown model: {model}")
        
        # Validate boolean flags
        boolean_fields = ['enable_nlp', 'enable_classification', 'enable_ollama']
        for field in boolean_fields:
            if field in request:
                if not isinstance(request[field], bool):
                    validation_result['errors'].append(f"{field} must be boolean")
                    validation_result['is_valid'] = False
        
        return validation_result
        
    except Exception as e:
        logger.error(f"ML analysis request validation error: {e}")
        return {
            'is_valid': False,
            'errors': [f"Validation exception: {str(e)}"],
            'warnings': []
        }


def validate_batch_urls(urls: List[str]) -> Dict[str, Any]:
    """
    Validate batch of URLs
    
    Args:
        urls: List of URL strings
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'valid_urls': [],
        'invalid_urls': [],
        'errors': [],
        'warnings': []
    }
    
    try:
        if not isinstance(urls, list):
            validation_result['errors'].append("URLs must be provided as a list")
            validation_result['is_valid'] = False
            return validation_result
        
        if len(urls) == 0:
            validation_result['errors'].append("No URLs provided")
            validation_result['is_valid'] = False
            return validation_result
        
        if len(urls) > settings.MAX_URLS_PER_BATCH:
            validation_result['errors'].append(f"Too many URLs (max: {settings.MAX_URLS_PER_BATCH})")
            validation_result['is_valid'] = False
            return validation_result
        
        # Validate each URL
        for url in urls:
            if validate_url(url):
                validation_result['valid_urls'].append(url)
            else:
                validation_result['invalid_urls'].append(url)
        
        # Check if we have any valid URLs
        if not validation_result['valid_urls']:
            validation_result['errors'].append("No valid URLs found")
            validation_result['is_valid'] = False
        
        # Warnings for invalid URLs
        if validation_result['invalid_urls']:
            validation_result['warnings'].append(f"{len(validation_result['invalid_urls'])} invalid URLs will be skipped")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Batch URL validation error: {e}")
        return {
            'is_valid': False,
            'valid_urls': [],
            'invalid_urls': [],
            'errors': [f"Validation exception: {str(e)}"],
            'warnings': []
        }


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    try:
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing whitespace and dots
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_length = 255 - len(ext) - 1 if ext else 255
            filename = name[:max_name_length] + ('.' + ext if ext else '')
        
        # Ensure it's not empty
        if not filename:
            filename = 'untitled'
        
        return filename
        
    except Exception as e:
        logger.error(f"Filename sanitization error: {e}")
        return 'untitled'


def validate_pagination(page: int, page_size: int, max_page_size: int = 1000) -> Dict[str, Any]:
    """
    Validate pagination parameters
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        max_page_size: Maximum allowed page size
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'normalized_page': page,
        'normalized_page_size': page_size
    }
    
    try:
        # Validate page number
        if not isinstance(page, int) or page < 1:
            validation_result['errors'].append("Page must be a positive integer")
            validation_result['is_valid'] = False
            validation_result['normalized_page'] = 1
        
        # Validate page size
        if not isinstance(page_size, int) or page_size < 1:
            validation_result['errors'].append("Page size must be a positive integer")
            validation_result['is_valid'] = False
            validation_result['normalized_page_size'] = 10
        elif page_size > max_page_size:
            validation_result['errors'].append(f"Page size too large (max: {max_page_size})")
            validation_result['is_valid'] = False
            validation_result['normalized_page_size'] = max_page_size
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Pagination validation error: {e}")
        return {
            'is_valid': False,
            'errors': [f"Validation exception: {str(e)}"],
            'normalized_page': 1,
            'normalized_page_size': 10
        }


def validate_date_range(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Validate date range parameters
    
    Args:
        start_date: Start date string (ISO format)
        end_date: End date string (ISO format)
        
    Returns:
        Dictionary with validation results
    """
    from datetime import datetime, timedelta
    
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'parsed_start': None,
        'parsed_end': None
    }
    
    try:
        # Parse start date
        if start_date:
            try:
                parsed_start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                validation_result['parsed_start'] = parsed_start
            except ValueError:
                validation_result['errors'].append("Invalid start date format")
                validation_result['is_valid'] = False
        
        # Parse end date
        if end_date:
            try:
                parsed_end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                validation_result['parsed_end'] = parsed_end
            except ValueError:
                validation_result['errors'].append("Invalid end date format")
                validation_result['is_valid'] = False
        
        # Validate date range logic
        if validation_result['parsed_start'] and validation_result['parsed_end']:
            if validation_result['parsed_start'] > validation_result['parsed_end']:
                validation_result['errors'].append("Start date must be before end date")
                validation_result['is_valid'] = False
            
            # Check for reasonable date range (not more than 1 year)
            date_diff = validation_result['parsed_end'] - validation_result['parsed_start']
            if date_diff.days > 365:
                validation_result['warnings'].append("Date range spans more than 1 year")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Date range validation error: {e}")
        return {
            'is_valid': False,
            'errors': [f"Validation exception: {str(e)}"],
            'warnings': [],
            'parsed_start': None,
            'parsed_end': None
        }


def validate_filter_parameters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate filter parameters for data queries
    
    Args:
        filters: Dictionary of filter parameters
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'sanitized_filters': {}
    }
    
    try:
        if not isinstance(filters, dict):
            validation_result['errors'].append("Filters must be a dictionary")
            validation_result['is_valid'] = False
            return validation_result
        
        # Define allowed filter fields and their types
        allowed_filters = {
            'status': str,
            'engine_type': str,
            'category': str,
            'min_quality_score': float,
            'max_quality_score': float,
            'min_word_count': int,
            'max_word_count': int,
            'has_errors': bool,
            'start_date': str,
            'end_date': str
        }
        
        for key, value in filters.items():
            if key not in allowed_filters:
                validation_result['warnings'].append(f"Unknown filter: {key}")
                continue
            
            expected_type = allowed_filters[key]
            
            # Type validation and conversion
            try:
                if expected_type == str:
                    sanitized_value = str(value).strip()
                    if not sanitized_value:
                        validation_result['warnings'].append(f"Empty value for filter: {key}")
                        continue
                elif expected_type == int:
                    sanitized_value = int(value)
                    if sanitized_value < 0:
                        validation_result['warnings'].append(f"Negative value for {key}")
                elif expected_type == float:
                    sanitized_value = float(value)
                    if key in ['min_quality_score', 'max_quality_score']:
                        if not 0 <= sanitized_value <= 1:
                            validation_result['warnings'].append(f"Quality score must be between 0 and 1: {key}")
                elif expected_type == bool:
                    sanitized_value = bool(value)
                else:
                    sanitized_value = value
                
                validation_result['sanitized_filters'][key] = sanitized_value
                
            except (ValueError, TypeError):
                validation_result['errors'].append(f"Invalid type for filter {key}: expected {expected_type.__name__}")
                validation_result['is_valid'] = False
        
        # Cross-field validation
        sanitized = validation_result['sanitized_filters']
        
        # Quality score range validation
        if 'min_quality_score' in sanitized and 'max_quality_score' in sanitized:
            if sanitized['min_quality_score'] > sanitized['max_quality_score']:
                validation_result['errors'].append("min_quality_score cannot be greater than max_quality_score")
                validation_result['is_valid'] = False
        
        # Word count range validation
        if 'min_word_count' in sanitized and 'max_word_count' in sanitized:
            if sanitized['min_word_count'] > sanitized['max_word_count']:
                validation_result['errors'].append("min_word_count cannot be greater than max_word_count")
                validation_result['is_valid'] = False
        
        # Date range validation
        if 'start_date' in sanitized and 'end_date' in sanitized:
            date_validation = validate_date_range(sanitized['start_date'], sanitized['end_date'])
            if not date_validation['is_valid']:
                validation_result['errors'].extend(date_validation['errors'])
                validation_result['is_valid'] = False
            validation_result['warnings'].extend(date_validation['warnings'])
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Filter validation error: {e}")
        return {
            'is_valid': False,
            'errors': [f"Validation exception: {str(e)}"],
            'warnings': [],
            'sanitized_filters': {}
        }


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format (if using API authentication)
    
    Args:
        api_key: API key string
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Basic format validation
        api_key = api_key.strip()
        
        # Check minimum length
        if len(api_key) < 32:
            return False
        
        # Check for valid characters (alphanumeric and some symbols)
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', api_key):
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"API key validation error: {e}")
        return False


def validate_search_query(query: str, max_length: int = 1000) -> Dict[str, Any]:
    """
    Validate search query parameters
    
    Args:
        query: Search query string
        max_length: Maximum query length
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'sanitized_query': ''
    }
    
    try:
        if not query or not isinstance(query, str):
            validation_result['errors'].append("Query must be a non-empty string")
            validation_result['is_valid'] = False
            return validation_result
        
        # Sanitize query
        sanitized_query = query.strip()
        
        # Check length
        if len(sanitized_query) < 1:
            validation_result['errors'].append("Query cannot be empty")
            validation_result['is_valid'] = False
            return validation_result
        
        if len(sanitized_query) > max_length:
            validation_result['warnings'].append(f"Query truncated to {max_length} characters")
            sanitized_query = sanitized_query[:max_length]
        
        # Remove potentially dangerous characters
        dangerous_patterns = [
            r'[<>]',  # HTML/XML tags
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Control characters
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized_query):
                sanitized_query = re.sub(pattern, '', sanitized_query)
                validation_result['warnings'].append("Removed potentially dangerous characters")
        
        validation_result['sanitized_query'] = sanitized_query
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Search query validation error: {e}")
        return {
            'is_valid': False,
            'errors': [f"Validation exception: {str(e)}"],
            'warnings': [],
            'sanitized_query': ''
        }


# Validation decorators
def validate_request(validation_func):
    """
    Decorator for request validation
    
    Args:
        validation_func: Function to validate request data
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract request data from args/kwargs
            request_data = kwargs.get('request') or (args[0] if args else None)
            
            if request_data:
                validation_result = validation_func(request_data)
                if not validation_result['is_valid']:
                    raise ValueError(f"Validation failed: {validation_result['errors']}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Batch validation utilities
def validate_batch_content(contents: List[str]) -> Dict[str, Any]:
    """
    Validate batch of content items
    
    Args:
        contents: List of content strings
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'valid_contents': [],
        'invalid_contents': [],
        'errors': [],
        'warnings': []
    }
    
    try:
        if not isinstance(contents, list):
            validation_result['errors'].append("Contents must be provided as a list")
            validation_result['is_valid'] = False
            return validation_result
        
        if len(contents) == 0:
            validation_result['errors'].append("No content provided")
            validation_result['is_valid'] = False
            return validation_result
        
        # Validate each content item
        for i, content in enumerate(contents):
            if validate_content(content):
                validation_result['valid_contents'].append(content)
            else:
                validation_result['invalid_contents'].append({
                    'index': i,
                    'content': content[:100] + '...' if len(content) > 100 else content
                })
        
        # Check if we have any valid content
        if not validation_result['valid_contents']:
            validation_result['errors'].append("No valid content found")
            validation_result['is_valid'] = False
        
        # Warnings for invalid content
        if validation_result['invalid_contents']:
            validation_result['warnings'].append(f"{len(validation_result['invalid_contents'])} invalid content items will be skipped")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Batch content validation error: {e}")
        return {
            'is_valid': False,
            'valid_contents': [],
            'invalid_contents': [],
            'errors': [f"Validation exception: {str(e)}"],
            'warnings': []
        }