"""
AWS S3 client for file storage and exports
"""

import boto3
import asyncio
import aiofiles
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import mimetypes
from botocore.exceptions import ClientError, NoCredentialsError
import uuid

from config.settings import settings

logger = logging.getLogger(__name__)


class S3Client:
    """AWS S3 client for file operations"""
    
    def __init__(self):
        self.bucket_name = settings.S3_BUCKET_NAME
        self.region = settings.AWS_REGION
        self.s3_client = None
        self.s3_resource = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize S3 client and resource"""
        try:
            # Create S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            
            # Create S3 resource
            self.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            
            logger.info("S3 client initialized successfully")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    async def upload_file(
        self,
        file_path: Union[str, Path],
        s3_key: str,
        metadata: Dict[str, str] = None,
        content_type: str = None
    ) -> str:
        """
        Upload file to S3
        
        Args:
            file_path: Local file path
            s3_key: S3 object key
            metadata: Optional metadata
            content_type: Content type override
            
        Returns:
            S3 URL of uploaded file
        """
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine content type
            if not content_type:
                content_type, _ = mimetypes.guess_type(str(file_path))
                if not content_type:
                    content_type = 'application/octet-stream'
            
            # Prepare metadata
            upload_metadata = {
                'uploaded_at': datetime.now().isoformat(),
                'original_filename': file_path.name,
                'file_size': str(file_path.stat().st_size)
            }
            
            if metadata:
                upload_metadata.update(metadata)
            
            # Upload file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._upload_file_sync,
                str(file_path),
                s3_key,
                content_type,
                upload_metadata
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"File uploaded successfully: {s3_url}")
            return s3_url
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise
    
    def _upload_file_sync(self, file_path: str, s3_key: str, content_type: str, metadata: Dict):
        """Synchronous file upload"""
        self.s3_client.upload_file(
            file_path,
            self.bucket_name,
            s3_key,
            ExtraArgs={
                'ContentType': content_type,
                'Metadata': metadata
            }
        )
    
    async def upload_json(self, data: Dict[str, Any], s3_key: str) -> str:
        """
        Upload JSON data to S3
        
        Args:
            data: Dictionary to upload as JSON
            s3_key: S3 object key
            
        Returns:
            S3 URL of uploaded file
        """
        
        try:
            # Convert data to JSON
            json_content = json.dumps(data, indent=2, default=str)
            
            # Upload JSON content
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._upload_json_sync,
                json_content,
                s3_key
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"JSON data uploaded successfully: {s3_url}")
            return s3_url
            
        except Exception as e:
            logger.error(f"JSON upload failed: {e}")
            raise
    
    def _upload_json_sync(self, json_content: str, s3_key: str):
        """Synchronous JSON upload"""
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=json_content,
            ContentType='application/json',
            Metadata={
                'uploaded_at': datetime.now().isoformat(),
                'content_type': 'json'
            }
        )
    
    async def download_file(self, s3_key: str, local_path: Union[str, Path]) -> bool:
        """
        Download file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local destination path
            
        Returns:
            Success status
        """
        
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._download_file_sync,
                s3_key,
                str(local_path)
            )
            
            logger.info(f"File downloaded successfully: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"File download failed: {e}")
            return False
    
    def _download_file_sync(self, s3_key: str, local_path: str):
        """Synchronous file download"""
        self.s3_client.download_file(self.bucket_name, s3_key, local_path)
    
    async def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Success status
        """
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._delete_file_sync,
                s3_key
            )
            
            logger.info(f"File deleted successfully: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"File deletion failed: {e}")
            return False
    
    def _delete_file_sync(self, s3_key: str):
        """Synchronous file deletion"""
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
    
    async def list_files(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List files in S3 bucket
        
        Args:
            prefix: Key prefix to filter files
            max_keys: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._list_files_sync,
                prefix,
                max_keys
            )
            
            # Process response
            files = []
            for obj in response.get('Contents', []):
                file_info = {
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag'],
                    'url': f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{obj['Key']}"
                }
                files.append(file_info)
            
            logger.info(f"Listed {len(files)} files with prefix: {prefix}")
            return files
            
        except Exception as e:
            logger.error(f"File listing failed: {e}")
            return []
    
    def _list_files_sync(self, prefix: str, max_keys: int):
        """Synchronous file listing"""
        return self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix,
            MaxKeys=max_keys
        )
    
    async def get_file_metadata(self, s3_key: str) -> Dict[str, Any]:
        """
        Get file metadata from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            File metadata dictionary
        """
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._get_file_metadata_sync,
                s3_key
            )
            
            metadata = {
                'content_length': response['ContentLength'],
                'content_type': response['ContentType'],
                'last_modified': response['LastModified'].isoformat(),
                'etag': response['ETag'],
                'metadata': response.get('Metadata', {})
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get file metadata: {e}")
            return {}
    
    def _get_file_metadata_sync(self, s3_key: str):
        """Synchronous metadata retrieval"""
        return self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
    
    async def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        method: str = 'GET'
    ) -> str:
        """
        Generate presigned URL for S3 object
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
            method: HTTP method (GET, PUT, etc.)
            
        Returns:
            Presigned URL
        """
        
        try:
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(
                None,
                self._generate_presigned_url_sync,
                s3_key,
                expiration,
                method
            )
            
            logger.info(f"Generated presigned URL for: {s3_key}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return ""
    
    def _generate_presigned_url_sync(self, s3_key: str, expiration: int, method: str):
        """Synchronous presigned URL generation"""
        return self.s3_client.generate_presigned_url(
            f'{method.lower()}_object',
            Params={'Bucket': self.bucket_name, 'Key': s3_key},
            ExpiresIn=expiration
        )
    
    async def create_export_folder(self, job_id: str) -> str:
        """
        Create a folder structure for exports
        
        Args:
            job_id: Job identifier
            
        Returns:
            Folder prefix
        """
        
        timestamp = datetime.now().strftime("%Y/%m/%d")
        folder_prefix = f"exports/{timestamp}/{job_id}/"
        
        # Create a marker file to ensure folder exists
        marker_key = f"{folder_prefix}.folder_marker"
        
        try:
            await self.upload_json(
                {
                    "job_id": job_id,
                    "created_at": datetime.now().isoformat(),
                    "type": "folder_marker"
                },
                marker_key
            )
            
            logger.info(f"Created export folder: {folder_prefix}")
            return folder_prefix
            
        except Exception as e:
            logger.error(f"Failed to create export folder: {e}")
            return ""
    
    async def health_check(self) -> bool:
        """Check S3 connectivity and permissions"""
        try:
            # Try to list objects (should work with read permissions)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._health_check_sync
            )
            
            return True
            
        except Exception as e:
            logger.error(f"S3 health check failed: {e}")
            return False
    
    def _health_check_sync(self):
        """Synchronous health check"""
        # Simple operation to test connectivity
        self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
    
    async def get_bucket_stats(self) -> Dict[str, Any]:
        """Get bucket statistics"""
        try:
            # List all objects to calculate stats
            files = await self.list_files(max_keys=10000)  # Limit to prevent timeout
            
            if not files:
                return {
                    'total_files': 0,
                    'total_size': 0,
                    'total_size_mb': 0
                }
            
            total_size = sum(file['size'] for file in files)
            
            stats = {
                'total_files': len(files),
                'total_size': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'bucket_name': self.bucket_name,
                'region': self.region
            }
            
            # File type breakdown
            file_types = {}
            for file in files:
                ext = Path(file['key']).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            
            stats['file_types'] = file_types
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get bucket stats: {e}")
            return {'error': str(e)}
    
    async def cleanup_old_exports(self, days: int = 7) -> int:
        """
        Clean up old export files
        
        Args:
            days: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        
        try:
            # List all export files
            export_files = await self.list_files(prefix="exports/")
            
            # Filter old files
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            old_files = []
            for file in export_files:
                file_date = datetime.fromisoformat(file['last_modified'].replace('Z', '+00:00'))
                if file_date.replace(tzinfo=None) < cutoff_date:
                    old_files.append(file['key'])
            
            # Delete old files
            deleted_count = 0
            for file_key in old_files:
                if await self.delete_file(file_key):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old export files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Export cleanup failed: {e}")
            return 0