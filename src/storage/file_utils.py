"""
File utility functions for cloud storage operations.

This module provides utility functions for file validation, type detection,
unique filename generation, MIME type detection, and thumbnail generation.
"""

import hashlib
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from .cloud_storage import FileType


class FileUtils:
    """Utility class for file operations and validation."""
    
    # Supported file extensions mapped to FileType
    FILE_TYPE_MAPPING: Dict[str, FileType] = {
        # 3D Model formats
        '.obj': FileType.OBJ,
        '.gltf': FileType.GLTF,
        '.glb': FileType.GLB,
        '.fbx': FileType.FBX,
        '.stl': FileType.STL,
        '.ply': FileType.PLY,
        '.dae': FileType.DAE,
        
        # Image formats
        '.png': FileType.PNG,
        '.jpg': FileType.JPG,
        '.jpeg': FileType.JPEG,
        '.webp': FileType.WEBP,
        
        # Data formats
        '.json': FileType.JSON,
        '.xml': FileType.XML,
        '.yaml': FileType.YAML,
        '.yml': FileType.YAML,
        
        # Archive formats
        '.zip': FileType.ZIP,
        '.tar': FileType.TAR,
        '.tar.gz': FileType.TAR,
        '.tgz': FileType.TAR,
    }
    
    # Allowed file types for upload
    ALLOWED_FILE_TYPES: Set[FileType] = {
        FileType.OBJ, FileType.GLTF, FileType.GLB, FileType.FBX,
        FileType.STL, FileType.PLY, FileType.DAE,
        FileType.PNG, FileType.JPG, FileType.JPEG, FileType.WEBP,
        FileType.JSON, FileType.XML, FileType.YAML,
        FileType.ZIP, FileType.TAR,
    }
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES: Dict[FileType, int] = {
        # 3D Models (up to 500MB)
        FileType.OBJ: 500 * 1024 * 1024,
        FileType.GLTF: 500 * 1024 * 1024,
        FileType.GLB: 500 * 1024 * 1024,
        FileType.FBX: 500 * 1024 * 1024,
        FileType.STL: 100 * 1024 * 1024,
        FileType.PLY: 100 * 1024 * 1024,
        FileType.DAE: 100 * 1024 * 1024,
        
        # Images (up to 50MB)
        FileType.PNG: 50 * 1024 * 1024,
        FileType.JPG: 50 * 1024 * 1024,
        FileType.JPEG: 50 * 1024 * 1024,
        FileType.WEBP: 50 * 1024 * 1024,
        
        # Data files (up to 10MB)
        FileType.JSON: 10 * 1024 * 1024,
        FileType.XML: 10 * 1024 * 1024,
        FileType.YAML: 10 * 1024 * 1024,
        
        # Archives (up to 1GB)
        FileType.ZIP: 1024 * 1024 * 1024,
        FileType.TAR: 1024 * 1024 * 1024,
    }
    
    def __init__(self):
        """Initialize file utilities."""
        # Initialize mimetypes
        mimetypes.init()
        
        # Add custom MIME types for 3D formats
        self._add_custom_mime_types()
    
    def _add_custom_mime_types(self) -> None:
        """Add custom MIME types for 3D file formats."""
        custom_types = {
            '.obj': 'model/obj',
            '.gltf': 'model/gltf+json',
            '.glb': 'model/gltf-binary',
            '.fbx': 'model/fbx',
            '.stl': 'model/stl',
            '.ply': 'model/ply',
            '.dae': 'model/vnd.collada+xml',
        }
        
        for extension, mime_type in custom_types.items():
            mimetypes.add_type(mime_type, extension)
    
    def get_file_type(self, file_path: Path) -> FileType:
        """
        Determine file type from file extension.
        
        Args:
            file_path: Path to the file
        
        Returns:
            FileType enum value
        """
        extension = file_path.suffix.lower()
        
        # Handle compound extensions like .tar.gz
        if file_path.name.lower().endswith(('.tar.gz', '.tgz')):
            return FileType.TAR
        
        return self.FILE_TYPE_MAPPING.get(extension, FileType.UNKNOWN)
    
    def is_file_type_allowed(self, file_type: FileType) -> bool:
        """
        Check if file type is allowed for upload.
        
        Args:
            file_type: FileType to check
        
        Returns:
            True if file type is allowed, False otherwise
        """
        return file_type in self.ALLOWED_FILE_TYPES
    
    def validate_file_size(self, file_path: Path, file_type: FileType) -> bool:
        """
        Validate file size against limits.
        
        Args:
            file_path: Path to the file
            file_type: Type of the file
        
        Returns:
            True if file size is within limits, False otherwise
        """
        if not file_path.exists():
            return False
        
        file_size = file_path.stat().st_size
        max_size = self.MAX_FILE_SIZES.get(file_type, 10 * 1024 * 1024)  # Default 10MB
        
        return file_size <= max_size
    
    def get_content_type(self, file_path: Path) -> str:
        """
        Get MIME content type for a file.
        
        Args:
            file_path: Path to the file
        
        Returns:
            MIME content type string
        """
        content_type, _ = mimetypes.guess_type(str(file_path))
        return content_type or 'application/octet-stream'
    
    def generate_unique_filename(
        self,
        original_filename: str,
        prefix: str = "",
        suffix: str = "",
        use_uuid: bool = True,
    ) -> str:
        """
        Generate a unique filename.
        
        Args:
            original_filename: Original filename
            prefix: Prefix to add to filename
            suffix: Suffix to add before extension
            use_uuid: Whether to include UUID for uniqueness
        
        Returns:
            Unique filename string
        """
        path = Path(original_filename)
        name = path.stem
        extension = path.suffix
        
        # Build new filename
        parts = []
        
        if prefix:
            parts.append(prefix)
        
        parts.append(name)
        
        if suffix:
            parts.append(suffix)
        
        if use_uuid:
            parts.append(str(uuid.uuid4())[:8])
        
        new_name = "_".join(parts)
        return f"{new_name}{extension}"
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe storage.
        
        Args:
            filename: Original filename
        
        Returns:
            Sanitized filename
        """
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        
        # Limit length
        if len(filename) > 255:
            path = Path(filename)
            name = path.stem[:200]  # Leave room for extension and UUID
            filename = f"{name}{path.suffix}"
        
        return filename
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Calculate hash of a file.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use (md5, sha1, sha256)
        
        Returns:
            Hexadecimal hash string
        """
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def get_image_dimensions(self, file_path: Path) -> Optional[Tuple[int, int]]:
        """
        Get image dimensions.
        
        Args:
            file_path: Path to the image file
        
        Returns:
            Tuple of (width, height) or None if not an image
        """
        try:
            with Image.open(file_path) as img:
                return img.size
        except Exception:
            return None
    
    def create_thumbnail(
        self,
        image_path: Path,
        output_path: Path,
        size: Tuple[int, int] = (256, 256),
        quality: int = 85,
    ) -> bool:
        """
        Create a thumbnail from an image.
        
        Args:
            image_path: Path to the source image
            output_path: Path for the thumbnail
            size: Thumbnail size (width, height)
            quality: JPEG quality (1-100)
        
        Returns:
            True if thumbnail was created successfully, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (for PNG with transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Save thumbnail
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                
                return True
                
        except Exception:
            return False
    
    def validate_3d_model(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a 3D model file.
        
        Args:
            file_path: Path to the 3D model file
        
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'file_type': self.get_file_type(file_path),
            'size_bytes': 0,
            'errors': [],
            'warnings': [],
        }
        
        if not file_path.exists():
            result['errors'].append('File does not exist')
            return result
        
        file_size = file_path.stat().st_size
        result['size_bytes'] = file_size
        
        file_type = result['file_type']
        
        # Check if file type is supported
        if not self.is_file_type_allowed(file_type):
            result['errors'].append(f'File type {file_type} is not supported')
            return result
        
        # Check file size
        if not self.validate_file_size(file_path, file_type):
            max_size = self.MAX_FILE_SIZES.get(file_type, 0)
            result['errors'].append(f'File size {file_size} exceeds limit {max_size}')
            return result
        
        # Basic format validation
        try:
            if file_type == FileType.OBJ:
                result.update(self._validate_obj_file(file_path))
            elif file_type in [FileType.GLTF, FileType.GLB]:
                result.update(self._validate_gltf_file(file_path))
            elif file_type == FileType.STL:
                result.update(self._validate_stl_file(file_path))
            else:
                # For other formats, just check if file is readable
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Read first 1KB to check readability
                result['valid'] = True
                
        except Exception as e:
            result['errors'].append(f'File validation error: {str(e)}')
        
        return result
    
    def _validate_obj_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate OBJ file format."""
        result = {'valid': False, 'vertex_count': 0, 'face_count': 0}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                vertex_count = 0
                face_count = 0
                
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if line.startswith('v '):
                        vertex_count += 1
                    elif line.startswith('f '):
                        face_count += 1
                    
                    # Don't read too many lines for large files
                    if line_num > 10000:
                        break
                
                result['vertex_count'] = vertex_count
                result['face_count'] = face_count
                result['valid'] = vertex_count > 0
                
                if vertex_count == 0:
                    result.setdefault('warnings', []).append('No vertices found in OBJ file')
                
        except UnicodeDecodeError:
            result.setdefault('errors', []).append('OBJ file contains invalid characters')
        except Exception as e:
            result.setdefault('errors', []).append(f'OBJ validation error: {str(e)}')
        
        return result
    
    def _validate_gltf_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate glTF file format."""
        result: Dict[str, Any] = {'valid': False}
        
        try:
            if file_path.suffix.lower() == '.gltf':
                # JSON-based glTF
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for required glTF structure
                if 'asset' in data and 'version' in data['asset']:
                    result['valid'] = True
                    result['version'] = data['asset']['version']
                else:
                    if 'errors' not in result:
                        result['errors'] = []
                    result['errors'].append('Invalid glTF structure')
            
            elif file_path.suffix.lower() == '.glb':
                # Binary glTF
                with open(file_path, 'rb') as f:
                    # Check GLB header
                    magic = f.read(4)
                    if magic == b'glTF':
                        result['valid'] = True
                    else:
                        if 'errors' not in result:
                            result['errors'] = []
                        result['errors'].append('Invalid GLB magic number')
            
        except json.JSONDecodeError:
            if 'errors' not in result:
                result['errors'] = []
            result['errors'].append('Invalid JSON in glTF file')
        except Exception as e:
            if 'errors' not in result:
                result['errors'] = []
            result['errors'].append(f'glTF validation error: {str(e)}')
        
        return result
    
    def _validate_stl_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate STL file format."""
        result = {'valid': False, 'triangle_count': 0}
        
        try:
            with open(file_path, 'rb') as f:
                # Read first 80 bytes
                header = f.read(80)
                
                # Check if it's ASCII STL
                if header.startswith(b'solid'):
                    # ASCII STL
                    f.seek(0)
                    content = f.read(1024).decode('utf-8', errors='ignore')
                    if 'facet normal' in content:
                        result['valid'] = True
                        result['format'] = 'ASCII'
                else:
                    # Binary STL
                    triangle_count_bytes = f.read(4)
                    if len(triangle_count_bytes) == 4:
                        triangle_count = int.from_bytes(triangle_count_bytes, byteorder='little')
                        result['triangle_count'] = triangle_count
                        result['valid'] = triangle_count > 0
                        result['format'] = 'Binary'
                
        except Exception as e:
            result.setdefault('errors', []).append(f'STL validation error: {str(e)}')
        
        return result


# Create a singleton instance for convenience
file_utils = FileUtils()

# Export commonly used functions
__all__ = [
    'FileUtils',
    'file_utils',
]
