# input_funnel.py

import os
import logging
import mimetypes
import tempfile
from typing import Dict, Any, Optional, Union, BinaryIO, Tuple
from pathlib import Path
import magic  # python-magic for better file type detection
import requests
import io

# Import processing modules
from ocr.image_to_text import OCRProcessor
from audio.asr import ASRProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InputFunnel:
    """
    CasaLingua - Input Funnel Module
    This module detects the input type (text, audio, image, pdf) and routes 
    it to the appropriate preprocessing handler.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 ocr_processor: Optional[OCRProcessor] = None,
                 asr_processor: Optional[ASRProcessor] = None):
        """
        Initialize the Input Funnel.
        
        Args:
            config_path: Path to configuration file
            ocr_processor: Optional preconfigured OCR processor
            asr_processor: Optional preconfigured ASR processor
        """
        self.config = {}
        if config_path and os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded input funnel configuration from {config_path}")
            
        # Initialize OCR processor
        self.ocr_processor = ocr_processor
        if not self.ocr_processor:
            logger.info("Initializing OCR processor")
            self.ocr_processor = OCRProcessor(
                lang=self.config.get("ocr_languages", "eng"),
                dpi=self.config.get("ocr_dpi", 300),
                preprocess=self.config.get("ocr_preprocess", True)
            )
            
        # Initialize ASR processor
        self.asr_processor = asr_processor
        if not self.asr_processor:
            logger.info("Initializing ASR processor")
            self.asr_processor = ASRProcessor(
                model_size=self.config.get("asr_model_size", "base"),
                device=self.config.get("asr_device")
            )
            
        # Initialize mime type detection
        mimetypes.init()
        
        # Define supported file types
        self.supported_types = {
            # Text files
            "text/plain": "text",
            "text/csv": "text",
            "text/markdown": "text",
            "application/json": "text",
            "application/xml": "text",
            
            # Image files
            "image/jpeg": "image",
            "image/png": "image",
            "image/tiff": "image",
            "image/bmp": "image",
            "image/gif": "image",
            
            # PDF files
            "application/pdf": "pdf",
            
            # Audio files
            "audio/wav": "audio",
            "audio/x-wav": "audio",
            "audio/mp3": "audio",
            "audio/mpeg": "audio",
            "audio/m4a": "audio",
            "audio/mp4": "audio",
            "audio/x-m4a": "audio",
            "audio/ogg": "audio",
            "audio/webm": "audio"
        }
        
        logger.info("Input funnel initialized")
    
    def _detect_file_type(self, file_path: str) -> Tuple[str, str]:
        """
        Detect the MIME type and category of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (mime_type, category)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Try to detect mime type using python-magic
        try:
            mime_type = magic.from_file(file_path, mime=True)
        except Exception as e:
            logger.warning(f"Failed to detect mime type with python-magic: {str(e)}")
            
            # Fall back to mimetypes module
            mime_type, _ = mimetypes.guess_type(file_path)
            
        if not mime_type:
            # Last resort: use file extension
            ext = os.path.splitext(file_path)[-1].lower()
            if ext in [".txt", ".csv", ".md", ".json", ".xml"]:
                mime_type = "text/plain"
            elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"]:
                mime_type = f"image/{ext[1:]}"
            elif ext == ".pdf":
                mime_type = "application/pdf"
            elif ext in [".wav", ".mp3", ".m4a", ".ogg"]:
                mime_type = f"audio/{ext[1:]}"
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
                
        # Get category from mime type
        category = self.supported_types.get(mime_type)
        
        if not category:
            raise ValueError(f"Unsupported MIME type: {mime_type}")
            
        return mime_type, category
    
    def process_file(self, file_path: str, detect_language: bool = False) -> Dict[str, Any]:
        """
        Process a file through the input funnel.
        
        Args:
            file_path: Path to the file
            detect_language: Whether to detect the language of the text
            
        Returns:
            Dictionary containing:
                - text: Extracted text
                - file_type: MIME type of the file
                - file_category: Category of the file (text, image, pdf, audio)
                - language: Detected language (if requested)
        """
        logger.info(f"Processing file: {file_path}")
        
        # Detect file type
        mime_type, category = self._detect_file_type(file_path)
        logger.info(f"Detected file type: {mime_type} (category: {category})")
        
        # Process file based on category
        if category == "text":
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
                
        elif category == "image":
            logger.info("Extracting text from image")
            result = self.ocr_processor.extract_text_from_image(file_path)
            text = result["text"]
            
        elif category == "pdf":
            logger.info("Extracting text from PDF")
            result = self.ocr_processor.extract_text_from_pdf(file_path)
            text = result["text"]
            
        elif category == "audio":
            logger.info("Transcribing audio")
            result = self.asr_processor.transcribe_audio(file_path)
            text = result["text"]
            
        else:
            raise ValueError(f"Unsupported file category: {category}")
            
        # Detect language if requested
        language = None
        if detect_language:
            if category == "audio" and hasattr(self.asr_processor, "detect_language"):
                language = self.asr_processor.detect_language(file_path)
            else:
                # Use translation module for language detection
                # This would be implemented elsewhere
                pass
                
        return {
            "text": text,
            "file_type": mime_type,
            "file_category": category,
            "language": language
        }
    
    def process_text(self, text: str, detect_language: bool = False) -> Dict[str, Any]:
        """
        Process direct text input.
        
        Args:
            text: Text input
            detect_language: Whether to detect the language of the text
            
        Returns:
            Dictionary containing:
                - text: Input text
                - file_type: "text/plain"
                - file_category: "text"
                - language: Detected language (if requested)
        """
        logger.info("Processing direct text input")
        
        language = None
        if detect_language:
            # Use translation module for language detection
            # This would be implemented elsewhere
            pass
            
        return {
            "text": text,
            "file_type": "text/plain",
            "file_category": "text",
            "language": language
        }
    
    def process_url(self, url: str, detect_language: bool = False) -> Dict[str, Any]:
        """
        Process a URL by downloading and processing the content.
        
        Args:
            url: URL to process
            detect_language: Whether to detect the language of the text
            
        Returns:
            Dictionary containing processed content
        """
        logger.info(f"Processing URL: {url}")
        
        try:
            # Download the content
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Get content type
            content_type = response.headers.get('content-type', '').split(';')[0]
            
            # Check if supported
            if content_type not in self.supported_types:
                raise ValueError(f"Unsupported content type: {content_type}")
                
            category = self.supported_types[content_type]
            
            # Save to temporary file for processing
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            
            try:
                temp_file.write(response.content)
                temp_file.close()
                
                # Process the temporary file
                result = self.process_file(temp_file.name, detect_language)
                
                # Add URL to result
                result["url"] = url
                
                return result
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                    
        except Exception as e:
            logger.error(f"Failed to process URL: {str(e)}")
            raise
    
    def process_binary_data(self, data: Union[bytes, BinaryIO], 
                          filename: Optional[str] = None,
                          mime_type: Optional[str] = None,
                          detect_language: bool = False) -> Dict[str, Any]:
        """
        Process binary data (e.g., from a file upload).
        
        Args:
            data: Binary data as bytes or file-like object
            filename: Optional filename (used for mime type detection)
            mime_type: Optional explicit MIME type
            detect_language: Whether to detect the language of the text
            
        Returns:
            Dictionary containing processed content
        """
        logger.info("Processing binary data")
        
        # Convert file-like object to bytes if needed
        if hasattr(data, 'read'):
            data = data.read()
            
        # Save to temporary file for processing
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        try:
            if filename:
                # Use provided filename extension
                ext = os.path.splitext(filename)[-1]
                if not ext:
                    ext = ".bin"
                    
                temp_path = temp_file.name + ext
                os.rename(temp_file.name, temp_path)
                temp_file.name = temp_path
                
            temp_file.write(data)
            temp_file.close()
            
            # If mime_type is provided, validate it
            if mime_type and mime_type not in self.supported_types:
                raise ValueError(f"Unsupported MIME type: {mime_type}")
                
            # Process the temporary file
            result = self.process_file(temp_file.name, detect_language)
            
            # Override with provided mime_type if any
            if mime_type:
                result["file_type"] = mime_type
                result["file_category"] = self.supported_types[mime_type]
                
            return result
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def process_input(self, input_data: Union[str, bytes, BinaryIO, Dict[str, Any]],
                    input_type: Optional[str] = None,
                    detect_language: bool = False) -> Dict[str, Any]:
        """
        Main entry point for processing any type of input.
        
        Args:
            input_data: Input data (file path, URL, text, binary data)
            input_type: Optional hint about input type ("file", "url", "text", "binary")
            detect_language: Whether to detect the language of the text
            
        Returns:
            Dictionary containing processed content
        """
        logger.info(f"Processing input with type hint: {input_type}")
        
        # Handle input based on type
        if isinstance(input_data, str):
            if input_type == "file" or (not input_type and os.path.exists(input_data)):
                return self.process_file(input_data, detect_language)
                
            elif input_type == "url" or (not input_type and input_data.startswith(("http://", "https://"))):
                return self.process_url(input_data, detect_language)
                
            else:
                # Assume it's plain text
                return self.process_text(input_data, detect_language)
                
        elif isinstance(input_data, (bytes, io.IOBase)):
            filename = None
            mime_type = None
            
            if isinstance(input_data, dict) and "filename" in input_data:
                filename = input_data["filename"]
                
            if isinstance(input_data, dict) and "mime_type" in input_data:
                mime_type = input_data["mime_type"]
                
            if isinstance(input_data, dict) and "data" in input_data:
                input_data = input_data["data"]
                
            return self.process_binary_data(input_data, filename, mime_type, detect_language)
            
        elif isinstance(input_data, dict):
            # Process structured input
            if "file_path" in input_data:
                return self.process_file(input_data["file_path"], detect_language)
                
            elif "url" in input_data:
                return self.process_url(input_data["url"], detect_language)
                
            elif "text" in input_data:
                return self.process_text(input_data["text"], detect_language)
                
            elif "data" in input_data:
                return self.process_binary_data(
                    input_data["data"],
                    input_data.get("filename"),
                    input_data.get("mime_type"),
                    detect_language
                )
                
            else:
                raise ValueError("Invalid input data dictionary")
                
        else:
            raise TypeError("Unsupported input type")