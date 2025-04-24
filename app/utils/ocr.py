# ocr/image_to_text.py

import os
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union, BinaryIO
from pathlib import Path

import pytesseract
from PIL import Image
import pdf2image
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    CasaLingua - OCR Tools
    Handles image and PDF text extraction using OCR.
    """
    
    def __init__(self, 
                 tesseract_cmd: Optional[str] = None, 
                 lang: str = "eng",
                 dpi: int = 300,
                 preprocess: bool = True):
        """
        Initialize OCR processor.
        
        Args:
            tesseract_cmd: Path to tesseract executable (if not in PATH)
            lang: Language code(s) for OCR (e.g., 'eng', 'spa', 'eng+spa')
            dpi: DPI for PDF to image conversion
            preprocess: Whether to apply image preprocessing for better OCR results
        """
        # Set tesseract command path if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        self.lang = lang
        self.dpi = dpi
        self.preprocess = preprocess
        
        # Check if tesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract initialized successfully with language: {lang}")
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {str(e)}")
            raise RuntimeError(f"Tesseract initialization failed: {str(e)}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR results.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply thresholding to get black text on white background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Apply noise removal
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        return denoised
    
    def extract_text_from_image(self, 
                               image_input: Union[str, np.ndarray, Image.Image, BinaryIO],
                               lang: Optional[str] = None,
                               preprocess: Optional[bool] = None) -> Dict[str, Any]:
        """
        Extract text from an image using OCR.
        
        Args:
            image_input: Path to image file, OpenCV image, PIL image, or file-like object
            lang: Language code (overrides init setting if provided)
            preprocess: Whether to preprocess the image (overrides init setting if provided)
            
        Returns:
            Dictionary containing:
                - text: Full extracted text
                - confidence: Overall OCR confidence score
                - blocks: List of text blocks with position data
        """
        logger.info("Starting OCR text extraction from image")
        
        # Use provided settings or fall back to init settings
        ocr_lang = lang or self.lang
        do_preprocess = preprocess if preprocess is not None else self.preprocess
        
        # Handle different input types
        temp_file = None
        try:
            # Case 1: Path string
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                img = cv2.imread(image_input)
                
            # Case 2: File-like object
            elif hasattr(image_input, 'read'):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_file.write(image_input.read())
                temp_file.close()
                img = cv2.imread(temp_file.name)
                
            # Case 3: OpenCV image (numpy array)
            elif isinstance(image_input, np.ndarray):
                img = image_input.copy()
                
            # Case 4: PIL Image
            elif isinstance(image_input, Image.Image):
                img = np.array(image_input)
                # Convert RGB to BGR if needed
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                raise TypeError("Unsupported image input type")
            
            # Preprocess image if enabled
            if do_preprocess:
                img = self._preprocess_image(img)
            
            # Convert to PIL Image for pytesseract
            pil_img = Image.fromarray(img)
            
            # Extract text
            ocr_result = pytesseract.image_to_data(pil_img, lang=ocr_lang, output_type=pytesseract.Output.DICT)
            
            # Extract full text
            text_parts = []
            blocks = []
            confidence_values = []
            
            for i in range(len(ocr_result["text"])):
                # Skip empty text
                if not ocr_result["text"][i].strip():
                    continue
                    
                text_parts.append(ocr_result["text"][i])
                confidence_values.append(ocr_result["conf"][i])
                
                # Add block data
                blocks.append({
                    "text": ocr_result["text"][i],
                    "conf": ocr_result["conf"][i],
                    "x": ocr_result["left"][i],
                    "y": ocr_result["top"][i],
                    "width": ocr_result["width"][i],
                    "height": ocr_result["height"][i],
                    "level": ocr_result["level"][i],
                    "page_num": ocr_result["page_num"][i] if "page_num" in ocr_result else 1
                })
            
            # Calculate average confidence (excluding zeros)
            filtered_conf = [c for c in confidence_values if c > 0]
            avg_confidence = sum(filtered_conf) / len(filtered_conf) if filtered_conf else 0
            
            # Create result dict
            result = {
                "text": " ".join(text_parts),
                "confidence": avg_confidence,
                "blocks": blocks,
                "language": ocr_lang
            }
            
            logger.info(f"OCR completed with {len(blocks)} text blocks and {avg_confidence:.2f}% confidence")
            return result
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise
            
        finally:
            # Clean up temporary file if created
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def extract_text_from_pdf(self, 
                             pdf_path: str, 
                             pages: Optional[Union[int, List[int], str]] = None,
                             lang: Optional[str] = None,
                             preprocess: Optional[bool] = None) -> Dict[str, Any]:
        """
        Extract text from a PDF using OCR.
        
        Args:
            pdf_path: Path to PDF file
            pages: Pages to process (int, list of ints, or 'all')
                   If None, processes all pages
            lang: Language code (overrides init setting if provided)
            preprocess: Whether to preprocess the image (overrides init setting if provided)
            
        Returns:
            Dictionary containing:
                - text: Full extracted text
                - pages: List of page dictionaries with text and metadata
                - confidence: Overall OCR confidence score
        """
        logger.info(f"Starting OCR text extraction from PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Use provided settings or fall back to init settings
        ocr_lang = lang or self.lang
        do_preprocess = preprocess if preprocess is not None else self.preprocess
        
        try:
            # Determine pages to process
            if pages is None or pages == 'all':
                page_numbers = None  # All pages
            elif isinstance(pages, int):
                page_numbers = [pages]
            elif isinstance(pages, list):
                page_numbers = pages
            else:
                raise ValueError("Invalid 'pages' parameter. Use int, list of ints, or 'all'")
                
            # Convert PDF to images
            logger.info(f"Converting PDF to images with DPI={self.dpi}")
            images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=self.dpi, 
                first_page=min(page_numbers) if page_numbers else None,
                last_page=max(page_numbers) if page_numbers else None
            )
            
            # Process each page
            all_text = []
            page_results = []
            total_confidence = 0
            
            for i, image in enumerate(images):
                # Calculate current page number (1-based)
                page_num = (min(page_numbers) + i) if page_numbers else (i + 1)
                
                logger.info(f"Processing page {page_num}")
                
                # Extract text from page image
                page_result = self.extract_text_from_image(
                    image, 
                    lang=ocr_lang,
                    preprocess=do_preprocess
                )
                
                # Add page number to result
                page_result["page_num"] = page_num
                
                # Add to results
                all_text.append(page_result["text"])
                page_results.append(page_result)
                total_confidence += page_result["confidence"]
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(images) if images else 0
            
            # Create final result
            result = {
                "text": "\n\n".join(all_text),
                "pages": page_results,
                "confidence": avg_confidence,
                "page_count": len(images),
                "language": ocr_lang
            }
            
            logger.info(f"PDF OCR completed with {len(images)} pages and {avg_confidence:.2f}% confidence")
            return result
            
        except Exception as e:
            logger.error(f"PDF OCR processing failed: {str(e)}")
            raise
    
    def get_simple_text(self, file_path: str) -> str:
        """
        Simple wrapper to extract text from file and return just the text.
        Automatically detects file type (PDF or image) based on extension.
        
        Args:
            file_path: Path to file (PDF or image)
            
        Returns:
            Extracted text as a string
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            result = self.extract_text_from_pdf(file_path)
        else:  # Assume image for all other extensions
            result = self.extract_text_from_image(file_path)
            
        return result["text"]