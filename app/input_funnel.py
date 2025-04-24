import os
import logging
import mimetypes
import tempfile
from typing import Dict, Any, Optional, Union, BinaryIO, Tuple
from pathlib import Path
import magic                     # python-magic
import requests
import io

from app.utils.ocr import OCRProcessor

from app.utils.audio_tools import ASRProcessor

logger = logging.getLogger("input_funnel")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s"))
logger.addHandler(ch)

class InputFunnel:
    def __init__(self):
        logger.debug("Initializing InputFunnel...")
        self.ocr_processor = OCRProcessor()
        self.asr_processor = ASRProcessor()
        logger.debug("OCRProcessor and ASRProcessor ready.")

    def process_file(self, file_stream: BinaryIO, filename: str, detect_language=False) -> Dict[str, Any]:
        logger.info(f"Processing file {filename}...")
        # Determine mime & category
        kind = magic.from_buffer(file_stream.read(2048), mime=True)
        file_stream.seek(0)
        logger.debug(f"Detected MIME type: {kind}")

        if kind.startswith("image/"):
            category = "image"
        elif kind in ("application/pdf",):
            category = "pdf"
        elif kind.startswith("audio/"):
            category = "audio"
        else:
            raise ValueError(f"Unsupported MIME type: {kind}")

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(file_stream.read())
            tmp.flush()
            tmp_path = tmp.name
        logger.debug(f"Written to temp file {tmp_path}")

        # Dispatch
        if category in ("image", "pdf"):
            text = self.ocr_processor.extract_text(tmp_path)
        elif category == "audio":
            text = self.asr_processor.transcribe(tmp_path)
        else:
            text = ""

        logger.info(f"Raw text length: {len(text)}")

        language = None
        if detect_language and category == "audio":
            language = self.asr_processor.detect_language(tmp_path)
            logger.info(f"Detected audio language: {language}")

        # Clean up
        os.unlink(tmp_path)
        logger.debug(f"Cleaned up temp file {tmp_path}")

        return {"text": text, "language": language}