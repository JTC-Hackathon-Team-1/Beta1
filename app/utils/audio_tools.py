# audio/asr.py

import os
import tempfile
from typing import Optional, Dict, Any, Union, BinaryIO
import whisper
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASRProcessor:
    """
    CasaLingua - Audio Tools
    Handles ASR (Automatic Speech Recognition) using Whisper or similar.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize the ASR processor with specified model size.
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run model on ('cuda', 'cpu'). If None, will use CUDA if available.
        """
        self.model_size = model_size
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Whisper model '{model_size}' on {self.device}")
        
        try:
            self.model = whisper.load_model(model_size).to(self.device)
            logger.info(f"Successfully loaded Whisper model '{model_size}'")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
            
    def transcribe_audio(self, 
                         audio_input: Union[str, BinaryIO], 
                         language: Optional[str] = None, 
                         task: str = "transcribe") -> Dict[str, Any]:
        """
        Transcribe audio file to text using Whisper ASR.
        
        Args:
            audio_input: Path to audio file or file-like object
            language: Language code (e.g., 'en', 'es'). If None, will be auto-detected.
            task: Either 'transcribe' or 'translate' (to English)
            
        Returns:
            Dictionary containing transcription results including:
                - text: Full transcription text
                - segments: List of segment dictionaries with timestamps
                - language: Detected language code
        """
        logger.info(f"Starting transcription of audio, language={language}, task={task}")
        
        # Handle file-like objects by saving to temporary file
        temp_file = None
        audio_path = audio_input
        
        try:
            if not isinstance(audio_input, str):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_file.write(audio_input.read())
                temp_file.close()
                audio_path = temp_file.name
            
            # Set transcription options
            options = {
                "task": task,
                "fp16": self.device == "cuda",  # Use fp16 if on CUDA
            }
            
            # Add language if specified
            if language:
                options["language"] = language
                
            # Perform transcription
            result = self.model.transcribe(audio_path, **options)
            
            logger.info(f"Transcription completed successfully: {len(result['text'])} characters")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
            
        finally:
            # Clean up temporary file if created
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def transcribe_file(self, audio_path: str, **kwargs) -> str:
        """
        Simple wrapper to transcribe a file and return just the text.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments to pass to transcribe_audio
            
        Returns:
            Transcribed text as a string
        """
        result = self.transcribe_audio(audio_path, **kwargs)
        return result["text"]
        
    def detect_language(self, audio_path: str) -> str:
        """
        Detect language from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Language code (e.g., 'en', 'es')
        """
        logger.info(f"Detecting language from audio file: {audio_path}")
        
        # Use the first 30 seconds for language detection
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        
        # Detect language
        _, probs = self.model.detect_language(mel)
        language_code = max(probs, key=probs.get)
        
        logger.info(f"Detected language: {language_code}")
        return language_code