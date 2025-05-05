#!/usr/bin/env python3
"""
Podcast Preprocessing Module

This module handles the audio preprocessing pipeline for the podcast-to-blog generator.
It provides functionality for:
1. Loading and transcribing audio files using Groq's Whisper models
2. Processing and cleaning transcription text
3. Language detection and mapping to appropriate model codes

The module relies on Groq's API for transcription services and supports multiple languages.

Dependencies:
- groq: API client for Groq services
- dotenv: For loading environment variables
- logging: For detailed operation logging
"""

import logging
import os
import json
from groq import Groq
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(log_dir, "preprocessing.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() 
    ]
)

logger = logging.getLogger(__name__)

def map_language_code(language):
    """
    Map common language names to Groq Whisper model language codes.
    
    This function converts user-friendly language names to the ISO language
    codes needed for the Whisper transcription model.
    
    Args:
        language (str): Language name in English (e.g., "english", "french")
        
    Returns:
        str or None: Two-letter ISO language code if mapping exists, None otherwise
        
    Example:
        >>> map_language_code("english")
        'en'
        >>> map_language_code("german")
        'de'
    """
    logger.debug(f"Mapping language: {language}")
    # This function can be expanded to include more languages as needed
    language = language.lower()
    language_map = {
        "english": "en",
        "french": "fr",
        "spanish": "es",
        "hindi": "hi",
        "german": "de",
        "italian": "it",
    }
    mapped_code = language_map.get(language, None)
    if mapped_code:
        logger.info(f"Mapped '{language}' to language code '{mapped_code}'")
    else:
        logger.warning(f"Could not map language: {language}")
    return mapped_code


def process_transcript(transcript):
    """
    Process a raw transcript to clean and format it for better readability.
    
    This function performs text normalization operations to prepare the transcript
    for further processing and content generation:
    - Removes newline and carriage return characters
    - Normalizes whitespace
    - Additional text cleanup as needed
    
    Args:
        transcript (str): Raw transcript text from the transcription service
        
    Returns:
        str: Cleaned and formatted transcript text
        
    Note:
        This function can be expanded to include more sophisticated text 
        normalization if needed, such as speaker diarization or paragraph 
        segmentation.
    """
    logger.info("Processing transcript...")
    # Remove unwanted characters
    cleaned_transcript = transcript.replace("\n", " ").replace("\r", "")
    logger.debug("Removed newline and carriage return characters.")

    # Additional processing can be implemented here:
    # - Remove repeated spaces
    # - Fix common transcription artifacts
    # - Normalize punctuation
    # - Format speaker labels if available
    
    logger.info("Transcript processing complete.")
    return cleaned_transcript

def load_transcript(model_type="whisper-large-v3-turbo", file_path=None, language=None):
    """
    Load and transcribe an audio file using Groq's Whisper API.
    
    This function handles the entire audio transcription process:
    1. Validates the audio file existence
    2. Initializes the Groq client
    3. Handles language specification or auto-detection
    4. Transcribes the audio file
    5. Processes the transcription result
    
    Args:
        model_type (str, optional): The Whisper model variant to use.
            Defaults to "whisper-large-v3-turbo".
        file_path (str, optional): Path to the audio file to transcribe.
            Must be a supported audio format (.mp3, .wav, .m4a, .ogg).
        language (str, optional): Language of the audio content.
            If provided, forces transcription in that language.
            If None, language auto-detection is used.
            
    Returns:
        str: Processed transcript text
        
    Raises:
        FileNotFoundError: If the audio file doesn't exist at the specified path
        ValueError: If an unsupported language is specified
        Exception: For other API or processing errors
        
    Note:
        This function requires a valid Groq API key to be set in the environment.
    """
    logger.info(f"Loading transcript from '{file_path}' using model '{model_type}'. Specified language: {language}")
    if not file_path or not os.path.exists(file_path):
        logger.error(f"Audio file not found at path: {file_path}")
        raise FileNotFoundError(f"Audio file not found at path: {file_path}")

    # Initialize the Groq client
    logger.debug("Initializing Groq client")
    client = Groq()
    logger.info("Groq client initialized")

    # The `language` parameter can be specified to force a language,
    # otherwise will attempt auto-detection.
    whisper_language_code = None
    if language:
        whisper_language_code = map_language_code(language)
        if not whisper_language_code:
            logger.error(f"Unsupported language specified: {language}")
            raise ValueError(f"Unsupported language: {language}")
        logger.info(f"Using specified language code for transcription: {whisper_language_code}")
    else:
        # Auto-detect the language
        logger.info("Language not specified, auto-detection will be used")
        whisper_language_code = None  # Explicitly set to None for clarity

    # Load and transcribe
    logger.info(f"Starting transcription for '{file_path}'...")
    try:
        with open(file_path, "rb") as file:
            # Create a transcription of the audio file
            response = client.audio.transcriptions.create(
                file=file,  # Required audio file
                model=model_type,  # Required model to use for transcription
                response_format="text",  # Get simple text response
                language=whisper_language_code,  # Optional language parameter
                temperature=0.0  # Lower temperature for more deterministic results
            )
            
            # The response is already a string when using response_format="text"
            transcript = response
            logger.info("Transcription successful.")
            return process_transcript(transcript)
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        raise  # Re-raise the exception after logging


# Example usage
if __name__ == "__main__":
    """
    Example script to demonstrate the module's functionality.
    
    When run directly, this script:
    1. Takes a sample audio file
    2. Transcribes it using the specified model and language
    3. Saves the processed transcript to a text file
    
    This provides a convenient way to test the transcription pipeline
    without invoking the full API.
    """
    logger.info("Starting preprocessing script example.")
    # Load a transcript from a file
    file_path = "audio/Clean code challenge - Silicon Valley Season 5, Ep6.mp3"
    language = 'english'  # Set to a specific language code if needed (e.g., "en", "fr")
    output_file = "transcript.txt"

    try:
        logger.info(f"Attempting to load transcript for: {file_path}")
        transcript = load_transcript(model_type="whisper-large-v3-turbo", file_path=file_path, language=language)
        # Save the transcript to a file
        logger.info(f"Saving transcript to '{output_file}'")
        with open(output_file, "w") as f:
            f.write(transcript)
        logger.info("Transcript saved successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found error during example execution: {e}")
        print(f"Error: {e}")
    except ValueError as e:
        logger.error(f"Value error during example execution: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during example execution: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")

    logger.info("Preprocessing script example finished.")
