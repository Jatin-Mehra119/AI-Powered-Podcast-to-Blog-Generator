import whisper
import logging
import os

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
    Map language codes to Whisper model language codes.
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
        logger.info(f"Mapped '{language}' to Whisper code '{mapped_code}'")
    else:
        logger.warning(f"Could not map language: {language}")
    return mapped_code


def process_transcript(transcript):
    """
    Process the transcript to remove unwanted characters and format it.
    """
    logger.info("Processing transcript...")
    # Remove unwanted characters
    cleaned_transcript = transcript.replace("\n", " ").replace("\r", "")
    logger.debug("Removed newline and carriage return characters.")

    # Further processing can be done here if needed
    logger.info("Transcript processing complete.")
    return cleaned_transcript

def load_transcript(model_type="base", file_path=None, language=None):
    """
    Load the transcript from a file using a specified Whisper model.
    Detects language automatically if not specified.
    """
    logger.info(f"Loading transcript from '{file_path}' using model '{model_type}'. Specified language: {language}")
    if not file_path or not os.path.exists(file_path):
        logger.error(f"Audio file not found at path: {file_path}")
        raise FileNotFoundError(f"Audio file not found at path: {file_path}")

    # Use a multilingual model by default (e.g., "base", "small", "medium", "large")
    # Removed ".en" to allow multilingual capabilities
    logger.debug(f"Loading Whisper model: {model_type}")
    model = whisper.load_model(model_type)
    logger.info(f"Whisper model '{model_type}' loaded.")

    # The `language` parameter can be specified to force a language,
    # otherwise Whisper will attempt auto-detection.
    whisper_language_code = None
    if language:
        whisper_language_code = map_language_code(language)
        if not whisper_language_code:
            logger.error(f"Unsupported language specified: {language}")
            raise ValueError(f"Unsupported language: {language}")
        logger.info(f"Using specified language code for transcription: {whisper_language_code}")
    else:
        # Auto-detect the language
        logger.info("Language not specified, Whisper will attempt auto-detection.")
        whisper_language_code = None # Explicitly set to None for clarity

    # Load
    logger.info(f"Starting transcription for '{file_path}'...")
    try:
        result = model.transcribe(file_path, language=whisper_language_code)
        transcript = result["text"]
        detected_language = result.get("language", "N/A") # Get detected language if available
        logger.info(f"Transcription successful. Detected language: {detected_language}")
        return process_transcript(transcript)
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        raise # Re-raise the exception after logging


# Example usage
if __name__ == "__main__":
    logger.info("Starting preprocessing script example.")
    # Load a transcript from a file
    file_path = "audio/Clean code challenge - Silicon Valley Season 5, Ep6.mp3"
    language = 'english'  # Set to a specific language code if needed (e.g., "en", "fr")
    output_file = "transcript.txt"

    try:
        logger.info(f"Attempting to load transcript for: {file_path}")
        transcript = load_transcript(model_type="base", file_path=file_path, language=language)
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



