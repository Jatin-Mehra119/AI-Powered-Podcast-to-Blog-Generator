import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from modules.preprocessing import load_transcript, process_transcript
from modules.content_generation import (
    generate_seo_elements,
    generate_faq,
    generate_social_media,
    generate_newsletter,
    extract_quotes
)
from modules.generate_blog import generate_blog

# Create output directory
OUTPUT_DIR = Path("output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "blog_generator.log")
logger = logging.getLogger("podcast_to_blog")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def save_output(content, filename, format="md"):
    filepath = OUTPUT_DIR / f"{filename}.{format}"
    with open(filepath, "w", encoding="utf-8") as f:
        if format == "json":
            json.dump(content, f, indent=2)
        else:
            f.write(str(content))
    logger.info(f"Saved {filename}.{format}")

def main():
    parser = argparse.ArgumentParser(description="Generate content from audio")
    parser.add_argument("--audio", required=True, help="audio/Clean code challenge - Silicon Valley Season 5, Ep6.mp3")
    parser.add_argument("--model", default="meta-llama/llama-4-scout-17b-16e-instruct", help="LLM model name")
    parser.add_argument("--content", nargs="+", default=["blog", "seo", "faq", "social", "newsletter", "quotes"],
                        help="Content types to generate")
    args = parser.parse_args()
    audio_path = args.audio
    model_name = args.model
    content_types = args.content

    # Load environment variables
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not set in .env file")
        sys.exit(1)
    use_tavily = bool(os.getenv("TAVILY_API_KEY"))

    # Initialize LLM
    llm = ChatGroq(temperature=0.2, model_name=model_name)

    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_base = f"{timestamp}_{base_filename}"

    try:
        # Transcribe audio
        logger.info(f"Transcribing audio: {audio_path}")
        transcript = load_transcript(model_type="whisper-large-v3-turbo", file_path=audio_path, language="english")
        cleaned_transcript = process_transcript(transcript)
        save_output(cleaned_transcript, f"{output_base}_transcript")

        # Generate blog post
        if "blog" in content_types:
            logger.info("Generating blog post...")
            blog_content = generate_blog(cleaned_transcript, llm, use_tavily)
            save_output(blog_content, f"{output_base}_blog")

        # Generate other content types
        if "seo" in content_types and "blog" in content_types:
            seo_elements = generate_seo_elements(llm, blog_content)
            save_output(seo_elements, f"{output_base}_seo", format="json")
        if "faq" in content_types:
            faq_content = generate_faq(llm, cleaned_transcript)
            save_output(faq_content, f"{output_base}_faq")
        if "social" in content_types and "blog" in content_types:
            social_content = generate_social_media(llm, blog_content)
            save_output(social_content, f"{output_base}_social")
        if "newsletter" in content_types and "blog" in content_types:
            newsletter_content = generate_newsletter(llm, blog_content)
            save_output(newsletter_content, f"{output_base}_newsletter")
        if "quotes" in content_types:
            quotes_content = extract_quotes(llm, cleaned_transcript)
            save_output(quotes_content, f"{output_base}_quotes")

        logger.info(f"All selected content generated and saved to '{OUTPUT_DIR}'")
        print(f"All selected content has been generated and saved to '{OUTPUT_DIR}'")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()