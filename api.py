import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import your existing modules
from modules.preprocessing import load_transcript, process_transcript
from modules.content_generation import (
    generate_seo_elements,
    generate_faq,
    generate_social_media,
    generate_newsletter,
    extract_quotes
)
from modules.generate_blog import generate_blog

# Import necessary libraries
from langchain_groq import ChatGroq
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "api.log")
logger = logging.getLogger("podcast_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Create output directory
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Podcast to Blog API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if GROQ API key is set
if not os.getenv("GROQ_API_KEY"):
    logger.error("GROQ_API_KEY not set in .env file")
    raise ValueError("GROQ_API_KEY not set in .env file")

if not os.getenv("TAVILY_API_KEY"):
    logger.error("TAVILY_API_KEY not set in .env file")
    raise ValueError("TAVILY_API_KEY not set in .env file")

use_tavily = bool(os.getenv("TAVILY_API_KEY"))

# Function to process the audio file and generate content
async def process_audio(
    file_path: str, 
    content_types: List[str], 
    model_name: str,
    job_id: str
):
    try:
        # Initialize LLM
        llm = ChatGroq(temperature=0.2, model_name=model_name)
        
        # Create timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_base = f"{job_id}_{base_filename}"
        
        # Dictionary to store output file paths
        output_files = {}
        
        # Transcribe audio
        logger.info(f"Transcribing audio: {file_path}")
        transcript = load_transcript(model_type="whisper-large-v3-turbo", file_path=file_path, language="english")
        cleaned_transcript = process_transcript(transcript)
        
        # Save transcript
        transcript_path = save_output(cleaned_transcript, f"{output_base}_transcript")
        output_files["transcript"] = os.path.basename(transcript_path)
        
        # Generate blog post if selected
        blog_content = None
        if "blog" in content_types:
            logger.info("Generating blog post...")
            blog_content = generate_blog(cleaned_transcript, llm, use_tavily)
            blog_path = save_output(blog_content, f"{output_base}_blog")
            output_files["blog"] = os.path.basename(blog_path)
        
        # Generate other content types
        if "seo" in content_types and blog_content:
            seo_elements = generate_seo_elements(llm, blog_content)
            seo_path = save_output(seo_elements, f"{output_base}_seo", format="json")
            output_files["seo"] = os.path.basename(seo_path)
        
        if "faq" in content_types:
            faq_content = generate_faq(llm, cleaned_transcript)
            faq_path = save_output(faq_content, f"{output_base}_faq")
            output_files["faq"] = os.path.basename(faq_path)
        
        if "social" in content_types and blog_content:
            social_content = generate_social_media(llm, blog_content)
            social_path = save_output(social_content, f"{output_base}_social")
            output_files["social"] = os.path.basename(social_path)
        
        if "newsletter" in content_types and blog_content:
            newsletter_content = generate_newsletter(llm, blog_content)
            newsletter_path = save_output(newsletter_content, f"{output_base}_newsletter")
            output_files["newsletter"] = os.path.basename(newsletter_path)
        
        if "quotes" in content_types:
            quotes_content = extract_quotes(llm, cleaned_transcript)
            quotes_path = save_output(quotes_content, f"{output_base}_quotes")
            output_files["quotes"] = os.path.basename(quotes_path)
        
        # Update job status to completed
        job_status[job_id] = {
            "status": "completed",
            "files": output_files
        }
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred while processing job {job_id}: {str(e)}", exc_info=True)
        # Update job status to failed
        job_status[job_id] = {
            "status": "failed",
            "error": str(e)
        }

# Dictionary to store job status
job_status = {}

# Helper function to save output
def save_output(content, filename, format="md"):
    filepath = OUTPUT_DIR / f"{filename}.{format}"
    with open(filepath, "w", encoding="utf-8") as f:
        if format == "json":
            import json
            json.dump(content, f, indent=2)
        else:
            f.write(str(content))
    logger.info(f"Saved {filename}.{format}")
    return str(filepath)

# Routes
@app.post("/api/upload")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    content_types: Optional[List[str]] = Form(["blog", "seo", "faq", "social", "newsletter", "quotes"]),
    model: Optional[str] = Form("meta-llama/llama-4-scout-17b-16e-instruct")
):
    # Check file size (limit to 20MB)
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB
    content = b""
    
    while chunk := await file.read(chunk_size):
        content += chunk
        file_size += len(chunk)
        if file_size > 20 * 1024 * 1024:  # 20MB limit
            return JSONResponse(
                status_code=400,
                content={"error": "File size exceeds the 20MB limit"}
            )
    
    # Check file type
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg')):
        return JSONResponse(
            status_code=400,
            content={"error": "Only audio files (.mp3, .wav, .m4a, .ogg) are supported"}
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save file temporarily
    temp_file_path = os.path.join(TEMP_DIR, f"{job_id}_{file.filename}")
    with open(temp_file_path, "wb") as f:
        f.write(content)
    
    # Initialize job status
    job_status[job_id] = {
        "status": "processing",
        "filename": file.filename
    }
    
    # Process the audio in the background
    background_tasks.add_task(
        process_audio,
        temp_file_path,
        content_types,
        model,
        job_id
    )
    
    return {"job_id": job_id, "message": "Audio file uploaded and processing started"}

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    return job_status[job_id]

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

# Mount static files for frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)