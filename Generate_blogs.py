from dotenv import load_dotenv
import os
import logging
from preprocessing import load_transcript, process_transcript

# Set up logging
logging.basicConfig(level=logging.INFO)
# Create logs directory if it doesn't exist
log_dir = "logs/generate_blogs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "generate_blogs.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() 
    ]
)
# Preprocess  the audio file

Audio_file = "audio/Clean code challenge - Silicon Valley Season 5, Ep6.mp3"
# Load the transcript using the Whisper model
transcript = load_transcript(model_type="base", file_path=Audio_file, language="english")
# Process the transcript to remove unwanted characters and format it
cleaned_transcript = process_transcript(transcript)

# Load environment variables from .env file
load_dotenv() 
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable not set.")


# Model initialization
from langchain.chat_models import init_chat_model

model = init_chat_model("llama3-8b-8192", model_provider="groq")

# PROMPT
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(
    """
    You are a helpful assistant that generates blog posts based on the provided transcript.
    The blog post should be informative, engaging, and well-structured.
    """
)
HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(
    """
    Given the following transcript, generate a blog post:
    {transcript}
    """
)
PROMPT = ChatPromptTemplate.from_messages(
    [
        SYSTEM_PROMPT,
        HUMAN_PROMPT,
        MessagesPlaceholder(variable_name="history"),
    ]
)

# Generate the blog post
from langchain.chains import LLMChain


chain = LLMChain(
    llm=model,
    prompt=PROMPT,
    verbose=True,
    memory=None,
)
# Generate the blog post
response = chain.run(transcript=cleaned_transcript)
# Print the generated blog post
print("Generated Blog Post:")
print(response)
# Save the generated blog post to a file
output_file = "generated_blog_post.txt"
with open(output_file, "w") as f:
    f.write(response)
print(f"Blog post saved to '{output_file}'")