from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from preprocessing import load_transcript, process_transcript
import os
import logging
from dotenv import load_dotenv

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "blog_generator.log")

# Create a custom logger for this file instead of using basicConfig
logger = logging.getLogger("blog_generator")
logger.setLevel(logging.INFO)

# Check if handlers are already configured to avoid duplicate handlers
if not logger.handlers:
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Verify API keys
if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    logger.error("GROQ_API_KEY or TAVILY_API_KEY not set in .env file")
    exit(1)

# Initialize the language model
try:
    llm = ChatGroq(temperature=0, model_name="meta-llama/llama-4-scout-17b-16e-instruct") 
    logger.info("Grok model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Grok model: {e}")
    exit(1)

# Preprocess the audio file
audio_file = "audio/Clean code challenge - Silicon Valley Season 5, Ep6.mp3"
try:
    transcript = load_transcript(model_type="whisper-large-v3-turbo", file_path=audio_file, language="english")
    cleaned_transcript = process_transcript(transcript)
    transcript = cleaned_transcript
    logger.info("Transcript processed successfully")
except Exception as e:
    logger.error(f"Error during transcription: {e}")
    exit(1)

# Chunk transcript if too long
def chunk_transcript(transcript, max_length=10000):
    return [transcript[i:i+max_length] for i in range(0, len(transcript), max_length)]

transcript_chunks = chunk_transcript(transcript)
if len(transcript_chunks) > 1:
    logger.warning(f"Transcript split into {len(transcript_chunks)} chunks due to length")
    transcript = transcript_chunks[0]  # Use first chunk for simplicity

# Define tools
tools = [TavilySearchResults(max_results=5)]
logger.info("Tools initialized")

# Define the prompt using {input} instead of {transcript}
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful assistant that generates blog posts based on provided transcripts. The blog post MUST:
- Be informative, engaging, and well-structured with an introduction, 2-3 subheadings, bullet points where relevant, and a conclusion.
- Include at least one hyperlink to a credible source from Tavily Search results to support claims or add depth.
- Be written in a professional yet approachable tone, suitable for software developers and tech enthusiasts.
Instructions:
1. Extract key themes from the transcript provided in the input and craft a narrative.
2. Use Tavily Search to find relevant information.
3. Explicitly cite at least one search result with a hyperlink in the blog post.
4. Structure the blog post clearly with subheadings and bullet points for key insights or recommendations.
    """),
    ("human", "{input}"), 
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Construct the tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)
logger.info("Agent created")

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
logger.info("Agent executor initialized")

# Invoke the agent
try:
    blog_result = agent_executor.invoke({
        "input": f"Generate a blog post based on this transcript: {transcript}"
    })
    logger.info("Blog post generated successfully")
except Exception as e:
    logger.error(f"Error during agent execution: {e}")
    exit(1)

# Save the blog post
output_file = "blog_post.md"
try:
    with open(output_file, "w") as f:
        f.write(blog_result["output"])
    logger.info(f"Blog post saved to {output_file}")
except Exception as e:
    logger.error(f"Error saving blog post: {e}")