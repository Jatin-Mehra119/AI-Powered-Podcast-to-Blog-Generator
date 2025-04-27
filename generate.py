from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from preprocessing import load_transcript, process_transcript
import os
from dotenv import load_dotenv

# Load environment variables (e.g., Tavily API key)
load_dotenv()

# Initialize the language model
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# Preprocess the audio file
audio_file = "audio/Clean code challenge - Silicon Valley Season 5, Ep6.mp3"
transcript = load_transcript(model_type="base", file_path=audio_file, language="english")
cleaned_transcript = process_transcript(transcript)
transcript = cleaned_transcript

# Define tools (corrected to use TavilySearchResults)
tools = [TavilySearchResults()]

# Define the prompt using ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful assistant that generates blog posts based on transcripts.
The blog post should be informative, engaging, and well-structured.

You can use the following tools to enhance your blog post:
- Tavily Search: Use this tool to search for relevant information and resources to support your blog post.
    """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Construct the tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Invoke the agent with the transcript embedded in the input
agent_executor.invoke({
    "input": f"Generate a blog post based on this transcript: {transcript}"
})