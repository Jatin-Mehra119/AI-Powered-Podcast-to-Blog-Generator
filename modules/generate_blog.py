from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from modules.models import ContentOutput
import logging

logger = logging.getLogger("blog_generator")

def generate_blog(transcript: str, llm, use_tavily: bool = True, max_length: int = 10000) -> str:
    # Set up tools and prompt
    tools = [TavilySearchResults(max_results=5)] if use_tavily else []
    search_instruction = (
        "Extract key themes from the transcript and use Tavily Search to find recent articles, examples, or data that support or expand on those themes. Include at least one hyperlink to a credible source in the blog post."
        if use_tavily
        else "Rely only on the transcript to generate the blog post."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        You are a helpful assistant that generates blog posts based on provided transcripts. The blog post MUST:
        - Be informative, engaging, and well-structured with an introduction, 2-3 subheadings, bullet points where relevant, and a conclusion.
        - Be written in a professional yet approachable tone, suitable for software developers and tech enthusiasts.
        - Format your answer in clean markdown with no CODE blocks.
        
        Instructions:
        1. {search_instruction}
        2. Structure the blog post clearly with subheadings and bullet points for key insights or recommendations.
        """),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    if len(transcript) > max_length:
        logger.info("Transcript is long, summarizing chunks")
        chunks = [transcript[i:i+max_length] for i in range(0, len(transcript), max_length)]
        summaries = []
        for chunk in chunks:
            summary_prompt = ChatPromptTemplate.from_messages([
                ("human", "Summarize this transcript chunk: {chunk}")
            ])
            chain = summary_prompt | llm
            summary = chain.invoke({"chunk": chunk})
            summaries.append(summary["content"])
        combined_summary = "\n".join(summaries)
        input_text = f"Generate a blog post based on these summaries: {combined_summary}"
    else:
        input_text = f"Generate a blog post based on this transcript: {transcript}"
    
    try:
        blog_result = agent_executor.invoke({"input": input_text})
        output = ContentOutput.from_llm_response(blog_result["output"])
        return output.content
    except Exception as e:
        logger.error(f"Error generating blog post: {e}")
        raise