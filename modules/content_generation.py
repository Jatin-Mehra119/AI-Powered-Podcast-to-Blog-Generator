#!/usr/bin/env python3
"""
Content Generation Module

This module handles the generation of various content types from podcast transcripts
and blog posts using Large Language Models (LLMs). It provides specialized functions
for creating:

1. SEO elements - titles, meta descriptions, tags, and keywords
2. Frequently Asked Questions (FAQs)
3. Social media posts for Twitter, LinkedIn, and Instagram
4. Newsletter summaries
5. Quotable highlights from transcripts

Each function uses carefully crafted prompts to generate high-quality content
optimized for its specific purpose and platform.

Dependencies:
- langchain: For prompt templating and LLM interactions
- Pydantic models: For data validation and JSON structure
- logging: For tracking the generation process
"""

import json
import re
from langchain.prompts import ChatPromptTemplate
import logging
from typing import Dict, Any

from modules.models import ContentOutput, SeoElements, FaqItem, SocialMediaPosts, BlogQuote

logger = logging.getLogger("content_generation")

def generate_seo_elements(llm, blog_content: str) -> Dict[str, Any]:
    """
    Generate SEO elements for a blog post using an LLM.
    
    This function creates search engine optimization (SEO) elements including
    a title, meta description, tags, and keywords that are optimized for 
    discoverability based on the blog content.
    
    Args:
        llm: The language model instance to use for generation.
        blog_content (str): The full content of the blog post.
        
    Returns:
        Dict[str, Any]: A dictionary containing the following SEO elements:
            - title: An SEO-friendly title (under 60 characters)
            - meta_description: A meta description (under 160 characters)
            - tags: A list of 5-10 tags
            - keywords: A list of 5-7 keywords
            
    Raises:
        JSONDecodeError: If the LLM response cannot be parsed as valid JSON
        Exception: For other processing errors
    """
    logger.info("Generating SEO elements...")
    seo_prompt = ChatPromptTemplate.from_messages([
        ("human", """
        Given this blog post, return a JSON object with:
        - "title": An SEO-friendly title (under 60 characters)
        - "meta_description": A meta description (under 160 characters)
        - "tags": A list of 5-10 tags
        - "keywords": A list of 5-7 keywords
        Blog post: {blog_content}
        do not include any other text or explanation, just return the JSON object
        """)
    ])
    chain = seo_prompt | llm
    result = chain.invoke({"blog_content": blog_content})
    
    try:
        # Extract response text
        response_text = result.content if hasattr(result, 'content') else str(result)
        
        # Clean up the response to extract just the JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        parsed_json = json.loads(response_text)
        
        # Create a Pydantic model to validate the structure
        seo = SeoElements(**parsed_json)
        return seo.model_dump()
    except json.JSONDecodeError:
        logger.error("LLM response is not valid JSON")
        return {"error": "Invalid JSON format", "raw_response": response_text}
    except Exception as e:
        logger.error(f"Error processing SEO elements: {e}")
        return {"error": str(e), "raw_response": response_text}

def generate_faq(llm, transcript: str) -> str:
    """
    Generate Frequently Asked Questions (FAQs) from a podcast transcript.
    
    This function identifies key topics and questions from the transcript
    and creates a set of Q&A pairs that capture the main insights.
    The output is formatted as markdown for easy integration into websites.
    
    Args:
        llm: The language model instance to use for generation.
        transcript (str): The transcript text from which to generate FAQs.
        
    Returns:
        str: A markdown-formatted string containing 3-5 FAQ items, each with
             a question heading and detailed answer paragraph.
             
    Example output:
        ## What are the benefits of meditation?
        
        Research shows that regular meditation can reduce stress, improve focus,
        and enhance overall well-being. Our guest explained that just 10 minutes
        per day can make a significant difference in mental clarity.
        
        ## How can beginners start a meditation practice?
        
        ...
    """
    logger.info("Generating FAQ...")
    faq_prompt = ChatPromptTemplate.from_messages([
        ("human", """
        Return a JSON list of 3-5 objects, each with "question" and "answer" keys, based on this transcript: {transcript}
        do not include any other text or explanation, just return the JSON object
        """)
    ])
    chain = faq_prompt | llm
    result = chain.invoke({"transcript": transcript})
    
    try:
        # Extract content from result
        content = result.content if hasattr(result, 'content') else str(result)
        
        # Clean up the response to extract just the JSON array
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
            
        # Parse the JSON and validate with Pydantic
        faq_items = json.loads(content)
        
        # Convert to FAQ items and validate
        validated_faqs = [FaqItem(**item) for item in faq_items]
        
        # Format as markdown
        md_content = ""
        for faq in validated_faqs:
            md_content += f"## {faq.question}\n\n{faq.answer}\n\n"
            
        return md_content
    except Exception as e:
        logger.error(f"Error generating FAQ: {e}")
        # Create clean output with the error included
        output = ContentOutput.from_llm_response(result)
        return output.content

def generate_social_media(llm, blog_content: str) -> str:
    """
    Generate social media posts for multiple platforms based on a blog post.
    
    Creates tailored content for three different social media platforms, each
    optimized for the platform's character limits and audience expectations:
    - Twitter/X: Short, engaging post under 280 characters
    - LinkedIn: Professional, detailed post of 200-300 words
    - Instagram: Visual-friendly caption with appropriate length
    
    Args:
        llm: The language model instance to use for generation.
        blog_content (str): The content of the blog post to transform.
        
    Returns:
        str: A markdown-formatted string containing the three social media posts,
             organized under appropriate headings.
             
    Note:
        The output is structured to make it easy for users to copy and paste
        each platform's content separately.
    """
    logger.info("Generating social media posts...")
    social_prompt = ChatPromptTemplate.from_messages([
        ("human", """
        Return a JSON object with:
        - "twitter": A post (<280 characters)
        - "linkedin": A post (200-300 words)
        - "instagram": A caption (50-100 words)
        Based on this blog post: {blog_content}
        do not include any other text or explanation, just return the JSON object
        """)
    ])
    chain = social_prompt | llm
    result = chain.invoke({"blog_content": blog_content})
    
    try:
        # Extract content from result
        content = result.content if hasattr(result, 'content') else str(result)
        
        # Clean up the response to extract just the JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
            
        # Parse the JSON and validate with Pydantic
        social_data = json.loads(content)
        social_posts = SocialMediaPosts(**social_data)
        
        # Format as markdown
        md_content = (
            f"# Social Media Content\n\n"
            f"## Twitter Post\n\n{social_posts.twitter}\n\n"
            f"## LinkedIn Post\n\n{social_posts.linkedin}\n\n"
            f"## Instagram Caption\n\n{social_posts.instagram}\n"
        )
        
        return md_content
    except Exception as e:
        logger.error(f"Error generating social media posts: {e}")
        # Create clean output with the error included
        output = ContentOutput.from_llm_response(result)
        return output.content

def generate_newsletter(llm, blog_content: str) -> str:
    """
    Generate a concise newsletter summary from a blog post.
    
    Creates a condensed version of the blog content that's suitable for
    email newsletters, focusing on the core message and key takeaways.
    The summary maintains the tone and perspective of the original post
    while being brief enough for email consumption.
    
    Args:
        llm: The language model instance to use for generation.
        blog_content (str): The content of the blog post to summarize.
        
    Returns:
        str: A markdown-formatted string containing a 100-150 word newsletter
             summary with an appropriate heading.
             
    Note:
        The summary is intentionally kept short to maximize engagement
        in email format, where attention spans are typically shorter.
    """
    logger.info("Generating newsletter...")
    newsletter_prompt = ChatPromptTemplate.from_messages([
        ("human", "Return a 100-150 word summary of this blog post for a newsletter: {blog_content}")
    ])
    chain = newsletter_prompt | llm
    result = chain.invoke({"blog_content": blog_content})
    
    # Create clean output
    output = ContentOutput.from_llm_response(result)
    
    # Format as markdown
    md_content = f"# Newsletter Summary\n\n{output.content}"
    return md_content

def extract_quotes(llm, transcript: str) -> str:
    """
    Extract memorable quotable content from a podcast transcript.
    
    Analyzes the transcript to identify insightful, powerful, or otherwise
    noteworthy quotes that capture key moments. These quotes can be used
    for promotional materials, social media highlights, or as callouts
    within the blog post itself.
    
    Args:
        llm: The language model instance to use for generation.
        transcript (str): The transcript from which to extract quotes.
        
    Returns:
        str: A markdown-formatted string containing 3-5 quotes with attributed
             speakers, formatted in blockquote style.
             
    Example output:
        # Memorable Quotes
        
        > The key to innovation isn't having new ideas, it's connecting existing ones.
        >
        > — Jane Smith
        
        > When we focus on user problems instead of technology, that's when the magic happens.
        >
        > — John Doe
    """
    logger.info("Extracting quotes...")
    quote_prompt = ChatPromptTemplate.from_messages([
        ("human", """
        Extract 3-5 memorable quotes from this transcript and return them as a JSON array.
        Each quote should have a "quote" field with the actual quote text and a "speaker" field 
        (use "The Author" if speaker is unknown):
        {transcript}
        """)
    ])
    chain = quote_prompt | llm
    result = chain.invoke({"transcript": transcript})
    
    try:
        # Extract content from result
        content = result.content if hasattr(result, 'content') else str(result)
        
        # Clean up the response to extract just the JSON array
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
            
        # Parse the JSON and validate with Pydantic
        quotes_data = json.loads(content)
        validated_quotes = [BlogQuote(**quote) for quote in quotes_data]
        
        # Format as markdown
        md_content = "# Memorable Quotes\n\n"
        for quote in validated_quotes:
            speaker = quote.speaker if quote.speaker else "Unknown"
            md_content += f"> {quote.quote}\n>\n> — {speaker}\n\n"
            
        return md_content
    except Exception as e:
        logger.error(f"Error extracting quotes: {e}")
        # Create clean output with the error included
        output = ContentOutput.from_llm_response(result)
        return output.content