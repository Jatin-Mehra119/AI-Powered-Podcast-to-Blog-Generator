import json
import re
from langchain.prompts import ChatPromptTemplate
import logging
from typing import Dict, Any, List

from modules.models import ContentOutput, SeoElements, FaqItem, SocialMediaPosts, BlogQuote

logger = logging.getLogger("content_generation")

def generate_seo_elements(llm, blog_content: str) -> Dict[str, Any]:
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