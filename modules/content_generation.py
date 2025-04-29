import json
from langchain.prompts import ChatPromptTemplate
import logging
from typing import Dict, Any

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
        # Fix: Handle AIMessage object correctly
        response_text = result.content if hasattr(result, 'content') else str(result)
        parsed_json = json.loads(response_text)
        return parsed_json
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
        content = result.content if hasattr(result, 'content') else str(result)
        faqs = json.loads(content)
        md_content = "\n".join([f"**Q: {faq['question']}**\nA: {faq['answer']}\n" for faq in faqs])
        return md_content
    except Exception as e:
        logger.error(f"Error generating FAQ: {e}")
        return str(result)

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
        content = result.content if hasattr(result, 'content') else str(result)
        social_posts = json.loads(content)
        md_content = (
            f"**Twitter Post:**\n{social_posts['twitter']}\n\n"
            f"**LinkedIn Post:**\n{social_posts['linkedin']}\n\n"
            f"**Instagram Caption:**\n{social_posts['instagram']}\n"
        )
        return md_content
    except Exception as e:
        logger.error(f"Error generating social media posts: {e}")
        return str(result)

def generate_newsletter(llm, blog_content: str) -> str:
    logger.info("Generating newsletter...")
    newsletter_prompt = ChatPromptTemplate.from_messages([
        ("human", "Return a 100-150 word summary of this blog post for a newsletter: {blog_content}")
    ])
    chain = newsletter_prompt | llm
    result = chain.invoke({"blog_content": blog_content})
    return result.content if hasattr(result, 'content') else str(result)

def extract_quotes(llm, transcript: str) -> str:
    logger.info("Extracting quotes...")
    quote_prompt = ChatPromptTemplate.from_messages([
        ("human", "Return a JSON list of 3-5 memorable quotes from this transcript: {transcript}")
    ])
    chain = quote_prompt | llm
    result = chain.invoke({"transcript": transcript})
    try:
        content = result.content if hasattr(result, 'content') else str(result)
        quotes = json.loads(content)
        md_content = "\n".join([f"- {quote}" for quote in quotes])
        return md_content
    except Exception as e:
        logger.error(f"Error extracting quotes: {e}")
        return str(result)