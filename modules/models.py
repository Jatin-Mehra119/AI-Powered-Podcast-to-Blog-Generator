from typing import List, Optional
from pydantic import BaseModel, Field

# Simple schema for the LLM response

class SeoElements(BaseModel):
    """Model for SEO elements of a blog post"""
    title: str = Field(..., description="SEO-friendly title")
    meta_description: str = Field(..., description="Meta description for the blog post")
    tags: List[str] = Field(..., description="List of tags for the blog post")
    keywords: List[str] = Field(..., description="List of keywords for the blog post")


class FaqItem(BaseModel):
    """Model for a single FAQ item"""
    question: str
    answer: str


class SocialMediaPosts(BaseModel):
    """Model for social media posts"""
    twitter: str = Field(..., description="Twitter/X post")
    linkedin: str = Field(..., description="LinkedIn post")
    instagram: str = Field(..., description="Instagram caption")


class BlogQuote(BaseModel):
    """Model for a memorable quote from a blog or podcast"""
    quote: str
    speaker: Optional[str] = None


class ContentOutput(BaseModel):
    """Base model for all generated content"""
    content: str = Field(..., description="The cleaned content")
    
    @classmethod
    def from_llm_response(cls, response):
        """
        Create a ContentOutput instance from an LLM response.
        Extract just the content, removing metadata.
        """
        if hasattr(response, 'content'):
            return cls(content=response.content)
        elif isinstance(response, str):
            # Try to clean up the string if it contains metadata
            if response.startswith('content='):
                # Extract just the content inside quotes if possible
                import re
                content_match = re.search(r"content='(.*?)'( additional_kwargs=|\Z)", response, re.DOTALL)
                if content_match:
                    return cls(content=content_match.group(1))
            return cls(content=response)
        else:
            # Fallback for other response types
            return cls(content=str(response))