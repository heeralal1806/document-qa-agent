"""
Document Q&A AI Agent - LLM Service
===================================
Integration with OpenAI and Gemini APIs for intelligent Q&A.

This module provides:
- Unified interface for multiple LLM providers
- Context optimization and token management
- Streaming responses
- Enterprise-grade error handling
"""

import os
import logging
from typing import Optional, Dict, Any, List, Generator, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI, AsyncOpenAI
from google.generativeai import GenerativeModel
from google.generativeai import types as google_types
import google.generativeai as genai

from config import settings

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    model: str
    provider: str
    tokens_used: int
    finish_reason: str
    response_time: float
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "tokens_used": self.tokens_used,
            "finish_reason": self.finish_reason,
            "response_time": self.response_time,
            "cached": self.cached
        }


@dataclass
class ChatMessage:
    """Chat message structure."""
    role: str  # 'system', 'user', 'assistant'
    content: str


class BaseLLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
        """Async generate response from prompt."""
        pass
    
    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, LLMResponse]:
        """Stream response from prompt."""
        pass


class OpenAIService(BaseLLMService):
    """OpenAI GPT integration service."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize OpenAI service.
        
        Args:
            api_key: OpenAI API key (defaults to env var)
            model: Model name to use
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.MODEL_NAME
        self.temperature = settings.TEMPERATURE
        self.max_tokens = settings.MAX_TOKENS
        self.top_p = settings.TOP_P
        
        # Initialize clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        logger.info(f"OpenAI service initialized with model: {self.model}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4
    
    def _trim_context(self, context: str, max_tokens: int = 100000) -> str:
        """
        Trim context to fit within token limits.
        
        OpenAI models have context windows up to 128K tokens.
        We trim to be safe.
        """
        max_chars = max_tokens * 4  # Rough character estimate
        
        if len(context) > max_chars:
            # Keep the beginning and end (most important parts)
            beginning = context[:max_chars // 2]
            end = context[-max_chars // 2:]
            return beginning + "\n\n[...content truncated for length...]\n\n" + end
        
        return context
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 context: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate response using OpenAI.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            context: Additional context to include
            **kwargs: Additional model parameters
            
        Returns:
            LLMResponse with generated text
        """
        import time
        start_time = time.time()
        
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            # Add context as a separate message
            context_prompt = f"""Based on the following document content, please answer the user's question:

DOCUMENT CONTENT:
{self._trim_context(context)}

USER QUESTION: {prompt}
"""
            messages.append({"role": "user", "content": context_prompt})
        else:
            messages.append({"role": "user", "content": prompt})
        
        # Override with kwargs if provided
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                stream=False
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                text=response.choices[0].message.content,
                model=self.model,
                provider=LLMProvider.OPENAI.value,
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else 0,
                finish_reason=response.choices[0].finish_reason,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None,
                       context: Optional[str] = None, **kwargs) -> LLMResponse:
        """Async generate response using OpenAI."""
        import time
        start_time = time.time()
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            context_prompt = f"""Based on the following document content, please answer the user's question:

DOCUMENT CONTENT:
{self._trim_context(context)}

USER QUESTION: {prompt}
"""
            messages.append({"role": "user", "content": context_prompt})
        else:
            messages.append({"role": "user", "content": prompt})
        
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                stream=False
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                text=response.choices[0].message.content,
                model=self.model,
                provider=LLMProvider.OPENAI.value,
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else 0,
                finish_reason=response.choices[0].finish_reason,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"OpenAI async API error: {e}")
            raise
    
    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None,
                       context: Optional[str] = None, **kwargs) -> Generator[str, None, LLMResponse]:
        """
        Stream response from OpenAI.
        
        Yields:
            Chunks of generated text
        """
        import time
        start_time = time.time()
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            context_prompt = f"""Based on the following document content, please answer the user's question:

DOCUMENT CONTENT:
{self._trim_context(context)}

USER QUESTION: {prompt}
"""
            messages.append({"role": "user", "content": context_prompt})
        else:
            messages.append({"role": "user", "content": prompt})
        
        temperature = kwargs.get('temperature', self.temperature)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            response_time = time.time() - start_time
            
            yield LLMResponse(
                text=full_response,
                model=self.model,
                provider=LLMProvider.OPENAI.value,
                tokens_used=self._estimate_tokens(full_response),
                finish_reason="stop",
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise


class GeminiService(BaseLLMService):
    """Google Gemini integration service."""
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: Optional[str] = None):
        """
        Initialize Gemini service.
        
        Args:
            api_key: Google Gemini API key (defaults to env var)
            model: Model name to use
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL_NAME
        self.temperature = settings.TEMPERATURE
        self.max_output_tokens = settings.MAX_TOKENS
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.client = GenerativeModel(self.model)
        
        logger.info(f"Gemini service initialized with model: {self.model}")
    
    def _trim_context(self, context: str, max_tokens: int = 100000) -> str:
        """Trim context to fit within token limits."""
        max_chars = max_tokens * 4
        
        if len(context) > max_chars:
            beginning = context[:max_chars // 2]
            end = context[-max_chars // 2:]
            return beginning + "\n\n[...content truncated for length...]\n\n" + end
        
        return context
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 context: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate response using Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            context: Additional context to include
            **kwargs: Additional model parameters
            
        Returns:
            LLMResponse with generated text
        """
        import time
        start_time = time.time()
        
        # Build the full prompt
        full_prompt = ""
        
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        if context:
            full_prompt += f"""Based on the following document content, please answer the user's question:

DOCUMENT CONTENT:
{self._trim_context(context)}

USER QUESTION: {prompt}
"""
        else:
            full_prompt = prompt
        
        temperature = kwargs.get('temperature', self.temperature)
        max_output_tokens = kwargs.get('max_tokens', self.max_output_tokens)
        
        generation_config = google_types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=settings.TOP_P
        )
        
        try:
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=False
            )
            
            response_time = time.time() - start_time
            
            # Extract text from response
            text = ""
            if hasattr(response, 'text'):
                text = response.text
            elif hasattr(response, 'parts'):
                text = "".join([part.text for part in response.parts])
            
            return LLMResponse(
                text=text,
                model=self.model,
                provider=LLMProvider.GEMINI.value,
                tokens_used=len(text) // 4,  # Rough estimate
                finish_reason="stop",
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None,
                       context: Optional[str] = None, **kwargs) -> LLMResponse:
        """Async generate response using Gemini."""
        # Gemini doesn't have native async, so we use the sync version
        return self.generate(prompt, system_prompt, context, **kwargs)
    
    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None,
                       context: Optional[str] = None, **kwargs) -> Generator[str, None, LLMResponse]:
        """
        Stream response from Gemini.
        
        Yields:
            Chunks of generated text
        """
        import time
        start_time = time.time()
        
        full_prompt = ""
        
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        if context:
            full_prompt += f"""Based on the following document content, please answer the user's question:

DOCUMENT CONTENT:
{self._trim_context(context)}

USER QUESTION: {prompt}
"""
        else:
            full_prompt = prompt
        
        temperature = kwargs.get('temperature', self.temperature)
        
        generation_config = google_types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=settings.TOP_P
        )
        
        try:
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            response_time = time.time() - start_time
            
            yield LLMResponse(
                text=full_response,
                model=self.model,
                provider=LLMProvider.GEMINI.value,
                tokens_used=len(full_response) // 4,
                finish_reason="stop",
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise


class LLMServiceFactory:
    """Factory for creating LLM services."""
    
    _services: Dict[LLMProvider, BaseLLMService] = {}
    
    @classmethod
    def get_service(cls, provider: Optional[str] = None) -> BaseLLMService:
        """
        Get LLM service for the specified provider.
        
        Args:
            provider: Provider name ('openai' or 'gemini')
            
        Returns:
            LLM service instance
        """
        provider = (provider or settings.LLM_PROVIDER).lower()
        
        if provider == LLMProvider.OPENAI.value:
            if LLMProvider.OPENAI not in cls._services:
                cls._services[LLMProvider.OPENAI] = OpenAIService()
            return cls._services[LLMProvider.OPENAI]
        
        elif provider == LLMProvider.GEMINI.value:
            if LLMProvider.GEMINI not in cls._services:
                cls._services[LLMProvider.GEMINI] = GeminiService()
            return cls._services[LLMProvider.GEMINI]
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    @classmethod
    def set_provider(cls, provider: str):
        """Set the default LLM provider."""
        settings.LLM_PROVIDER = provider.lower()
        # Clear cached service
        if provider.lower() == LLMProvider.OPENAI.value:
            cls._services.pop(LLMProvider.OPENAI, None)
        elif provider.lower() == LLMProvider.GEMINI.value:
            cls._services.pop(LLMProvider.GEMINI, None)
    
    @classmethod
    def get_current_provider(cls) -> str:
        """Get the current LLM provider."""
        return settings.LLM_PROVIDER


def get_llm_service(provider: Optional[str] = None) -> BaseLLMService:
    """
    Convenience function to get LLM service.
    
    Args:
        provider: Provider name (optional, uses default if not specified)
        
    Returns:
        LLM service instance
    """
    return LLMServiceFactory.get_service(provider)


# System prompts for different query types
SYSTEM_PROMPTS = {
    "general": """You are a helpful AI assistant that answers questions about documents.
Please provide accurate, concise answers based on the document content provided.
If the answer is not in the document, say so clearly.""",
    
    "summarization": """You are an expert at summarizing academic papers and documents.
Please provide a clear, structured summary that includes:
1. Main contributions
2. Key findings
3. Methodology
4. Conclusions
Be concise but comprehensive.""",
    
    "extraction": """You are an expert at extracting specific information from documents.
Please extract the requested information accurately.
If the information is not found, say so clearly.
Format your answer clearly.""",
    
    "comparison": """You are an expert at comparing information across documents.
Please provide a clear comparison highlighting similarities and differences.
Use tables or bullet points for clarity.""",
    
    "explanation": """You are an expert at explaining complex concepts.
Please explain the requested concept clearly and thoroughly.
Use examples from the document where possible."""
}

