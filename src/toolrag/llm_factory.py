"""
LLM Factory - Unified interface for multiple LLM providers

Supports:
- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude)
- Google Gemini
- Ollama (local models)
"""

import os
from typing import Optional, Literal
from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv("../.env", override=True)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def create_chat_model(self, model: str, temperature: float = 0.7) -> BaseChatModel:
        """Create and return a chat model instance"""
        pass
    
    @abstractmethod
    def validate_api_key(self) -> bool:
        """Validate that required API keys/configs are set"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider"""
    
    def validate_api_key(self) -> bool:
        """Check if OpenAI API key is set"""
        return bool(os.getenv("OPENAI_API_KEY"))
    
    def create_chat_model(self, model: str = "gpt-4o-mini", temperature: float = 0.7) -> BaseChatModel:
        """Create OpenAI chat model"""
        if not self.validate_api_key():
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM Provider"""
    
    def validate_api_key(self) -> bool:
        """Check if Anthropic API key is set"""
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    
    def create_chat_model(self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.7) -> BaseChatModel:
        """Create Anthropic Claude chat model"""
        if not self.validate_api_key():
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        from langchain_anthropic import ChatAnthropic
        
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )


class GeminiProvider(LLMProvider):
    """Google Gemini LLM Provider"""
    
    def validate_api_key(self) -> bool:
        """Check if Gemini API key is set"""
        return bool(os.getenv("GOOGLE_API_KEY"))
    
    def create_chat_model(self, model: str = "gemini-2.0-flash", temperature: float = 0.7) -> BaseChatModel:
        """Create Google Gemini chat model"""
        if not self.validate_api_key():
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GOOGLE_API_KEY")
        )


class OllamaProvider(LLMProvider):
    """Ollama Local LLM Provider"""
    
    def validate_api_key(self) -> bool:
        """Check if Ollama is configured (endpoint should be accessible)"""
        # Ollama typically runs locally, so we just check if the env var is set or use default
        return True  # Ollama doesn't require API keys
    
    def create_chat_model(self, model: str = "gpt-oss:20b", temperature: float = 0.7) -> BaseChatModel:
        """Create Ollama chat model"""
        from langchain_ollama import ChatOllama
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature
        )


# Provider registry
PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-2.0-flash",
    "ollama": "gpt-oss:20b",
}


class LLMFactory:
    """Factory for creating LLM instances from different providers"""
    
    _default_provider: str = "openai"
    
    @classmethod
    def set_default_provider(cls, provider: Literal["openai", "anthropic", "gemini", "ollama"]):
        """Set the default LLM provider"""
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")
        cls._default_provider = provider
    
    @classmethod
    def get_default_provider(cls) -> str:
        """Get the current default provider"""
        return cls._default_provider
    
    @classmethod
    def create(
        cls,
        provider: Optional[Literal["openai", "anthropic", "gemini", "ollama"]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> BaseChatModel:
        """
        Create an LLM instance
        
        Args:
            provider: LLM provider ("openai", "anthropic", "gemini", "ollama")
                     Uses default if not specified
            model: Model name. Uses provider default if not specified
            temperature: Model temperature (0.0-1.0)
        
        Returns:
            A BaseChatModel instance ready to use
            
        Examples:
            # Using default provider
            llm = LLMFactory.create()
            
            # Using specific provider
            llm = LLMFactory.create(provider="anthropic")
            
            # Using specific model
            llm = LLMFactory.create(provider="openai", model="gpt-4")
        """
        if provider is None:
            provider = cls._default_provider
        
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")
        
        provider_class = PROVIDERS[provider]
        provider_instance = provider_class()
        
        # Validate API key/config
        if not provider_instance.validate_api_key():
            raise ValueError(
                f"Required credentials for {provider} not found. "
                f"Please set the appropriate environment variable(s)."
            )
        
        # Use default model if not specified
        if model is None:
            model = DEFAULT_MODELS[provider]
        
        return provider_instance.create_chat_model(model=model, temperature=temperature)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all available providers"""
        return list(PROVIDERS.keys())
    
    @classmethod
    def get_available_providers(cls) -> dict[str, bool]:
        """Get available providers and their API key status"""
        available = {}
        for name, provider_class in PROVIDERS.items():
            provider_instance = provider_class()
            available[name] = provider_instance.validate_api_key()
        return available


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7
) -> BaseChatModel:
    """
    Convenience function to create an LLM instance
    
    Args:
        provider: LLM provider name ("openai", "anthropic", "gemini", "ollama")
        model: Model name
        temperature: Model temperature
        
    Returns:
        A BaseChatModel instance
    """
    return LLMFactory.create(provider=provider, model=model, temperature=temperature)
