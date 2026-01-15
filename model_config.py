"""
Model Configuration and Provider Management
Centralizes all LLM model configurations and provider settings
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelTier(Enum):
    """Model performance/cost tiers"""
    RECOMMENDED = "Recommended"
    BUDGET = "Budget"
    PREMIUM = "Premium"
    LEGACY = "Legacy"


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model"""
    id: str
    name: str
    provider: LLMProvider
    tier: ModelTier
    description: str
    context_window: int
    supports_streaming: bool
    supports_function_calling: bool
    cost_per_1m_input: float
    cost_per_1m_output: float
    recommended_for_sql: bool = True
    
    def get_display_name(self) -> str:
        """Get formatted display name for UI"""
        return f"{self.tier.value} {self.name} - {self.description}"
    
    def get_cost_info(self) -> str:
        """Get formatted cost information"""
        return f"${self.cost_per_1m_input:.2f}/${self.cost_per_1m_output:.2f} per 1M tokens"
    
    def get_provider_name(self) -> str:
        """Get provider display name"""
        return self.provider.value.capitalize()


# Model catalog with all supported models
MODEL_CATALOG: Dict[str, ModelConfig] = {
    # OpenAI Models
    "gpt-4o": ModelConfig(
        id="gpt-4o",
        name="GPT-4o",
        provider=LLMProvider.OPENAI,
        tier=ModelTier.RECOMMENDED,
        description="Best overall - Optimal balance of speed, accuracy, and cost",
        context_window=128000,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
        recommended_for_sql=True
    ),
    "gpt-4o-mini": ModelConfig(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider=LLMProvider.OPENAI,
        tier=ModelTier.BUDGET,
        description="Excellent value - Fast, accurate, and very affordable",
        context_window=128000,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        recommended_for_sql=True
    ),
    "gpt-4-turbo": ModelConfig(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider=LLMProvider.OPENAI,
        tier=ModelTier.PREMIUM,
        description="Maximum accuracy - Best for complex queries",
        context_window=128000,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1m_input=10.00,
        cost_per_1m_output=30.00,
        recommended_for_sql=True
    ),
    "gpt-4-turbo-2024-04-09": ModelConfig(
        id="gpt-4-turbo-2024-04-09",
        name="GPT-4 Turbo (Apr 2024)",
        provider=LLMProvider.OPENAI,
        tier=ModelTier.PREMIUM,
        description="Specific turbo version - Stable and reliable",
        context_window=128000,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1m_input=10.00,
        cost_per_1m_output=30.00,
        recommended_for_sql=True
    ),
    "gpt-3.5-turbo": ModelConfig(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider=LLMProvider.OPENAI,
        tier=ModelTier.LEGACY,
        description="Legacy - Fastest but less accurate (not recommended)",
        context_window=16385,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1m_input=0.50,
        cost_per_1m_output=1.50,
        recommended_for_sql=False
    ),
    
    # Anthropic Claude Models
    "claude-sonnet-4-20250514": ModelConfig(
        id="claude-sonnet-4-20250514",
        name="Claude 4 Sonnet",
        provider=LLMProvider.ANTHROPIC,
        tier=ModelTier.RECOMMENDED,
        description="Excellent reasoning - Great for complex SQL",
        context_window=200000,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
        recommended_for_sql=True
    ),
    "claude-opus-4-1-20250805": ModelConfig(
        id="claude-opus-4-1-20250805",
        name="Claude 4 Opus",
        provider=LLMProvider.ANTHROPIC,
        tier=ModelTier.PREMIUM,
        description="Most capable - Best for very complex scenarios",
        context_window=200000,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1m_input=15.00,
        cost_per_1m_output=75.00,
        recommended_for_sql=True
    ),
    "claude-3-haiku-20240307": ModelConfig(
        id="claude-3-haiku-20240307",
        name="Claude 3 Haiku",
        provider=LLMProvider.ANTHROPIC,
        tier=ModelTier.BUDGET,
        description="Fast and economical - Good for simple queries",
        context_window=200000,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1m_input=0.25,
        cost_per_1m_output=1.25,
        recommended_for_sql=True
    ),
}


def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """
    Get configuration for a specific model
    
    Args:
        model_id: Model identifier
        
    Returns:
        ModelConfig if found, None otherwise
    """
    return MODEL_CATALOG.get(model_id)


def get_models_by_provider(provider: LLMProvider) -> List[ModelConfig]:
    """
    Get all models for a specific provider
    
    Args:
        provider: LLM provider
        
    Returns:
        List of ModelConfig for the provider
    """
    return [
        config for config in MODEL_CATALOG.values()
        if config.provider == provider
    ]


def get_recommended_models() -> List[ModelConfig]:
    """
    Get all recommended models (excluding legacy)
    
    Returns:
        List of recommended ModelConfig objects
    """
    return [
        config for config in MODEL_CATALOG.values()
        if config.recommended_for_sql and config.tier != ModelTier.LEGACY
    ]


def get_all_model_ids() -> List[str]:
    """Get list of all supported model IDs"""
    return list(MODEL_CATALOG.keys())


def get_default_model_id() -> str:
    """Get the default/recommended model ID"""
    return "gpt-4o-mini"  # Best value proposition


def get_models_for_ui() -> Dict[str, str]:
    """
    Get models formatted for UI display
    
    Returns:
        Dictionary mapping model_id to display string
    """
    return {
        model_id: config.get_display_name()
        for model_id, config in MODEL_CATALOG.items()
    }


def get_provider_for_model(model_id: str) -> Optional[LLMProvider]:
    """
    Get the provider for a specific model
    
    Args:
        model_id: Model identifier
        
    Returns:
        LLMProvider if found, None otherwise
    """
    config = get_model_config(model_id)
    return config.provider if config else None


def validate_model_id(model_id: str) -> bool:
    """
    Check if a model ID is valid and supported
    
    Args:
        model_id: Model identifier to validate
        
    Returns:
        True if valid, False otherwise
    """
    return model_id in MODEL_CATALOG


def get_model_info_text(model_id: str) -> str:
    """
    Get detailed information text for a model
    
    Args:
        model_id: Model identifier
        
    Returns:
        Formatted information string
    """
    config = get_model_config(model_id)
    if not config:
        return "Unknown model"
    
    return f"""
**{config.name}** ({config.get_provider_name()})

{config.description}

- **Context Window:** {config.context_window:,} tokens
- **Streaming:** {'Ok' if config.supports_streaming else 'Fail'}
- **Function Calling:** {'Ok' if config.supports_function_calling else 'Fail'}
- **Cost:** {config.get_cost_info()}
- **Recommended for SQL:** {'Ok' if config.recommended_for_sql else 'Fail'}
    """.strip()


# Provider-specific configuration
PROVIDER_CONFIGS = {
    LLMProvider.OPENAI: {
        "api_key_env_var": "OPENAI_API_KEY",
        "api_key_label": "OpenAI API Key",
        "documentation_url": "https://platform.openai.com/docs/models",
        "requires_separate_key": False,
    },
    LLMProvider.ANTHROPIC: {
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "api_key_label": "Anthropic API Key",
        "documentation_url": "https://docs.anthropic.com/claude/docs",
        "requires_separate_key": True,
    }
}


def get_provider_config(provider: LLMProvider) -> dict:
    """Get configuration for a provider"""
    return PROVIDER_CONFIGS.get(provider, {})


def get_required_api_keys(model_ids: List[str]) -> List[LLMProvider]:
    """
    Determine which API keys are required for a list of models
    
    Args:
        model_ids: List of model identifiers
        
    Returns:
        List of required LLMProvider enums
    """
    providers = set()
    for model_id in model_ids:
        provider = get_provider_for_model(model_id)
        if provider:
            providers.add(provider)
    return list(providers)