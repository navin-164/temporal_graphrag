# ==============================================================================
# Temporal Hybrid GraphRAG with Multi-Agent AI — Settings & Configurations
# Target: Python 3.11.x (Pydantic v2)
# ==============================================================================

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Centralized configuration management for paths, database credentials,
    and LLM provider API keys using Pydantic v2 BaseSettings.
    """
    # Project Root Directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # LLM & Embedding Providers API Keys
    GROQ_API_KEY: str = Field(default="", env="GROQ_API_KEY")
    GOOGLE_API_KEY: str = Field(default="", env="GOOGLE_API_KEY")
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")

    # Neo4j Graph Database Configuration
    NEO4J_URI: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    NEO4J_USER: str = Field(default="neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field(default="password", env="NEO4J_PASSWORD")

    # Model defaults
    DEFAULT_LLM_PROVIDER: str = Field(default="groq", env="DEFAULT_LLM_PROVIDER")
    DEFAULT_EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        env="DEFAULT_EMBEDDING_MODEL"
    )

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate global settings object
settings = Settings()