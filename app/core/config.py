"""
Configuration and constants for Resonate Microservice.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4.1-mini"
    
    # Server Configuration
    PORT: int = int(os.getenv("PORT", 10000))
    HOST: str = "0.0.0.0"
    
    # Request Configuration
    PDF_DOWNLOAD_TIMEOUT: int = 20
    
    # AI Temperature Settings
    TEMPERATURE_EXTRACTION: float = 0.0  # Precise extraction
    TEMPERATURE_CREATIVE: float = 0.7    # Creative generation
    TEMPERATURE_ANALYSIS: float = 0.5    # Balanced analysis
    
    # PDF Processing
    PDF_PREVIEW_PAGES: int = 2  # Pages to use for classification
    PDF_RENDER_SCALE: int = 2   # Render quality multiplier

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration on startup."""
        missing = []
        
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please check your .env file."
            )


settings = Settings()
