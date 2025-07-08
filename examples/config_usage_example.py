"""
Example application demonstrating the configuration system usage.

This example shows how to:
1. Initialize the configuration system
2. Use configuration in different parts of the application
3. Handle configuration-based conditional logic
4. Access nested configuration values
"""

import asyncio
import logging
from pathlib import Path

# Import our configuration system
from src.utils.config_init import initialize_configuration, get_configuration_summary
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class LLMService:
    """Example service that uses LLM configuration."""
    
    def __init__(self):
        self.config = get_config().llm
        primary_provider = self.config.get_primary_provider()
        logger.info(f"LLM Service initialized with primary provider: {primary_provider.name}, model: {primary_provider.model}")
    
    async def generate_description(self, prompt: str) -> str:
        """Generate enhanced description using configured LLM settings."""
        primary_provider = self.config.get_primary_provider()
        fallback_provider = self.config.get_fallback_provider()
        
        if not primary_provider.api_key and (not fallback_provider or not fallback_provider.api_key):
            raise ValueError("No LLM providers configured with API keys")
        
        # Use configuration values from primary provider
        max_tokens = primary_provider.max_tokens
        temperature = primary_provider.temperature
        timeout = primary_provider.timeout
        
        logger.info(f"Generating with {primary_provider.name}: max_tokens={max_tokens}, temperature={temperature}")
        
        # Simulate API call
        await asyncio.sleep(0.1)
        return f"Enhanced description for: {prompt}"


class ObjectStorageService:
    """Example service that uses object storage configuration."""
    
    def __init__(self):
        self.config = get_config().object_storage
        logger.info(f"Object Storage initialized: provider={self.config.provider}, bucket={self.config.bucket_name}")
    
    async def upload_asset(self, file_path: str, asset_name: str) -> str:
        """Upload asset to configured object storage."""
        if not self.config.access_key_id:
            logger.warning("Object storage credentials not configured, using mock upload")
            return f"mock://{self.config.bucket_name}/{asset_name}"
        
        # Check file size
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        if file_size > self.config.max_file_size:
            raise ValueError(f"File size {file_size} exceeds limit {self.config.max_file_size}")
        
        # Check file extension
        file_ext = Path(file_path).suffix
        if file_ext not in self.config.allowed_extensions:
            raise ValueError(f"File extension {file_ext} not allowed")
        
        logger.info(f"Uploading {file_path} to {self.config.bucket_name} via {self.config.provider}")
        
        # Simulate upload
        await asyncio.sleep(0.2)
        
        # Return URL using the configuration method
        return self.config.get_public_url(asset_name)


class ThreeDGenerationService:
    """Example service that uses 3D generation configuration."""
    
    def __init__(self):
        self.config = get_config().threed_generation
        logger.info(f"3D Generation Service initialized with primary: {self.config.primary_service}")
    
    async def generate_3d_model(self, description: str) -> dict:
        """Generate 3D model using configured service."""
        if self.config.primary_service == "meshy" and not self.config.meshy_api_key:
            if self.config.fallback_service == "kaedim" and self.config.kaedim_api_key:
                logger.warning("Primary service unavailable, using fallback")
                service = "kaedim"
            else:
                raise ValueError("No 3D generation service available")
        else:
            service = self.config.primary_service
        
        logger.info(f"Generating 3D model using {service} with quality: {self.config.quality_preset}")
        
        # Simulate generation based on configuration
        timeout = self.config.generation_timeout
        polling_interval = self.config.polling_interval
        
        logger.info(f"Generation timeout: {timeout}s, polling interval: {polling_interval}s")
        
        # Simulate async generation
        await asyncio.sleep(0.5)
        
        return {
            "service": service,
            "description": description,
            "quality": self.config.quality_preset,
            "formats": self.config.output_formats,
            "texture_resolution": self.config.texture_resolution,
            "model_url": f"https://example.com/models/generated-model.{self.config.output_formats[0]}"
        }


class GradioApp:
    """Example Gradio application using configuration."""
    
    def __init__(self):
        self.config = get_config().gradio
        self.llm_service = LLMService()
        self.storage_service = ObjectStorageService()
        self.generation_service = ThreeDGenerationService()
        
        logger.info(f"Gradio app configured for {self.config.host}:{self.config.port}")
    
    async def process_asset_request(self, user_input: str) -> dict:
        """Process a complete asset generation request."""
        try:
            # Step 1: Enhance description using LLM
            enhanced_description = await self.llm_service.generate_description(user_input)
            
            # Step 2: Generate 3D model
            model_data = await self.generation_service.generate_3d_model(enhanced_description)
            
            # Step 3: Upload to cloud storage (simulate)
            model_url = await self.storage_service.upload_asset(
                "/tmp/mock_model.obj", 
                "generated_model.obj"
            )
            
            return {
                "status": "success",
                "original_input": user_input,
                "enhanced_description": enhanced_description,
                "model_data": model_data,
                "download_url": model_url
            }
            
        except Exception as e:
            logger.error(f"Asset generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_interface_config(self) -> dict:
        """Get Gradio interface configuration."""
        return {
            "title": self.config.title,
            "description": self.config.description,
            "theme": self.config.theme,
            "show_error": self.config.show_error,
            "max_file_size": self.config.max_file_size
        }


async def main():
    """Main application entry point."""
    try:
        # Initialize configuration system
        logger.info("Initializing configuration system...")
        config = initialize_configuration()
        
        logger.info(f"Configuration loaded for environment: {config.environment}")
        
        # Display configuration summary (safe, no sensitive data)
        summary = get_configuration_summary(config)
        logger.info(f"Configuration summary: {summary}")
        
        # Initialize services
        logger.info("Initializing services...")
        app = GradioApp()
        
        # Example usage
        logger.info("Processing example asset request...")
        result = await app.process_asset_request("magical healing potion")
        
        logger.info("Asset generation result:")
        for key, value in result.items():
            if key != "model_data":  # Don't log large model data
                logger.info(f"  {key}: {value}")
        
        # Show interface configuration
        interface_config = app.get_interface_config()
        logger.info(f"Gradio interface config: {interface_config}")
        
        # Example of environment-specific behavior
        if config.environment == "development":
            logger.info("Development mode: enabling debug features")
        elif config.environment == "production":
            logger.info("Production mode: optimizing for performance")
        
        logger.info("Application initialization complete!")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise


def demonstrate_configuration_access():
    """Demonstrate different ways to access configuration."""
    config = get_config()
    
    # Direct access to configuration sections
    print("=== Configuration Access Examples ===")
    print(f"Environment: {config.environment}")
    print(f"Debug mode: {config.debug}")
    print(f"App name: {config.app_name}")
    print(f"App version: {config.app_version}")
    
    # LLM configuration
    print(f"\nLLM Configuration:")
    primary_provider = config.llm.get_primary_provider()
    fallback_provider = config.llm.get_fallback_provider()
    print(f"  Primary provider: {config.llm.primary_provider}")
    print(f"  Primary model: {primary_provider.model}")
    print(f"  Primary base URL: {primary_provider.base_url or 'default'}")
    print(f"  Max tokens: {primary_provider.max_tokens}")
    print(f"  Temperature: {primary_provider.temperature}")
    print(f"  Has primary API key: {bool(primary_provider.api_key)}")
    if fallback_provider:
        print(f"  Fallback provider: {config.llm.fallback_provider}")
        print(f"  Has fallback API key: {bool(fallback_provider.api_key)}")
    
    # Object storage configuration
    print(f"\nObject Storage Configuration:")
    print(f"  Provider: {config.object_storage.provider}")
    print(f"  Region: {config.object_storage.region}")
    print(f"  Bucket: {config.object_storage.bucket_name}")
    print(f"  Endpoint URL: {config.object_storage.get_endpoint_url() or 'default'}")
    print(f"  Max file size: {config.object_storage.max_file_size / 1024 / 1024:.1f} MB")
    print(f"  Allowed extensions: {config.object_storage.allowed_extensions}")
    print(f"  Custom domain: {config.object_storage.custom_domain or 'none'}")
    
    # 3D generation configuration
    print(f"\n3D Generation Configuration:")
    print(f"  Primary service: {config.threed_generation.primary_service}")
    print(f"  Quality preset: {config.threed_generation.quality_preset}")
    print(f"  Output formats: {config.threed_generation.output_formats}")
    print(f"  Texture resolution: {config.threed_generation.texture_resolution}")
    
    # Gradio configuration
    print(f"\nGradio Configuration:")
    print(f"  Host: {config.gradio.host}")
    print(f"  Port: {config.gradio.port}")
    print(f"  Debug: {config.gradio.debug}")
    print(f"  Title: {config.gradio.title}")
    
    # Security configuration (be careful with sensitive data)
    print(f"\nSecurity Configuration:")
    print(f"  Secret key length: {len(config.security.secret_key)} chars")
    print(f"  Allowed hosts: {config.security.allowed_hosts}")
    print(f"  Rate limit: {config.security.rate_limit_requests} req/min")


if __name__ == "__main__":
    # Set up basic logging for this example
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Initialize configuration
        initialize_configuration()
        
        # Demonstrate configuration access
        demonstrate_configuration_access()
        
        # Run async main function
        asyncio.run(main())
        
    except Exception as e:
        logger.error(f"Example application failed: {e}")
        raise
