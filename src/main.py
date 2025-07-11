#!/usr/bin/env python3
"""
Main entry point for the AI 3D Asset Generator application.

This is the main entry point that launches the Gradio web application.
The application has been refactored into modular components for better maintainability.
"""

import asyncio
import logging
import sys
from pathlib import Path

import structlog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.core.app import AssetGenerationApp
    from src.ui import create_app_interface
    from src.utils.env_config import get_settings
except ImportError as e:
    logger.error("Failed to import required modules", error=str(e))
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed and the project structure is correct.")
    sys.exit(1)


async def initialize_app() -> AssetGenerationApp:
    """Initialize the application."""
    logger.info("Initializing AI 3D Asset Generator application")
    
    try:
        app = AssetGenerationApp()
        await app.initialize()
        logger.info("Application initialized successfully")
        return app
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise


def main():
    """Main entry point for the application."""
    print("🚀 Starting AI 3D Asset Generator (Refactored UI)")
    print("=" * 50)
    
    try:
        # Get configuration settings
        settings = get_settings()
        
        # Initialize the application
        app = asyncio.run(initialize_app())
        
        # Create the new refactored Gradio interface
        print("🌐 Creating refactored web interface...")
        interface = create_app_interface(app)
        
        # Launch the application with settings from environment
        print(f"🎉 Launching application on http://{settings.gradio_host}:{settings.gradio_port}")
        print("📱 Interface: Modern, simplified UI with improved UX")
        print("=" * 50)
        
        interface.launch(
            server_name=settings.gradio_host,
            server_port=settings.gradio_port,
            share=settings.gradio_share,
            debug=settings.gradio_debug,
            show_error=settings.gradio_show_error,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        logger.error("Application failed to start", error=str(e))
        print(f"❌ Error: {e}")
        print("💡 Tip: Check that all dependencies are installed and configuration is correct")
        sys.exit(1)
    finally:
        print("🧹 Cleaning up...")
        # Cleanup would happen here if needed


if __name__ == "__main__":
    main()
