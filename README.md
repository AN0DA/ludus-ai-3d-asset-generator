# Ludus AI 3D Asset Generator

An AI-powered application that generates 3D game assets from text descriptions. The application combines Large Language Models (LLMs) for description enhancement with specialized 3D generation APIs, providing a complete pipeline from text input to downloadable 3D models.

## 🚀 Quick Start

### Using Docker (Recommended)

The application is pre-built and available on GitHub Container Registry:

```bash
# Pull the latest image
docker pull ghcr.io/an0da/3d-asset-generator:latest

# Run the application
docker run --rm -p 7860:7860 -v "/path/to/your/.env:/app/.env" ghcr.io/an0da/3d-asset-generator:latest
```

**Note**: There's currently an issue with passing environment variables via `--env-file`. Mount your `.env` file directly to `/app/.env` as shown above.

The application will be available at `http://localhost:7860`

### Configuration

Create a `.env` file with the following required API keys and configuration:

```bash
# OpenAI-compatible LLM Provider
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # or your provider's URL
OPENAI_MODEL=gpt-4

# Meshy AI (3D Generation)
MESHY_API_KEY=your_meshy_api_key_here

# S3-Compatible Storage (optional)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
S3_BUCKET_NAME=your_bucket_name
S3_ENDPOINT_URL=your_s3_endpoint  # for non-AWS providers
```

## 🛠️ Technical Approach & Design Decisions

### LLM Integration

**Choice**: Generic OpenAI-compatible implementation

- **Why**: Maximum flexibility - supports multiple LLM providers (OpenAI, Anthropic, local models, etc.)
- **Development**: Primarily tested with GitHub Copilot and GPT-4.1

**Prompt Construction Strategy**:

The application uses a sophisticated multi-layered prompt engineering approach:

1. **Structured Templates**: Pydantic models define strict output schemas for consistent 3D asset descriptions
2. **Context-Aware Enhancement**: Base user descriptions are enriched with game-specific terminology, technical specifications, and visual details
3. **Iterative Refinement**: Multi-step prompt chains that first analyze the input, then generate comprehensive descriptions
4. **Format Validation**: Built-in validation ensures prompts meet downstream API requirements (length limits, format constraints)

The prompts are designed to transform simple descriptions like "medieval sword" into detailed specifications including materials, dimensions, style references, and technical parameters suitable for 3D generation.

### 3D Asset Generation

**Choice**: Meshy AI

- **Why**:
  - High-quality model output
  - Developer-friendly API
  - Generous trial period for development
  - Good documentation and support

**Implementation**:

- Asynchronous generation pipeline with status polling
- Automatic retry logic for failed generations
- Support for multiple output formats (GLB, OBJ, etc.)
- Progress tracking and user feedback

**Challenges**:

- **Prompt Length Limitations**: Meshy has strict prompt length limits. Current implementation truncates prompts, which affects both cost efficiency and output quality. This needs improvement with better prompt validation and optimization.
- **API Rate Limiting**: Implemented exponential backoff and queuing system

### Cloud Storage Integration

**Choice**: Generic S3-compatible implementation using boto3

- **Why**:
  - Universal compatibility - works with AWS S3, MinIO, DigitalOcean Spaces, etc.
  - Mature library with excellent async support
  - Easy configuration switching between providers

**Implementation**:

- Configurable endpoint URLs for different providers
- Automatic file organization by date and asset type
- Presigned URL generation for secure downloads
- Cleanup routines for temporary files

### Dynamic Asset Rendering & UI

**Framework**: Gradio 5.x

- **Components Used**:
  - `gr.Textbox` for user input
  - `gr.Dropdown` for asset selection and management
  - `gr.File` for downloads
  - `gr.Model3D` for 3D preview (when working properly)
  - Custom CSS for dark mode and responsive design

**Challenges**:

- **State Management**: Complex async state synchronization between generation, storage, and UI components
- **Hot Reloading**: UI doesn't refresh dynamically (see Known Issues)

## ⚠️ Known Issues

### MAJOR BUG - UI Refresh Problem

**Issue**: Results window does not refresh dynamically. To see newly generated models in the list, you need to restart the application.

**Additional UX Issues**:

- Even when a model is selected in the dropdown, you need to reselect it for it to load properly
- The overall user experience is currently poor due to these state management issues

### Other Issues

- Environment variable passing to Docker container needs improvement
- SSL certificate handling needs fixes
- Dark mode implementation is incomplete
- Prompt validation for Meshy length limits needs implementation

## 🧪 Development & Testing

### Local Development

```bash
# Install dependencies with uv
uv sync

# Run tests
make test

# Run linting
make lint
make mypy

# Start development server
uv run python src/main.py
```

### Project Structure

```text
src/
├── core/           # Core application logic and session management
├── generators/     # LLM and 3D asset generation modules
├── models/         # Pydantic data models
├── storage/        # Cloud storage implementations
├── ui/            # Gradio interface components
└── utils/         # Utilities and configuration
```

## 🚀 Future Development Ideas

### 1. Code Quality & Maintenance

- **Code Cleanup**: Remove obsolete functions, split oversized modules, merge related components
- **Error Resolution**: Fix remaining lint errors and mypy warnings

### 2. Bug Fixes & Core Improvements

- **Environment Variables**: Fix Docker container environment variable handling
- **Prompt Optimization**: Implement proper prompt validation for Meshy length limits instead of truncation
- **Module Decoupling**: Make LLM, asset generators, and storage providers loosely coupled with specific implementations
- **UI Overhaul**: Fix dynamic refreshing, improve UX, complete dark mode implementation
- **SSL**: Resolve SSL certificate issues
- **Testing**: Add comprehensive test coverage

### 3. New Features

- **Image-to-3D**: Support for generating 3D models from reference images
- **Settings Management**: In-app configuration of API keys and service settings
- **Multiple Providers**: Additional LLM, asset generation, and storage provider implementations
- **Asset Management**: Better organization, tagging, and search capabilities
- **Batch Processing**: Generate multiple assets from batch descriptions

### 4. Infrastructure

- **Documentation**: Comprehensive API documentation and user guides
- **Monitoring**: Application health monitoring and metrics
- **Scaling**: Multi-instance deployment support
