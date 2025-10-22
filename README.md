# Text-to-Image Generation App

A FastAPI-based intelligent image generation application that uses DSPy optimization pipeline with LLM evaluation to iteratively improve image generation prompts. The system combines Gemini 2.5 Pro for evaluation and FAL AI Imagen4 for high-quality image generation.

## Features

- **DSPy Optimization Pipeline**: Iterative prompt improvement using structured evaluation
- **Multi-Model Support**: Integration with various AI image generation models
- **LLM Evaluation**: Gemini 2.5 Pro provides detailed feedback on generated images
- **RAG-Enhanced Optimization**: Learn from historical successful prompts
- **Web Interface**: Clean, responsive UI for image generation and history viewing
- **Comprehensive Logging**: CSV-based tracking of all generations with detailed metrics

## Project Structure

```
text_to_image_generation/
├── app/                    # Main FastAPI application
│   ├── main.py            # FastAPI server entry point
│   ├── data/              # Configuration and templates
│   ├── imagesdata/        # Generated images and CSV logs
│   ├── models/            # Data models and managers
│   ├── routers/           # API route handlers
│   ├── services/          # Business logic and DSPy optimization
│   ├── static/            # CSS and JavaScript files
│   └── templates/         # HTML templates
├── modelexperiment/       # Multi-model testing tools
├── optimization/          # RAG-based prompt optimization
└── README.md             # This file
```

## App Directory Structure

### Core Application (`app/`)

- **`main.py`** - FastAPI server with Jinja2 templates and static file serving
- **`data/`** - Contains prompt templates and configuration files
- **`imagesdata/`** - Storage for generated images and CSV logs with evaluation data

### Models (`app/models/`)
- **`image_manager.py`** - Handles image storage, base64 conversion, and CSV logging
- **`prompt_manager.py`** - Manages prompt templates and formatting

### Routers (`app/routers/`)
- **`image_generator_router.py`** - HTTP request routing and error handling for image generation endpoints

### Services (`app/services/`)
- **`dspy_optimization.py`** - Core DSPy optimization pipeline with iterative improvement
- **`image_generator_service.py`** - Business logic for image generation workflow

### Frontend (`app/static/` & `app/templates/`)
- **`templates/image_generator.html`** - Main web interface
- **`static/style.css`** - Styling for the web interface
- **`static/script.js`** - Frontend JavaScript for API interactions

## Key Technologies

- **FastAPI** - Modern web framework for building APIs
- **DSPy** - Framework for optimizing language model pipelines
- **FAL AI** - Image generation service (Imagen4)
- **Google Gemini 2.5 Pro** - LLM for image evaluation and feedback
- **FAISS** - Vector database for similarity search
- **Upstage** - Language model for RAG optimization

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd text_to_image_generation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file with:
   ```
   FAL_KEY=your_fal_ai_api_key
   GOOGLE_API_KEY=your_gemini_api_key
   UPSTAGE_API_KEY=your_upstage_api_key
   ```

4. **Run the application**
   ```bash
   cd app
   python main.py
   ```

5. **Access the web interface**
   Open `http://localhost:8000` in your browser

## Usage

### Web Interface
1. Enter your image description in the text input
2. Click "Generate Image" to start the DSPy optimization process
3. View the iterative improvements and final result
4. Browse generation history and evaluation metrics

### API Endpoints
- `POST /api/generate_image` - Generate image with DSPy optimization
- `GET /api/get_images` - Retrieve image generation history
- `GET /api/get_csv_log` - Download CSV log of all generations

## DSPy Optimization Process

1. **Initial Generation** - Create image from user prompt
2. **LLM Evaluation** - Gemini 2.5 Pro evaluates image quality across multiple dimensions
3. **Feedback Analysis** - Identify areas for improvement
4. **Prompt Refinement** - Generate improved prompt based on feedback
5. **Iteration** - Repeat up to 5 times until optimal result

### Evaluation Criteria
- Subject accuracy and presence
- Art type/medium matching
- Art style consistency
- Art movement alignment
- Conflict detection
- Overall quality score (1-10)

## Additional Tools

### Model Experiments (`modelexperiment/`)
Test and compare different AI image generation models with the same prompts. See [modelexperiment/README.md](modelexperiment/README.md) for details.

### RAG Optimization (`optimization/`)
Enhance prompt generation using historical data and few-shot learning. See [optimization/README.md](optimization/README.md) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.