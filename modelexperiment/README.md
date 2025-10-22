# Model Experiment

This directory contains experimental tools for testing and comparing different AI image generation models.

## Files

- `experiment_fal_generator.py` - Multi-model image generation experiment tool using FAL AI

## FAL Experiment Generator

The `experiment_fal_generator.py` script allows you to test and compare multiple AI image generation models with the same prompt to evaluate their performance and output quality.

### Features

- Test multiple models with a single prompt
- Compare generation time and quality across models
- Save generated images locally with metadata
- Interactive CLI interface

### Supported Models

- Imagen4 Ultra
- Dreamina v3.1
- Seedream v4
- FLUX Pro v1.1 Ultra
- Ideogram v3

### Usage

1. Set your FAL AI API key in the `.env` file:
   ```
   FAL_KEY=your_api_key_here
   ```

2. Run the experiment:
   ```bash
   python experiment_fal_generator.py
   ```

3. Follow the interactive prompts to:
   - Enter your image generation prompt
   - Select which models to test
   - Specify output directory

### Output

Generated images are saved to `generated_images/` with:
- Timestamped filenames
- Model-specific naming
- Metadata including generation time and parameters

This tool is useful for model evaluation, performance benchmarking, and finding the best model for specific use cases.