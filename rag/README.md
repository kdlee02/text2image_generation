# Optimization

This directory contains RAG (Retrieval-Augmented Generation) tools for optimizing image generation prompts using historical data and few-shot learning techniques.

## Files

- `main.py` - Main script that demonstrates RAG-enhanced prompt optimization
- `rag_chatbot.py` - Core RAG chatbot implementation
- `chat_engine.py` - Chat interface using Upstage Solar Pro2 model
- `document_processor.py` - Document processing and vector storage management
- `vectorstore/` - FAISS vector database for storing processed documents

## Overview

The optimization module uses RAG to improve image generation prompts by learning from successful examples in the historical data. It processes CSV logs from previous image generations and creates a knowledge base of high-quality prompts.

## Key Features

### RAG Chatbot
- Processes historical image generation logs (CSV files)
- Creates vector embeddings for prompt similarity search
- Retrieves relevant successful prompts for few-shot learning
- Filters for high-quality examples (score > 8)

### Document Processing
- Supports CSV and PDF document loading
- Automatic text chunking and embedding generation
- FAISS vector store for efficient similarity search
- Persistent storage for processed documents

### Chat Engine
- Integration with Upstage Solar Pro2 language model
- Context-aware prompt enhancement
- Conversation history tracking

## Usage

### Basic Setup

1. Configure API keys in the respective files:
   - Upstage API key in `chat_engine.py`
   - Upstage API key in `document_processor.py`

2. Run the main optimization script:
   ```bash
   python main.py
   ```

### RAG Chatbot Usage

```python
from rag_chatbot import RAGChatbot

# Initialize chatbot
chatbot = RAGChatbot()

# Upload historical data
chatbot.upload_document('path/to/images_log.csv')

# Query for similar prompts
response = chatbot.send_message("water lily")

# Reset if needed
chatbot.reset_all()
```

### Document Processing

The system automatically:
- Filters CSV data for high-quality examples (overall_score > 8)
- Excludes unchanged prompts (original_user_prompt != prompt)
- Creates embeddings for similarity search
- Saves processed data to `vectorstore/`

## Integration with Main App

This optimization module enhances the main image generation pipeline by:
- Providing few-shot examples for DSPy prompt optimization
- Learning from successful prompt iterations
- Improving prompt quality through historical knowledge

The enhanced prompts are integrated into the main DSPy evaluation signature to improve image generation results.