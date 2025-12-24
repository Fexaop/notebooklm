# PDF to Vector Database Processing API

A robust FastAPI application for processing PDF documents, converting them to searchable vector embeddings, and storing them in Weaviate vector database.

## Features

- **PDF Processing**: Upload and convert PDFs to Markdown format
- **Intelligent Chunking**: Dynamic text splitting based on:
  - Headers (H1-H6)
  - Paragraphs
  - Lists
  - Code blocks
  - Tables
- **Image Processing**:
  - Automatic image extraction from PDFs
  - OCR using Tesseract
  - AI-powered image captioning using GPT-4 Vision
- **Vector Search**:
  - Semantic search using OpenAI embeddings
  - Hybrid search (vector + keyword)
  - Filter by document, page, or content type
- **Weaviate Integration**: Production-ready vector database storage

## Architecture

```
app/
├── api/                 # API routes
│   ├── documents.py     # Document upload/management endpoints
│   ├── search.py        # Search endpoints
│   └── routes.py        # Route aggregator
├── core/                # Core configuration
│   ├── config.py        # Settings management
│   ├── logging.py       # Logging configuration
│   └── exceptions.py    # Custom exceptions
├── db/                  # Database layer
│   └── weaviate_client.py  # Weaviate operations
├── models/              # Data models
│   └── schemas.py       # Pydantic schemas
├── services/            # Business logic
│   ├── pdf_processor.py    # PDF extraction
│   ├── chunker.py          # Text chunking
│   ├── ocr_processor.py    # Image OCR
│   ├── openai_service.py   # OpenAI integration
│   └── orchestrator.py     # Processing pipeline
└── main.py              # Application factory
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Tesseract OCR (for image text extraction)
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd MIC-2025
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. **Start with Docker Compose**
```bash
docker-compose up -d
```

4. **Or run locally**
```bash
# Install dependencies
pip install -r requirements.txt

# Install Tesseract (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# Start Weaviate
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  semitechnologies/weaviate:1.23.7

# Run the API
python run.py
```

## API Usage

### Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "title=My Document" \
  -F "tags=research,important"
```

### Search Documents

```bash
# Semantic search
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning algorithms", "limit": 10}'

# Quick search
curl "http://localhost:8000/api/v1/search/quick?q=neural+networks&limit=5"

# Hybrid search
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "deep learning", "alpha": 0.7, "limit": 10}'
```

### List Documents

```bash
curl "http://localhost:8000/api/v1/documents"
```

### Get Document Details

```bash
curl "http://localhost:8000/api/v1/documents/{document_id}"
curl "http://localhost:8000/api/v1/documents/{document_id}/chunks"
curl "http://localhost:8000/api/v1/documents/{document_id}/images"
```

### Delete Document

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/{document_id}"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `WEAVIATE_HOST` | Weaviate host | localhost |
| `WEAVIATE_PORT` | Weaviate HTTP port | 8080 |
| `CHUNK_MIN_SIZE` | Minimum chunk size | 100 |
| `CHUNK_MAX_SIZE` | Maximum chunk size | 2000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `MAX_UPLOAD_SIZE_MB` | Max file upload size | 100 |
| `DEBUG` | Enable debug mode | false |

### Chunking Strategies

The chunker supports multiple strategies:

1. **Hybrid** (default): Combines semantic and size-based chunking
2. **Semantic**: Splits on structural boundaries only
3. **Fixed Size**: Simple size-based splitting
4. **Recursive**: Hierarchical splitting with multiple separators

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "weaviate_connected": true,
  "openai_configured": true
}
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black app/
isort app/
```

### Type Checking

```bash
mypy app/
```

## Production Deployment

1. **Set environment to production**
```bash
DEBUG=false
ENVIRONMENT=production
```

2. **Use proper secrets management** for API keys

3. **Configure CORS** appropriately
```bash
ALLOWED_HOSTS=["https://yourdomain.com"]
```

4. **Scale with multiple workers**
```bash
uvicorn app.main:app --workers 4
```

5. **Add authentication** middleware for API security

## Troubleshooting

### Weaviate Connection Issues
- Ensure Weaviate is running: `docker ps`
- Check logs: `docker logs weaviate`

### OCR Not Working
- Verify Tesseract installation: `tesseract --version`
- Check language packs: `tesseract --list-langs`

### OpenAI Errors
- Verify API key is set correctly
- Check rate limits and quotas

## License

MIT License
