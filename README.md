# PDF to Markdown & Intelligent Chunking

A pipeline for converting PDF documents to Markdown using Mistral OCR and then chunking the text with AI-powered metadata enrichment.

## Features

- **Mistral OCR Integration**: High-quality PDF to Markdown conversion including image extraction and table preservation.
- **Intelligent Chunking**: Structural chunking of Markdown files based on headers and paragraphs.
- **AI Enrichment**: Each chunk is enriched with:
  - A concise summary
  - Hypothetical questions the chunk answers
  - Key entities and keywords
- **Metadata Preservation**: Maintains header paths for each chunk to provide structural context.

## Project Structure

- `pdf_to_md.py`: Script to convert PDF files to Markdown using Mistral AI's OCR API.
- `chunking.py`: Script to process Markdown files, split them into logical chunks, and enrich them using OpenAI.
- `in/`: Input directory for PDF files.
- `output/`: Output directory for generated Markdown and images.
- `chunks/`: Output directory for enriched JSON chunks.

## Setup

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:

   Create a `.env` file with your API keys:

   ```env
   MISTRAL_API_KEY=your_mistral_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### 1. Convert PDF to Markdown

Place your PDF in the `in/` folder and run:

```bash
python pdf_to_md.py
```

This will generate a Markdown file and an `images/` folder in `output/<pdf_name>/`.

### 2. Chunk and Enrich

Run the chunking script to process the generated Markdown:

```bash
python chunking.py
```

This will create enriched JSON chunks in the `chunks/` directory.

## License

MIT
