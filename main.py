"""
PDF Vector Processing Application
Consolidated single-file implementation.
"""

import os
import sys
import json
import uuid
import time
import asyncio
import logging
import hashlib
import shutil
import re
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import fitz  # PyMuPDF
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.util import generate_uuid5
from openai import AsyncOpenAI
from PIL import Image
import pytesseract
from pydantic_settings import BaseSettings
from pydantic import Field, BaseModel, validator
from dotenv import load_dotenv
import base64
import typer
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

# Load environment variables
load_dotenv()

# ==========================================
# Configuration
# ==========================================

class Settings(BaseSettings):
    """Application settings."""
    APP_NAME: str = "PDF Vector Processing API"
    DEBUG: bool = False
    
    # Weaviate Settings
    WEAVIATE_HOST: str = Field(default="localhost", description="Weaviate host")
    WEAVIATE_PORT: int = Field(default=8080, description="Weaviate port")
    WEAVIATE_GRPC_PORT: int = Field(default=50051, description="Weaviate gRPC port")
    WEAVIATE_API_KEY: Optional[str] = Field(default=None, description="Weaviate API key")
    WEAVIATE_SCHEME: str = Field(default="http", description="Weaviate connection scheme")
    
    # OpenAI Settings
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_BASE_URL: Optional[str] = Field(default=None, description="OpenAI base URL")
    OPENAI_MODEL: str = Field(default="gpt-4o", description="OpenAI model for image captioning")
    
    # Embedding Settings
    OPENAI_EMBEDDING_API_KEY: Optional[str] = Field(default=None, description="API key for embeddings (defaults to OPENAI_API_KEY)")
    OPENAI_EMBEDDING_BASE_URL: Optional[str] = Field(default=None, description="Base URL for embeddings (defaults to OPENAI_BASE_URL)")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    
    # Chunking Settings
    CHUNK_MIN_SIZE: int = Field(default=100, description="Minimum chunk size")
    CHUNK_MAX_SIZE: int = Field(default=2000, description="Maximum chunk size")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlap between chunks")
    
    # Image Settings
    IMAGES_DIR: str = Field(default="./uploads/images", description="Directory to save extracted images")
    
    # OCR Settings
    TESSERACT_CMD: Optional[str] = Field(default=None, description="Path to tesseract command")
    
    @property
    def weaviate_url(self) -> str:
        return f"{self.WEAVIATE_SCHEME}://{self.WEAVIATE_HOST}:{self.WEAVIATE_PORT}"

    @validator("IMAGES_DIR", pre=True)
    def create_directories(cls, v):
        os.makedirs(v, exist_ok=True)
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# ==========================================
# Logging
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pdf-processor")

class LoggerMixin:
    @property
    def logger(self):
        return logging.getLogger(self.__class__.__name__)

# ==========================================
# Models & Schemas
# ==========================================

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    GENERATING_EMBEDDINGS = "generating_embeddings"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"

class ChunkType(str, Enum):
    HEADER = "header"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    MIXED = "mixed"

@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    chunk_type: ChunkType
    page_number: int
    position: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

@dataclass
class PDFMetadata:
    title: Optional[str]
    author: Optional[str]
    subject: Optional[str]
    page_count: int
    file_size: int
    has_toc: bool

@dataclass
class ExtractedPDFContent:
    metadata: PDFMetadata
    full_text: str
    full_markdown: str
    all_images: List[Dict[str, Any]]

class ProcessingTaskResponse(BaseModel):
    task_id: str
    document_id: str
    status: ProcessingStatus
    progress: float
    current_step: str
    error_message: Optional[str] = None

# ==========================================
# Services
# ==========================================

class OpenAIService(LoggerMixin):
    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None
        self._embedding_client: Optional[AsyncOpenAI] = None
        
    def _ensure_clients(self):
        # Main client (for Chat/Vision)
        if not self._client:
            if not settings.OPENAI_API_KEY:
                raise Exception("OPENAI_API_KEY not set")
            self._client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL
            )
            
        # Embedding client
        if not self._embedding_client:
            api_key = settings.OPENAI_EMBEDDING_API_KEY or settings.OPENAI_API_KEY
            base_url = settings.OPENAI_EMBEDDING_BASE_URL or settings.OPENAI_BASE_URL
            
            if not api_key:
                raise Exception("API key for embeddings not set")
                
            self._embedding_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
    
    async def generate_embedding(self, text: str) -> List[float]:
        self._ensure_clients()
        try:
            text = text[:8000]
            response = await self._embedding_client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Embedding error: {e}")
            raise

    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        self._ensure_clients()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:8000] for t in batch]
            try:
                response = await self._embedding_client.embeddings.create(
                    model=settings.OPENAI_EMBEDDING_MODEL,
                    input=batch
                )
                sorted_data = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend([d.embedding for d in sorted_data])
            except Exception as e:
                self.logger.error(f"Batch embedding error: {e}")
                raise
        return all_embeddings

    async def caption_image(self, image_data: bytes) -> str:
        self._ensure_clients()
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            response = await self._client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail for a blind user."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Captioning error: {e}")
            return "Image captioning failed."

openai_service = OpenAIService()

class PDFProcessor(LoggerMixin):
    async def process_pdf(
        self,
        file_path: str,
        extract_images: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> ExtractedPDFContent:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            doc = fitz.open(file_path)
        except Exception:
            raise Exception("Corrupted PDF")

        metadata = PDFMetadata(
            title=doc.metadata.get("title"),
            author=doc.metadata.get("author"),
            subject=doc.metadata.get("subject"),
            page_count=len(doc),
            file_size=path.stat().st_size,
            has_toc=len(doc.get_toc()) > 0
        )

        full_text_parts = []
        all_images = []

        for page_num in range(len(doc)):
            if progress_callback:
                progress_callback(page_num + 1, len(doc))
            
            page = doc[page_num]
            text = page.get_text()
            full_text_parts.append(text)
            
            if extract_images:
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    all_images.append({
                        "data": image_bytes,
                        "page": page_num + 1,
                        "ext": base_image["ext"]
                    })

        return ExtractedPDFContent(
            metadata=metadata,
            full_text="\n\n".join(full_text_parts),
            full_markdown="\n\n".join(full_text_parts), # Simplified for now
            all_images=all_images
        )

pdf_processor = PDFProcessor()

class DocumentChunker(LoggerMixin):
    def __init__(self):
        self.min_size = settings.CHUNK_MIN_SIZE
        self.max_size = settings.CHUNK_MAX_SIZE
        self.overlap = settings.CHUNK_OVERLAP

    async def chunk_document(self, content: str, document_id: str) -> List[DocumentChunk]:
        # Simple recursive character splitting for now to keep it single-file manageable
        # In a real scenario, we'd use the complex regex logic from the original file
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', content)
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.max_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        content=current_chunk,
                        chunk_type=ChunkType.PARAGRAPH,
                        page_number=1, # Simplified
                        position=chunk_index
                    ))
                    chunk_index += 1
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else ""
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    # Paragraph itself is too big, force split
                    chunks.append(DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        content=para[:self.max_size],
                        chunk_type=ChunkType.PARAGRAPH,
                        page_number=1,
                        position=chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = "" # Reset

        if current_chunk:
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                content=current_chunk,
                chunk_type=ChunkType.PARAGRAPH,
                page_number=1,
                position=chunk_index
            ))
            
        return chunks

document_chunker = DocumentChunker()

class WeaviateClient(LoggerMixin):
    def __init__(self):
        self._client: Optional[weaviate.WeaviateClient] = None
        self._connected = False

    async def connect(self):
        try:
            if settings.WEAVIATE_API_KEY:
                self._client = weaviate.connect_to_custom(
                    http_host=settings.WEAVIATE_HOST,
                    http_port=settings.WEAVIATE_PORT,
                    http_secure=(settings.WEAVIATE_SCHEME == "https"),
                    grpc_host=settings.WEAVIATE_HOST,
                    grpc_port=settings.WEAVIATE_GRPC_PORT,
                    grpc_secure=(settings.WEAVIATE_SCHEME == "https"),
                    auth_credentials=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY)
                )
            else:
                self._client = weaviate.connect_to_local(
                    host=settings.WEAVIATE_HOST,
                    port=settings.WEAVIATE_PORT,
                    grpc_port=settings.WEAVIATE_GRPC_PORT
                )
            
            if self._client.is_ready():
                self._connected = True
                self.logger.info("Connected to Weaviate")
            else:
                raise Exception("Weaviate not ready")
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            raise

    async def disconnect(self):
        if self._client:
            self._client.close()
            self._connected = False

    async def initialize_schema(self):
        if not self._client.collections.exists("Document"):
            self._client.collections.create(
                name="Document",
                properties=[
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.DATE),
                ]
            )
        
        if not self._client.collections.exists("DocumentChunk"):
            self._client.collections.create(
                name="DocumentChunk",
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                ),
                properties=[
                    Property(name="document_id", data_type=DataType.TEXT, index_filterable=True),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="chunk_type", data_type=DataType.TEXT),
                    Property(name="position", data_type=DataType.INT),
                ]
            )
            
        if not self._client.collections.exists("DocumentImage"):
            self._client.collections.create(
                name="DocumentImage",
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                ),
                properties=[
                    Property(name="document_id", data_type=DataType.TEXT, index_filterable=True),
                    Property(name="caption", data_type=DataType.TEXT),
                    Property(name="image_path", data_type=DataType.TEXT),
                    Property(name="page_number", data_type=DataType.INT),
                ]
            )

    async def insert_document(self, data: Dict[str, Any]):
        collection = self._client.collections.get("Document")
        # Remove 'id' from properties as it is reserved
        properties = data.copy()
        doc_uuid = properties.pop("id")
        collection.data.insert(properties=properties, uuid=doc_uuid)

    async def insert_chunks(self, chunks: List[DocumentChunk]):
        collection = self._client.collections.get("DocumentChunk")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "chunk_type": chunk.chunk_type.value,
                        "position": chunk.position
                    },
                    vector=chunk.embedding,
                    uuid=chunk.chunk_id
                )

    async def insert_image(self, document_id: str, image_path: str, caption: str, page_number: int, embedding: Optional[List[float]] = None):
        collection = self._client.collections.get("DocumentImage")
        collection.data.insert(
            properties={
                "document_id": document_id,
                "caption": caption,
                "image_path": image_path,
                "page_number": page_number
            },
            vector=embedding,
            uuid=generate_uuid5(image_path)
        )

    async def search(self, query_text: str, limit: int = 5):
        # Generate embedding for the query
        query_embedding = await openai_service.generate_embedding(query_text)
        
        results = {}
        
        # Search chunks
        chunks_collection = self._client.collections.get("DocumentChunk")
        chunk_results = chunks_collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )
        
        results["chunks"] = []
        for obj in chunk_results.objects:
            results["chunks"].append({
                "content": obj.properties["content"],
                "document_id": obj.properties["document_id"],
                "distance": obj.metadata.distance
            })
            
        # Search images
        images_collection = self._client.collections.get("DocumentImage")
        image_results = images_collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )
        
        results["images"] = []
        for obj in image_results.objects:
            results["images"].append({
                "caption": obj.properties["caption"],
                "image_path": obj.properties["image_path"],
                "document_id": obj.properties["document_id"],
                "distance": obj.metadata.distance
            })
            
        return results

weaviate_client = WeaviateClient()

class ProcessingTask:
    def __init__(self, document_id: str, filename: str):
        self.task_id = str(uuid.uuid4())
        self.document_id = document_id
        self.filename = filename
        self.status = ProcessingStatus.PENDING
        self.progress = 0.0
        self.current_step = ""
        self.error_message = None

    def update(self, status=None, progress=None, step=None):
        if status: self.status = status
        if progress is not None: self.progress = progress
        if step: self.current_step = step

class Orchestrator(LoggerMixin):
    async def process_document(
        self,
        file_path: str,
        filename: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: List[str] = None,
        process_images: bool = True,
        progress_callback: Optional[Callable[[ProcessingTask], None]] = None
    ) -> ProcessingTask:
        
        document_id = str(uuid.uuid4())
        task = ProcessingTask(document_id, filename)
        
        try:
            # 1. Extract
            task.update(ProcessingStatus.PROCESSING, 10, "Extracting PDF content")
            if progress_callback: progress_callback(task)
            
            def pdf_progress(current, total):
                p = 10 + (current / total * 20) # 10-30%
                task.update(progress=p, step=f"Extracting page {current}/{total}")
                if progress_callback: progress_callback(task)

            pdf_content = await pdf_processor.process_pdf(
                file_path, 
                extract_images=process_images,
                progress_callback=pdf_progress
            )
            
            # 2. Create Record
            task.update(progress=30, step="Creating document record")
            if progress_callback: progress_callback(task)
            
            await weaviate_client.insert_document({
                "id": document_id,
                "filename": filename,
                "title": title or pdf_content.metadata.title or filename,
                "created_at": datetime.now(timezone.utc).isoformat()
            })
            
            # 3. Process Images (Save, Caption, Embed, Store)
            if process_images and pdf_content.all_images:
                task.update(ProcessingStatus.PROCESSING, 35, f"Processing {len(pdf_content.all_images)} images")
                if progress_callback: progress_callback(task)
                
                for i, img_data in enumerate(pdf_content.all_images):
                    try:
                        # Save image
                        img_filename = f"{document_id}_p{img_data['page']}_{i}.{img_data['ext']}"
                        img_path = os.path.join(settings.IMAGES_DIR, img_filename)
                        
                        with open(img_path, "wb") as f:
                            f.write(img_data["data"])
                            
                        # Caption
                        caption = await openai_service.caption_image(img_data["data"])
                        
                        # Embed caption
                        embedding = await openai_service.generate_embedding(caption)
                        
                        # Store in Weaviate
                        await weaviate_client.insert_image(
                            document_id=document_id,
                            image_path=img_path,
                            caption=caption,
                            page_number=img_data["page"],
                            embedding=embedding
                        )
                        
                        task.update(progress=35 + (i / len(pdf_content.all_images) * 5), step=f"Processed image {i+1}/{len(pdf_content.all_images)}")
                        if progress_callback: progress_callback(task)
                        
                    except Exception as e:
                        self.logger.error(f"Image processing failed for image {i}: {e}")
                        # Continue with other images
            
            # 4. Chunk
            task.update(ProcessingStatus.CHUNKING, 40, "Chunking document")
            if progress_callback: progress_callback(task)
            
            chunks = await document_chunker.chunk_document(pdf_content.full_text, document_id)
            
            # 5. Embed
            task.update(ProcessingStatus.GENERATING_EMBEDDINGS, 60, "Generating embeddings")
            if progress_callback: progress_callback(task)
            
            chunk_texts = [c.content for c in chunks]
            embeddings = await openai_service.generate_embeddings_batch(chunk_texts)
            
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
                
            # 6. Store
            task.update(ProcessingStatus.STORING, 80, "Storing vectors")
            if progress_callback: progress_callback(task)
            
            await weaviate_client.insert_chunks(chunks)
            
            task.update(ProcessingStatus.COMPLETED, 100, "Completed")
            if progress_callback: progress_callback(task)
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            task.status = ProcessingStatus.FAILED
            task.error_message = str(e)
            if progress_callback: progress_callback(task)
            
        return task

processing_orchestrator = Orchestrator()

# ==========================================
# CLI
# ==========================================

app = typer.Typer(help="PDF Vector Processing CLI")
console = Console()

async def setup_services():
    """Initialize database connection and schema."""
    try:
        await weaviate_client.connect()
        await weaviate_client.initialize_schema()
        return True
    except Exception as e:
        console.print(f"[bold red]Error connecting to Weaviate:[/bold red] {e}")
        return False

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, help="Number of results to return")
):
    """Search for documents and images."""
    async def run_search():
        if not await setup_services():
            return

        with console.status("[bold green]Searching..."):
            try:
                results = await weaviate_client.search(query, limit=limit)
                
                # Display Text Results
                console.print("\n[bold blue]Text Results:[/bold blue]")
                if results["chunks"]:
                    for i, chunk in enumerate(results["chunks"], 1):
                        panel = Panel(
                            chunk["content"],
                            title=f"Result {i} (Distance: {chunk['distance']:.4f})",
                            subtitle=f"Document ID: {chunk['document_id']}",
                            border_style="blue"
                        )
                        console.print(panel)
                else:
                    console.print("[italic]No text results found.[/italic]")

                # Display Image Results
                console.print("\n[bold magenta]Image Results:[/bold magenta]")
                if results["images"]:
                    for i, img in enumerate(results["images"], 1):
                        panel = Panel(
                            f"Caption: {img['caption']}\nPath: {img['image_path']}",
                            title=f"Image {i} (Distance: {img['distance']:.4f})",
                            subtitle=f"Document ID: {img['document_id']}",
                            border_style="magenta"
                        )
                        console.print(panel)
                else:
                    console.print("[italic]No image results found.[/italic]")
                    
            except Exception as e:
                console.print(f"[bold red]Search failed:[/bold red] {e}")
            finally:
                await weaviate_client.disconnect()

    asyncio.run(run_search())

if __name__ == "__main__":
    app()
