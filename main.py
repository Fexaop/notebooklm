import os
import json
import base64
from typing import List, Optional
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich import box
from PIL import Image
import io

load_dotenv()

console = Console()


# Pydantic models for query generation
class GeneratedQueries(BaseModel):
    """Generated queries for keyword and hypothetical question searches"""
    keyword_queries: List[str] = Field(
        description="List of keyword queries derived from the user's question"
    )
    hypothetical_queries: List[str] = Field(
        description="List of hypothetical questions that would be answered by relevant content"
    )


class WeaviateManager:
    """Manages Weaviate database connections and operations"""
    
    def __init__(
        self, 
        host: Optional[str] = None, 
        port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        scheme: Optional[str] = None
    ):
        # Load Weaviate configuration from environment
        weaviate_host = host or os.getenv("WEAVIATE_HOST", "localhost")
        weaviate_port = port or int(os.getenv("WEAVIATE_PORT", "8080"))
        weaviate_grpc_port = grpc_port or int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        weaviate_scheme = scheme or os.getenv("WEAVIATE_SCHEME", "http")
        
        # Connect to Weaviate
        self.client = weaviate.connect_to_local(
            host=weaviate_host,
            port=weaviate_port,
            grpc_port=weaviate_grpc_port
        )
        
        # Load OpenAI configuration from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize OpenAI client with custom base URL if provided
        openai_kwargs = {"api_key": openai_api_key}
        if openai_base_url:
            openai_kwargs["base_url"] = openai_base_url
        
        self.openai_client = instructor.from_openai(OpenAI(**openai_kwargs))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Initialize embedding client
        embedding_api_key = os.getenv("OPENAI_EMBEDDING_API_KEY") or openai_api_key
        embedding_base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL") or openai_base_url
        self.embedding_client = OpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
    
    def create_collections(self):
        """Create 5 collections in Weaviate database"""
        
        collections = [
            ("Content", "Content"),
            ("HypotheticalQuestion", "Hypothetical Questions"),
            ("Keyword", "Keywords"),
            ("ImageCaption", "Image Captions"),
            ("Image", "Images")
        ]
        
        # Collection 1: Content
        if not self.client.collections.exists("Content"):
            self.client.collections.create(
                name="Content",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="header_path", data_type=DataType.TEXT),
                    Property(name="source_file", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="global_chunk_index", data_type=DataType.INT),
                    Property(name="summary", data_type=DataType.TEXT),
                ]
            )
            console.print(f"[green]✓[/green] Created 'Content' collection")
        
        # Collection 2: Hypothetical Questions
        if not self.client.collections.exists("HypotheticalQuestion"):
            self.client.collections.create(
                name="HypotheticalQuestion",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="question", data_type=DataType.TEXT),
                    Property(name="chunk_reference", data_type=DataType.INT),
                    Property(name="global_chunk_index", data_type=DataType.INT),
                ]
            )
            console.print(f"[green]✓[/green] Created 'HypotheticalQuestion' collection")
        
        # Collection 3: Keywords
        if not self.client.collections.exists("Keyword"):
            self.client.collections.create(
                name="Keyword",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="keyword", data_type=DataType.TEXT),
                    Property(name="chunk_reference", data_type=DataType.INT),
                    Property(name="global_chunk_index", data_type=DataType.INT),
                ]
            )
            console.print(f"[green]✓[/green] Created 'Keyword' collection")
        
        # Collection 4: Image Captions (with vector support)
        if not self.client.collections.exists("ImageCaption"):
            self.client.collections.create(
                name="ImageCaption",
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                ),
                properties=[
                    Property(name="caption", data_type=DataType.TEXT),
                    Property(name="image_path", data_type=DataType.TEXT),
                    Property(name="chunk_reference", data_type=DataType.INT),
                    Property(name="global_chunk_index", data_type=DataType.INT),
                ]
            )
            console.print(f"[green]✓[/green] Created 'ImageCaption' collection with vector support")
        
        # Collection 5: Images (with vector support)
        if not self.client.collections.exists("Image"):
            self.client.collections.create(
                name="Image",
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                ),
                properties=[
                    Property(name="image_path", data_type=DataType.TEXT),
                    Property(name="image_base64", data_type=DataType.TEXT),
                    Property(name="chunk_reference", data_type=DataType.INT),
                    Property(name="global_chunk_index", data_type=DataType.INT),
                ]
            )
            console.print(f"[green]✓[/green] Created 'Image' collection with vector support")
    
    def load_chunks_to_weaviate(self, chunks_dir: str = "chunks"):
        """Load all chunk JSON files into Weaviate collections"""
        chunks_path = Path(chunks_dir)
        chunk_files = sorted(chunks_path.glob("chunk_*.json"))
        
        content_collection = self.client.collections.get("Content")
        hypo_collection = self.client.collections.get("HypotheticalQuestion")
        keyword_collection = self.client.collections.get("Keyword")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Loading {len(chunk_files)} chunks...", total=len(chunk_files))
            
            for chunk_file in chunk_files:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                # Insert into Content collection
                content_uuid = content_collection.data.insert({
                    "content": chunk_data.get("content", ""),
                    "header_path": chunk_data.get("header_path", ""),
                    "source_file": chunk_data.get("source_file", ""),
                    "chunk_index": chunk_data.get("chunk_index", 0),
                    "global_chunk_index": chunk_data.get("global_chunk_index", 0),
                    "summary": chunk_data.get("summary", ""),
                })
                
                # Insert hypothetical questions
                for question in chunk_data.get("hypothetical_questions", []):
                    hypo_collection.data.insert({
                        "question": question,
                        "chunk_reference": chunk_data.get("chunk_index", 0),
                        "global_chunk_index": chunk_data.get("global_chunk_index", 0),
                    })
                
                # Insert keywords
                for keyword in chunk_data.get("keywords", []):
                    keyword_collection.data.insert({
                        "keyword": keyword,
                        "chunk_reference": chunk_data.get("chunk_index", 0),
                        "global_chunk_index": chunk_data.get("global_chunk_index", 0),
                    })
                
                progress.advance(task)
        
        console.print(f"[green]✓[/green] Loaded all chunks successfully")
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API"""
        try:
            text = text.replace("\n", " ")
            response = self.embedding_client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            console.print(f"[red]Error generating embedding: {e}[/red]")
            return []
    
    def extract_and_encode_images(self, chunks_dir: str = "chunks", output_dir: str = "output"):
        """Load images from chunks/images/*.json files and add them with embeddings to Weaviate"""
        images_path = Path(chunks_dir) / "images"
        
        if not images_path.exists():
            console.print(f"[yellow]Warning: {images_path} not found. Skipping image loading.[/yellow]")
            return
        
        image_files = sorted(images_path.glob("image_*.json"))
        
        if not image_files:
            console.print(f"[yellow]Warning: No image JSON files found in {images_path}[/yellow]")
            return
        
        image_collection = self.client.collections.get("Image")
        caption_collection = self.client.collections.get("ImageCaption")
        
        images_loaded = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Loading {len(image_files)} images...", total=len(image_files))
            
            for image_file in image_files:
                try:
                    with open(image_file, 'r', encoding='utf-8') as f:
                        image_data = json.load(f)
                    
                    # Extract data from JSON
                    source_image = image_data.get("source_image", "")
                    caption = image_data.get("caption", "")
                    key_elements = image_data.get("key_elements", [])
                    image_type = image_data.get("image_type", "")
                    scientific_context = image_data.get("scientific_context", "")
                    text_embedding = image_data.get("text_embedding", [])
                    
                    # Build full caption with context
                    full_caption = caption
                    if scientific_context:
                        full_caption += f" {scientific_context}"
                    if key_elements:
                        full_caption += f" Key elements: {', '.join(key_elements)}"
                    
                    # Read and encode the actual image file
                    image_path = Path(source_image)
                    image_base64 = ""
                    
                    if image_path.exists():
                        try:
                            with Image.open(image_path) as img:
                                # Resize if too large
                                if img.size[0] > 1024 or img.size[1] > 1024:
                                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                                
                                # Convert to RGB if needed
                                if img.mode not in ('RGB', 'L'):
                                    img = img.convert('RGB')
                                
                                # Convert to base64
                                buffer = io.BytesIO()
                                img.save(buffer, format='PNG')
                                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not encode image {image_path}: {e}[/yellow]")
                    
                    # Use existing embedding or generate new one
                    if not text_embedding:
                        text_embedding = self.get_text_embedding(full_caption)
                    
                    if text_embedding:
                        # Insert into Image collection with embedding
                        image_collection.data.insert(
                            properties={
                                "image_path": source_image,
                                "image_base64": image_base64,
                                "chunk_reference": 0,  # Not directly linked to chunks
                                "global_chunk_index": images_loaded,
                            },
                            vector=text_embedding
                        )
                        
                        # Insert caption with embedding
                        caption_collection.data.insert(
                            properties={
                                "caption": full_caption,
                                "image_path": source_image,
                                "chunk_reference": 0,
                                "global_chunk_index": images_loaded,
                            },
                            vector=text_embedding
                        )
                        
                        images_loaded += 1
                
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not process {image_file}: {e}[/yellow]")
                
                progress.advance(task)
        
        console.print(f"[green]✓[/green] Loaded {images_loaded} images with embeddings")
    
    def generate_queries(self, user_query: str, model: Optional[str] = None) -> GeneratedQueries:
        """Use instructor to generate keyword and hypothetical queries"""
        
        # Use model from environment or override
        model_name = model or self.model
        
        response = self.openai_client.chat.completions.create(
            model=model_name,
            response_model=GeneratedQueries,
            messages=[
                {
                    "role": "system",
                    "content": """You are a query expansion assistant. Given a user's question, generate:
1. Keyword queries: Extract key terms and concepts that would help find relevant content
2. Hypothetical queries: Generate questions that, if answered, would help answer the user's question

Be specific and focused. Generate 3-5 queries for each type."""
                },
                {
                    "role": "user",
                    "content": f"User question: {user_query}"
                }
            ]
        )
        
        return response
    
    def search_content(self, query: str, limit: int = 5):
        """Direct search in Content collection"""
        content_collection = self.client.collections.get("Content")
        
        response = content_collection.query.bm25(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(score=True)
        )
        
        return response.objects
    
    def search_hypothetical_questions(self, queries: List[str], limit: int = 5):
        """Search using generated hypothetical questions"""
        hypo_collection = self.client.collections.get("HypotheticalQuestion")
        
        all_results = []
        for query in queries:
            response = hypo_collection.query.bm25(
                query=query,
                limit=limit,
                return_metadata=MetadataQuery(score=True)
            )
            all_results.extend(response.objects)
        
        # Sort by score and deduplicate
        all_results.sort(key=lambda x: x.metadata.score, reverse=True)
        seen_indices = set()
        unique_results = []
        
        for result in all_results:
            global_idx = result.properties.get("global_chunk_index")
            if global_idx not in seen_indices:
                seen_indices.add(global_idx)
                unique_results.append(result)
        
        return unique_results[:limit]
    
    def search_keywords(self, queries: List[str], limit: int = 5):
        """Search using generated keyword queries"""
        keyword_collection = self.client.collections.get("Keyword")
        
        all_results = []
        for query in queries:
            response = keyword_collection.query.bm25(
                query=query,
                limit=limit,
                return_metadata=MetadataQuery(score=True)
            )
            all_results.extend(response.objects)
        
        # Sort by score and deduplicate
        all_results.sort(key=lambda x: x.metadata.score, reverse=True)
        seen_indices = set()
        unique_results = []
        
        for result in all_results:
            global_idx = result.properties.get("global_chunk_index")
            if global_idx not in seen_indices:
                seen_indices.add(global_idx)
                unique_results.append(result)
        
        return unique_results[:limit]
    
    def search_image_captions(self, query: str, limit: int = 5, use_vector: bool = True):
        """Search in ImageCaption collection using vector similarity or BM25"""
        caption_collection = self.client.collections.get("ImageCaption")
        
        if use_vector:
            # Generate query embedding
            query_embedding = self.get_text_embedding(query)
            if query_embedding:
                response = caption_collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True)
                )
                return response.objects
        
        # Fallback to BM25
        response = caption_collection.query.bm25(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(score=True)
        )
        
        return response.objects
    
    def search_images(self, query: str, limit: int = 5, use_vector: bool = True):
        """Search in Image collection using vector similarity (text-to-image) or BM25"""
        image_collection = self.client.collections.get("Image")
        
        if use_vector:
            # Generate query embedding from text
            query_embedding = self.get_text_embedding(query)
            if query_embedding:
                response = image_collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True)
                )
                return response.objects
        
        # Fallback to BM25
        response = image_collection.query.bm25(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(score=True)
        )
        
        return response.objects
    
    def hybrid_search(self, user_query: str, limit: int = 10):
        """
        Perform hybrid search across all collections:
        - Generate queries for keywords and hypothetical questions using instructor
        - Direct search for content, captions, and images
        """
        console.print(Panel(f"[bold cyan]🔍 Searching for:[/bold cyan] {user_query}", box=box.ROUNDED))
        
        # Generate queries using instructor
        console.print("\n[yellow]📝 Generating optimized queries...[/yellow]")
        generated = self.generate_queries(user_query)
        
        console.print(f"\n[cyan]Keyword queries:[/cyan] {', '.join(generated.keyword_queries)}")
        console.print(f"[cyan]Hypothetical queries:[/cyan] {', '.join(generated.hypothetical_queries)}")
        
        # Search all collections
        console.print("\n[yellow]🔎 Searching collections...[/yellow]")
        
        content_results = self.search_content(user_query, limit=limit)
        hypo_results = self.search_hypothetical_questions(generated.hypothetical_queries, limit=limit)
        keyword_results = self.search_keywords(generated.keyword_queries, limit=limit)
        caption_results = self.search_image_captions(user_query, limit=limit)
        image_results = self.search_images(user_query, limit=limit)
        
        results = {
            "content": content_results,
            "hypothetical_questions": hypo_results,
            "keywords": keyword_results,
            "image_captions": caption_results,
            "images": image_results
        }
        
        return results
    
    def display_results(self, results: dict):
        """Display search results in a readable format"""
        
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]                              SEARCH RESULTS                                   [/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]")
        
        # Content results
        if results["content"]:
            console.print("\n[bold green]📄 CONTENT MATCHES[/bold green]")
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="dim", width=4)
            table.add_column("Score", width=8)
            table.add_column("Header", width=30)
            table.add_column("Content Preview", width=50)
            table.add_column("Chunk", width=6)
            
            for i, obj in enumerate(results["content"][:5], 1):
                props = obj.properties
                score = f"{obj.metadata.score:.2f}" if hasattr(obj.metadata, 'score') and obj.metadata.score is not None else 'N/A'
                content_preview = props.get('content', 'N/A')[:100] + "..."
                
                table.add_row(
                    str(i),
                    score,
                    props.get('header_path', 'N/A')[:30],
                    content_preview,
                    str(props.get('global_chunk_index', 'N/A'))
                )
            console.print(table)
        
        # Hypothetical question results
        if results["hypothetical_questions"]:
            console.print("\n[bold yellow]❓ HYPOTHETICAL QUESTION MATCHES[/bold yellow]")
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="dim", width=4)
            table.add_column("Score", width=8)
            table.add_column("Question", width=70)
            table.add_column("Chunk", width=6)
            
            for i, obj in enumerate(results["hypothetical_questions"][:5], 1):
                props = obj.properties
                score = f"{obj.metadata.score:.2f}" if hasattr(obj.metadata, 'score') and obj.metadata.score is not None else 'N/A'
                
                table.add_row(
                    str(i),
                    score,
                    props.get('question', 'N/A'),
                    str(props.get('global_chunk_index', 'N/A'))
                )
            console.print(table)
        
        # Keyword results
        if results["keywords"]:
            console.print("\n[bold blue]🔑 KEYWORD MATCHES[/bold blue]")
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="dim", width=4)
            table.add_column("Score", width=8)
            table.add_column("Keyword", width=40)
            table.add_column("Chunk", width=6)
            
            for i, obj in enumerate(results["keywords"][:5], 1):
                props = obj.properties
                score = f"{obj.metadata.score:.2f}" if hasattr(obj.metadata, 'score') and obj.metadata.score is not None else 'N/A'
                
                table.add_row(
                    str(i),
                    score,
                    props.get('keyword', 'N/A'),
                    str(props.get('global_chunk_index', 'N/A'))
                )
            console.print(table)
        
        # Image caption results
        if results["image_captions"]:
            console.print("\n[bold magenta]🖼️  IMAGE CAPTION MATCHES[/bold magenta]")
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="dim", width=4)
            table.add_column("Score/Distance", width=12)
            table.add_column("Caption", width=60)
            table.add_column("Image Path", width=40)
            
            for i, obj in enumerate(results["image_captions"][:5], 1):
                props = obj.properties
                # Handle both score (BM25) and distance (vector)
                if hasattr(obj.metadata, 'score') and obj.metadata.score is not None:
                    score = f"{obj.metadata.score:.2f}"
                elif hasattr(obj.metadata, 'distance') and obj.metadata.distance is not None:
                    score = f"{obj.metadata.distance:.4f}"
                else:
                    score = 'N/A'
                
                caption = props.get('caption', 'N/A')
                # Truncate long captions
                if len(caption) > 150:
                    caption = caption[:147] + "..."
                
                table.add_row(
                    str(i),
                    score,
                    caption,
                    props.get('image_path', 'N/A')
                )
            console.print(table)
        
        # Image results
        if results["images"]:
            console.print("\n[bold red]🎨 IMAGE MATCHES[/bold red]")
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="dim", width=4)
            table.add_column("Score/Distance", width=12)
            table.add_column("Image Path", width=80)
            
            for i, obj in enumerate(results["images"][:5], 1):
                props = obj.properties
                # Handle both score (BM25) and distance (vector)
                if hasattr(obj.metadata, 'score') and obj.metadata.score is not None:
                    score = f"{obj.metadata.score:.2f}"
                elif hasattr(obj.metadata, 'distance') and obj.metadata.distance is not None:
                    score = f"{obj.metadata.distance:.4f}"
                else:
                    score = 'N/A'
                
                table.add_row(
                    str(i),
                    score,
                    props.get('image_path', 'N/A')
                )
            console.print(table)


def main():
    """Main execution function"""
    
    console.print(Panel.fit(
        "[bold cyan]🚀 Weaviate Database Manager[/bold cyan]",
        border_style="cyan"
    ))
    
    with WeaviateManager() as weaviate_mgr:
        # Create collections
        console.print("\n[yellow]📦 Creating Weaviate collections...[/yellow]")
        weaviate_mgr.create_collections()
        
        # Load chunks from chunks folder
        console.print("\n[yellow]📥 Loading chunks into Weaviate...[/yellow]")
        weaviate_mgr.load_chunks_to_weaviate()
        
        # Extract and encode images
        console.print("\n[yellow]🖼️  Extracting and encoding images...[/yellow]")
        weaviate_mgr.extract_and_encode_images()
        
        # Example queries
        example_queries = [
            "What is magnetism?",
            "Explain magnetic fields",
            "How do magnets work?"
        ]
        
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]                            EXAMPLE SEARCHES                                   [/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]")
        
        for query in example_queries:
            results = weaviate_mgr.hybrid_search(query, limit=5)
            weaviate_mgr.display_results(results)
            console.print()
        
        # Interactive mode
        console.print(Panel(
            "[bold yellow]💬 Interactive Query Mode[/bold yellow]\n"
            "Type your query and press Enter\n"
            "Type 'exit', 'quit', or 'q' to quit",
            border_style="yellow"
        ))
        
        while True:
            user_input = Prompt.ask("\n[bold green]Enter your query[/bold green]")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("\n[bold cyan]👋 Goodbye![/bold cyan]")
                break
            
            if not user_input:
                continue
            
            results = weaviate_mgr.hybrid_search(user_input, limit=5)
            weaviate_mgr.display_results(results)


if __name__ == "__main__":
    main()
