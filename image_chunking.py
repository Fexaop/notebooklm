import os
import json
import asyncio
import base64
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import instructor
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from PIL import Image
import io

load_dotenv()

# Configurable defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_IMAGE_SIZE = (1024, 1024)  # Max dimensions for image processing

class ImageCaption(BaseModel):
    """Structured output for image caption generation."""
    caption: str = Field(..., description="A detailed caption describing the image content, context, and any visible text or data.")
    key_elements: List[str] = Field(..., description="List of 3-7 key elements, objects, or concepts visible in the image.")
    image_type: str = Field(..., description="Type of image: graph, diagram, chart, photograph, illustration, table, equation, etc.")
    scientific_context: Optional[str] = Field(None, description="Scientific or technical context if applicable.")

class ImageChunker:
    def __init__(self,
                 embedding_model: Optional[str] = None,
                 vision_model: Optional[str] = None,
                 embedding_api_key: Optional[str] = None,
                 embedding_base_url: Optional[str] = None,
                 vision_api_key: Optional[str] = None,
                 vision_base_url: Optional[str] = None,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 max_image_size: tuple = DEFAULT_MAX_IMAGE_SIZE):
        
        # Embedding settings
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        embedding_api_key = embedding_api_key or os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        embedding_base_url = embedding_base_url or os.getenv("OPENAI_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        
        self.embedding_client = OpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
        self.async_embedding_client = AsyncOpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
        
        # Vision model settings
        self.vision_model = vision_model or os.getenv("VISION_MODEL", "gpt-4o")
        vision_api_key = vision_api_key or os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY")
        vision_base_url = vision_base_url or os.getenv("VISION_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        
        self.vision_client = instructor.from_openai(
            OpenAI(api_key=vision_api_key, base_url=vision_base_url),
            mode=instructor.Mode.JSON
        )
        self.async_vision_client = instructor.from_openai(
            AsyncOpenAI(api_key=vision_api_key, base_url=vision_base_url),
            mode=instructor.Mode.JSON
        )
        
        # Configuration
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.max_image_size = max_image_size
        
        self.console = Console()
        self.failed_captions = []
        
    def resize_image_if_needed(self, image_path: Path) -> bytes:
        """Resize image if it exceeds max dimensions, return as bytes."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Resize if needed
                if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                    img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                    self.console.log(f"Resized {image_path.name} from original to {img.size}")
                
                # Convert to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                return buffer.getvalue()
        except Exception as e:
            self.console.print(f"[red]Error processing image {image_path}: {e}[/red]")
            raise
    
    def encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    async def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (caption)."""
        try:
            text = text.replace("\n", " ")
            response = await self.async_embedding_client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            self.console.print(f"[red]Error generating text embedding: {e}[/red]")
            return []
    
    async def generate_image_caption(self, image_path: Path, retry_count: int = 0) -> Optional[ImageCaption]:
        """Generate a structured caption for an image using vision model with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Prepare image
                image_bytes = self.resize_image_if_needed(image_path)
                base64_image = self.encode_image_to_base64(image_bytes)
                
                # Generate caption using vision model with structured output
                response = await self.async_vision_client.chat.completions.create(
                    model=self.vision_model,
                    response_model=ImageCaption,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at analyzing scientific and technical images. Provide detailed, accurate descriptions."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this image and provide a detailed caption, key elements, type, and scientific context if applicable."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
                
                if isinstance(response, ImageCaption):
                    return response
                    
            except Exception as e:
                error_str = str(e)
                if attempt < self.max_retries - 1:
                    self.console.print(f"[yellow]Attempt {attempt + 1} failed for {image_path.name}, retrying...[/yellow]")
                    await asyncio.sleep(1)
                else:
                    self.console.print(f"[red]Failed to caption {image_path.name} after {self.max_retries} attempts: {error_str[:200]}[/red]")
                    failure_info = {
                        "image_path": str(image_path),
                        "error": error_str[:500] if len(error_str) > 500 else error_str
                    }
                    self.failed_captions.append(failure_info)
        
        return None
    
    async def process_single_image(self, image_path: Path, output_dir: Path, global_index: int) -> Optional[dict]:
        """Process a single image: generate caption and embedding."""
        try:
            # Generate caption with retry
            caption_obj = await self.generate_image_caption(image_path)
            
            if caption_obj is None:
                return None
            
            caption_dict = caption_obj.model_dump()
            
            # Generate text embedding from the caption
            caption_text = caption_dict["caption"]
            text_embedding = await self.get_text_embedding(caption_text)
            
            # Prepare image chunk data
            image_data = {
                "type": "image",
                "source_image": str(image_path),
                "caption": caption_dict["caption"],
                "key_elements": caption_dict["key_elements"],
                "image_type": caption_dict["image_type"],
                "scientific_context": caption_dict.get("scientific_context"),
                "text_embedding": text_embedding,
                "global_chunk_index": global_index,
                "chunk_index": global_index
            }
            
            # Save to JSON
            output_file = output_dir / f"image_{global_index:03d}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(image_data, f, indent=2, ensure_ascii=False)
            
            return image_data
            
        except Exception as e:
            self.console.print(f"[red]Error processing {image_path}: {e}[/red]")
            return None
    
    async def process_images(self, input_dir: Path = Path("output"), output_dir: Path = Path("chunks/images")) -> List[dict]:
        """Find and process all images in the input directory recursively."""
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(input_dir.rglob(f"*{ext}"))
            image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_paths = sorted(set(image_paths))
        
        if not image_paths:
            self.console.print("[yellow]No images found in the input directory.[/yellow]")
            return []
        
        self.console.print(f"[green]Found {len(image_paths)} images to process[/green]")
        
        # Process images in batches
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Processing images", total=len(image_paths))
            
            for i in range(0, len(image_paths), self.batch_size):
                batch = image_paths[i:i + self.batch_size]
                batch_tasks = [
                    self.process_single_image(img_path, output_dir, i + j)
                    for j, img_path in enumerate(batch)
                ]
                
                batch_results = await asyncio.gather(*batch_tasks)
                all_results.extend([r for r in batch_results if r is not None])
                
                progress.update(
                    task,
                    advance=len(batch),
                    description=f"Processing images {i + len(batch)}/{len(image_paths)}"
                )
        
        self.console.print(f"[green]Successfully processed {len(all_results)} images[/green]")
        
        if self.failed_captions:
            self.console.print(f"\n[bold red]Failed to process {len(self.failed_captions)} images:[/bold red]")
            for idx, failure in enumerate(self.failed_captions, 1):
                self.console.print(f"\n[bold]Failure #{idx}:[/bold]")
                self.console.print(f"  Image: [cyan]{failure['image_path']}[/cyan]")
                self.console.print(f"  Error: [red]{failure['error']}[/red]")
        
        return all_results
    
    def run(self, input_dir: Path = Path("output"), output_dir: Path = Path("chunks/images")):
        """Main entry point for image processing."""
        return asyncio.run(self.process_images(input_dir, output_dir))

if __name__ == "__main__":
    chunker = ImageChunker()
    chunker.run()
