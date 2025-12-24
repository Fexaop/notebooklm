import os
import re
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import numpy as np
import instructor
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.traceback import install as rich_install
from pydantic import BaseModel, Field
from typing import List, Optional
import bisect

load_dotenv()

# Configurable defaults
DEFAULT_EMBEDDING_BATCH_SIZE = 100
DEFAULT_ENRICH_BATCH_SIZE = 10
DEFAULT_MAX_DYNAMIC_SIZE = 2000
DEFAULT_MIN_DYNAMIC_SIZE = 300
DEFAULT_LONG_PARAGRAPH_LENGTH = 500
DEFAULT_MAX_RETRIES = 3

class ChunkMetadata(BaseModel):
    summary: str = Field(..., description="A concise 1-sentence summary of the text.")
    hypothetical_questions: List[str] = Field(..., description="A list of 2-4 questions this text answers.")
    keywords: List[str] = Field(..., description="A list of 3-5 key entities/keywords found in the text.")

class Chunker:
    def __init__(self,
                 embedding_model: Optional[str] = None,
                 chat_model: Optional[str] = None,
                 embedding_api_key: Optional[str] = None,
                 embedding_base_url: Optional[str] = None,
                 chat_api_key: Optional[str] = None,
                 chat_base_url: Optional[str] = None,
                 embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
                 enrich_batch_size: int = DEFAULT_ENRICH_BATCH_SIZE,
                 max_dynamic_size: int = DEFAULT_MAX_DYNAMIC_SIZE,
                 min_dynamic_size: int = DEFAULT_MIN_DYNAMIC_SIZE,
                 long_paragraph_length: int = DEFAULT_LONG_PARAGRAPH_LENGTH,
                 max_retries: int = DEFAULT_MAX_RETRIES):

        self.model_name = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        embedding_api_key = embedding_api_key or os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        embedding_base_url = embedding_base_url or os.getenv("OPENAI_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
        self.async_client = AsyncOpenAI(api_key=embedding_api_key, base_url=embedding_base_url)

        # Tunable parameters (defaults are defined at module level)
        self.embedding_batch_size = embedding_batch_size
        self.enrich_batch_size = enrich_batch_size
        self.max_dynamic_size = max_dynamic_size
        self.min_dynamic_size = min_dynamic_size
        self.long_paragraph_length = long_paragraph_length
        self.max_retries = max_retries

        chat_api_key = chat_api_key or os.getenv("OPENAI_API_KEY")
        chat_base_url = chat_base_url or os.getenv("OPENAI_BASE_URL")
        self.chat_model = chat_model or os.getenv("OPENAI_MODEL", "gpt-4o")

        self.chat_client = instructor.from_openai(
            OpenAI(api_key=chat_api_key, base_url=chat_base_url),
            mode=instructor.Mode.JSON
        )
        self.async_chat_client = instructor.from_openai(
            AsyncOpenAI(api_key=chat_api_key, base_url=chat_base_url),
            mode=instructor.Mode.JSON
        )

        rich_install()
        self.console = Console()
        self.failed_enrichments = []

    async def get_embeddings(self, texts: List[str], model: Optional[str] = None, batch_size: Optional[int] = None) -> List[List[float]]:
        model = model or self.model_name
        texts = [t.replace("\n", " ") for t in texts]
        batch_size = batch_size or self.embedding_batch_size

        async def fetch_batch(batch):
            response = await self.async_client.embeddings.create(input=batch, model=model)
            return [data.embedding for data in response.data]

        tasks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tasks.append(fetch_batch(batch))

        results = await asyncio.gather(*tasks)
        all_embeddings: List[List[float]] = []
        for batch_embeddings in results:
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    @staticmethod
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get_markdown_units(self, content: str, long_paragraph_length: Optional[int] = None) -> List[dict]:
        units = []
        long_paragraph_length = long_paragraph_length or self.long_paragraph_length
        
        # Map character index to line number
        line_offsets = [0]
        for i, char in enumerate(content):
            if char == '\n':
                line_offsets.append(i + 1)
        
        def get_line_number(char_index):
            # bisect_right returns the insertion point.
            # If char_index is before the first newline (offset < line_offsets[1]), it returns 1.
            return bisect.bisect_right(line_offsets, char_index)

        # Split by blank lines to get paragraphs, preserving offsets
        parts = []
        last_end = 0
        for match in re.finditer(r'\n\s*\n', content):
            start = last_end
            end = match.start()
            if end > start:
                parts.append((start, end, content[start:end]))
            last_end = match.end()
        if last_end < len(content):
            parts.append((last_end, len(content), content[last_end:]))

        header_stack = []
        current_header_path = ""

        for start_offset, end_offset, part in parts:
            # Calculate line range for the whole part
            # We use end_offset - 1 because the range is inclusive of the last character's line
            
            # Adjust for stripping
            lstripped = part.lstrip()
            leading_ws = len(part) - len(lstripped)
            real_start_offset = start_offset + leading_ws
            
            stripped = lstripped.rstrip()
            real_end_offset = real_start_offset + len(stripped)
            
            if not stripped:
                continue
                
            part = stripped
            part_start_line = get_line_number(real_start_offset)
            part_end_line = get_line_number(real_end_offset - 1)

            if part.startswith('#'):
                level = len(part.split()[0])
                header_text = part.lstrip('#').strip()
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, header_text))
                current_header_path = " > ".join([h[1] for h in header_stack])
                units.append({
                    "text": part, 
                    "type": "header", 
                    "header_path": current_header_path,
                    "line_start": part_start_line,
                    "line_end": part_end_line
                })
                continue

            current_header_path = " > ".join([h[1] for h in header_stack])

            if part.startswith('|'):
                units.append({
                    "text": part, 
                    "type": "table", 
                    "header_path": current_header_path,
                    "line_start": part_start_line,
                    "line_end": part_end_line
                })
                continue

            if len(part) > long_paragraph_length:
                sentences = re.split(r'(?<=[.!?])\s+', part)
                current_rel_pos = 0
                for s in sentences:
                    if not s.strip():
                        current_rel_pos += len(s)
                        continue
                    
                    s_stripped = s.strip()
                    found_pos = part.find(s_stripped, current_rel_pos)
                    if found_pos == -1:
                        found_pos = current_rel_pos
                    
                    s_start_offset = real_start_offset + found_pos
                    s_end_offset = s_start_offset + len(s_stripped)
                    
                    s_start_line = get_line_number(s_start_offset)
                    s_end_line = get_line_number(s_end_offset - 1)
                    
                    units.append({
                        "text": s_stripped, 
                        "type": "text", 
                        "header_path": current_header_path,
                        "line_start": s_start_line,
                        "line_end": s_end_line
                    })
                    current_rel_pos = found_pos + len(s_stripped)
            else:
                units.append({
                    "text": part, 
                    "type": "text", 
                    "header_path": current_header_path,
                    "line_start": part_start_line,
                    "line_end": part_end_line
                })

        return units

    async def chunk_text(self, content: str, max_dynamic_size: Optional[int] = None, min_dynamic_size: Optional[int] = None, long_paragraph_length: Optional[int] = None) -> List[dict]:
        units = self.get_markdown_units(content)
        if not units:
            return []

        unit_texts = [u["text"] for u in units]

        def get_chunk_line_ranges(chunk_units):
            ranges = []
            for u in chunk_units:
                ranges.append((u["line_start"], u["line_end"]))
            
            if not ranges:
                return []
            
            ranges.sort(key=lambda x: x[0])
            
            merged = []
            current_start, current_end = ranges[0]
            for start, end in ranges[1:]:
                if start <= current_end + 1:
                    current_end = max(current_end, end)
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = start, end
            merged.append((current_start, current_end))
            return merged

        if len(units) == 1:
            return [{
                "content": units[0]["text"], 
                "header_path": units[0]["header_path"],
                "line_ranges": get_chunk_line_ranges([units[0]])
            }]

        self.console.log(f"Semantic chunking: generating embeddings for {len(units)} units...")
        embeddings = np.array(await self.get_embeddings(unit_texts))

        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self.cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        if not similarities:
            return [{
                "content": u["text"], 
                "header_path": u["header_path"],
                "line_ranges": get_chunk_line_ranges([u])
            } for u in units]

        threshold = np.percentile(similarities, 40)

        chunks = []
        current_chunk_units = [units[0]]

        MAX_DYNAMIC_SIZE = max_dynamic_size or self.max_dynamic_size
        MIN_DYNAMIC_SIZE = min_dynamic_size or self.min_dynamic_size

        for i in range(len(similarities)):
            sim = similarities[i]
            next_unit = units[i + 1]

            current_text_len = sum(len(u["text"]) + 2 for u in current_chunk_units)

            is_topic_shift = sim < threshold
            is_big_enough = current_text_len >= MIN_DYNAMIC_SIZE
            will_be_too_big = current_text_len + len(next_unit["text"]) > MAX_DYNAMIC_SIZE

            if (is_topic_shift and is_big_enough) or will_be_too_big:
                chunk_content = "\n\n".join([u["text"] for u in current_chunk_units])
                chunk_header_path = current_chunk_units[0]["header_path"]
                line_ranges = get_chunk_line_ranges(current_chunk_units)

                chunks.append({
                    "content": chunk_content, 
                    "header_path": chunk_header_path,
                    "line_ranges": line_ranges
                })

                last_unit = current_chunk_units[-1]
                if last_unit["type"] != "header":
                    current_chunk_units = [last_unit, next_unit]
                else:
                    current_chunk_units = [next_unit]
            else:
                current_chunk_units.append(next_unit)

        if current_chunk_units:
            chunk_content = "\n\n".join([u["text"] for u in current_chunk_units])
            chunk_header_path = current_chunk_units[0]["header_path"]
            line_ranges = get_chunk_line_ranges(current_chunk_units)
            chunks.append({
                "content": chunk_content, 
                "header_path": chunk_header_path,
                "line_ranges": line_ranges
            })

        return chunks

    async def enrich_chunk_metadata(self, chunk_text: str, header_path: str, chunk_index: Optional[int] = None) -> dict:
        for attempt in range(self.max_retries):
            try:
                raw = await self.async_chat_client.chat.completions.create(
                    model=self.chat_model,
                    response_model=ChunkMetadata,
                    messages=[
                        {"role": "system", "content": "Analyze the provided scientific text and extract the requested metadata."},
                        {"role": "user", "content": f"Context/Header Path: {header_path}\n\nText:\n{chunk_text[:10000]}"}
                    ]
                )

                if isinstance(raw, ChunkMetadata):
                    data = raw.model_dump()
                    
                    data["hypothetical_questions"] = data["hypothetical_questions"][:4]
                    data["keywords"] = data["keywords"][:5]
                    return data

            except Exception as e:
                error_str = str(e)
                if attempt < self.max_retries - 1:
                    self.console.print(f"[yellow]Attempt {attempt + 1} failed for chunk {chunk_index if chunk_index is not None else 'unknown'}, retrying...[/yellow]")
                    await asyncio.sleep(1)  # short delay before retry
                else:
                    self.console.print(f"[red]Failed to enrich chunk {chunk_index if chunk_index is not None else 'unknown'} after {self.max_retries} attempts: {error_str[:200]}[/red]")
                    failure_info = {
                        "chunk_index": chunk_index,
                        "header_path": header_path,
                        "chunk_preview": chunk_text[:150].strip() + "...",
                        "error": error_str[:500] if len(error_str) > 500 else error_str
                    }
                    self.failed_enrichments.append(failure_info)

        # Return empty metadata if all retries failed
        return {"summary": "", "hypothetical_questions": [], "keywords": []}

    async def _process_file(self, md_file: Path, content: str, start_index: int = 0) -> List[dict]:
        file_chunks = await self.chunk_text(content)
        result = []
        for i, chunk_obj in enumerate(file_chunks):
            result.append({
                "content": chunk_obj["content"],
                "header_path": chunk_obj["header_path"],
                "line_ranges": chunk_obj.get("line_ranges", []),
                "source_file": str(md_file),
                "chunk_index": start_index + i,
                "total_chunks_in_file": len(file_chunks)
            })
        self.console.print(f"Processed {md_file}: {len(file_chunks)} chunks")
        return result

    async def process_and_save(self, input_dir: Path = Path("output"), chunks_dir: Path = Path("chunks"), enrich_batch_size: Optional[int] = None) -> None:
        all_chunks_data: List[dict] = []
        
        md_files = list(input_dir.rglob("*.md"))
        tasks = []
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    tasks.append(self._process_file(md_file, content))
            except Exception as e:
                self.console.print(f"[red]Error reading {md_file}: {e}[/red]")
        
        file_results = await asyncio.gather(*tasks)
        for file_chunks in file_results:
            all_chunks_data.extend(file_chunks)

        if not all_chunks_data:
            self.console.print("[yellow]No chunks found. Exiting.[/yellow]")
            return
        chunks_dir.mkdir(exist_ok=True)

        self.console.print(f"\nSaving chunks to [bold]{chunks_dir}[/bold]...")

        async def enrich_and_save(idx, chunk_data):
            metadata = await self.enrich_chunk_metadata(chunk_data["content"], chunk_data["header_path"], chunk_index=idx)
            chunk_data.update(metadata)
            chunk_data["global_chunk_index"] = idx
            chunk_data["chunk_index"] = idx
            chunk_filename = chunks_dir / f"chunk_{idx+1:03d}.json"
            with open(chunk_filename, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), console=self.console) as progress:
            task = progress.add_task("Enriching chunks", total=len(all_chunks_data))
            
            batch_size = enrich_batch_size or self.enrich_batch_size
            for i in range(0, len(all_chunks_data), batch_size):
                batch = all_chunks_data[i:i + batch_size]
                batch_tasks = [enrich_and_save(i + j, chunk_data) for j, chunk_data in enumerate(batch)]
                await asyncio.gather(*batch_tasks)
                progress.update(task, advance=len(batch), description=f"Enriching chunks {i+len(batch)}/{len(all_chunks_data)}")

        self.console.print(f"[green]Saved {len(all_chunks_data)} chunks.[/green]")
        
        if self.failed_enrichments:
            self.console.print(f"\n[bold red]Failed to enrich {len(self.failed_enrichments)} chunks:[/bold red]")
            for idx, failure in enumerate(self.failed_enrichments, 1):
                self.console.print(f"\n[bold]Failure #{idx} (Chunk {failure.get('chunk_index', 'unknown')}):[/bold]")
                self.console.print(f"  Header: [cyan]{failure['header_path']}[/cyan]")
                self.console.print(f"  Preview: {failure['chunk_preview']}")
                self.console.print(f"  Error: [red]{failure['error']}[/red]")

    def run(self) -> None:
        asyncio.run(self.process_and_save())

if __name__ == "__main__":
    Chunker().run()

