#!/usr/bin/env python3
"""
CLI tool for uploading PDFs to the Vector Processing API.
"""

import typer
import asyncio
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import application services
from main import processing_orchestrator, ProcessingTask, weaviate_client, ProcessingStatus

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

async def process_file(
    file_path: Path,
    title: Optional[str],
    description: Optional[str],
    tags: Optional[str],
    process_images: bool,
    progress: Progress,
    overall_task_id: int
):
    """Process a single file."""
    task_id = progress.add_task(f"Starting {file_path.name}...", total=100)
    
    def update_progress(task: ProcessingTask):
        # Ensure we don't overwrite the final status
        if task.status == ProcessingStatus.COMPLETED:
            progress.update(task_id, completed=100, description=f"[green]Completed[/green] {file_path.name}")
        elif task.status == ProcessingStatus.FAILED:
            progress.update(task_id, description=f"[red]Failed[/red] {file_path.name}: {task.error_message}")
        else:
            # Show detailed step info
            step_info = f": {task.current_step}" if task.current_step else ""
            progress.update(task_id, completed=task.progress, description=f"[{task.status.value}] {file_path.name}{step_info}")
            
            # Print verbose logs to console above the progress bar
            if task.current_step:
                progress.console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')} - {task.current_step}[/dim]")

    try:
        # Parse tags
        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        
        # Run processing
        result_task = await processing_orchestrator.process_document(
            file_path=str(file_path),
            filename=file_path.name,
            title=title,
            description=description,
            tags=tag_list,
            process_images=process_images,
            progress_callback=update_progress
        )
        
        if result_task.status == ProcessingStatus.COMPLETED:
            return True, result_task.task_id
        else:
            return False, result_task.error_message
            
    except Exception as e:
        progress.console.print(f"[red]Exception processing {file_path.name}: {e}[/red]")
        return False, str(e)
    finally:
        progress.remove_task(task_id)

@app.command()
def upload(
    path: Path = typer.Argument(..., help="Path to PDF file or directory containing PDFs"),
    title: Optional[str] = typer.Option(None, help="Document title (for single file)"),
    description: Optional[str] = typer.Option(None, help="Document description"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated tags"),
    process_images: bool = typer.Option(True, help="Extract and process images"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for PDFs in directories"),
):
    """
    Upload and process PDF(s) directly.
    """
    # Run async main function
    asyncio.run(async_upload(path, title, description, tags, process_images, recursive))

async def async_upload(path: Path, title: Optional[str], description: Optional[str], tags: Optional[str], process_images: bool, recursive: bool):
    
    # Setup services
    if not await setup_services():
        return

    files_to_process = []
    
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            files_to_process.append(path)
        else:
            console.print(f"[bold red]Error:[/bold red] File {path} is not a PDF.")
            return
    elif path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        files_to_process = list(path.glob(pattern))
        if not files_to_process:
            console.print(f"[yellow]No PDF files found in {path}[/yellow]")
            return
    else:
        console.print(f"[bold red]Error:[/bold red] Path {path} does not exist.")
        return

    console.print(Panel(f"Found [bold green]{len(files_to_process)}[/bold green] PDF files to process", title="Processing Job"))

    success_count = 0
    fail_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        overall_task = progress.add_task("[bold blue]Overall Progress", total=len(files_to_process))
        
        for pdf_file in files_to_process:
            progress.update(overall_task, description=f"[bold blue]Processing {pdf_file.name}...")
            
            file_title = title if title and len(files_to_process) == 1 else None
            
            success, result = await process_file(
                pdf_file, 
                file_title, 
                description, 
                tags, 
                process_images, 
                progress, 
                overall_task
            )
            
            if success:
                success_count += 1
                progress.console.print(f"[green]✓[/green] Processed [bold]{pdf_file.name}[/bold]")
            else:
                fail_count += 1
                progress.console.print(f"[red]✗[/red] Failed [bold]{pdf_file.name}[/bold]: {result}")
            
            progress.advance(overall_task)

    # Cleanup
    await weaviate_client.disconnect()

    # Summary
    table = Table(title="Processing Summary")
    table.add_column("Status", style="bold")
    table.add_column("Count")
    
    table.add_row("[green]Success[/green]", str(success_count))
    table.add_row("[red]Failed[/red]", str(fail_count))
    table.add_row("Total", str(len(files_to_process)))
    
    console.print(table)

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
