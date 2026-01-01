#!/usr/bin/env python3
"""
Standalone script to process images from the output folder.
Creates image chunks with captions and embeddings in chunks/images/
"""

from pathlib import Path
from image_chunking import ImageChunker

def main():
    """Process all images from output folder."""
    input_dir = Path("output")
    output_dir = Path("chunks/images")
    
    chunker = ImageChunker()
    results = chunker.run(input_dir, output_dir)
    
    print(f"\n✅ Processing complete! Created {len(results)} image chunks in {output_dir}")

if __name__ == "__main__":
    main()
