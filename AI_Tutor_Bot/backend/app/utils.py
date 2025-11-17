import fitz  # PyMuPDF
import hashlib
import json
import os
from typing import List, Dict, Any

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with page information"""
    doc = fitz.open(pdf_path)
    pages_text = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages_text.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })
    
    doc.close()
    return pages_text

def create_text_chunks(pages_text: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Split text into overlapping chunks"""
    chunks = []
    chunk_id = 0
    
    for page in pages_text:
        text = page["text"]
        page_num = page["page_number"]
        
        # Simple sentence-based splitting (you can improve this)
        sentences = text.split('. ')
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": current_chunk.strip(),
                        "page": page_num
                    })
                    chunk_id += 1
                    # Overlap: keep last few sentences
                    current_chunk = ". ".join(current_chunk.split(". ")[-3:]) + sentence + ". "
                else:
                    current_chunk = sentence + ". "
        
        # Add the last chunk for the page
        if current_chunk.strip():
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": current_chunk.strip(),
                "page": page_num
            })
            chunk_id += 1
    
    return chunks

def generate_topic_id(pdf_filename: str) -> str:
    """Generate unique topic ID based on filename and timestamp"""
    import time
    timestamp = str(int(time.time()))
    base_name = os.path.splitext(pdf_filename)[0]
    return f"{base_name}_{timestamp}"

def save_embeddings(topic_id: str, chunks: List[Dict], image_metadata: List[Dict]):
    """Save embeddings and metadata to JSON files"""
    os.makedirs("data/embeddings", exist_ok=True)
    
    # Save text chunks
    with open(f"data/embeddings/{topic_id}_chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    
    # Save image metadata
    with open(f"data/embeddings/{topic_id}_images.json", "w") as f:
        json.dump(image_metadata, f, indent=2)

def load_embeddings(topic_id: str) -> tuple:
    """Load embeddings and metadata from JSON files"""
    try:
        with open(f"data/embeddings/{topic_id}_chunks.json", "r") as f:
            chunks = json.load(f)
        
        with open(f"data/embeddings/{topic_id}_images.json", "r") as f:
            images = json.load(f)
        
        return chunks, images
    except FileNotFoundError:
        return None, None

def discover_images_from_folder(images_folder: str = "images") -> List[Dict]:
    """Automatically discover images from folder and generate metadata"""
    import re
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    image_metadata = []
    
    # Try multiple possible paths (project root or relative to backend)
    possible_paths = [
        images_folder,  # Relative to current directory
        f"../{images_folder}",  # One level up (if running from backend/)
        os.path.join(os.path.dirname(os.path.dirname(__file__)), images_folder)  # Project root
    ]
    
    actual_path = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_path = path
            break
    
    if not actual_path:
        print(f"Warning: Images folder '{images_folder}' not found in any of these locations: {possible_paths}")
        return []
    
    images_folder = actual_path
    
    # Try to load from images_metadata.json first (priority), then demo_images.json
    metadata_paths = [
        "images_metadata.json",
        "../images_metadata.json",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "images_metadata.json"),
        "demo_images.json",
        "../demo_images.json"
    ]
    
    for metadata_path in metadata_paths:
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata_data = json.load(f)
                    if "images" in metadata_data:
                        print(f"Loaded image metadata from: {metadata_path}")
                        return metadata_data["images"]
            except Exception as e:
                print(f"Could not load {metadata_path}: {e}")
                continue
    
    # Scan folder for images
    image_files = []
    for filename in os.listdir(images_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in image_extensions:
            image_files.append(filename)
    
    # Generate metadata for each image
    for idx, filename in enumerate(sorted(image_files), start=1):
        # Extract title from filename (remove extension, split by caps/underscores)
        base_name = os.path.splitext(filename)[0]
        # Convert CamelCase or snake_case to readable title
        title = re.sub(r'([a-z])([A-Z])', r'\1 \2', base_name)
        title = title.replace('_', ' ').replace('-', ' ')
        title = ' '.join(word.capitalize() for word in title.split())
        
        # Generate keywords from filename
        keywords = []
        # Split by capital letters, underscores, hyphens
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', base_name)
        keywords.extend([w.lower() for w in words if len(w) > 2])
        
        # Generate description
        description = f"Diagram or illustration related to {title.lower()}"
        
        image_metadata.append({
            "id": f"img_{idx:03d}",
            "filename": filename,
            "title": title,
            "keywords": keywords,
            "description": description
        })
    
    return image_metadata