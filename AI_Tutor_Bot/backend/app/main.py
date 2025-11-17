from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import shutil
from typing import List

from app.models import *
from app.utils import *
from app.rag import rag_pipeline
from app.image_retrieval import image_retriever

app = FastAPI(title="AI Tutor API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for images
app.mount("/static", StaticFiles(directory="data/images"), name="static")

# Ensure directories exist
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/embeddings", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Copy images to data/images on startup
def copy_images_on_startup():
    """Copy images from images folder to data/images for serving"""
    image_metadata = discover_images_from_folder("images")
    if image_metadata:
        images_base_path = None
        possible_image_paths = ["images", "../images", os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")]
        for path in possible_image_paths:
            if os.path.exists(path):
                images_base_path = path
                break
        
        if images_base_path:
            copied_count = 0
            for img in image_metadata:
                src_path = os.path.join(images_base_path, img['filename'])
                dst_path = f"data/images/{img['filename']}"
                if os.path.exists(src_path):
                    if not os.path.exists(dst_path) or os.path.getmtime(src_path) > os.path.getmtime(dst_path):
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
            if copied_count > 0:
                print(f"Copied {copied_count} images to data/images on startup")

# Copy images on startup
copy_images_on_startup()

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and process it for RAG"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process PDF
    try:
        # Extract text
        pages_text = extract_text_from_pdf(file_path)
        
        # Create chunks
        chunks = create_text_chunks(pages_text)
        
        # Generate topic ID
        topic_id = generate_topic_id(file.filename)
        
        # Create embeddings
        rag_pipeline.create_embeddings(chunks)
        
        # Automatically discover images from images folder
        image_metadata = discover_images_from_folder("images")
        
        if not image_metadata:
            print("Warning: No images found in images folder. Using empty list.")
        
        # Load image metadata into retriever
        image_retriever.load_image_metadata(image_metadata)
        
        # Save embeddings and metadata
        save_embeddings(topic_id, chunks, image_metadata)
        
        # Copy images to static directory
        # Find the actual images folder path
        images_base_path = None
        possible_image_paths = ["images", "../images", os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")]
        for path in possible_image_paths:
            if os.path.exists(path):
                images_base_path = path
                break
        
        if images_base_path:
            for img in image_metadata:
                src_path = os.path.join(images_base_path, img['filename'])
                dst_path = f"data/images/{img['filename']}"
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied image: {img['filename']}")
                else:
                    print(f"Warning: Image not found at {src_path}")
        else:
            print("Warning: Could not locate images folder for copying")
        
        return UploadResponse(
            topic_id=topic_id,
            message="PDF processed successfully",
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat questions and return answers with relevant images"""
    
    # Load embeddings for the topic
    chunks, images = load_embeddings(request.topic_id)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    # Set up RAG pipeline with loaded chunks
    rag_pipeline.chunks = chunks
    rag_pipeline.create_embeddings(chunks)
    
    # Always reload images from metadata file to ensure we have the latest/correct images
    # This ensures we use the correct filenames even if old embeddings exist
    image_metadata = discover_images_from_folder("images")
    if image_metadata:
        image_retriever.load_image_metadata(image_metadata)
        print(f"Loaded {len(image_metadata)} images for retrieval")
        
        # Ensure images are copied to data/images directory for serving
        images_base_path = None
        possible_image_paths = ["images", "../images", os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")]
        for path in possible_image_paths:
            if os.path.exists(path):
                images_base_path = path
                break
        
        if images_base_path:
            for img in image_metadata:
                src_path = os.path.join(images_base_path, img['filename'])
                dst_path = f"data/images/{img['filename']}"
                # Only copy if source exists and destination doesn't exist or is older
                if os.path.exists(src_path):
                    if not os.path.exists(dst_path) or os.path.getmtime(src_path) > os.path.getmtime(dst_path):
                        os.makedirs("data/images", exist_ok=True)
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied image for serving: {img['filename']}")
    elif images:
        # Fallback to saved images if metadata file not found
        image_retriever.load_image_metadata(images)
        print(f"Loaded {len(images)} images from saved embeddings")
    else:
        print("Warning: No images found for retrieval")
    
    # Retrieve relevant chunks
    relevant_chunks = rag_pipeline.retrieve_chunks(request.question, k=4)
    
    # Generate answer
    answer = rag_pipeline.generate_answer(request.question, relevant_chunks)
    
    # Find relevant image based on answer and question
    relevant_image = image_retriever.find_relevant_image(answer, request.question)
    
    # Log image retrieval for debugging
    if relevant_image:
        print(f"Found relevant image: {relevant_image['filename']} (similarity match)")
    else:
        print("No relevant image found above threshold")
    
    # Prepare sources
    sources = [f"Page {chunk['page']}" for chunk in relevant_chunks]
    
    return ChatResponse(
        answer=answer,
        image_id=relevant_image["id"] if relevant_image else None,
        image_filename=relevant_image["filename"] if relevant_image else None,
        image_title=relevant_image["title"] if relevant_image else None,
        sources=list(set(sources))  # Remove duplicates
    )

@app.get("/images/{topic_id}")
async def get_image_metadata(topic_id: str):
    """Get image metadata for a topic"""
    chunks, images = load_embeddings(topic_id)
    if images is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    return images

@app.get("/images/file/{filename}")
async def get_image_file(filename: str):
    """Serve image files"""
    # Try multiple possible paths
    possible_paths = [
        f"data/images/{filename}",
        f"../data/images/{filename}",
        os.path.join("data", "images", filename),
        os.path.join(os.path.dirname(__file__), "..", "data", "images", filename)
    ]
    
    for file_path in possible_paths:
        abs_path = os.path.abspath(file_path)
        if os.path.exists(abs_path):
            return FileResponse(abs_path)
    
    # If not found, try to copy it from source
    image_metadata = discover_images_from_folder("images")
    if image_metadata:
        images_base_path = None
        possible_image_paths = ["images", "../images", os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")]
        for path in possible_image_paths:
            if os.path.exists(path):
                images_base_path = path
                break
        
        if images_base_path:
            for img in image_metadata:
                if img['filename'] == filename:
                    src_path = os.path.join(images_base_path, filename)
                    dst_path = f"data/images/{filename}"
                    if os.path.exists(src_path):
                        os.makedirs("data/images", exist_ok=True)
                        shutil.copy2(src_path, dst_path)
                        if os.path.exists(dst_path):
                            return FileResponse(os.path.abspath(dst_path))
    
    raise HTTPException(status_code=404, detail=f"Image not found: {filename}")

@app.get("/")
async def root():
    return {"message": "AI Tutor API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)