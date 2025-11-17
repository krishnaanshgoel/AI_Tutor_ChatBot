import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import google.generativeai as genai
import os
import time

class RAGPipeline:
    def __init__(self):
        # Using open-source embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = None
        self.chunks = []
        
        # Gemini setup
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "your_api_key_here")
        genai.configure(api_key=self.gemini_api_key)
        
        # Configure safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        # Initialize Gemini model with generation config
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 8192,  # Increased to allow longer responses
        }
        
        # Try different model names based on API version compatibility
        model_names = ['gemini-2.5-flash','gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        self.llm_model = None
        
        for model_name in model_names:
            try:
                self.llm_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                # Test if model works by checking if it can be accessed
                break
            except Exception as e:
                print(f"Failed to initialize model {model_name}: {e}")
                continue
        
        if self.llm_model is None:
            raise RuntimeError("Failed to initialize any Gemini model. Please check your API key and model availability.")
    
    def create_embeddings(self, chunks: List[Dict]) -> None:
        """Create embeddings for text chunks and build FAISS index"""
        self.chunks = chunks
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatL2(dimension)
        self.vector_store.add(embeddings.astype('float32'))
    
    def retrieve_chunks(self, query: str, k: int = 4) -> List[Dict]:
        """Retrieve top-k most relevant chunks"""
        if self.vector_store is None:
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
        # Return relevant chunks
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                relevant_chunks.append(self.chunks[idx])
        
        return relevant_chunks
    
    def generate_answer_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate answer with retry logic for rate limiting"""
        for attempt in range(max_retries):
            try:
                response = self.llm_model.generate_content(prompt)
                
                # Check if response was blocked
                if not response.parts:
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        return f"I cannot answer this question due to content safety restrictions. Please try a different question."
                
                # Get full response text
                response_text = response.text
                
                # Log the full Gemini response to terminal
                print("\n" + "="*80)
                print("GEMINI RESPONSE:")
                print("="*80)
                print(response_text)
                print("="*80 + "\n")
                
                # Check if response was cut off (incomplete)
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    if finish_reason:
                        finish_reason_str = str(finish_reason).upper()
                        if 'MAX_TOKENS' in finish_reason_str or finish_reason == 2:
                            print(f"WARNING: Response was truncated due to max_output_tokens limit. Finish reason: {finish_reason}")
                        elif 'SAFETY' in finish_reason_str or finish_reason == 3:
                            print(f"WARNING: Response was blocked due to safety settings. Finish reason: {finish_reason}")
                        elif 'RECITATION' in finish_reason_str or finish_reason == 4:
                            print(f"WARNING: Response was blocked due to recitation. Finish reason: {finish_reason}")
                        elif 'STOP' in finish_reason_str or finish_reason == 1:
                            print(f"Response completed normally. Finish reason: {finish_reason}")
                        else:
                            print(f"Response finish reason: {finish_reason}")
                
                return response_text
                
            except Exception as e:
                error_str = str(e).lower()
                if "404" in error_str or "not found" in error_str or "not supported" in error_str:
                    # Model not found - try to reinitialize with fallback models
                    if attempt == 0:  # Only try once to avoid infinite loops
                        model_names = ['gemini-1.5-pro', 'gemini-pro']
                        generation_config = {
                            "temperature": 0.3,
                            "top_p": 0.8,
                            "top_k": 40,
                            "max_output_tokens": 8192,  # Increased to allow longer responses
                        }
                        safety_settings = [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        ]
                        model_switched = False
                        for model_name in model_names:
                            try:
                                self.llm_model = genai.GenerativeModel(
                                    model_name=model_name,
                                    generation_config=generation_config,
                                    safety_settings=safety_settings
                                )
                                print(f"Switched to model: {model_name}")
                                model_switched = True
                                break  # Exit inner loop and retry with new model
                            except:
                                continue
                        if model_switched:
                            continue  # Retry the outer loop with new model
                    return "Model configuration error. Please update google-generativeai package: pip install --upgrade google-generativeai"
                elif "quota" in error_str or "rate limit" in error_str or "resource exhausted" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                elif "safety" in error_str or "blocked" in error_str:
                    return "I cannot provide an answer to this question due to content safety guidelines."
                else:
                    print(f"Gemini API error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
        
        return "I apologize, but I'm currently experiencing technical difficulties. Please try again in a moment."
    
    def generate_answer(self, question: str, relevant_chunks: List[Dict]) -> str:
        """Generate answer using Gemini with retrieved context"""
        
        # Prepare context
        context = "\n\n".join([f"Source {i+1} (Page {chunk['page']}): {chunk['text']}" 
                              for i, chunk in enumerate(relevant_chunks)])
        
        system_prompt = """You are an AI tutor helping students understand educational content. 
Your role is to provide clear, accurate, and helpful explanations based on the provided textbook context."""

        user_prompt = f"""Based on the following context from the textbook, please answer the student's question:

CONTEXT FROM TEXTBOOK:
{context}

STUDENT'S QUESTION: {question}

Please provide a helpful answer that:
1. Directly addresses the question using information from the context
2. Is educational and easy to understand
3. Uses examples when helpful
4. Is structured clearly
5. Does not mention that you are using retrieved context
6. Stays focused on the educational content

Answer:"""
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        return self.generate_answer_with_retry(full_prompt)

# Global RAG instance
rag_pipeline = RAGPipeline()