import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json

class ImageRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_metadata = []
        self.image_embeddings = None
    
    def load_image_metadata(self, image_metadata: List[Dict]):
        """Load image metadata and create embeddings"""
        self.image_metadata = image_metadata
        
        # Create embeddings from descriptions and keywords
        texts_to_embed = []
        for img in image_metadata:
            text = f"{img['description']} {' '.join(img['keywords'])} {img['title']}"
            texts_to_embed.append(text)
        
        if texts_to_embed:
            print(f"Generating embeddings for {len(texts_to_embed)} images...")
            self.image_embeddings = self.embedding_model.encode(texts_to_embed)
            print(f"✓ Generated {len(self.image_embeddings)} image embeddings (dimension: {self.image_embeddings.shape[1]})")
        else:
            print("Warning: No image metadata provided for embedding generation")
    
    def find_relevant_image(self, answer_text: str, question: str = "") -> Optional[Dict]:
        """Find the most relevant image for the answer using hybrid keyword + embedding matching"""
        if not self.image_metadata or self.image_embeddings is None:
            return None
        
        # Combine question and answer for better context
        search_text = f"{question} {answer_text}"
        search_embedding = self.embedding_model.encode([search_text])
        
        # Normalize embeddings for cosine similarity
        search_norm = search_embedding / (np.linalg.norm(search_embedding, axis=1, keepdims=True) + 1e-8)
        image_norms = self.image_embeddings / (np.linalg.norm(self.image_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarities
        similarities = np.dot(image_norms, search_norm.T).flatten()
        
        # Extract keywords from question and answer (simple keyword extraction)
        import re
        question_lower = question.lower()
        answer_lower = answer_text.lower()
        combined_text = f"{question_lower} {answer_lower}"
        
        # Keyword boost: Check for exact keyword matches and word variations
        keyword_scores = []
        for img in self.image_metadata:
            keyword_match_score = 0.0
            
            # Check if any keywords from image appear in question/answer
            for keyword in img['keywords']:
                keyword_lower = keyword.lower().strip('.,!?;:')
                keyword_stem = keyword_lower.rstrip('s')  # Remove plural 's'
                
                # Exact match
                if keyword_lower in combined_text:
                    keyword_match_score += 0.25
                # Check for word variations (e.g., "reflect" matches "reflection", "reflects")
                elif len(keyword_lower) > 3:
                    # Check if keyword stem is contained in any word in the text
                    words_in_text = combined_text.split()
                    for word in words_in_text:
                        word_clean = word.lower().strip('.,!?;:')
                        word_stem = word_clean.rstrip('s')
                        # Check various matching conditions
                        if (keyword_lower in word_clean or word_clean in keyword_lower or
                            keyword_stem in word_clean or word_stem in keyword_lower or
                            (len(keyword_stem) > 4 and keyword_stem[:4] == word_stem[:4])):
                            keyword_match_score += 0.2
                            break
            
            # Also check title words (higher weight for title matches)
            title_words = [w.lower().strip('.,!?;:') for w in img['title'].lower().split()]
            for word in title_words:
                if len(word) > 3:
                    if word in combined_text:
                        keyword_match_score += 0.25  # Title matches are important
                    # Check for variations
                    for text_word in combined_text.split():
                        if len(text_word) > 3 and (word in text_word or text_word in word):
                            keyword_match_score += 0.15
            
            # Check description for important words
            desc_words = [w.lower().strip('.,!?;:') for w in img['description'].lower().split()]
            for word in desc_words:
                if len(word) > 5 and word in combined_text:  # Only longer, meaningful words
                    keyword_match_score += 0.1
            
            keyword_scores.append(keyword_match_score)
        
        keyword_scores = np.array(keyword_scores)
        
        # Normalize keyword scores to 0-1 range
        if keyword_scores.max() > 0:
            keyword_scores = keyword_scores / max(keyword_scores.max(), 1.0)
        
        # Combine embedding similarity (0-1) with keyword boost (0-1)
        # Weight: 60% embedding similarity, 40% keyword matching (increased keyword weight)
        final_scores = 0.6 * similarities + 0.4 * keyword_scores
        
        # Get most relevant image
        best_match_idx = np.argmax(final_scores)
        best_similarity = final_scores[best_match_idx]
        best_embedding_score = similarities[best_match_idx]
        best_keyword_score = keyword_scores[best_match_idx]
        
        # Log all scores for debugging
        print("\n" + "="*80)
        print("IMAGE RETRIEVAL SCORES:")
        print("="*80)
        for idx, img in enumerate(self.image_metadata):
            print(f"{idx+1}. {img['filename']:40s} | "
                  f"Embedding: {similarities[idx]:.3f} | "
                  f"Keywords: {keyword_scores[idx]:.3f} | "
                  f"Final: {final_scores[idx]:.3f}")
        print("="*80)
        print(f"Selected: {self.image_metadata[best_match_idx]['filename']}")
        print(f"  - Embedding similarity: {best_embedding_score:.3f}")
        print(f"  - Keyword match score: {best_keyword_score:.3f}")
        print(f"  - Final combined score: {best_similarity:.3f}")
        print("="*80 + "\n")
        
        # Always return the best match
        return self.image_metadata[best_match_idx]

# Global image retriever instance
image_retriever = ImageRetriever()