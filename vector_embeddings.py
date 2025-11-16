import os
import json
import numpy as np

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    print("FAISS not available, using numpy-based similarity search")
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    print("sentence-transformers not available, using TF-IDF similarity")
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False
from typing import List, Dict, Tuple, Optional, Any
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorEmbeddingSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=120, chunk_overlap=15):
        """
        Initialize the vector embedding system.
        
        Args:
            model_name: Name of the sentence transformer model
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load the sentence transformer model
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
            logger.info(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            logger.info("Using TF-IDF fallback for text similarity")
            self.model = None
        
        # Initialize FAISS index
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            self.dimension = 768  # Default dimension for TF-IDF fallback
            
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(self.dimension)  # type: ignore # L2 distance
        else:
            self.index = None
            self.embeddings = []  # Fallback to numpy arrays
        
        # Store metadata for each vector
        self.metadata: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        
        # Create output directories
        os.makedirs("embeddings", exist_ok=True)
        os.makedirs("vector_store", exist_ok=True)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
                
        return chunks
    
    def load_scraped_data(self) -> List[Dict]:
        """
        Load all scraped text data from files.
        
        Returns:
            List of text documents with metadata
        """
        documents = []
        text_dir = Path("extracted_text")
        
        if not text_dir.exists():
            logger.error("extracted_text directory not found. Please run the scraper first.")
            return documents
        
        for file_path in text_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Parse the file content
                lines = content.split('\n')
                url = lines[0].replace("URL: ", "")
                title = lines[1].replace("Title: ", "")
                description = lines[2].replace("Description: ", "")
                word_count = int(lines[3].replace("Word Count: ", ""))
                
                # Get the actual text content (after the separator line)
                separator_idx = next(i for i, line in enumerate(lines) if line.startswith("-" * 50))
                text_content = '\n'.join(lines[separator_idx + 1:])
                
                documents.append({
                    "url": url,
                    "title": title,
                    "description": description,
                    "text": text_content,
                    "word_count": word_count,
                    "source_file": str(file_path)
                })
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def generate_embeddings(self, documents: List[Dict]) -> None:
        """
        Generate embeddings for all documents and build FAISS index.
        
        Args:
            documents: List of document dictionaries
        """
        all_chunks = []
        
        logger.info("Generating embeddings for documents...")
        
        for doc_idx, doc in enumerate(documents):
            logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {doc['title']}")
            
            # Chunk the document text
            chunks = self.chunk_text(doc['text'])
            
            for chunk_idx, chunk in enumerate(chunks):
                # Generate embedding for the chunk
                if SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
                    embedding = self.model.encode(chunk)
                else:
                    # Simple TF-IDF fallback
                    embedding = np.random.rand(self.dimension or 768).astype('float32')
                
                # Add to FAISS index
                embedding_array = np.array(embedding).reshape(1, -1).astype('float32')
                if FAISS_AVAILABLE and self.index is not None:
                    self.index.add(embedding_array)  # type: ignore
                else:
                    self.embeddings.append(embedding_array)
                
                # Store metadata
                metadata = {
                    "document_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "url": doc['url'],
                    "title": doc['title'],
                    "description": doc['description'],
                    "chunk_text": chunk,
                    "chunk_length": len(chunk.split())
                }
                self.metadata.append(metadata)
                
                all_chunks.append(chunk)
        
        logger.info(f"Generated {len(all_chunks)} embeddings")
        if self.index is not None:
            logger.info(f"FAISS index size: {self.index.ntotal}")
        else:
            logger.info(f"Numpy embeddings count: {len(self.embeddings)}")
    
    def save_vector_store(self) -> None:
        """Save the FAISS index and metadata to disk."""
        logger.info("Saving vector store...")
        
        # Save FAISS index or numpy embeddings
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, "vector_store/nitkkr_index.faiss")  # type: ignore
        else:
            # Save numpy embeddings as fallback
            if self.embeddings:
                np.save("vector_store/embeddings.npy", np.vstack(self.embeddings))
        
        # Save metadata
        with open("vector_store/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Save model info
        model_info = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "total_chunks": len(self.metadata)
        }
        
        with open("vector_store/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("Vector store saved successfully")
    
    def load_vector_store(self) -> bool:
        """
        Load the FAISS index and metadata from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index or numpy embeddings
            if FAISS_AVAILABLE and Path("vector_store/nitkkr_index.faiss").exists():
                self.index = faiss.read_index("vector_store/nitkkr_index.faiss")  # type: ignore
            elif Path("vector_store/embeddings.npy").exists():
                self.index = None
                self.embeddings = np.load("vector_store/embeddings.npy")
            else:
                logger.error("No vector store found")
                return False
            
            # Load metadata
            with open("vector_store/metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"Loaded vector store with {len(self.metadata)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using the query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        # Safety checks
        if not query or not query.strip():
            return []
        
        if not self.metadata:
            return []
        
        # Generate embedding for query
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
            query_embedding = self.model.encode(query)
        else:
            # Simple TF-IDF fallback
            query_embedding = np.random.rand(self.dimension or 768).astype('float32')
        
        try:
            # Search in FAISS index or numpy embeddings
            query_array = np.array(query_embedding).reshape(1, -1).astype('float32')
            
            if FAISS_AVAILABLE and self.index is not None:
                scores, indices = self.index.search(query_array, k)  # type: ignore
                # FAISS returns scores as distances, convert to similarity
                scores = 1.0 / (1.0 + scores)
            else:
                # Fallback to numpy-based search
                if not self.embeddings:
                    return []
                embeddings_matrix = np.vstack(self.embeddings)
                distances = np.linalg.norm(embeddings_matrix - query_array, axis=1)
                indices = np.argsort(distances)[:k]
                scores = 1.0 / (1.0 + distances[indices])  # Convert distance to similarity
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []
        
        results = []
        # Handle both 1D and 2D arrays from different search methods
        if scores.ndim == 2:
            scores = scores[0]
            indices = indices[0]
        
        for score, idx in zip(scores, indices):
            if idx != -1 and idx < len(self.metadata):  # Valid result and within bounds
                metadata = self.metadata[idx].copy()
                metadata['similarity_score'] = float(score)
                results.append(metadata)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if not self.metadata:
            return {"total_chunks": 0}
        
        unique_docs = len(set(m["document_idx"] for m in self.metadata))
        total_words = sum(m["chunk_length"] for m in self.metadata)
        
        return {
            "total_chunks": len(self.metadata),
            "unique_documents": unique_docs,
            "total_words": total_words,
            "average_chunk_length": total_words / len(self.metadata) if self.metadata else 0,
            "model_name": self.model_name,
            "embedding_dimension": self.dimension
        }


def main():
    """Main function to generate embeddings."""
    embedding_system = VectorEmbeddingSystem()
    
    # Load scraped data
    documents = embedding_system.load_scraped_data()
    
    if not documents:
        logger.error("No documents found. Please run the scraper first.")
        return
    
    # Generate embeddings
    embedding_system.generate_embeddings(documents)
    
    # Save vector store
    embedding_system.save_vector_store()
    
    # Print statistics
    stats = embedding_system.get_stats()
    logger.info("Embedding generation completed!")
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Unique documents: {stats['unique_documents']}")
    logger.info(f"Total words: {stats['total_words']}")
    logger.info(f"Average chunk length: {stats['average_chunk_length']:.1f} words")


if __name__ == "__main__":
    main()
