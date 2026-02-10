#!/usr/bin/env python3
"""
WINDSURF CHROMA FULL REBUILD FROM ELASTICSEARCH
Source of truth: Elasticsearch index rag_files_v1
Target: persistent Chroma dir on /media/felix/RAG/1/volumes/chroma
"""

import os
import sys
import json
import hashlib
import time
import logging
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
from pathlib import Path

import requests
import chromadb
from chromadb.config import Settings
import tiktoken

# Configuration
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_files_v1")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/media/felix/RAG/1/volumes/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_files_v1_chunks")
BATCH_DOCS = int(os.getenv("BATCH_DOCS", "200"))
BATCH_UPSERT = int(os.getenv("BATCH_UPSERT", "1000"))
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1200"))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", "200"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{CHROMA_PERSIST_DIR}/rebuild.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ChromaRebuilder:
    def __init__(self):
        self.es_url = ES_URL
        self.es_index = ES_INDEX
        self.chroma_dir = Path(CHROMA_PERSIST_DIR)
        self.collection_name = COLLECTION_NAME
        self.batch_docs = BATCH_DOCS
        self.batch_upsert = BATCH_UPSERT
        self.chunk_size = CHUNK_SIZE_CHARS
        self.overlap = OVERLAP_CHARS
        
        # State file for resume capability
        self.state_file = self.chroma_dir / "_rebuild_state.json"
        self.state = self.load_state()
        
        # Initialize Chroma
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(allow_reset=True)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Found existing collection: {self.collection_name}")
            # Clear collection for full rebuild
            self.chroma_client.delete_collection(self.collection_name)
            logger.info("Cleared existing collection for full rebuild")
        except Exception:
            logger.info("Creating new collection")
        
        self.collection = self.chroma_client.create_collection(self.collection_name)
        
        # Initialize embedding provider (placeholder - would use actual embedding model)
        self.embedding_provider = self.get_embedding_provider()
        
    def load_state(self) -> Dict[str, Any]:
        """Load rebuild state for resume capability"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")
        return {
            "last_search_after": None,
            "docs_processed": 0,
            "chunks_written": 0,
            "start_time": None,
            "chunk_params": {
                "chunk_size_chars": self.chunk_size,
                "overlap_chars": self.overlap
            }
        }
    
    def save_state(self):
        """Save current rebuild state"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def get_embedding_provider(self):
        """Get embedding provider - use same as agent"""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "all-MiniLM-L6-v2"
            model = SentenceTransformer(model_name)
            logger.info(f"Using sentence-transformers model: {model_name}")
            
            class SentenceTransformerProvider:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    embeddings = self.model.encode(texts, convert_to_numpy=True)
                    return embeddings.tolist()
            
            return SentenceTransformerProvider(model)
        except ImportError as e:
            logger.error(f"Failed to import sentence-transformers: {e}")
            logger.warning("Falling back to mock embeddings - THIS WILL NOT WORK PROPERLY")
            # Fallback to mock
            class MockEmbeddingProvider:
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    import random
                    return [[random.random() for _ in range(384)] for _ in texts]
            return MockEmbeddingProvider()
    
    def fetch_es_documents(self, search_after: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Fetch documents from Elasticsearch using search_after for stability"""
        query = {
            "size": self.batch_docs,
            "_source": ["content", "file", "path", "meta"],
            "query": {"match_all": {}},
            "sort": [{"_doc": {"order": "asc"}}]
        }
        
        if search_after:
            query["search_after"] = [search_after]
        
        while True:
            try:
                response = requests.post(
                    f"{self.es_url}/{self.es_index}/_search",
                    json=query,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                
                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break
                
                for hit in hits:
                    yield hit
                
                # Update search_after for next iteration
                if len(hits) < self.batch_docs:
                    break
                
                last_hit = hits[-1]
                search_after = last_hit["sort"][0]
                query["search_after"] = [search_after]
                
            except Exception as e:
                logger.error(f"Error fetching documents: {e}")
                break
    
    def normalize_text(self, text: str) -> str:
        """Normalize text content"""
        if not text:
            return ""
        
        # Replace Windows line endings with Unix
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize multiple consecutive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Preserve table structure and important line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines
        text = re.sub(r'\n(?=[A-Za-z0-9])', ' ', text)  # Single newlines in text
        
        return text.strip()
    
    def is_tabular_content(self, text: str) -> bool:
        """Detect if content is tabular (many tabs or short lines)"""
        lines = text.split('\n')
        tab_count = sum(line.count('\t') for line in lines)
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        # Consider tabular if many tabs or very short average lines
        return tab_count > len(lines) * 2 or avg_line_length < 50
    
    def chunk_text(self, text: str, es_id: str) -> List[Dict[str, Any]]:
        """Chunk text deterministically"""
        if not text:
            return []
        
        text = self.normalize_text(text)
        is_tabular = self.is_tabular_content(text)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Find best split point
            if end < len(text):
                split_point = self.find_split_point(text, start, end, is_tabular)
                if split_point > start:
                    end = split_point
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_index = len(chunks)
                chunk_id = self.generate_chunk_id(es_id, chunk_index, chunk_text)
                
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "es_id": es_id,
                    "chunk_index": chunk_index,
                    "chunk_start_char": start,
                    "chunk_end_char": end
                })
            
            # Calculate next start with overlap
            start = max(start + 1, end - self.overlap)
        
        return chunks
    
    def find_split_point(self, text: str, start: int, end: int, is_tabular: bool) -> int:
        """Find best split point for chunking"""
        window = text[start:end]
        
        if is_tabular:
            # For tabular content, prefer newlines
            split_chars = ['\n\n', '\n', '\t']
        else:
            # For normal content, prefer paragraphs
            split_chars = ['\n\n', '\n', '. ', ' ', '\t']
        
        for char in split_chars:
            last_pos = window.rfind(char)
            if last_pos > len(window) * 0.7:  # Don't split too early
                return start + last_pos + len(char)
        
        return end
    
    def generate_chunk_id(self, es_id: str, chunk_index: int, chunk_text: str) -> str:
        """Generate deterministic chunk ID"""
        content_hash = hashlib.sha1(chunk_text.encode('utf-8')).hexdigest()[:16]
        return hashlib.sha1(f"{es_id}:{chunk_index}:{content_hash}".encode('utf-8')).hexdigest()
    
    def process_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single document into chunks with metadata"""
        es_id = doc["_id"]
        source = doc["_source"]
        
        content = source.get("content", "")
        if not content or not content.strip():
            logger.debug(f"Skipping empty content for doc {es_id}")
            return []
        
        # Extract metadata
        file_info = source.get("file", {})
        path_info = source.get("path", {})
        meta_info = source.get("meta", {})
        
        # Create chunks
        chunks = self.chunk_text(content, es_id)
        
        # Add metadata to each chunk
        for chunk in chunks:
            chunk.update({
                "path": path_info.get("real", file_info.get("url", "")),
                "filename": file_info.get("filename", ""),
                "extension": file_info.get("extension", ""),
                "content_type": file_info.get("content_type", ""),
                "filesize": file_info.get("filesize", 0),
                "checksum": file_info.get("checksum", ""),
                "author": meta_info.get("author", ""),
                "created": file_info.get("created", ""),
                "last_modified": file_info.get("last_modified", ""),
                "source": f"{path_info.get('real', file_info.get('url', ''))}"
            })
        
        return chunks
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """Upsert chunks to Chroma in batches"""
        if not chunks:
            return
        
        # Prepare batch data
        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {k: v for k, v in chunk.items() if k not in ["id", "text"]}
            metadatas.append(metadata)
        
        # Generate embeddings
        embeddings = self.embedding_provider.embed_documents(texts)
        
        # Upsert in batches
        for i in range(0, len(ids), self.batch_upsert):
            batch_end = min(i + self.batch_upsert, len(ids))
            
            self.collection.upsert(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                documents=texts[i:batch_end]
            )
            
            logger.info(f"Upserted batch {i//self.batch_upsert + 1}: {batch_end - i} chunks")
    
    def rebuild(self):
        """Main rebuild process"""
        logger.info("Starting Chroma rebuild from Elasticsearch")
        logger.info(f"ES: {self.es_url}/{self.es_index}")
        logger.info(f"Chroma: {self.chroma_dir}")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Chunk size: {self.chunk_size}, Overlap: {self.overlap}")
        
        self.state["start_time"] = datetime.now().isoformat()
        self.save_state()
        
        try:
            # Resume from last position if state exists
            search_after = self.state.get("last_search_after")
            docs_processed = self.state.get("docs_processed", 0)
            chunks_written = self.state.get("chunks_written", 0)
            
            logger.info(f"Resuming from doc {docs_processed}, chunks {chunks_written}")
            
            # Process documents
            batch_chunks = []
            
            for doc in self.fetch_es_documents(search_after):
                chunks = self.process_document(doc)
                batch_chunks.extend(chunks)
                
                # Upsert when batch is full
                if len(batch_chunks) >= self.batch_upsert:
                    self.upsert_chunks(batch_chunks)
                    chunks_written += len(batch_chunks)
                    batch_chunks = []
                
                docs_processed += 1
                
                # Update state periodically
                if docs_processed % 1000 == 0:
                    self.state["docs_processed"] = docs_processed
                    self.state["chunks_written"] = chunks_written
                    self.state["last_search_after"] = doc.get("sort", [None])[0]
                    self.save_state()
                    
                    logger.info(f"Processed {docs_processed} docs, {chunks_written} chunks")
            
            # Process remaining chunks
            if batch_chunks:
                self.upsert_chunks(batch_chunks)
                chunks_written += len(batch_chunks)
            
            # Final state update
            self.state["docs_processed"] = docs_processed
            self.state["chunks_written"] = chunks_written
            self.state["end_time"] = datetime.now().isoformat()
            self.state["status"] = "completed"
            self.save_state()
            
            logger.info(f"Rebuild completed: {docs_processed} docs, {chunks_written} chunks")
            
            # Generate report
            self.generate_report(docs_processed, chunks_written)
            
        except Exception as e:
            logger.error(f"Rebuild failed: {e}")
            self.state["status"] = "failed"
            self.state["error"] = str(e)
            self.state["end_time"] = datetime.now().isoformat()
            self.save_state()
            raise
    
    def generate_report(self, docs_processed: int, chunks_written: int):
        """Generate rebuild report"""
        report = {
            "rebuild_info": {
                "timestamp": datetime.now().isoformat(),
                "es_url": self.es_url,
                "es_index": self.es_index,
                "chroma_dir": str(self.chroma_dir),
                "collection_name": self.collection_name
            },
            "chunking_params": {
                "chunk_size_chars": self.chunk_size,
                "overlap_chars": self.overlap,
                "batch_docs": self.batch_docs,
                "batch_upsert": self.batch_upsert
            },
            "results": {
                "docs_processed": docs_processed,
                "chunks_written": chunks_written,
                "avg_chunks_per_doc": chunks_written / max(docs_processed, 1)
            },
            "state": self.state
        }
        
        # Save small report
        with open(self.chroma_dir / "rebuild_report_small.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {self.chroma_dir / 'rebuild_report_small.json'}")

def main():
    """Main entry point"""
    try:
        rebuilder = ChromaRebuilder()
        rebuilder.rebuild()
    except KeyboardInterrupt:
        logger.info("Rebuild interrupted by user")
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
