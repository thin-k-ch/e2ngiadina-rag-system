#!/usr/bin/env python3
"""
WINDSURF CHROMA VALIDATION
Read-only validation of Chroma rebuild from Elasticsearch
"""

import os
import sys
import json
import random
import hashlib
from typing import List, Dict, Any
from pathlib import Path

import requests
import chromadb
from chromadb.config import Settings

# Configuration
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_files_v1")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/media/felix/RAG/1/volumes/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_files_v1_chunks")

class ChromaValidator:
    def __init__(self):
        self.es_url = ES_URL
        self.es_index = ES_INDEX
        self.chroma_dir = Path(CHROMA_PERSIST_DIR)
        self.collection_name = COLLECTION_NAME
        
        # Initialize Chroma
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(allow_reset=False)
        )
        
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except Exception as e:
            print(f"ERROR: Cannot access collection {self.collection_name}: {e}")
            sys.exit(1)
    
    def get_es_count(self) -> int:
        """Get total document count from Elasticsearch"""
        try:
            response = requests.get(f"{self.es_url}/{self.es_index}/_count")
            response.raise_for_status()
            return response.json()["count"]
        except Exception as e:
            print(f"ERROR: Cannot get ES count: {e}")
            return 0
    
    def get_chroma_distinct_es_ids(self) -> set:
        """Get distinct ES IDs from Chroma metadata"""
        try:
            result = self.collection.get(
                include=["metadatas"]
            )
            es_ids = set()
            for metadata in result["metadatas"]:
                if metadata and "es_id" in metadata:
                    es_ids.add(metadata["es_id"])
            return es_ids
        except Exception as e:
            print(f"ERROR: Cannot get Chroma ES IDs: {e}")
            return set()
    
    def validate_small_suite(self) -> Dict[str, Any]:
        """Run small suite validation"""
        print("=== SMALL SUITE VALIDATION ===")
        
        results = {}
        
        # a) ES count vs Chroma distinct es_id count
        print("a) Checking document coverage...")
        es_count = self.get_es_count()
        chroma_es_ids = self.get_chroma_distinct_es_ids()
        chroma_count = len(chroma_es_ids)
        
        coverage = chroma_count / max(es_count, 1) * 100
        results["document_coverage"] = {
            "es_count": es_count,
            "chroma_distinct_es_ids": chroma_count,
            "coverage_percent": coverage
        }
        
        print(f"   ES docs: {es_count}")
        print(f"   Chroma distinct ES IDs: {chroma_count}")
        print(f"   Coverage: {coverage:.2f}%")
        
        # b) Random sample validation
        print("b) Validating random sample...")
        sample_size = min(20, chroma_count)
        sample_es_ids = random.sample(list(chroma_es_ids), sample_size)
        
        valid_chunks = 0
        for es_id in sample_es_ids:
            chunks = self.collection.get(
                where={"es_id": es_id},
                include=["documents", "metadatas"]
            )
            
            if chunks["documents"] and any(doc.strip() for doc in chunks["documents"]):
                valid_chunks += 1
        
        results["sample_validation"] = {
            "sample_size": sample_size,
            "valid_docs": valid_chunks,
            "valid_percent": valid_chunks / sample_size * 100
        }
        
        print(f"   Sample size: {sample_size}")
        print(f"   Valid docs: {valid_chunks}")
        print(f"   Valid percent: {valid_chunks / sample_size * 100:.2f}%")
        
        # c) Query sanity check
        print("c) Query sanity check...")
        query = "Projektleitung Konzepthase"
        
        try:
            chroma_results = self.collection.query(
                query_texts=[query],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            # Check if expected file is in results
            expected_file = "Sockelkosten Konzeptphase.xlsx"
            found_expected = False
            
            for metadata_list in chroma_results["metadatas"]:
                for metadata in metadata_list:
                    if metadata and metadata.get("filename") == expected_file:
                        found_expected = True
                        break
                if found_expected:
                    break
            
            results["query_sanity"] = {
                "query": query,
                "expected_file": expected_file,
                "found_expected": found_expected,
                "results_count": len(chroma_results["documents"][0]) if chroma_results["documents"] else 0
            }
            
            print(f"   Query: '{query}'")
            print(f"   Expected file: {expected_file}")
            print(f"   Found expected: {found_expected}")
            print(f"   Results count: {len(chroma_results['documents'][0]) if chroma_results['documents'] else 0}")
            
        except Exception as e:
            results["query_sanity"] = {"error": str(e)}
            print(f"   ERROR: {e}")
        
        return results
    
    def validate_release_train(self) -> Dict[str, Any]:
        """Run release train validation"""
        print("=== RELEASE TRAIN VALIDATION ===")
        
        results = {}
        
        # a) Coverage analysis
        print("a) Detailed coverage analysis...")
        
        # Get ES docs with content
        es_response = requests.post(
            f"{self.es_url}/{self.es_index}/_search",
            json={
                "size": 0,
                "query": {
                    "bool": {
                        "must": [
                            {"exists": {"field": "content"}},
                            {"script": {"script": {"source": "doc['content'].value.length() > 0"}}}
                        ]
                    }
                }
            }
        )
        es_response.raise_for_status()
        es_content_docs = es_response.json()["hits"]["total"]["value"]
        
        # Get Chroma stats
        chroma_es_ids = self.get_chroma_distinct_es_ids()
        
        # Get chunk statistics
        all_chunks = self.collection.get(include=["metadatas"])
        chunk_lengths = [len(doc) for doc in all_chunks["documents"] if doc]
        
        results["coverage_analysis"] = {
            "es_total_docs": self.get_es_count(),
            "es_content_docs": es_content_docs,
            "chroma_distinct_es_ids": len(chroma_es_ids),
            "content_coverage": len(chroma_es_ids) / max(es_content_docs, 1) * 100
        }
        
        print(f"   ES total docs: {results['coverage_analysis']['es_total_docs']}")
        print(f"   ES content docs: {es_content_docs}")
        print(f"   Chroma distinct ES IDs: {len(chroma_es_ids)}")
        print(f"   Content coverage: {results['coverage_analysis']['content_coverage']:.2f}%")
        
        # b) Chunk statistics
        print("b) Chunk statistics...")
        if chunk_lengths:
            import statistics
            results["chunk_stats"] = {
                "total_chunks": len(chunk_lengths),
                "avg_length": statistics.mean(chunk_lengths),
                "median_length": statistics.median(chunk_lengths),
                "min_length": min(chunk_lengths),
                "max_length": max(chunk_lengths)
            }
            
            print(f"   Total chunks: {len(chunk_lengths)}")
            print(f"   Avg length: {results['chunk_stats']['avg_length']:.1f}")
            print(f"   Median length: {results['chunk_stats']['median_length']}")
            print(f"   Min length: {results['chunk_stats']['min_length']}")
            print(f"   Max length: {results['chunk_stats']['max_length']}")
        
        # c) MIME/Extension breakdown
        print("c) MIME/Extension breakdown...")
        mime_counts = {}
        ext_counts = {}
        
        for metadata in all_chunks["metadatas"]:
            if metadata:
                mime = metadata.get("content_type", "unknown")
                ext = metadata.get("extension", "unknown")
                mime_counts[mime] = mime_counts.get(mime, 0) + 1
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
        results["mime_breakdown"] = dict(sorted(mime_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        results["extension_breakdown"] = dict(sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        print("   Top MIME types:")
        for mime, count in list(results["mime_breakdown"].items())[:5]:
            print(f"     {mime}: {count}")
        
        print("   Top extensions:")
        for ext, count in list(results["extension_breakdown"].items())[:5]:
            print(f"     {ext}: {count}")
        
        # d) Retrieval spot-check
        print("d) Retrieval spot-check...")
        test_queries = [
            "Projektleitung Konzepthase",
            "Tabelle1",
            "SBB TFK",
            "Engineering Konzeptphase",
            "Gotthard Basistunnel"
        ]
        
        spot_check_results = []
        for query in test_queries:
            try:
                chroma_results = self.collection.query(
                    query_texts=[query],
                    n_results=3,
                    include=["documents", "metadatas"]
                )
                
                result = {
                    "query": query,
                    "results_count": len(chroma_results["documents"][0]) if chroma_results["documents"] else 0,
                    "top_sources": []
                }
                
                for metadata_list in chroma_results["metadatas"]:
                    for metadata in metadata_list:
                        if metadata and metadata.get("source"):
                            result["top_sources"].append(metadata["source"])
                
                spot_check_results.append(result)
                
            except Exception as e:
                spot_check_results.append({"query": query, "error": str(e)})
        
        results["retrieval_spot_check"] = spot_check_results
        
        print("   Spot-check results:")
        for result in spot_check_results:
            if "error" not in result:
                print(f"     '{result['query']}': {result['results_count']} results")
            else:
                print(f"     '{result['query']}': ERROR - {result['error']}")
        
        return results
    
    def run_validation(self, suite: str = "small"):
        """Run validation suite"""
        print(f"Running {suite} validation suite...")
        
        if suite == "small":
            results = self.validate_small_suite()
            output_file = self.chroma_dir / "validation_report_small.json"
        elif suite == "release":
            results = {
                "small_suite": self.validate_small_suite(),
                "release_train": self.validate_release_train()
            }
            output_file = self.chroma_dir / "validation_report_release.json"
        else:
            print(f"Unknown suite: {suite}")
            return
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nReport saved to {output_file}")
        
        # Summary
        if suite == "small":
            coverage = results["document_coverage"]["coverage_percent"]
            valid_percent = results["sample_validation"]["valid_percent"]
            found_expected = results["query_sanity"].get("found_expected", False)
            
            print(f"\n=== SUMMARY ===")
            print(f"Document coverage: {coverage:.2f}%")
            print(f"Sample validation: {valid_percent:.2f}%")
            print(f"Query sanity: {'PASS' if found_expected else 'FAIL'}")
            
            if coverage > 95 and valid_percent > 95 and found_expected:
                print("✅ SMALL SUITE: PASSED")
            else:
                print("❌ SMALL SUITE: FAILED")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Chroma rebuild")
    parser.add_argument("--suite", choices=["small", "release"], default="small",
                       help="Validation suite to run")
    
    args = parser.parse_args()
    
    try:
        validator = ChromaValidator()
        validator.run_validation(args.suite)
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
