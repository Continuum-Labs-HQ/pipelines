"""
title: Academic RAG Filter
author: Assistant
date: 2024-11-14
version: 1.0
license: MIT
description: Filter for academic paper search and context injection
requirements: pymilvus, requests, pydantic, cachetools
"""

import os
import time
from typing import List, Union, Generator, Iterator, Dict, Any, Optional
from pydantic import BaseModel, Field
import requests
from pymilvus import connections, Collection, utility
import logging
from cachetools import TTLCache
from datetime import datetime

class Pipeline:
    """Filter pipeline for academic paper search and context injection"""
    
    class Valves(BaseModel):
        """Configuration parameters for the filter"""
        # Required filter configuration
        pipelines: List[str] = Field(default=["*"], description="Target pipelines")
        priority: int = Field(default=0, description="Filter priority")
        
        # Milvus configuration
        milvus_host: str = Field(
            default=os.getenv("MILVUS_HOST", "localhost"),  # Replace hardcoded IP with localhost or remove fallback
,            description="Milvus server host"
        )
        milvus_port: str = Field(
            default=os.getenv("MILVUS_PORT", "19530"),
            description="Milvus server port"
        )
        milvus_user: str = Field(
            default=os.getenv("MILVUS_USER", "thannon"),
            description="Milvus username"
        )
        milvus_password: str = Field(
            default=os.getenv("MILVUS_PASSWORD", "chaeBio7!!!"),
            description="Milvus password"
        )
        milvus_collection: str = Field(
            default=os.getenv("MILVUS_COLLECTION", "arxiv_documents"),
            description="Milvus collection name"
        )
        
        # Embedding configuration
        embedding_endpoint: str = Field(
            default=os.getenv("EMBEDDING_ENDPOINT", "http://192.168.13.50:30000/v1/embeddings"),
            description="Embedding service endpoint"
        )
        embedding_model: str = Field(
            default=os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-mistral-7b-v2"),
            description="Embedding model name"
        )
        
        # Search configuration
        top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
        score_threshold: float = Field(default=2.0, description="Maximum distance threshold")
        
        # Rate limiting
        requests_per_minute: int = Field(default=60, description="Maximum requests per minute")
        cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

        class Config:
            """Pydantic configuration"""
            json_schema_extra = {
                "title": "Academic RAG Filter Configuration",
                "description": "Settings for academic paper search and retrieval"
            }

    def __init__(self):
        """Initialize the filter pipeline"""
        self.type = "filter"
        self.name = "Academic RAG Filter"
        self.valves = self.Valves()
        self._collection = None
        self._request_times = []
        self._cache = TTLCache(maxsize=1000, ttl=self.valves.cache_ttl)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.name)
        self.logger.info(f"Initialized {self.name}")

    def _check_rate_limit(self) -> bool:
        """Check if within rate limit"""
        now = time.time()
        minute_ago = now - 60
        self._request_times = [t for t in self._request_times if t > minute_ago]
        if len(self._request_times) >= self.valves.requests_per_minute:
            return False
        self._request_times.append(now)
        return True

    async def on_startup(self):
        """Server startup hook"""
        try:
            # Initialize Milvus connection
            connections.connect(
                alias="default",
                host=self.valves.milvus_host,
                port=self.valves.milvus_port,
                user=self.valves.milvus_user,
                password=self.valves.milvus_password
            )
            self._collection = Collection(self.valves.milvus_collection)
            self._collection.load()
            self.logger.info(f"Started {self.name}")
        except Exception as e:
            self.logger.error(f"Startup error: {str(e)}", exc_info=True)
            raise

    async def on_shutdown(self):
        """Server shutdown hook"""
        try:
            if self._collection:
                self._collection.release()
            connections.disconnect("default")
            self.logger.info(f"Shut down {self.name}")
        except Exception as e:
            self.logger.error(f"Shutdown error: {str(e)}", exc_info=True)

    def search_papers(self, query: str) -> Dict[str, Any]:
        """Search for relevant papers using the query"""
        # Check cache first
        cache_key = f"search:{query}"
        if cache_key in self._cache:
            self.logger.info("Returning cached results")
            return self._cache[cache_key]

        # Check rate limit
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded")
            return {"success": False, "error": "Rate limit exceeded"}

        try:
            # Get embedding
            response = requests.post(
                self.valves.embedding_endpoint,
                headers={"Content-Type": "application/json"},
                json={
                    "input": [query],
                    "model": self.valves.embedding_model,
                    "input_type": "query",
                },
                timeout=10
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]

            # Search Milvus
            results = self._collection.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 16}},
                limit=self.valves.top_k,
                output_fields=["source_file", "abstract", "key_points"]
            )

            # Process results
            papers = []
            for hits in results:
                for hit in hits:
                    if hit.distance <= self.valves.score_threshold:
                        papers.append({
                            "source": getattr(hit.entity, "source_file", "Unknown"),
                            "abstract": getattr(hit.entity, "abstract", "No abstract"),
                            "key_points": getattr(hit.entity, "key_points", "No key points"),
                            "score": float(hit.distance),
                            "timestamp": datetime.now().isoformat()
                        })

            result = {"success": True, "papers": papers}
            self._cache[cache_key] = result
            return result

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {str(e)}", exc_info=True)
            return {"success": False, "error": "Embedding service unavailable"}
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def inlet(self, body: Dict, user: Optional[Dict] = None) -> Dict:
        """Process incoming messages"""
        try:
            # Skip if not a proper message body
            if not isinstance(body, dict) or "messages" not in body:
                return body

            # Get the last user message
            last_message = body["messages"][-1]["content"]
            
            # Search for relevant papers
            result = self.search_papers(last_message)
            
            if not result["success"]:
                self.logger.error(f"Search failed: {result.get('error')}")
                return body

            papers = result["papers"]
            if not papers:
                return body

            # Format context
            context = "\n\n".join([
                f"Document {i+1}:\n"
                f"Source: {paper['source']}\n"
                f"Abstract: {paper['abstract']}\n"
                f"Key Points: {paper['key_points']}\n"
                f"Relevance: {paper['score']:.2f}\n"
                f"Retrieved: {paper['timestamp']}\n"
                "---"
                for i, paper in enumerate(papers)
            ])

            # Add context to system message
            system_msg = {
                "role": "system",
                "content": (
                    "You are a research assistant. Use these academic papers as context "
                    f"for your response:\n\n{context}"
                )
            }
            
            body["messages"].insert(0, system_msg)
            return body

        except Exception as e:
            self.logger.error(f"Inlet error: {str(e)}", exc_info=True)
            return body

    async def outlet(self, response: str, user: Optional[Dict] = None) -> str:
        """Process outgoing messages"""
        return response
