from fastapi import APIRouter
import httpx
import os

router = APIRouter()
ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_chunks_v1")

@router.post("/proxy/es")
async def proxy_es_search(request: dict):
    """Proxy für Elasticsearch Anfragen (um CORS zu umgehen)"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{ES_URL}/{ES_INDEX}/_search",
                json=request
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}
