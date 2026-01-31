#!/usr/bin/env python3
"""
Streaming Backend Example - Real-time progress updates (v0.3.10)

Demonstrates how to stream REAL progress updates from orchestrator to frontend using Server-Sent Events (SSE).

NEW in v0.3.10: Uses progress_callback parameter for genuine real-time progress!
Frontend receives actual progress as enable-ai processes the query:
- "Understanding your question..." (when parsing)
- "Finding the right API..." (when matching)
- "Calling /inventory/..." (when executing)
- "Preparing your results..." (when summarizing)
- "Done! ✓" (when complete)
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enable_ai import APIOrchestrator, ResponseFormatter
from enable_ai.progress_tracker import ProgressTracker, ProgressStage, create_progress_message

app = FastAPI(title="Enable AI - Streaming API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    format_type: str = "auto"
    api_token: Optional[str] = None


# Initialize components (load schemas from config.json)
orchestrator = APIOrchestrator()
formatter = ResponseFormatter()


@app.post("/query")
async def query_api(request: QueryRequest):
    """
    Standard query endpoint (non-streaming).
    Returns final result only.
    """
    try:
        # Process query
        result = orchestrator.process(
            request.query,
            access_token=request.api_token
        )
        
        # Format result
        formatted = formatter.format_response(
            data=result.get('data'),
            query=request.query,
            format_type=request.format_type
        )
        
        return {
            "success": True,
            "summary": formatted['summary'],
            "formatted": formatted['formatted'],
            "format": formatted['format'],
            "data": formatted['raw_data']
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/query/stream")
async def query_api_stream(request: QueryRequest):
    """
    Streaming query endpoint using Server-Sent Events (SSE).
    
    Streams progress updates:
    - "Finding intent..."
    - "Calling user API..."
    - "Calling order API..."
    - "Summarizing results..."
    - Final result
    
    Frontend receives real-time updates!
    """
    
    async def event_generator():
        """Generate Server-Sent Events with progress updates (v0.3.10)."""
        
        try:
            # NEW in v0.3.10: Use progress_callback parameter!
            # The orchestrator will call this for each progress update
            progress_queue = asyncio.Queue()
            
            def progress_callback(stage, message, progress, metadata=None):
                """Callback receives real-time progress from orchestrator."""
                update = {
                    "stage": stage,
                    "message": message,
                    "progress": progress,
                    "metadata": metadata or {}
                }
                # Put update in queue (thread-safe)
                asyncio.run_coroutine_threadsafe(
                    progress_queue.put(update),
                    asyncio.get_event_loop()
                )
            
            # Start processing in background task
            async def process_query():
                """Process query and send result to queue when done."""
                try:
                    result = orchestrator.process(
                        request.query,
                        access_token=request.api_token,
                        progress_callback=progress_callback  # NEW! Real-time progress
                    )
                    await progress_queue.put({"type": "result", "data": result})
                except Exception as e:
                    await progress_queue.put({"type": "error", "error": str(e)})
            
            # Start background task
            task = asyncio.create_task(process_query())
            
            # Stream progress updates as they arrive
            while True:
                try:
                    update = await asyncio.wait_for(progress_queue.get(), timeout=30.0)
                    
                    if update.get("type") == "result":
                        # Process complete, format and send result
                        result = update["data"]
                        formatted = formatter.format_response(
                            data=result.get('data'),
                            query=request.query,
                            format_type=request.format_type
                        )
                        
                        final_result = {
                            "success": True,
                            "summary": formatted['summary'],
                            "formatted": formatted['formatted'],
                            "format": formatted['format'],
                            "data": formatted['raw_data']
                        }
                        yield format_sse_message("result", final_result)
                        break
                    
                    elif update.get("type") == "error":
                        yield format_sse_message("error", {"error": update["error"]})
                        break
                    
                    else:
                        # Progress update from orchestrator
                        yield format_sse_message("progress", update)
                
                except asyncio.TimeoutError:
                    yield format_sse_message("error", {"error": "Request timeout"})
                    break
            
            # Wait for background task to complete
            await task
            
        except Exception as e:
            # Send error
            error_data = {
                "success": False,
                "error": str(e)
            }
            yield format_sse_message("error", error_data)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


def format_sse_message(event: str, data: dict) -> str:
    """Format message for Server-Sent Events."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Enable AI - Streaming API",
        "version": "0.2.0",
        "endpoints": {
            "/query": "Standard query (non-streaming)",
            "/query/stream": "Streaming query with progress updates (SSE)",
        },
        "features": [
            "Real-time progress updates",
            "Server-Sent Events (SSE)",
            "Smart response formatting",
            "Multi-step API execution"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  Enable AI - Streaming Backend with Progress Updates     ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print("")
    print("📡 Endpoints:")
    print("   POST /query          - Standard query")
    print("   POST /query/stream   - Streaming query (SSE)")
    print("")
    print("🌐 Server starting on http://localhost:8000")
    print("")
    print("📖 API Docs: http://localhost:8000/docs")
    print("")
    print("Press Ctrl+C to stop")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
