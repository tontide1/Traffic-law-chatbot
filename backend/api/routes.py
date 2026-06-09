from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from backend.api.schemas import ChatRequest, ChatResponse, ComparisonResponse, UploadResponse
from fastapi.responses import StreamingResponse
import json
import asyncio
from lightrag import LightRAG, QueryParam
from backend.core.rag_engine import RAGEngine
from backend.core.document_parser import parse_pdf_to_markdown
import shutil
import os
from backend.config import settings
from backend.core.indexing_provider_store import IndexingProviderStore

router = APIRouter()

SUPPORTED_INDEXING_PROVIDERS = {"deepseek", "google_studio"}

def resolve_indexing_provider(provider: str | None) -> str:
    value = (provider or "deepseek").strip().lower()
    if value not in SUPPORTED_INDEXING_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported indexing provider: {provider}")
    return value


from typing import Union

@router.post("/chat")
async def chat(request: ChatRequest):
    rag = RAGEngine.get_query_instance()
    print(f"DEBUG: Chat request received. message='{request.message[:20]}...', comparison_mode={request.comparison_mode}, stream={request.stream}")
    
    system_prompt = (
        "STRICT INSTRUCTION: Output ONLY the relevant information. "
        "DO NOT use introductory phrases like 'Dựa trên thông tin được cung cấp...', 'Dưới đây là...', etc. "
        "Directly provide the answer based on the context."
    )
    
    full_query = f"{request.message}\n\n{system_prompt}"
    
    if not request.stream:
        try:
            if request.comparison_mode:
                naive_response = await rag.aquery(full_query, param=QueryParam(mode="naive"))
                hybrid_response = await rag.aquery(full_query, param=QueryParam(mode="hybrid"))
                return ComparisonResponse(
                    naive=ChatResponse(response=naive_response, mode="naive"),
                    hybrid=ChatResponse(response=hybrid_response, mode="hybrid")
                )
            else:
                response = await rag.aquery(full_query, param=QueryParam(mode="hybrid"))
                return ChatResponse(response=response, mode="hybrid")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Streaming Implementation
    async def event_generator():
        queue = asyncio.Queue()
        pending_tasks = set()

        async def stream_wrapper(gen_func, mode):
            try:
                await queue.put(f"data: {json.dumps({'type': 'start', 'mode': mode})}\n\n")
                generator = await gen_func
                if hasattr(generator, '__aiter__'):
                    async for chunk in generator:
                        await queue.put(f"data: {json.dumps({'type': 'chunk', 'mode': mode, 'content': chunk})}\n\n")
                else:
                    await queue.put(f"data: {json.dumps({'type': 'chunk', 'mode': mode, 'content': str(generator)})}\n\n")
            except Exception as e:
                print(f"STREAM ERROR ({mode}): {str(e)}")
                await queue.put(f"data: {json.dumps({'type': 'error', 'mode': mode, 'message': str(e)})}\n\n")

        try:
            if request.comparison_mode:
                # Start both in parallel
                t1 = asyncio.create_task(stream_wrapper(rag.aquery(full_query, param=QueryParam(mode="naive", stream=True)), "naive"))
                t2 = asyncio.create_task(stream_wrapper(rag.aquery(full_query, param=QueryParam(mode="hybrid", stream=True)), "hybrid"))
                pending_tasks.update([t1, t2])
                
                while pending_tasks:
                    # Wait for items in queue or for tasks to finish
                    while not queue.empty():
                        yield await queue.get()
                    
                    done, pending_tasks = await asyncio.wait(pending_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                    
                    # Yield any new items added during wait
                    while not queue.empty():
                        yield await queue.get()
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            else:
                # Standard single stream
                generator = await rag.aquery(full_query, param=QueryParam(mode="hybrid", stream=True))
                if hasattr(generator, '__aiter__'):
                    async for chunk in generator:
                        yield f"data: {json.dumps({'type': 'chunk', 'mode': 'hybrid', 'content': chunk})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'chunk', 'mode': 'hybrid', 'content': str(generator)})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/documents")
async def list_documents():
    rag = RAGEngine.get_query_instance()
    provider_store = IndexingProviderStore(settings.LIGHTRAG_WORKING_DIR)
    try:
        docs_tuple, _ = await rag.doc_status.get_docs_paginated()
        all_providers = await provider_store.get_all_providers()
        result = []
        for doc_id, status_obj in docs_tuple:
            status_str = "unknown"
            if hasattr(status_obj.status, "value"):
                status_str = status_obj.status.value
            elif isinstance(status_obj.status, str):
                status_str = status_obj.status

            source = status_obj.file_path or "unknown"
            result.append(
                {
                    "id": doc_id,
                    "status": status_str,
                    "source": source,
                    "content_summary": (status_obj.content_summary[:100] + "...") if status_obj.content_summary else "",
                    "indexed_provider": all_providers.get(source, "legacy"),
                }
            )
        return result
    except Exception as e:
        print(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), provider: str = Form("deepseek")):
    resolved_provider = resolve_indexing_provider(provider)

    if not file.filename.endswith(".pdf") and not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    file_path = os.path.join(settings.LIGHTRAG_WORKING_DIR, file.filename)
    os.makedirs(settings.LIGHTRAG_WORKING_DIR, exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        rag = await RAGEngine.get_indexing_instance(resolved_provider)
        provider_store = IndexingProviderStore(settings.LIGHTRAG_WORKING_DIR)

        if file.filename.endswith(".pdf"):
            content = await parse_pdf_to_markdown(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        if not content.strip():
            raise ValueError("File is empty or no text could be extracted")

        await rag.ainsert(content, file_paths=[file.filename])
        try:
            await provider_store.set_provider(file.filename, resolved_provider)
        except Exception as e:
            print(f"WARNING: Failed to save document provider metadata for {file.filename}: {e}")

        return UploadResponse(
            filename=file.filename,
            status="success",
            message=f"File uploaded and indexed with {resolved_provider}",
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error indexing uploaded file {file.filename}: {e}")
        return UploadResponse(
            filename=file.filename,
            status="error",
            message=f"Failed to index file: {str(e)}",
        )

@router.get("/health")
async def health():
    return {"status": "healthy"}
