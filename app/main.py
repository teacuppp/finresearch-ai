from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.services.rag_service import (
    create_application_services,
)
from app.api.rag import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = create_application_services()

    app.state.rag_pipeline = services.rag_pipeline
    app.state.document_service = services.document_service

    yield

    app.state.rag_pipeline = None
    app.state.document_service = None

app = FastAPI(
    title="FinResearch AI",
    description="Agentic RAG platform for financial research",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "project": "FinResearch AI",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
    }


app.include_router(documents_router)
app.include_router(rag_router)


#    uvicorn app.main:app --reload