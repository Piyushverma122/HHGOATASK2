from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.voice import router as voice_router
from app.api.v1.rag import router as rag_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(voice_router)
api_v1_router.include_router(rag_router)
