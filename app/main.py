from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.api.documents import router as documents_router, search_router
from app.api.chat import router as chat_router

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise RAG application API",
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
)

# Register CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(documents_router, prefix="/documents")
app.include_router(search_router)
app.include_router(chat_router, prefix="/chat")




@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "NexusAI",
    }
