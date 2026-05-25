from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.database import Base, get_engine
from app.config import init_dirs
from app.routers import documents, query, history, profile

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_dirs()
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="RAG 知识库", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"code": "SUCCESS", "message": "RAG 知识库服务运行中"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"code": "SERVER_ERROR", "message": "服务器内部错误"},
    )

app.include_router(documents.router)
app.include_router(query.router)
app.include_router(history.router)
app.include_router(profile.router)
