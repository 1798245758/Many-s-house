from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.database import init_db
from app.config import init_dirs
from app.routers import documents, query, history, profile, tasks, diagnosis, memory
from app.models.query_trace import QueryTrace  # noqa: F401  确保 init_db 的 create_all 建出检索轨迹表
from app.services.tasks.runner import cleanup_stale_processing
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_dirs()
    # init_db 含存量库轻量迁移（documents 补 visibility / task_id 列）
    init_db()
    # 重启兜底：上次进程被中断的"处理中"文档回置失败，不残留假状态
    cleanup_stale_processing()
    yield

app = FastAPI(title="企业问答助手知识库", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"code": "SUCCESS", "message": "企业问答助手知识库服务运行中"}

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
app.include_router(tasks.router)
app.include_router(diagnosis.router)
app.include_router(memory.router)
