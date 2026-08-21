from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = DATA_DIR / "knowledge.db"

# ChromaDB向量库配置
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "chunks")

# BGE模型配置
USE_BGE_MODEL = os.getenv("USE_BGE_MODEL", "true").lower() == "true"
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-base-zh-v1.5")
BGE_MODEL_TIMEOUT = int(os.getenv("BGE_MODEL_TIMEOUT", "120"))  # 超时时间（秒），首次下载需要更长时间

# 角色权限配置：文件名含以下关键词的文档自动标记为经理专属
MANAGER_KEYWORDS = ["经理"]


def init_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
