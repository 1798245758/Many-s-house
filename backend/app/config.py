from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = DATA_DIR / "knowledge.db"

# BGE模型配置
USE_BGE_MODEL = os.getenv("USE_BGE_MODEL", "true").lower() == "true"
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-base-zh-v1.5")
BGE_MODEL_TIMEOUT = int(os.getenv("BGE_MODEL_TIMEOUT", "120"))  # 超时时间（秒），首次下载需要更长时间


def init_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
