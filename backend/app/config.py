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

# 扫描件 OCR 兜底（Tesseract）：未安装时保持显式报错行为
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_TESSERACT_CMD = os.getenv("OCR_TESSERACT_CMD", "")  # 空=自动定位（PATH→默认安装路径）
OCR_LANG = os.getenv("OCR_LANG", "chi_sim+eng")         # 中文简体+英文（需安装对应语言包）

# 角色权限配置：文件名含以下关键词的文档自动标记为经理专属
MANAGER_KEYWORDS = ["经理"]

# 多跳检索循环配置（依赖驱动，单次请求内迭代检索）
HOP_MAX_HOPS = int(os.getenv("HOP_MAX_HOPS", "3"))            # 总检索跳数上限（首跳+最多2跳）
HOP_TOP_K_PER_HOP = int(os.getenv("HOP_TOP_K_PER_HOP", "5"))   # 每跳混合检索候选数
HOP_TOP_K_CONTEXT = int(os.getenv("HOP_TOP_K_CONTEXT", "6"))   # 统一重排后入上下文的chunk数
VERIFY_MISSING_STREAK = int(os.getenv("VERIFY_MISSING_STREAK", "2"))  # 连续相同缺失方面即硬停的轮数

# 证据不足时的原文页回读（page_tools 接入 Agent 链）
PAGE_READ_ENABLED = os.getenv("PAGE_READ_ENABLED", "true").lower() == "true"
PAGE_READ_MAX_PAGES = int(os.getenv("PAGE_READ_MAX_PAGES", "2"))     # 最多回读页数
PAGE_READ_MAX_CHARS = int(os.getenv("PAGE_READ_MAX_CHARS", "1500"))  # 每页截断长度

# 长期记忆（LangGraph SqliteStore，跨重启持久化，独立于 knowledge.db）
LTM_DB_PATH = DATA_DIR / "long_term_memory.db"
MEMORY_MAX_ITEMS = int(os.getenv("MEMORY_MAX_ITEMS", "30"))  # 条目上限，超了按 updated_at 删最旧


def init_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
