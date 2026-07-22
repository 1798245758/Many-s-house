from app.services.ingestion.loaders.base import DocumentResult, BaseLoader
from app.services.ingestion.loaders.manager import DocumentLoaderManager
from app.services.ingestion.loaders.pdf_loader import PDFLoader

# 注册PDF加载器
DocumentLoaderManager.register("pdf", PDFLoader)

__all__ = ["DocumentResult", "BaseLoader", "DocumentLoaderManager", "PDFLoader"]
