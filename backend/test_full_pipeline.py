"""Full pipeline diagnostic test - tests each component independently"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

results = []

def log(msg, ok=None):
    status = "PASS" if ok is True else ("FAIL" if ok is False else "INFO")
    line = f"[{status}] {msg}"
    results.append(line)
    print(line)

# Test 1: Basic imports
log("=== Test 1: Module Imports ===")
try:
    from app.services.ingestion.loaders.base import BaseLoader, DocumentResult
    log("Import base.py: OK", True)
except Exception as e:
    log(f"Import base.py: {e}", False)

try:
    from app.services.ingestion.loaders.manager import DocumentLoaderManager
    log("Import manager.py: OK", True)
except Exception as e:
    log(f"Import manager.py: {e}", False)

try:
    from app.services.ingestion.extractor import extract_text, extract_document
    log("Import extractor.py: OK", True)
except Exception as e:
    log(f"Import extractor.py: {e}", False)

try:
    from app.services.ingestion.cleaner import clean_text
    log("Import cleaner.py: OK", True)
except Exception as e:
    log(f"Import cleaner.py: {e}", False)

try:
    from app.services.ingestion.chunker import semantic_chunk
    log("Import chunker.py: OK", True)
except Exception as e:
    log(f"Import chunker.py: {e}", False)

try:
    from app.services.ingestion.embedder import vectorize_chunks
    log("Import embedder.py: OK", True)
except Exception as e:
    log(f"Import embedder.py: {e}", False)

try:
    from app.services.ingestion.indexer import save_chunks
    log("Import indexer.py: OK", True)
except Exception as e:
    log(f"Import indexer.py: {e}", False)

# Test 2: DocumentLoaderManager creation
log("\n=== Test 2: DocumentLoaderManager ===")
try:
    manager = DocumentLoaderManager()
    log("Create manager: OK", True)
    log(f"Supported types: {manager.get_supported_types()}", True)
except Exception as e:
    log(f"Create manager: {e}", False)

# Test 3: Individual loaders
log("\n=== Test 3: Individual Loaders ===")

# PDF Loader
try:
    from app.services.ingestion.loaders.pdf_loader import PDFLoader
    loader = PDFLoader("/nonexistent.pdf")
    log("PDFLoader instantiation: OK", True)
except Exception as e:
    log(f"PDFLoader instantiation: {e}", False)

try:
    from app.services.ingestion.loaders.pdf_loader import PDFLoader
    loader = PDFLoader("/nonexistent.pdf")
    loader.load()
    log("PDFLoader.load nonexistent: should raise", False)
except FileNotFoundError:
    log("PDFLoader.load nonexistent raises FileNotFoundError: OK", True)
except Exception as e:
    log(f"PDFLoader.load nonexistent wrong exception: {type(e).__name__}: {e}", False)

# Markdown Loader
try:
    from app.services.ingestion.loaders.markdown_loader import MarkdownLoader
    loader = MarkdownLoader("/nonexistent.md")
    log("MarkdownLoader instantiation: OK", True)
except Exception as e:
    log(f"MarkdownLoader instantiation: {e}", False)

try:
    from app.services.ingestion.loaders.markdown_loader import MarkdownLoader
    loader = MarkdownLoader("/nonexistent.md")
    loader.load()
    log("MarkdownLoader.load nonexistent: should raise", False)
except FileNotFoundError:
    log("MarkdownLoader.load nonexistent raises FileNotFoundError: OK", True)
except Exception as e:
    log(f"MarkdownLoader.load nonexistent wrong exception: {type(e).__name__}: {e}", False)

# Test 4: Markdown loading with real file
log("\n=== Test 4: Markdown Loading (real file) ===")
try:
    import tempfile
    from app.services.ingestion.loaders.markdown_loader import MarkdownLoader
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write("# Title\n\nSome content.\n\n## Section\n\nMore content.")
        tmp_path = f.name
    
    loader = MarkdownLoader(tmp_path)
    result = loader.load()
    log(f"Markdown content loaded: {len(result.content)} chars", True)
    log(f"Markdown file_type: {result.file_type}", result.file_type == "md")
    log(f"Markdown has sections in structure: {'sections' in result.structure}", True)
    os.unlink(tmp_path)
except Exception as e:
    log(f"Markdown loading failed: {type(e).__name__}: {e}", False)

# Test 5: PDF loading with real file
log("\n=== Test 5: PDF Loading (real file) ===")
try:
    import tempfile
    from app.services.ingestion.loaders.pdf_loader import PDFLoader
    
    # Check if pdfplumber is installed
    import pdfplumber
    log("pdfplumber installed: OK", True)
except ImportError:
    log("pdfplumber NOT installed", False)

# Test 6: Clean text
log("\n=== Test 6: Clean Text ===")
try:
    result = clean_text("Hello\n\n\n\n\nWorld")
    expected = "Hello\n\nWorld"
    log(f"clean_text: {repr(result)}", result == expected)
except Exception as e:
    log(f"clean_text failed: {type(e).__name__}: {e}", False)

# Test 7: Semantic chunk
log("\n=== Test 7: Semantic Chunk ===")
try:
    text = "Paragraph A.\n\nParagraph B.\n\nParagraph C."
    chunks = semantic_chunk(text, max_tokens=1000)
    log(f"semantic_chunk produced {len(chunks)} chunks", len(chunks) == 3)
except Exception as e:
    log(f"semantic_chunk failed: {type(e).__name__}: {e}", False)

# Test 8: Embedding
log("\n=== Test 8: Embedding ===")
try:
    from app.services.embedding import embed_text, EMBEDDING_DIM
    result = embed_text("test text")
    log(f"embed_text returned {len(result)} bytes, expected {EMBEDDING_DIM * 4}", len(result) == EMBEDDING_DIM * 4)
except Exception as e:
    log(f"embed_text failed: {type(e).__name__}: {e}", False)

# Test 9: vectorize_chunks
log("\n=== Test 9: Vectorize Chunks ===")
try:
    chunks = ["chunk one", "chunk two"]
    embeddings = vectorize_chunks(chunks)
    log(f"vectorize_chunks returned {len(embeddings)} embeddings", len(embeddings) == 2)
except Exception as e:
    log(f"vectorize_chunks failed: {type(e).__name__}: {e}", False)

# Test 10: Full extraction pipeline (no DB)
log("\n=== Test 10: Full Extraction (extract_text) ===")
try:
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write("# Test Document\n\nThis is test content for extraction.")
        tmp_path = f.name
    
    from app.services.ingestion.extractor import extract_text
    text = extract_text(tmp_path)
    log(f"extract_text: {len(text)} chars", True)
    os.unlink(tmp_path)
except Exception as e:
    log(f"extract_text failed: {type(e).__name__}: {e}", False)

# Test 11: Full extraction with metadata
log("\n=== Test 11: Full Extraction (extract_document) ===")
try:
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write("# Test\n\nContent here.")
        tmp_path = f.name
    
    from app.services.ingestion.extractor import extract_document
    doc = extract_document(tmp_path)
    log(f"extract_document keys: {list(doc.keys())}", "content" in doc and "metadata" in doc)
    os.unlink(tmp_path)
except Exception as e:
    log(f"extract_document failed: {type(e).__name__}: {e}", False)

# Test 12: Database + full pipeline
log("\n=== Test 12: Database Init ===")
try:
    from app.database import init_db, get_engine, reset_engine
    from app.config import BASE_DIR
    import tempfile as tf
    
    test_db = tf.NamedTemporaryFile(suffix='.db', delete=False)
    test_db.close()
    reset_engine()
    init_db(test_db.name)
    log(f"Database init: OK", True)
    
    from app.database import get_session_local
    session = get_session_local()()
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write("# Test Doc\n\nThis is content for the full pipeline test.")
        tmp_path = f.name
    
    from app.services.ingestion.pipeline import ingest_document
    doc = ingest_document(session, tmp_path, "test.md", "md")
    log(f"Full pipeline doc status: {doc.status}", doc.status == "ready")
    log(f"Full pipeline chunk_count: {doc.chunk_count}", doc.chunk_count > 0)
    
    # Verify chunks in DB
    from app.models.chunk import Chunk
    chunks = session.query(Chunk).filter(Chunk.document_id == doc.id).all()
    log(f"Chunks in DB: {len(chunks)}", len(chunks) > 0)
    
    os.unlink(tmp_path)
    session.close()
    reset_engine()
    os.unlink(test_db.name)
except Exception as e:
    import traceback
    log(f"Full pipeline failed: {type(e).__name__}: {e}", False)
    log(f"Traceback: {traceback.format_exc()}", False)

# Summary
log("\n=== SUMMARY ===")
failures = [r for r in results if r.startswith("[FAIL]")]
passes = [r for r in results if r.startswith("[PASS]")]
log(f"Passed: {len(passes)}, Failed: {len(failures)}")
if failures:
    log("\nFailed tests:")
    for f in failures:
        log(f"  {f}")
