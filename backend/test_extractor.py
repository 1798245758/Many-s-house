import sys
sys.path.insert(0, '.')
from app.services.ingestion.extractor import extract_text, extract_document, get_supported_types
print('Extractor updated successfully')
print('Supported types:', get_supported_types())
