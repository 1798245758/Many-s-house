from pathlib import Path

def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif suffix == ".md":
        return file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        import pdfplumber
        text_parts = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")
