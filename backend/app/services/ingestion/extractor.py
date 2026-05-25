from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import json


def _extract_xmind(file_path: Path) -> str:
    parts = []
    with zipfile.ZipFile(str(file_path), "r") as z:
        if "content.json" in z.namelist():
            try:
                data = json.loads(z.read("content.json").decode("utf-8"))
                for sheet in data:
                    root = sheet.get("rootTopic", {})
                    _walk_json_topic(root, parts, 0)
            except Exception:
                pass
        elif "content.xml" in z.namelist():
            try:
                xml_content = z.read("content.xml")
                root = ET.fromstring(xml_content)
                ns = {"xmap": "urn:xmind:xmap:xmlns:content:2.0"}
                for sheet in root.findall(".//xmap:sheet", ns) or root.findall(".//sheet"):
                    topic = sheet.find("xmap:topic", ns) or sheet.find("topic")
                    if topic is not None:
                        _walk_xml_topic(topic, parts, 0, ns)
            except Exception:
                pass
    return "\n".join(parts) if parts else "[XMind 文件为空或无法解析]"


def _walk_json_topic(topic, parts, depth):
    prefix = "  " * depth
    title = topic.get("title", "")
    if title:
        parts.append(f"{prefix}- {title}")
    for child in topic.get("children", {}).get("attached", []):
        _walk_json_topic(child, parts, depth + 1)


def _walk_xml_topic(topic, parts, depth, ns):
    prefix = "  " * depth
    title = topic.get("title", "")
    if title:
        parts.append(f"{prefix}- {title}")
    for child in topic.findall("xmap:children", ns) or topic.findall("children"):
        for t in child.findall("xmap:topics", ns) or child.findall("topics"):
            for subtopic in t.findall("xmap:topic", ns) or t.findall("topic"):
                _walk_xml_topic(subtopic, parts, depth + 1, ns)


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif suffix == ".md":
        return file_path.read_text(encoding="utf-8")
    elif suffix == ".xmind":
        return _extract_xmind(file_path)
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
