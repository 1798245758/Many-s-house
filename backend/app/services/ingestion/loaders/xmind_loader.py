import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List
from app.services.ingestion.loaders.base import BaseLoader, DocumentResult


class XmindLoader(BaseLoader):
    """XMind文件加载器"""

    def load(self) -> DocumentResult:
        """加载XMind文档"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        metadata = self._get_file_metadata()
        content_parts = []
        structure: Dict[str, Any] = {"topics": []}

        with zipfile.ZipFile(str(self.file_path), "r") as z:
            if "content.json" in z.namelist():
                try:
                    data = json.loads(z.read("content.json").decode("utf-8"))
                    for sheet in data:
                        root = sheet.get("rootTopic", {})
                        self._walk_json_topic(root, content_parts, 0, structure["topics"])
                except Exception:
                    pass
            elif "content.xml" in z.namelist():
                try:
                    xml_content = z.read("content.xml")
                    root = ET.fromstring(xml_content)
                    ns = {"xmap": "urn:xmind:xmap:xmlns:content:2.0"}
                    sheets = root.findall(".//xmap:sheet", ns) or root.findall(".//sheet")
                    for sheet in sheets:
                        topic = sheet.find("xmap:topic", ns) or sheet.find("topic")
                        if topic is not None:
                            self._walk_xml_topic(topic, content_parts, 0, ns, structure["topics"])
                except Exception:
                    pass

        content = "\n".join(content_parts) if content_parts else "[XMind 文件为空或无法解析]"

        return DocumentResult(
            content=content,
            metadata=metadata,
            structure=structure,
            source=str(self.file_path),
            file_type="xmind",
        )

    def _walk_json_topic(self, topic: Dict, parts: List[str], depth: int, topics: List[Dict]):
        """遍历JSON格式的主题"""
        prefix = "  " * depth
        title = topic.get("title", "")
        if title:
            parts.append(f"{prefix}- {title}")
            topics.append({"title": title, "level": depth})
        for child in topic.get("children", {}).get("attached", []):
            self._walk_json_topic(child, parts, depth + 1, topics)

    def _walk_xml_topic(self, topic, parts: List[str], depth: int, ns: Dict, topics: List[Dict]):
        """遍历XML格式的主题"""
        prefix = "  " * depth
        title = topic.get("title", "")
        if title:
            parts.append(f"{prefix}- {title}")
            topics.append({"title": title, "level": depth})
        for child in topic.findall("xmap:children", ns) or topic.findall("children"):
            for t in child.findall("xmap:topics", ns) or child.findall("topics"):
                for subtopic in t.findall("xmap:topic", ns) or t.findall("topic"):
                    self._walk_xml_topic(subtopic, parts, depth + 1, ns, topics)
