import json
import zipfile
import pytest
from pathlib import Path
from app.services.ingestion.loaders.xmind_loader import XmindLoader


class TestXmindLoader:
    """XMind加载器测试"""

    def test_load_xmind_json_format(self, tmp_path):
        """测试加载JSON格式的XMind文件"""
        content_data = [
            {
                "rootTopic": {
                    "title": "Central Topic",
                    "children": {
                        "attached": [
                            {"title": "Sub Topic 1"},
                            {
                                "title": "Sub Topic 2",
                                "children": {
                                    "attached": [
                                        {"title": "Child 2.1"},
                                    ]
                                },
                            },
                        ]
                    },
                }
            }
        ]
        xmind_file = tmp_path / "test.xmind"
        with zipfile.ZipFile(str(xmind_file), "w") as zf:
            zf.writestr("content.json", json.dumps(content_data))

        loader = XmindLoader(str(xmind_file))
        result = loader.load()

        assert result.file_type == "xmind"
        assert "Central Topic" in result.content
        assert "Sub Topic 1" in result.content
        assert "Child 2.1" in result.content
        assert len(result.structure["topics"]) == 4

    def test_load_xmind_xml_format(self, tmp_path):
        """测试加载XML格式的XMind文件"""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<xmap:xmap xmlns:xmap="urn:xmind:xmap:xmlns:content:2.0">
  <sheet>
    <topic title="Root">
      <children>
        <topics>
          <topic title="Child A"/>
          <topic title="Child B"/>
        </topics>
      </children>
    </topic>
  </sheet>
</xmap:xmap>"""
        xmind_file = tmp_path / "test.xml.xmind"
        with zipfile.ZipFile(str(xmind_file), "w") as zf:
            zf.writestr("content.xml", xml_content)

        loader = XmindLoader(str(xmind_file))
        result = loader.load()

        assert result.file_type == "xmind"
        assert "Root" in result.content
        assert "Child A" in result.content
        assert len(result.structure["topics"]) == 3

    def test_load_empty_xmind(self, tmp_path):
        """测试加载空XMind文件"""
        content_data = [{"rootTopic": {"title": ""}}]
        xmind_file = tmp_path / "empty.xmind"
        with zipfile.ZipFile(str(xmind_file), "w") as zf:
            zf.writestr("content.json", json.dumps(content_data))

        loader = XmindLoader(str(xmind_file))
        result = loader.load()

        assert result.file_type == "xmind"
        assert result.content == "[XMind 文件为空或无法解析]"

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            loader = XmindLoader("/nonexistent/file.xmind")
            loader.load()

    def test_load_xmind_metadata(self, tmp_path):
        """测试XMind文件元数据"""
        content_data = [{"rootTopic": {"title": "Topic"}}]
        xmind_file = tmp_path / "meta.xmind"
        with zipfile.ZipFile(str(xmind_file), "w") as zf:
            zf.writestr("content.json", json.dumps(content_data))

        loader = XmindLoader(str(xmind_file))
        result = loader.load()

        assert result.source == str(xmind_file)
        assert result.metadata["file_name"] == "meta.xmind"
        assert result.metadata["file_type"] == ".xmind"
