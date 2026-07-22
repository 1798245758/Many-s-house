# LangChain DocumentLoader 优化设计文档

> 项目：RAG 知识库  
> 状态：设计阶段（已确认）  
> 日期：2026-07-22

---

## 1. 概述

优化现有上传文件功能，使用LangChain的DocumentLoader来提取多源数据，支持更多文件类型，提升文档处理能力和扩展性。

### 目标

- 使用LangChain DocumentLoader替换现有的自定义extractor
- 支持更多文件类型（PDF、TXT、MD、XMind、Word、Excel、PowerPoint）
- 提取内容、元数据和结构信息
- 保持向后兼容性（重新设计数据结构，清除历史数据）
- 采用容错模式，单个文件处理失败不影响其他文件

### 技术栈

| 层级 | 技术 |
|------|------|
| 文档处理 | LangChain DocumentLoader |
| 文件类型支持 | PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader |
| 自定义Loader | XmindLoader（保持现有实现） |

---

## 2. 架构设计

### 2.1 整体架构

```
用户上传文件
  → 保存到 uploads/
  → LangChain DocumentLoader 处理
    - 根据文件类型选择对应的Loader
    - 提取内容、元数据、结构信息
  → 文档清洗（cleaner）
  → 语义切分（chunker）
  → 向量化（embedder）
  → 索引构建（indexer）
  → 更新文档状态
```

### 2.2 组件设计

#### DocumentLoaderManager

负责管理所有DocumentLoader，根据文件类型选择合适的Loader。

```python
class DocumentLoaderManager:
    def __init__(self):
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.xmind': XmindLoader,
            '.docx': Docx2txtLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.pptx': UnstructuredPowerPointLoader,
        }
    
    def get_loader(self, file_type: str):
        return self.loaders.get(file_type)
    
    def load_document(self, file_path: Path) -> DocumentResult:
        loader = self.get_loader(file_path.suffix.lower())
        if not loader:
            raise ValueError(f"不支持的文件类型: {file_path.suffix}")
        return loader(str(file_path)).load()
```

#### DocumentResult

统一的文档处理结果格式。

```python
@dataclass
class DocumentResult:
    content: str          # 文本内容
    metadata: dict        # 元数据（作者、创建时间等）
    structure: dict       # 结构信息（标题、章节等）
    source: str           # 文件来源
    file_type: str        # 文件类型
```

---

## 3. 支持的文件类型

| 文件类型 | LangChain DocumentLoader | 说明 |
|---------|-------------------------|------|
| PDF | PyPDFLoader | 支持文本提取，保留页面信息 |
| TXT | TextLoader | 直接读取，支持编码检测 |
| MD | UnstructuredMarkdownLoader | 保留Markdown结构信息 |
| XMind | 自定义XmindLoader | 保持现有实现，支持JSON和XML格式 |
| Word | Docx2txtLoader | 支持.docx格式，提取文本和表格 |
| Excel | UnstructuredExcelLoader | 支持.xlsx格式，提取表格数据 |
| PowerPoint | UnstructuredPowerPointLoader | 支持.pptx格式，提取幻灯片内容 |

---

## 4. 数据结构

### 4.1 文档模型更新

```python
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)
    chunk_count = Column(Integer, default=0)
    status = Column(String, default="pending")  # pending/processing/ready/error
    metadata_json = Column(Text)  # 存储元数据JSON
    structure_json = Column(Text)  # 存储结构信息JSON
    created_at = Column(String, server_default=func.datetime('now'))
```

### 4.2 Chunk模型更新

```python
class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer)
    embedding = Column(BLOB)
    metadata_json = Column(Text)  # 存储chunk级别的元数据
    created_at = Column(String, server_default=func.datetime('now'))
```

---

## 5. 错误处理策略

### 5.1 容错模式

- 单个文件处理失败不影响其他文件处理
- 记录详细的错误信息到数据库
- 提供错误恢复机制

### 5.2 错误分类

```python
class ErrorType:
    UNSUPPORTED_FILE = "unsupported_file"  # 不支持的文件类型
    FILE_CORRUPTED = "file_corrupted"      # 文件损坏
    EXTRACTION_FAILED = "extraction_failed"  # 提取失败
    PROCESSING_FAILED = "processing_failed"  # 处理失败
```

### 5.3 错误处理流程

```python
def process_file_with_error_handling(file_path: Path) -> DocumentResult:
    try:
        return document_loader_manager.load_document(file_path)
    except ValueError as e:
        raise DocumentError(ErrorType.UNSUPPORTED_FILE, str(e))
    except Exception as e:
        raise DocumentError(ErrorType.EXTRACTION_FAILED, str(e))
```

---

## 6. 实现计划

### 阶段一：依赖安装和基础框架

1. 安装LangChain及相关依赖
2. 创建DocumentLoaderManager基础框架
3. 定义DocumentResult数据结构

### 阶段二：实现各个DocumentLoader

1. 实现PyPDFLoader封装
2. 实现TextLoader封装
3. 实现UnstructuredMarkdownLoader封装
4. 实现XmindLoader（保持现有实现）
5. 实现Docx2txtLoader封装
6. 实现UnstructuredExcelLoader封装
7. 实现UnstructuredPowerPointLoader封装

### 阶段三：集成到现有系统

1. 更新extractor.py使用新的DocumentLoaderManager
2. 更新数据模型支持元数据和结构信息
3. 更新pipeline.py处理流程

### 阶段四：测试和验证

1. 为每个DocumentLoader编写单元测试
2. 测试完整的上传处理流程
3. 验证错误处理机制

---

## 7. 依赖管理

### 新增依赖

```txt
langchain>=0.1.0
langchain-community>=0.0.1
pypdf>=3.17.0
unstructured>=0.10.0
docx2txt>=0.8.0
openpyxl>=3.1.2
python-pptx>=0.6.21
```

### 更新后的requirements.txt

```txt
fastapi==0.115.6
uvicorn==0.34.0
sqlalchemy==2.0.36
pydantic==2.13.4
python-multipart==0.0.19
pdfplumber==0.11.4
httpx==0.28.1
python-dotenv==1.0.1
langchain>=0.1.0
langchain-community>=0.0.1
pypdf>=3.17.0
unstructured>=0.10.0
docx2txt>=0.8.0
openpyxl>=3.1.2
python-pptx>=0.6.21
```

---

## 8. 测试策略

### 单元测试

为每个DocumentLoader编写单元测试：

1. **PyPDFLoader测试**
   - 测试PDF文件内容提取
   - 测试元数据提取
   - 测试错误处理

2. **TextLoader测试**
   - 测试TXT文件读取
   - 测试编码检测
   - 测试大文件处理

3. **UnstructuredMarkdownLoader测试**
   - 测试Markdown结构保留
   - 测试元数据提取

4. **XmindLoader测试**
   - 测试JSON格式解析
   - 测试XML格式解析
   - 测试空文件处理

5. **Docx2txtLoader测试**
   - 测试Word文档内容提取
   - 测试表格提取

6. **UnstructuredExcelLoader测试**
   - 测试Excel数据提取
   - 多工作表处理

7. **UnstructuredPowerPointLoader测试**
   - 测试幻灯片内容提取
   - 测试备注提取

### 集成测试

测试完整的上传处理流程：

1. 文件上传 → 解析 → 切片 → 索引全流程
2. 多文件并发处理
3. 错误恢复机制

### 测试覆盖率目标

- 单元测试覆盖率 ≥ 85%
- 集成测试覆盖率 ≥ 80%

---

## 9. 部署注意事项

### 数据库迁移

由于重新设计数据结构，需要：
1. 备份现有数据（如需要）
2. 删除旧的数据库文件
3. 重新创建数据库结构

### 依赖安装

需要安装新的依赖包，可能需要系统级别的依赖（如unstructured需要的一些库）。

### 性能考虑

- 大文件处理可能需要较长时间
- 建议添加进度反馈机制
- 考虑添加文件大小限制

---

## 附录 A：关键决策

| 决策项 | 选择 |
|--------|------|
| 实现方法 | 完全替换现有extractor |
| 文件类型支持 | PDF/TXT/MD/XMind/Word/Excel/PowerPoint |
| 错误处理 | 容错模式，单个文件失败不影响其他文件 |
| 元数据提取 | 内容 + 元数据 + 结构信息 |
| 向后兼容 | 重新设计数据结构，清除历史数据 |
| 测试策略 | 单元测试为主 |
