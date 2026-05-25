from sqlalchemy.orm import Session
from app.services.deepseek import DeepSeekClient
from app.services.retrieval.searcher import hybrid_search
from app.schemas.query import SourceInfo, QueryResponse

def generate_answer(db: Session, query_text: str, client: DeepSeekClient) -> QueryResponse:
    query_vec = client.embed(query_text)
    chunks = hybrid_search(db, query_vec, query_text, top_k=10)
    if not chunks:
        return QueryResponse(answer="当前知识库中没有相关信息，请先上传文档。", sources=[])
    context = "\n\n".join(c.content for c in chunks)
    answer = client.chat(context, query_text)
    sources = [
        SourceInfo(
            chunk_id=c.id,
            content_snippet=c.content[:200],
            document_name=c.document.filename if c.document else "未知",
        )
        for c in chunks[:3]
    ]
    return QueryResponse(answer=answer, sources=sources)
