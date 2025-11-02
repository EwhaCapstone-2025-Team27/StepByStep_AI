from typing import Optional
from fastapi import FastAPI, Body
from pydantic import BaseModel
import re, os

from app.core.config import cfg
from app.rag.chains.llm_chain import RAGLightHybrid
from app.rag.retrievers.hybrid_rrf import load_hybrid_from_env

app = FastAPI(title="StepByStep RAG & QUIZ (Light-Hybrid)", version="1.0.0")
hy = load_hybrid_from_env()

# 앱 기동 시 기본 체인 준비
default_chain = RAGLightHybrid()


class ChatIn(BaseModel):
    message: str
    userId: Optional[str] = None
    top_k: Optional[int] = None
    enable_bm25: Optional[bool] = None
    enable_rrf: Optional[bool] = None
    # (선택) corpus 이름으로 스위칭하고 싶다면 여기에 추가 가능:
    # corpus: Optional[str] = None


class ChatOut(BaseModel):
    answer: str
    citations: list[str] = []

def _pretty_source(src: str) -> str:
    name = os.path.basename((src or "").split("::")[0])
    name = re.sub(r"\.txt$|\.pdf$|\.md$", "", name)
    m = re.search(r"::chunk_(\d+)", src or "")
    return f"{name} (chunk {m.group(1)})" if m else (name or "자료")

def _mask_abs_path(s: str) -> str:
    return re.sub(r"/[^ ]+?/data/indexes/", "data/indexes/", s or "")

async def _search_impl(q: str, k: int):
    try:
        retr = load_hybrid_from_env(top_k=k, candidate_k=cfg.CANDIDATE_K)
        docs = retr._get_relevant_documents(q)
        items = []
        for d in docs:
            src = d.metadata.get("source") or "unknown_source"
            items.append({
                "source": _mask_abs_path(src),
                "label": _pretty_source(src),
                "chunk_id": d.metadata.get("chunk_id"),
                "score": d.metadata.get("score"),   # 있으면 표시
                "snippet": (d.page_content or "").replace("\n", " ")[:200],
            })
        return {"status":"ok", "q": q, "k": k, "items": items}
    except Exception as e:
        return {"status":"error", "message": str(e), "data": None}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "light-hybrid",
        "bm25": cfg.ENABLE_BM25,
        "rrf": cfg.ENABLE_RRF,
        "faiss_dir": cfg.FAISS_DIR,
        "bm25_dir": cfg.BM25_DIR,
        "model": cfg.OPENAI_MODEL,
    }


@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn = Body(...)):
    # 요청별 토글/파라미터 반영(없으면 기본값 사용)
    chain = RAGLightHybrid(
        top_k=body.top_k or cfg.TOP_K,
        candidate_k=cfg.CANDIDATE_K,
        enable_bm25=cfg.ENABLE_BM25 if body.enable_bm25 is None else body.enable_bm25,
        enable_rrf=cfg.ENABLE_RRF if body.enable_rrf is None else body.enable_rrf,
    )
    result = await chain.arun(body.message)
    return ChatOut(answer=result["answer"], citations=result["citations"])

@app.post("/v1/search")
def search_debug(q: str, k: int = 5):
    docs = hy._get_relevant_documents(q)
    out = []
    for i,d in enumerate(docs,1):
        out.append({
            "rank": i,
            "source": d.metadata.get("source"),
            "chunk_id": d.metadata.get("chunk_id"),
            "preview": (d.page_content or "")[:160]
        })
    return {"q": q, "k": k, "results": out}