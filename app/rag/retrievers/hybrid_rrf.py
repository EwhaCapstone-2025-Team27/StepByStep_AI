# app/rag/retrievers/hybrid_rrf.py
from __future__ import annotations
import os
from typing import List, Sequence, Optional
from pydantic import Field
try:
    # pydantic v2
    from pydantic import ConfigDict
    _MODEL_CONFIG = {"model_config": ConfigDict(arbitrary_types_allowed=True, extra="allow")}
except Exception:
    # pydantic v1 fallback
    _MODEL_CONFIG = {"Config": type("Config", (), {"arbitrary_types_allowed": True, "extra": "allow"})}

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from app.core.stores.faiss_store import FaissStore
from app.rag.retrievers.multi_bm25_adapter import MultiBM25Retriever, load_bm25_retrievers_from_env
from app.rag.retrievers.rrf import rrf_fuse
from app.rag.retrievers.utils import dict_to_doc


class HybridRRF(BaseRetriever):
    """FAISS(벡터) + BM25(키워드) 결과를 RRF로 융합하는 리트리버"""

    # ---- Pydantic 모델 필드 선언 (중요) ----
    faiss: FaissStore
    bm25: MultiBM25Retriever

    # 파라미터들
    top_k: int = Field(default=4)
    candidate_k: int = Field(default=6)
    rrf_k: int = Field(default=60)

    # 점수 정규화 여부(필요 시 확장)
    use_cosine: bool = Field(default=False)

    # pydantic 설정 (v2 / v1 호환)
    locals().update(_MODEL_CONFIG)

    # BaseRetriever 인터페이스
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1) FAISS 후보
        faiss_rows = self.faiss.search(query, top_k=self.candidate_k, normalize=self.use_cosine)
        faiss_docs = [dict_to_doc(r) for r in faiss_rows]

        # 2) BM25 후보 (이미 Document로 반환됨)
        bm25_docs = self.bm25._get_relevant_documents(query)

        # 3) RRF 융합
        fused = rrf_fuse([faiss_docs, bm25_docs], k=self.rrf_k, top_k=self.top_k)
        return fused

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def _read_dirs_from_file(path: str) -> List[str]:
    out: List[str] = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def load_hybrid_from_env(
    *,
    faiss_dir: Optional[str] = None,
    bm25_dirs: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
    candidate_k: Optional[int] = None,
) -> HybridRRF:
    """
    ENV 사용:
      - FAISS_DIR=./data/indexes/merged/faiss
      - BM25_DIRS=dir1,dir2,...  또는  BM25_DIRS_FILE=.bm25_dirs.txt
      - TOP_K / CANDIDATE_K / RRF_K
    """
    # FAISS
    faiss_dir = faiss_dir or os.getenv("FAISS_DIR")
    if not faiss_dir:
        raise ValueError("FAISS_DIR 환경변수가 비어 있습니다.")
    faiss_store = FaissStore.load(faiss_dir)

    # BM25
    if bm25_dirs is None:
        env_dirs = (os.getenv("BM25_DIRS") or "").strip()
        if env_dirs:
            dirs = [d.strip() for d in env_dirs.split(",") if d.strip()]
        else:
            # 파일 경유
            bm25_file = os.getenv("BM25_DIRS_FILE")
            dirs = _read_dirs_from_file(bm25_file) if bm25_file else []
        if not dirs:
            raise ValueError("BM25_DIRS / BM25_DIRS_FILE 둘 다 비어 있습니다.")
    else:
        dirs = list(bm25_dirs)

    final_candidate_k = candidate_k if candidate_k is not None else int(os.getenv("CANDIDATE_K", "6"))

    bm25_retriever = MultiBM25Retriever(dirs=dirs, candidate_k=int(os.getenv("CANDIDATE_K", "6")))

    return HybridRRF(
        faiss=faiss_store,
        bm25=bm25_retriever,
        top_k=top_k if top_k is not None else int(os.getenv("TOP_K", "4")),
        candidate_k=final_candidate_k,
        rrf_k=int(os.getenv("RRF_K", "60")),
        use_cosine=False,
    )