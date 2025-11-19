# app/rag/chains/llm_chain.py
from __future__ import annotations
from typing import List, Optional
import os, re
from openai import OpenAI

from app.core.config import cfg
from app.rag.prompts.prompts_lc import build_prompt
from app.rag.retrievers.hybrid_rrf import load_hybrid_from_env
from app.core.stores.faiss_store import FaissStore
from langchain_core.documents import Document

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _docs_to_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("src") or ""
        parts.append(f"- {src}\n{d.page_content}")
    return "\n\n".join(parts)


class _FaissOnlyRetriever:
    """BM25/RRF 비활성 시 fallback 용 간단 리트리버 (FaissStore 직접 사용)."""

    def __init__(self, faiss_dir: str, top_k: int):
        self.store = FaissStore.load(faiss_dir)
        self.top_k = top_k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        rows = self.store.search(query, top_k=self.top_k)
        docs = []
        for r in rows:
            meta = dict(r)
            text = meta.pop("text", "") or meta.pop("content", "") or ""
            docs.append(Document(page_content=text, metadata=meta))
        return docs


class RAGLightHybrid:
    def __init__(
            self,
            top_k: int | None = None,
            candidate_k: int | None = None,
            enable_bm25: Optional[bool] = None,
            enable_rrf: Optional[bool] = None,
    ):
        self.top_k = top_k or cfg.TOP_K
        self.candidate_k = candidate_k or cfg.CANDIDATE_K
        # 요청별 토글 우선, 없으면 .env 기본
        self.enable_bm25 = cfg.ENABLE_BM25 if enable_bm25 is None else enable_bm25
        self.enable_rrf = cfg.ENABLE_RRF if enable_rrf is None else enable_rrf

        # 리트리버 준비
        if self.enable_bm25 or self.enable_rrf:
            self.retriever = load_hybrid_from_env(top_k=self.top_k, candidate_k=self.candidate_k)
        else:
            self.retriever = _FaissOnlyRetriever(cfg.FAISS_DIR, top_k=self.top_k)

        self.prompt = build_prompt()

    def _pretty_source(self, src: str) -> str:
        # 파일명만 뽑고 확장자/청크 표시 정리
        name = os.path.basename(src.split("::")[0])  # .../파일명.txt::chunk_123 -> 파일명.txt
        name = re.sub(r"\.txt$|\.pdf$|\.md$", "", name)
        # chunk_숫자 있으면 붙여주기
        m = re.search(r"::chunk_(\d+)", src)
        if m:
            return f"{name} (chunk {m.group(1)})"
        return name or "자료"

    def _mask_abs_path(self, s: str) -> str:
        # /Users/... 같은 절대경로 제거
        return re.sub(r"/[^ ]+?/data/indexes/", "data/indexes/", s)

    async def arun(self, question: str):
        # 1. 검색 단계
        docs: List[Document] = self.retriever._get_relevant_documents(question)

        # 2. 컨텍스트 정규화 (clean_text 적용)
        _ZWS_RE = re.compile(r"[\u200B-\u200D\uFEFF\u2060]")
        _BAD_RE = re.compile(r"[■□◆◇●○▪︎▸►•·]{1,}")  # 점/불릿 잔재
        _MULTI_WS = re.compile(r"[ \t]{2,}")

        def clean_text(t: str) -> str:
            t = re.sub(r"[""\"ʼ'´`]+", '"', t)  # 따옴표류 통일
            t = _ZWS_RE.sub("", t)
            t = _BAD_RE.sub(" • ", t)
            t = t.replace("BO0I", "").replace("BOOI", "")  # OCR 잔재
            t = re.sub(r"\s+\n", "\n", t)
            t = _MULTI_WS.sub(" ", t)
            return t.strip()

        # 문서 내용들을 합치되, 각 문단 clean 후 4000자 제한
        context_slices = [d.page_content for d in docs]
        context = "\n\n".join(clean_text(x) for x in context_slices)[:4000]

        # 3. LLM 입력 준비 (prompt 포맷)
        lc_messages = self.prompt.format_messages(question=question, context=context)

        # LangChain 메시지를 OpenAI 포맷으로 변환
        role_map = {"system": "system", "human": "user", "ai": "assistant", "assistant": "assistant", "user": "user"}
        openai_messages = [
            {"role": role_map.get(m.type, "user"), "content": m.content}
            for m in lc_messages
        ]

        # 4. OpenAI API 호출 (실제 LLM)
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=openai_messages,
                temperature=0.2,
                max_tokens=500
            )
            final_text = response.choices[0].message.content.strip()
        except Exception as e:
            final_text = f"죄송해요, 답변 생성 중 오류가 발생했어요. 다시 시도해주세요. (오류: {str(e)})"

        # 5. Citation 정리 (pretty version)
        citations_raw = [d.metadata.get("source") for d in docs if d.metadata.get("source")]
        citations_pretty = []
        for c in citations_raw:
            c2 = self._mask_abs_path(c)
            citations_pretty.append(self._pretty_source(c2))

        # 6. 최종 응답 반환
        return {
            "answer": final_text,
            "citations": citations_pretty,
        }