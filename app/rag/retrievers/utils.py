# app/rag/retrievers/utils.py
from __future__ import annotations
from langchain_core.documents import Document

from langchain_core.documents import Document

def dict_to_doc(d: dict) -> Document:
    # 원본 dict 복사 (기존 유지)
    meta = dict(d)

    # 텍스트 필드 추출
    text = d.get("text") or d.get("content") or d.get("chunk") or ""
    for k in ("text", "content", "chunk"):
        meta.pop(k, None)

    #source 필드 정규화 (빈 경우 보완)
    if not meta.get("source"):
        src = (
            d.get("src")
            or d.get("source")
            or d.get("source_path")
            or d.get("doc")
            or d.get("file")
            or d.get("path")
        )
        if not src:
            cid = d.get("chunk_id") or d.get("id")
            src = f"unknown_source::chunk_{cid}" if cid is not None else "unknown_source"
        meta["source"] = src

    # 기본 메타 필드 세팅
    meta.setdefault("chunk_id", d.get("chunk_id") or d.get("id"))
    meta.setdefault("doc_id", d.get("id"))

    # 5최종 Document 반환
    return Document(page_content=text, metadata=meta)