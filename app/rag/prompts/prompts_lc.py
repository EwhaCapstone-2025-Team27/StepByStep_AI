# LangChain 기반 RAG용 프롬프트 빌더LLM에 넘길 “질문+문맥”을 구성
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

_SYSTEM_KO = """\
당신은 성교육 전문 어시스턴트입니다. 아래 CONTEXT 범위 내에서만 한국어로 정확하고 친절하게 답하세요.
- 근거 없는 추측 금지, 모르면 모른다고 말하고 추가 정보 요청
- 민감 주제는 안전·건강 관점으로 설명, 교육 목적의 비묘사적 설명만 사용(자극적 상세 묘사 금지)
- 답변 형식: 1) 핵심 요점 1~2문장, 2) 쉬운 설명/예시 불릿 3~6개, 3) 안전 팁(필요 시), 4) 근거(파일명/페이지 또는 chunk_id)
- 출처는 제공된 CONTEXT에서만 인용하고 새 출처를 지어내지 말 것. 여러 자료가 같은 내용을 말하면 근거 표시에 [1,3,5]처럼 묶어 표기
- 문서 원문 특수기호/OCR 노이즈 제거, 자연스러운 한국어로 정리
"""
_SYSTEM_KO += "\n[말투] 친구에게 말하듯 따뜻하고 존중하는 반말을 사용(훈계/과장/조롱 금지, 문장 간결)."

_USER_TMPL = """\
질문: {question}

[CONTEXT]
{context}
"""

def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_KO),
        ("human", _USER_TMPL),
    ])

def format_context(docs: List[Document], max_total_chars: int = 1800) -> tuple[str, list[dict]]:
    """
    CONTEXT 문자열과 citation 매핑을 함께 반환:
    - 문자열: [1] 제목 · chunk_id : 스니펫...
    - 매핑: [{"n":1,"source":..., "chunk_id":...}, ...]
    """
    lines, cits, used = [], [], 0
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        title = md.get("title") or md.get("source") or f"doc{i}"
        chunk_id = md.get("chunk_id")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        row = f"[{i}] {title}"
        if chunk_id is not None:
            row += f" · chunk_{chunk_id}"
        row += f" : {snippet}"
        if used + len(row) > max_total_chars and i > 1:
            break
        lines.append(row)
        cits.append({"n": i, "source": title, "chunk_id": chunk_id})
        used += len(row)
    return "\n".join(lines), cits