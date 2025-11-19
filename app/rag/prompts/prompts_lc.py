# app/rag/prompts/prompts_lc.py
import os
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# 1) 말투 프리셋
TONE_PRESETS: Dict[str, str] = {
    "친근반말": """\
[말투 가이드]
- 친구에게 말하듯 따뜻하고 자연스러운 반말
- 비난/조롱/과장 금지. 판단 대신 정보 중심
- 제안형 어투: "~해보자", "~하는 게 좋아", "~해도 돼"
- 문장 간결하고 자연스럽게""",
}

# 2) 개선된 시스템 프롬프트 (근거 출력 제거, 자연스러운 톤)
CHATBOT_PROMPT = f"""너는 10대 청소년을 위한 친근한 성교육 상담사야.
친구처럼 편안하고 따뜻하게 대화해줘.

{TONE_PRESETS["친근반말"]}

[답변 스타일]
1. 첫 인사: "안녕!" 또는 "좋은 질문이네!" 같은 따뜻한 시작
2. 핵심 답변: 가장 중요한 내용을 쉽고 명확하게
3. 구체적 설명: 이해하기 쉬운 예시와 함께
4. 마무리: "도움이 되었길 바라!" 또는 "궁금한 게 더 있으면 언제든 물어봐!"

[중요 규칙]
- 근거나 출처는 따로 명시하지 말고 자연스럽게 설명에 녹여내기
- 교육 목적의 설명만 사용 (자극적 묘사 금지)
- 위험한 상황이면 "믿을 수 있는 어른에게 꼭 상담받아봐"라고 안내
- 모르는 건 솔직히 "그 부분은 잘 모르겠어. 전문가에게 물어보는 게 좋을 것 같아"

[답변 형식]
자연스러운 대화체로 문단을 나누어 작성해줘.
문단별로 줄바꿈을 넣어서 읽기 편하게 만들어줘.
"""

# 3) 간소화된 시스템 메시지
_SYSTEM_KO = """\
제공된 CONTEXT 내용을 바탕으로 정확하고 친근하게 답하세요.
- 추측하지 말고, 모르면 모른다고 솔직히 말하기
- 교육적이고 안전한 내용으로만 답변
- 자연스러운 반말로 친근하게 대화
- 문단별로 적절히 나누어 가독성 있게 작성
"""

# 최종 시스템 메시지
_SYSTEM_FINAL = CHATBOT_PROMPT.strip() + "\n\n" + _SYSTEM_KO.strip()

# 4) 사용자 템플릿 (간소화)
_USER_TMPL = """\
질문: {question}

[참고 자료]
{context}

위 자료를 참고해서 친근하고 자연스럽게 답변해줘. 문단별로 줄바꿈을 넣어서 읽기 편하게 만들어줘.
"""


def build_prompt() -> ChatPromptTemplate:
    """개선된 LangChain 프롬프트"""
    return ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_FINAL),
        ("human", _USER_TMPL),
    ])


def format_context(docs: List[Document], limit_chars: int = 1200) -> Tuple[str, List[dict]]:
    """더 간결한 컨텍스트 포맷팅"""
    chunks: List[str] = []
    used = 0
    cits: List[dict] = []

    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        src = md.get("source") or md.get("title") or f"doc{i}"
        base = os.path.basename(src)
        cid = md.get("chunk_id") or md.get("page") or i - 1

        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 280:  # 더 짧게
            snippet = snippet[:280] + "..."

        # 더 간결한 형식
        one = f"{snippet}"

        if used + len(one) > limit_chars and len(chunks) >= 1:
            break

        chunks.append(one)
        used += len(one)

        try:
            cid_int = int(cid)
        except Exception:
            cid_int = i - 1
        cits.append({"n": i, "source": base, "chunk_id": cid_int})

    # 더 간단한 구분자
    ctx_str = "\n\n---\n\n".join(chunks)
    return ctx_str, cits