# 성큼성큼 - AI (RAG & Quiz)

StepByStep AI는 FastAPI 기반의 RAG(검색 증강 생성) 서비스와 관련 스크립트를 포함한 프로젝트입니다. 
이 README는 **AI 코드 실행 방법**을 중심으로 정리했습니다.

---

## 📁 프로젝트 구조

```
StepByStep_AI/
├── app/
│   ├── api/
│   │   └── serve_rag.py          # RAG API 서버 (FastAPI)
│   ├── core/                     # 설정, 헬스체크, 스토어 로직
│   ├── rag/                      # 체인, 리트리버, 프롬프트
│   └── main.py                   # (참고용) 기본 FastAPI 엔트리
├── scripts/                      # 인덱스 구축/평가 스크립트
│   ├── build_index.py
│   ├── preprocess_texts.py
│   └── run_dev.sh                # 로컬 개발 서버 실행 스크립트
├── utils/                        # 텍스트 정제 유틸
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧰 필수 준비 사항

- **Python 3.10+**
- **OpenAI API Key** (임베딩/LLM 호출용)
- FAISS/BM25 인덱스 파일 (없다면 아래의 `인덱스 구축` 단계 참고)

---

## ⚙️ 환경 설정

### 1️⃣ 가상환경 생성 및 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ 환경 변수 설정

프로젝트 루트에 `.env` 파일을 만들고, `.env.example`을 참고해 환경값을 입력합니다.

```bash
cp .env.example .env
```

필수 항목 예시:

```
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
```

인덱스 위치도 필요합니다.

```
FAISS_DIR=./data/indexes/merged/faiss
BM25_DIRS_FILE=.bm25_dirs.txt
```

---

## 🧱 인덱스 구축 (필요 시)

텍스트 파일을 기반으로 FAISS/BM25 인덱스를 생성합니다.

```bash
python scripts/build_index.py \
  --input ./data/raw \
  --outdir ./data/indexes
```

- `--input`: 텍스트 파일(.txt) 폴더 경로
- `--outdir`: 생성된 인덱스 저장 위치

여러 인덱스를 합치거나 전처리가 필요한 경우 `scripts/merge_faiss.py`, `scripts/preprocess_texts.py`도 참고하세요.

---

## 🚀 실행 방법

### 1️⃣ 개발 서버 실행 (FastAPI)

```bash
bash scripts/run_dev.sh
```

또는 직접 실행:

```bash
uvicorn app.api.serve_rag:app --host 0.0.0.0 --port 8000 --reload
```

서버가 정상 기동되면 아래에서 상태를 확인할 수 있습니다.

```
GET http://localhost:8000/health
```

---

## 🔍 API 테스트 예시

### 1) RAG 질의

**POST** `/v1/chat`

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "질문",
    "top_k": 5,
    "enable_bm25": true,
    "enable_rrf": true
  }'
```

---

## 📌 참고

- 주요 설정은 `app/core/config.py`에서 불러옵니다.
- RAG API는 `app/api/serve_rag.py` 기준으로 실행됩니다.
- OpenAI API 사용량이 발생할 수 있으므로 키와 비용에 유의하세요.
