#!/usr/bin/env python3
import os, sys, json, argparse, csv, math
from pathlib import Path
from typing import List, Dict, Tuple

# 프로젝트 루트 import 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.stores.faiss_store import FaissStore
from app.core.stores.bm25_store import BM25Store
from app.rag.retrievers.rrf import rrf_fuse
from langchain_core.documents import Document

# -------------------- 공용 유틸 --------------------
def load_qa(fp: Path) -> List[Dict]:
    rows = []
    for ln in fp.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln: continue
        obj = json.loads(ln)
        # 두 포맷 지원: {"q","gold_keywords"} 또는 {"question","doc_keywords"}
        q = obj.get("q") or obj.get("question")
        gold = obj.get("gold_keywords") or obj.get("doc_keywords")
        if isinstance(q, str) and isinstance(gold, list) and gold:
            rows.append({"q": q, "gold": gold})
    return rows

def keyword_hit(text: str, gold: List[str]) -> bool:
    t = (text or "").lower()
    return any(kw.lower() in t for kw in gold)

def metrics(hits: List[List[bool]]) -> Tuple[float, float, float]:
    n = len(hits) or 1
    recall = sum(any(h) for h in hits) / n
    mrr = 0.0
    ndcg = 0.0
    for hs in hits:
        # MRR
        rank = next((i + 1 for i, v in enumerate(hs) if v), None)
        if rank: mrr += 1.0 / rank
        # nDCG (단정답 가정)
        dcg = sum((1.0 / math.log2(i + 2)) for i, v in enumerate(hs) if v)
        idcg = 1.0
        ndcg += dcg / idcg
    return recall, mrr / n, ndcg / n

def to_doc(row: Dict) -> Document:
    meta = dict(row)
    text = meta.pop("text", "") or meta.pop("content", "") or meta.pop("chunk", "")
    return Document(page_content=text, metadata=meta)

def search_faiss(fa: FaissStore, q: str, k: int) -> List[Dict]:
    return fa.search(q, top_k=k)

def search_multi_bm25(bm25s: List[BM25Store], q: str, k: int, rrf_k: int) -> List[Dict]:
    bags = []
    for st in bm25s:
        rows = st.search(q, top_k=k)
        bags.append([to_doc(r) for r in rows])
    if not bags:
        return []
    fused_docs = rrf_fuse(bags, k=rrf_k, top_k=k)
    out = []
    for d in fused_docs:
        m = dict(d.metadata)
        m["text"] = d.page_content
        out.append(m)
    return out

def rrf_weighted(fa_rows: List[Dict], bm_rows: List[Dict], k: int, rrf_k: int, w_f: float, w_b: float) -> List[Dict]:
    def key_of(r: Dict):
        return r.get("doc_id") or (r.get("source"), r.get("chunk_id"))
    scores, seen, order = {}, {}, 0
    for lst, w in ((fa_rows, w_f), (bm_rows, w_b)):
        for i, r in enumerate(lst):
            key = key_of(r)
            if key not in scores:
                scores[key] = 0.0
                seen[key] = (order, r)
                order += 1
            scores[key] += w * (1.0 / (rrf_k + (i + 1)))
    top = sorted(scores.items(), key=lambda x: (-x[1], seen[x[0]][0]))[:k]
    return [seen[k][1] for k, _ in top]

# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser(description="Grid search for RRF hyperparameters")
    ap.add_argument("--qa", required=True, help="eval/*.jsonl")
    ap.add_argument("--k", default="5", help="comma sep (e.g., 3,5)")
    ap.add_argument("--rrf-grid", default="30,60,120", help="RRF k candidates")
    ap.add_argument("--w-faiss", default="0.5,0.6,0.7,0.8,0.9", help="weights for FAISS")
    ap.add_argument("--w-bm25", default="", help="weights for BM25 (optional, else complement)")
    ap.add_argument("--csv", default=None, help="write results to CSV")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    qa = load_qa(Path(args.qa))
    if not qa:
        print(f"[ERR] No QA loaded from {args.qa}")
        sys.exit(1)

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    rrf_ks = [int(x) for x in args.rrf_grid.split(",") if x.strip()]
    w_f_list = [float(x) for x in args.w_faiss.split(",") if x.strip()]
    if args.w_bm25.strip():
        w_b_list = [float(x) for x in args.w_bm25.split(",") if x.strip()]
    else:
        # 보완 가중치(1 - w_faiss)
        w_b_list = [round(1.0 - w, 3) for w in w_f_list]

    # stores
    faiss_dir = os.getenv("FAISS_DIR", "./data/indexes/merged/faiss")
    fa = FaissStore.load(faiss_dir)

    bm25_dirs = []
    if os.getenv("BM25_DIRS"):
        bm25_dirs = [d.strip() for d in os.getenv("BM25_DIRS").split(",") if d.strip()]
    elif os.getenv("BM25_DIRS_FILE"):
        bm25_dirs = [ln.strip() for ln in Path(os.getenv("BM25_DIRS_FILE")).read_text(encoding="utf-8").splitlines() if ln.strip()]
    bm25s = [BM25Store.load(d) for d in bm25_dirs]

    rows = []
    best = None  # (metric tuple, row)
    for k in ks:
        for rrf_k in rrf_ks:
            # 베이스: 개별 검색 결과
            for w_f, w_b in zip(w_f_list, w_b_list):
                per_hits = []
                for item in qa:
                    q, gold = item["q"], item["gold"]
                    fa_rows = search_faiss(fa, q, k)
                    bm_rows = search_multi_bm25(bm25s, q, k, rrf_k) if bm25s else []
                    fused = rrf_weighted(fa_rows, bm_rows, k=k, rrf_k=rrf_k, w_f=w_f, w_b=w_b)
                    flags = [keyword_hit((d.get("text","") + " " + str(d.get("source",""))), gold) for d in fused]
                    per_hits.append(flags)

                    if args.debug:
                        print(f"\nQ: {q}  | k={k}  rrf_k={rrf_k}  w_f={w_f:.2f}/w_b={w_b:.2f}")
                        for i, d in enumerate(fused, 1):
                            s = d.get("source") or d.get("src") or ""
                            t = (d.get("text") or "")[:80].replace("\n"," ")
                            print(f"  {i:>2}. {'✔' if flags[i-1] else '✘'} {s} :: {t}")

                recall, mrr, ndcg = metrics(per_hits)
                row = {
                    "k": k, "rrf_k": rrf_k, "w_faiss": w_f, "w_bm25": w_b,
                    "Recall@k": round(recall, 6), "MRR@k": round(mrr, 6), "nDCG@k": round(ndcg, 6),
                }
                rows.append(row)

                key = (recall, mrr, ndcg)
                if best is None or key > best[0]:
                    best = (key, row)

                print(f"[RRFw] k={k} rrf_k={rrf_k} wF={w_f:.2f}/wB={w_b:.2f} "
                      f"=> Recall@k={recall:.3f}  MRR@k={mrr:.3f}  nDCG@k={ndcg:.3f}")

    # 요약
    print("\n=== Best (Recall@k, tie=M RR@k) ===")
    print(best[1])

    # CSV 저장
    if args.csv:
        outp = Path(args.csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] wrote {outp}")

if __name__ == "__main__":
    main()