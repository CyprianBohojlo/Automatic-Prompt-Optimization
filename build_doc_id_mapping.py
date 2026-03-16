#!/usr/bin/env python3
"""
Build a mapping between dataset document names and Sturdy doc_ids
by sampling text chunks and matching.

Uses exact phrase matching (consecutive-word subsequences) for all datasets
instead of Jaccard word overlap, since financial documents share too much
common vocabulary for bag-of-words approaches to disambiguate them.

Usage:
    export STURDY_API_KEY="your-key"
    python build_doc_id_mapping.py
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

import ijson
import pandas as pd
import requests
from sturdystats.index import Index

STURDY_API_KEY = os.environ["STURDY_API_KEY"]
INDEX_NAME = "bulk_train_all"

DOCFINQA_URL = (
    "https://huggingface.co/datasets/kensho/DocFinQA/resolve/main/train.json"
)
FINDOCRAG_REPO_URL = (
    "https://gitlab-core.supsi.ch/dti-idsia/ai-finance-papers/findoc-rag.git"
)
FINDOCRAG_DOCS_SUBDIR = "FinDoc-RAG_data/documents"
FINANCEBENCH_QUESTIONS_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/"
    "main/data/financebench_open_source.jsonl"
)

CLONE_DIR = Path("./tmp_findocrag_clone")
OUTPUT_CSV = Path("./output/doc_id_mapping.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# =========================================================================
# TEXT SAMPLING & MATCHING
# =========================================================================

def normalize_text(text: str) -> str:
    """Collapse all whitespace (tabs, newlines, multiple spaces) into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def sample_chunks(text: str, n_chunks: int = 5, chunk_size: int = 20) -> list[str]:
    """Sample n_chunks of chunk_size words from text at evenly spaced positions."""
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    step = max(1, (len(words) - chunk_size) // (n_chunks - 1)) if n_chunks > 1 else 0
    for i in range(n_chunks):
        start = min(i * step, len(words) - chunk_size)
        chunks.append(" ".join(words[start:start + chunk_size]))
    return chunks


def chunks_overlap(chunks_a: list[str], chunks_b: list[str]) -> float:
    """Score how many words overlap between two sets of chunks."""
    words_a = set(" ".join(chunks_a).split())
    words_b = set(" ".join(chunks_b).split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def extract_phrases(text: str, n_phrases: int = 10, phrase_len: int = 8) -> list[str]:
    """Extract exact consecutive-word phrases from text at evenly spaced positions.

    Unlike sample_chunks (used for Jaccard), these phrases preserve exact word
    order and are used for substring matching -- much more discriminating for
    documents that share vocabulary (like financial filings).
    """
    normed = normalize_text(text)
    words = normed.split()
    if len(words) < phrase_len:
        return [normed] if words else []

    phrases = []
    step = max(1, (len(words) - phrase_len) // (n_phrases - 1)) if n_phrases > 1 else 0
    for i in range(n_phrases):
        start = min(i * step, len(words) - phrase_len)
        phrases.append(" ".join(words[start:start + phrase_len]))
    return phrases


def phrase_match_score(sturdy_phrases: list[str], dataset_text_normed: str) -> float:
    """Score how many of the Sturdy phrases appear as exact substrings in the dataset text.

    Returns fraction of phrases that matched (0.0 to 1.0).
    """
    if not sturdy_phrases or not dataset_text_normed:
        return 0.0
    hits = sum(1 for p in sturdy_phrases if p in dataset_text_normed)
    return hits / len(sturdy_phrases)


# =========================================================================
# STEP 1: Get all Sturdy doc_ids and sample their text
# =========================================================================

def get_sturdy_samples(idx: Index) -> list[dict]:
    """Get all doc_ids and text samples from the index using query()."""
    print("Step 1: Getting all doc_ids and text from Sturdy index...")

    # Paginate through all docs, 100 at a time
    all_results = []
    offset = 0
    page_size = 100
    while True:
        page = idx.query(
            search_query="",
            semantic_search_cutoff=0.0,
            semantic_search_weight=0.0,
            max_excerpts_per_doc=5,
            limit=page_size,
            offset=offset,
        )
        if len(page) == 0:
            break
        all_results.append(page)
        print(f"  Fetched offset {offset}: {len(page)} rows")
        offset += page_size

    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    print(f"  Got {len(results)} total rows from query")

    # Group text by doc_id
    doc_texts: dict[str, list[str]] = {}
    for _, row in results.iterrows():
        doc_id = row["doc_id"]
        if doc_id not in doc_texts:
            doc_texts[doc_id] = []
        doc_texts[doc_id].append(str(row["text"]))

    print(f"  Found {len(doc_texts)} unique doc_ids")

    # Build samples for each doc_id: both chunks (for Jaccard) and phrases (for exact matching)
    rows = []
    for doc_id, texts in doc_texts.items():
        full_text = " ".join(texts)
        rows.append({
            "sturdy_doc_id": doc_id,
            "chunks": sample_chunks(full_text),
            "phrases": extract_phrases(full_text),
        })

    return rows


# =========================================================================
# STEP 2: Load datasets and sample their text
# =========================================================================

def load_and_sample_datasets() -> list[dict]:
    """Load all three datasets and sample text from each document."""
    all_docs = []

    # --- DocFinQA ---
    try:
        print("\nStep 2a: Loading DocFinQA...")
        seen: OrderedDict = OrderedDict()
        ctx_text: dict[str, str] = {}
        total = 0
        with requests.get(DOCFINQA_URL, stream=True) as resp:
            resp.raise_for_status()
            for idx_val, item in enumerate(ijson.items(resp.raw, "item")):
                total = idx_val + 1
                ctx = item.get("Context", "")
                h = hashlib.sha256(ctx.encode("utf-8")).hexdigest()
                if h not in seen:
                    seen[h] = []
                    ctx_text[h] = ctx
                seen[h].append(idx_val)
                if total % 500 == 0:
                    print(f"    Streamed {total} rows...")

        for doc_index, (h, indices) in enumerate(seen.items()):
            text = ctx_text[h]
            all_docs.append({
                "dataset_name": "DocFinQA",
                "doc_index": doc_index,
                "doc_label": f"DocFinQA_{doc_index}",
                "original_row_indices": ",".join(str(i) for i in indices),
                "context_hash": h,
                "chunks": sample_chunks(text),
                "text_normed": normalize_text(text),
            })
        print(f"    {len(seen)} unique documents")
    except Exception as e:
        print(f"    Skipping DocFinQA: {e}")

    # --- FinDoc-RAG ---
    try:
        print("\nStep 2b: Loading FinDoc-RAG...")
        if CLONE_DIR.exists():
            shutil.rmtree(CLONE_DIR)
        subprocess.run(
            ["git", "clone", "--depth", "1", FINDOCRAG_REPO_URL, str(CLONE_DIR)],
            check=True, timeout=120,
        )
        docs_dir = CLONE_DIR / FINDOCRAG_DOCS_SUBDIR
        md_files = sorted(docs_dir.glob("*.md"))
        for doc_index, md_path in enumerate(md_files):
            text = md_path.read_text(encoding="utf-8")
            all_docs.append({
                "dataset_name": "FinDoc-RAG",
                "doc_index": doc_index,
                "doc_label": f"FinDoc-RAG_{md_path.name}",
                "filename": md_path.name,
                "context_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "chunks": sample_chunks(text),
                "text_normed": normalize_text(text),
            })
        shutil.rmtree(CLONE_DIR, ignore_errors=True)
        print(f"    {len(md_files)} documents")
    except Exception as e:
        print(f"    Skipping FinDoc-RAG: {e}")

    # --- FinanceBench ---
    try:
        print("\nStep 2c: Loading FinanceBench...")
        resp = requests.get(FINANCEBENCH_QUESTIONS_URL)
        resp.raise_for_status()
        questions = [json.loads(l) for l in resp.text.strip().split("\n") if l.strip()]

        doc_pages: dict[str, dict[int, str]] = {}
        doc_qids: dict[str, set] = {}
        for q in questions:
            fb_id = q.get("financebench_id", "")
            for ev in q.get("evidence", []):
                dn = ev.get("doc_name", "")
                pn = ev.get("evidence_page_num", 0)
                fp = ev.get("evidence_text_full_page", "")
                if not dn or not fp:
                    continue
                if dn not in doc_pages:
                    doc_pages[dn] = {}
                    doc_qids[dn] = set()
                doc_qids[dn].add(str(fb_id))
                if pn not in doc_pages[dn]:
                    doc_pages[dn][pn] = fp

        for doc_index, doc_name in enumerate(sorted(doc_pages.keys())):
            pages = doc_pages[doc_name]
            context = "\n\n".join(pages[p] for p in sorted(pages) if pages[p])
            all_docs.append({
                "dataset_name": "FinanceBench",
                "doc_index": doc_index,
                "doc_label": f"FinanceBench_{doc_name}",
                "doc_name": doc_name,
                "financebench_ids": ",".join(sorted(doc_qids[doc_name])),
                "context_hash": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                "chunks": sample_chunks(context),
                "text_normed": normalize_text(context),
            })
        print(f"    {len(doc_pages)} documents")
    except Exception as e:
        print(f"    Skipping FinanceBench: {e}")

    print(f"\n  Total dataset documents: {len(all_docs)}")
    return all_docs


# =========================================================================
# STEP 3: Match and write CSV
# =========================================================================

def match_by_phrases(docs: list[dict], sturdy_samples: list[dict], label: str) -> list[dict]:
    """Match documents using exact phrase matching.

    For each Sturdy doc_id, we extracted 10 exact 8-word phrases.
    For each dataset document, we check how many of those phrases appear
    as exact substrings in the normalized document text.
    An 8-word exact phrase is nearly unique across financial documents,
    so this avoids the false positives that Jaccard word overlap produces.
    """
    print(f"  Matching {label} with exact phrase matching...")
    results = []

    for i, doc in enumerate(docs):
        best_id = None
        best_score = 0.0
        doc_text = doc["text_normed"]

        for s in sturdy_samples:
            score = phrase_match_score(s["phrases"], doc_text)
            if score > best_score:
                best_score = score
                best_id = s["sturdy_doc_id"]

        results.append({
            "dataset_name": doc.get("dataset_name"),
            "doc_index": doc.get("doc_index"),
            "doc_label": doc.get("doc_label"),
            "doc_name": doc.get("doc_name", ""),
            "filename": doc.get("filename", ""),
            "original_row_indices": doc.get("original_row_indices", ""),
            "financebench_ids": doc.get("financebench_ids", ""),
            "context_hash": doc.get("context_hash"),
            "sturdy_doc_id": best_id if best_score >= 0.3 else None,
            "match_score": round(best_score, 4),
        })

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(docs)} matched")

    return results


def main():
    idx = Index(name=INDEX_NAME, API_key=STURDY_API_KEY)

    # Step 1: Get Sturdy doc_ids + text samples (includes both chunks and phrases)
    sturdy_samples = get_sturdy_samples(idx)

    # Step 2: Load datasets + text samples
    dataset_docs = load_and_sample_datasets()

    # Step 3: Match all datasets using exact phrase matching
    print("\nStep 3: Matching documents...")
    results = []
    for ds_name in ["DocFinQA", "FinDoc-RAG", "FinanceBench"]:
        ds_docs = [d for d in dataset_docs if d["dataset_name"] == ds_name]
        if ds_docs:
            results.extend(match_by_phrases(ds_docs, sturdy_samples, ds_name))

    df = pd.DataFrame(results)
    matched = df["sturdy_doc_id"].notna().sum()
    print(f"\n  Matched:   {matched}")
    print(f"  Unmatched: {len(df) - matched}")

    # Per-dataset summary
    for ds in ["DocFinQA", "FinDoc-RAG", "FinanceBench"]:
        subset = df[df["dataset_name"] == ds]
        m = subset["sturdy_doc_id"].notna().sum()
        scores = subset[subset["sturdy_doc_id"].notna()]["match_score"]
        if len(scores) > 0:
            print(f"  {ds}: {m}/{len(subset)} matched, scores {scores.min():.4f}-{scores.max():.4f} (mean {scores.mean():.4f})")
        else:
            print(f"  {ds}: {m}/{len(subset)} matched")

    # Check for duplicate sturdy_doc_id assignments
    matched_df = df[df["sturdy_doc_id"].notna()]
    dupes = matched_df[matched_df.duplicated(subset="sturdy_doc_id", keep=False)]
    if len(dupes) > 0:
        print(f"\n  WARNING: {len(dupes)} rows share a sturdy_doc_id with another row")
    else:
        print(f"\n  No duplicate sturdy_doc_id assignments")

    # Write CSV (drop the text_normed column -- it's only for matching)
    if "text_normed" in df.columns:
        df = df.drop(columns=["text_normed"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nMapping CSV written to: {OUTPUT_CSV}")

    # Show samples
    print("\n--- Sample matches ---")
    for _, row in df[df["sturdy_doc_id"].notna()].head(5).iterrows():
        print(f"  {row['doc_label']} -> {row['sturdy_doc_id']} (score: {row['match_score']})")


if __name__ == "__main__":
    main()
