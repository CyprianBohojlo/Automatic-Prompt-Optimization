"""Vector store builder + retriever factory for FinanceBench/FinDER/DocFinQA."""
import argparse
import os
import pathlib
import sys
from typing import Dict

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "API key not loaded!"


# ---------------------------------------------------------------------
ROOT = pathlib.Path(r"C:\Users\cypri\Desktop\Master Thesis")
DEFAULT_DATA_DIRS = {
    "financebench": ROOT / "data" / "FinanceBench",
    "finder": ROOT / "data" / "Finder",
    "docfinqa": ROOT / "data" / "DocFinQa",
    "findoc": ROOT / "data" / "FinDoc",
}
FINANCEBENCH_REF = ROOT / "references" / "FinanceBenchPdfs"
FINANCEBENCH_REF.mkdir(parents=True, exist_ok=True)
DEFAULT_REF_DIRS = {
    "financebench": FINANCEBENCH_REF,
    #"finder": pathlib.Path("C:\Users\cypri\Desktop\Master Thesis\references\Finder"),
    "docfinqa": None,
    #"findoc": pathlib.Path("C:\Users\cypri\Desktop\Master Thesis\references\FinDoc"),
}
FINANCEBENCH_VS = ROOT / "vectorstores"/ "FinanceBench"
FINANCEBENCH_VS.mkdir(parents=True, exist_ok=True)

VS_CHUNK_SIZE = 1024
VS_CHUNK_OVERLAP = 30
EMBEDDINGS = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
GH_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"

_CONFIG: Dict[str, pathlib.Path | str | None] = {
    "task": "financebench",
    "data_dir": DEFAULT_DATA_DIRS["financebench"],
    "dataset_path": DEFAULT_DATA_DIRS["financebench"] / "dataset_prepared.parquet",
    "ref_dir": FINANCEBENCH_REF,
    "vs_dir": FINANCEBENCH_VS,
}
_DOC_CONTEXTS: Dict[str, str] | None = None
# ---------------------------------------------------------------------


def configure(task: str = "financebench", data_dir: str | os.PathLike | None = None,
              ref_dir: str | os.PathLike | None = None,
              vs_dir: str | os.PathLike | None = None):
    """Configure dataset-dependent paths before building retrievers."""
    global _CONFIG, _DOC_CONTEXTS

    task = task.lower()
    data_root = pathlib.Path(data_dir) if data_dir else DEFAULT_DATA_DIRS.get(task, DEFAULT_DATA_DIRS["financebench"])
    dataset_path = pathlib.Path(data_root) / "dataset_prepared.parquet"

    if ref_dir is None:
        ref_dir_path = DEFAULT_REF_DIRS.get(task)
    elif ref_dir == "":
        ref_dir_path = None
    else:
        ref_dir_path = pathlib.Path(ref_dir)

    if ref_dir_path and "YOUR_PATH" not in str(ref_dir_path):
        ref_dir_path.mkdir(parents=True, exist_ok=True)

    vs_root = pathlib.Path(vs_dir) if vs_dir else FINANCEBENCH_VS
    vs_root.mkdir(parents=True, exist_ok=True)

    _CONFIG = {
        "task": task,
        "data_dir": data_root,
        "dataset_path": dataset_path,
        "ref_dir": ref_dir_path,
        "vs_dir": vs_root,
    }
    _DOC_CONTEXTS = None


def _sanitize_doc_name(name: str) -> str:
    return str(name).replace(os.sep, "_")


def _docfinqa_contexts() -> Dict[str, str]:
    global _DOC_CONTEXTS
    if _DOC_CONTEXTS is None:
        df = pd.read_parquet(_CONFIG["dataset_path"])
        if "context_text" not in df.columns:
            raise ValueError("DocFinQA dataset must contain a context_text column")
        _DOC_CONTEXTS = dict(zip(df["doc_name"].astype(str), df["context_text"].astype(str)))
    return _DOC_CONTEXTS


def _load_pdf_pages(doc_name: str):
    ref_dir = _CONFIG.get("ref_dir")
    if ref_dir is None:
        raise FileNotFoundError("Reference directory is not configured for this dataset.")
    base_name = str(doc_name)
    if _CONFIG["task"] == "findoc":
        filename = base_name if base_name.lower().endswith(".md") else f"{base_name}.md"
        path_doc = pathlib.Path(ref_dir) / filename
        if not path_doc.exists():
            raise FileNotFoundError(f"Missing markdown for {base_name} at {path_doc}.")
        text = pathlib.Path(path_doc).read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"doc_name": doc_name})]

    filename = base_name if base_name.lower().endswith(".pdf") else f"{base_name}.pdf"
    path_doc = pathlib.Path(ref_dir) / filename
    if not path_doc.exists():
        if _CONFIG["task"] == "financebench":
            r = requests.get(f"{GH_RAW}/{base_name}.pdf", timeout=40)
            r.raise_for_status()
            path_doc.write_bytes(r.content)
        else:
            raise FileNotFoundError(f"Missing PDF for {base_name} at {path_doc}.")
    return PyMuPDFLoader(str(path_doc)).load()


def get_pdf_text(doc_name: str):
    if _CONFIG["task"] == "docfinqa":
        ctx = _docfinqa_contexts().get(str(doc_name))
        if ctx is None:
            raise KeyError(f"No context stored for doc_name={doc_name}")
        return [Document(page_content=ctx, metadata={"doc_name": doc_name})]
    return _load_pdf_pages(doc_name)


def build_vectorstore_retriever(doc_name, embeddings=EMBEDDINGS):
    """Return a retriever backed by per-document Chroma stores."""
    safe_doc = _sanitize_doc_name(doc_name)
    db_path = pathlib.Path(_CONFIG["vs_dir"]) / _CONFIG["task"] / safe_doc
    if not (db_path / "chroma.sqlite3").exists():
        db_path.mkdir(parents=True, exist_ok=True)
        pages = get_pdf_text(doc_name)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=VS_CHUNK_SIZE, chunk_overlap=VS_CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(pages)
        Chroma.from_documents(chunks, embeddings, persist_directory=str(db_path)).persist()
    vectordb = Chroma(persist_directory=str(db_path), embedding_function=embeddings)
    return vectordb.as_retriever()


def parse_args():
    p = argparse.ArgumentParser("Build vectorstores for FinanceBench/FinDER/DocFinQA")
    p.add_argument("--task", choices=("financebench", "finder", "docfinqa", "findoc"), default="financebench")
    p.add_argument("--data_dir", required=False, help="Folder containing dataset_prepared.parquet")
    p.add_argument("--ref_dir", required=False, help="Folder with PDF references (if applicable)")
    p.add_argument("--vs_dir", required=False, help="Folder to store vectorstores")
    return p.parse_args()


def main():
    args = parse_args()
    configure(task=args.task, data_dir=args.data_dir, ref_dir=args.ref_dir, vs_dir=args.vs_dir)
    dataset_path = _CONFIG["dataset_path"]
    if not dataset_path.exists():
        sys.exit(f"Dataset not found at {dataset_path}. Run prepare_data.py first.")
    df = pd.read_parquet(dataset_path)
    for doc in tqdm(df.doc_name.astype(str).unique(), desc="VectorStores"):
        build_vectorstore_retriever(doc)
    print("All single-store vector DBs ready.")


if __name__ == "__main__":
    main()