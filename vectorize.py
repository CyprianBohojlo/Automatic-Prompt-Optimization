
import os, pathlib, requests, sys
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "API key not loaded!"

import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


ROOT    = pathlib.Path(r"C:\Users\cypri\OneDrive\Desktop\Master Thesis")
DATASET = ROOT / "data" / "dataset_prepared.parquet"
PATH_PDFS = ROOT / "pdfs";            PATH_PDFS.mkdir(exist_ok=True)
VS_DIR   = ROOT / "vectorstores";     VS_DIR.mkdir(exist_ok=True)

VS_CHUNK_SIZE      = 1024
VS_CHUNK_OVERLAP   = 30
EMBEDDINGS         = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
GH_RAW             = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"


def get_pdf_text(doc_name: str):
    path_doc = PATH_PDFS / f"{doc_name}.pdf"
    if not path_doc.exists():
        r = requests.get(f"{GH_RAW}/{doc_name}.pdf", timeout=40)
        r.raise_for_status()
        path_doc.write_bytes(r.content)
    return PyMuPDFLoader(str(path_doc)).load()

def build_vectorstore_retriever(docs, embeddings=EMBEDDINGS):
    """`docs` is either one doc_name or 'all'. For singleStore we always pass one."""
    db_path = VS_DIR / docs
    if not (db_path / "chroma.sqlite3").exists():
        db_path.mkdir(parents=True, exist_ok=True)
        pages = get_pdf_text(docs)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=VS_CHUNK_SIZE, chunk_overlap=VS_CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(pages)
        Chroma.from_documents(chunks, embeddings, persist_directory=str(db_path)).persist()
    vectordb = Chroma(persist_directory=str(db_path), embedding_function=embeddings)
    return vectordb.as_retriever()

def main():
    if not DATASET.exists():
        sys.exit("Run prepare_dataset.py first.")
    df = pd.read_parquet(DATASET)
    for doc in tqdm(df.doc_name.unique(), desc="VectorStores"):
        build_vectorstore_retriever(doc)
    print("All single-store vector DBs ready.")

if __name__ == "__main__":
    main()
