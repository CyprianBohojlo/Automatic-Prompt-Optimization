"""
Functions kept: get_model · get_answer
Uses only singleStore path
Writes results/answers.jsonl
"""
import json, pathlib, sys
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

ROOT      = pathlib.Path(r"C:\Users\cypri\OneDrive\Desktop\Master Thesis")
DATASET   = ROOT / "data" / "dataset_prepared.parquet"
VS_DIR    = ROOT / "vectorstores"
RESULTS   = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
OUT_PATH  = RESULTS / "answers.jsonl"

EMBEDDINGS = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

def get_model(model_name="gpt-4o-mini", temp=0.01, max_tokens=2048):
    return ChatOpenAI(model_name=model_name, temperature=temp, max_tokens=max_tokens)

def get_answer(model, question, retriever):
    qa = RetrievalQA.from_chain_type(
        llm=model, chain_type="stuff", retriever=retriever,
        return_source_documents=False
    )
    return qa(question)["result"]

def main():
    if not DATASET.exists(): sys.exit("Run prepare_data and vectorize first.")
    df = pd.read_parquet(DATASET).sort_values("doc_name")
    model = get_model()
    cache = {}
    with OUT_PATH.open("w", encoding="utf8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Answering"):
            doc = row.doc_name
            if doc not in cache:
                cache[doc] = Chroma(
                    persist_directory=str(VS_DIR/doc),
                    embedding_function=EMBEDDINGS
                ).as_retriever()
            answer = get_answer(model, row.question, cache[doc])
            f.write(json.dumps({
                "financebench_id": row.financebench_id,
                "question": row.question,
                "gold_answer": row.answer,
                "model_answer": answer,
                "doc_name": doc
            }, ensure_ascii=False) + "\n")
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    main()
