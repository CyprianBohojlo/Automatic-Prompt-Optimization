

import argparse, json, os, sys, pathlib
from collections import Counter
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv


from scorers import BEMScorer
from predictors import QA_Generator


load_dotenv()

ROOT     = pathlib.Path(r"C:\Users\cypri\OneDrive\Desktop\Master Thesis")
ANSWERS  = ROOT / "results" / "answers.jsonl"
GRADES   = ROOT / "results" / "grades_ce.jsonl"

def get_args():
    p = argparse.ArgumentParser("Grade FinanceBench answers")
    p.add_argument("--metric", choices=["gpt", "bem", "checkembed"], default="gpt",
                   help="Which judge to use.")
    p.add_argument("--threshold", type=float, default=0.56,
                   help="BEM equivalence probability threshold.")

    # RAG comparison arguments
    p.add_argument("--compare_rag", action="store_true",
                   help="Run both neural and Sturdy RAG on the dataset and "
                        "compare BEM accuracy side-by-side.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Folder with dataset_prepared.parquet (required for --compare_rag)")
    p.add_argument("--prompt_file", type=str, default=None,
                   help="Prompt template file (required for --compare_rag)")
    p.add_argument("--sturdy_manifest", type=str, default=None,
                   help="Path to trained_indices_manifest.csv (required for --compare_rag)")
    p.add_argument("--top_k", type=int, default=3,
                   help="Number of retrieved chunks/results per query.")
    p.add_argument("--n_examples", type=int, default=None,
                   help="Limit comparison to first N examples (useful for quick tests).")
    return p.parse_args()


judge = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=64)
SYS = (
    "You are a strict grader.\n"
    "Classify the candidate answer as one of:\n"
    "- CORRECT (factual and matches the gold answer)\n"
    "- INCORRECT (plausible but wrong)\n"
    "- REFUSED (the model refused to answer or hallucinated)\n"
    "Respond ONLY with one of those labels."
)

def verdict_gpt(question: str, gold: str, cand: str) -> str:
    prompt = f"{SYS}\nQ: {question}\nGold: {gold}\nCandidate: {cand}\nVerdict:"
    resp = judge.predict(prompt).strip().upper()
    if resp.startswith("CORRECT"):
        return "CORRECT"
    if resp.startswith("INCORRECT"):
        return "INCORRECT"
    if resp.startswith("REFUSED"):
        return "REFUSED"
    return "INCORRECT"


bem_judge = BEMScorer(None)

def verdict_bem(question: str, gold: str, cand: str, tau: float) -> str:
    if bem_judge.pair_equivalent(cand, gold, question):
        return "CORRECT"
    if cand.strip().lower() in {"", "i don't know", "idk"}:
        return "REFUSED"
    return "INCORRECT"





def compare_rag_methods(args):
    """Generate answers with both RAG backends and compare BEM accuracy."""
    if not args.data_dir:
        sys.exit("--data_dir is required for --compare_rag")
    if not args.prompt_file:
        sys.exit("--prompt_file is required for --compare_rag")
    if not args.sturdy_manifest:
        sys.exit("--sturdy_manifest is required for --compare_rag")

    prompt = pathlib.Path(args.prompt_file).read_text()
    df = pd.read_parquet(pathlib.Path(args.data_dir) / "dataset_prepared.parquet")
    examples = [
        {"id": int(i), "question": r.question, "answer": r.answer, "doc_name": r.doc_name}
        for i, r in df.iterrows()
    ]
    if args.n_examples:
        examples = examples[: args.n_examples]

    scorer = BEMScorer(None, bem_threshold=args.threshold)

    results = []
    for method in ("neural", "sturdy"):
        predictor = QA_Generator({
            "top_k": args.top_k,
            "rag_method": method,
            "sturdy_manifest": args.sturdy_manifest if method == "sturdy" else None,
        })
        hits = 0
        for ex in tqdm(examples, desc=f"RAG={method}"):
            try:
                pred = predictor.inference(ex, prompt)
            except Exception as e:
                print(f"  WARN [{method}] {ex['doc_name']}: {e}")
                pred = ""
            is_correct = scorer.pair_equivalent(pred, ex["answer"], ex["question"])
            results.append({
                "rag_method": method,
                "doc_name": ex["doc_name"],
                "question": ex["question"],
                "gold_answer": ex["answer"],
                "model_answer": pred,
                "bem_correct": is_correct,
            })
            hits += int(is_correct)
        acc = hits / len(examples)
        print(f"\n{method.upper()} RAG — BEM accuracy: {acc:.2%}  ({hits}/{len(examples)})")

    # Side-by-side summary
    df_results = pd.DataFrame(results)
    summary = df_results.groupby("rag_method")["bem_correct"].agg(["mean", "sum", "count"])
    summary.columns = ["accuracy", "correct", "total"]
    print("\n===== RAG COMPARISON =====")
    print(summary.to_string())

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    summary["accuracy"].plot.bar(ax=ax, color=["steelblue", "darkorange"])
    ax.set_ylabel("BEM Accuracy")
    ax.set_title("Neural vs Sturdy RAG — BEM Accuracy")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("rag_comparison.png", dpi=150)
    plt.show()

    # Save detailed results
    out_path = pathlib.Path("rag_comparison_results.jsonl")
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\nDetailed results saved to {out_path}")


def main():
    args = get_args()

    # ---- RAG comparison mode ----
    if args.compare_rag:
        compare_rag_methods(args)
        return

    # ---- Original grading mode ----
    if not ANSWERS.exists():
        sys.exit("Run generate.py first to produce answers.jsonl.")


    total_qas = sum(1 for _ in ANSWERS.open("r", encoding="utf-8"))

    graded = []
    with ANSWERS.open("r", encoding="utf-8") as fin, \
         tqdm(total=total_qas, desc="Grading", unit="qa") as pbar:
        for line in fin:
            rec = json.loads(line)
            if args.metric == "gpt":
                v = verdict_gpt(rec["question"], rec["gold_answer"], rec["model_answer"])
            elif args.metric =="bem":
                v = verdict_bem(rec["question"], rec["gold_answer"],
                                rec["model_answer"], args.threshold)
            graded.append({**rec, "verdict": v})
            pbar.update(1)


    GRADES.parent.mkdir(parents=True, exist_ok=True)
    with GRADES.open("w", encoding="utf-8") as fout:
        for g in graded:
            fout.write(json.dumps(g, ensure_ascii=False) + "\n")

    counts = Counter(g["verdict"] for g in graded)


    labels = ["CORRECT", "INCORRECT", "REFUSED"]
    values = [counts.get(l, 0) for l in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title(f"Verdict distribution ({args.metric.upper()})")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    total = len(graded)
    acc = counts.get("CORRECT", 0) / total
    print(f"Accuracy ({args.metric}): {acc:.2%}   Counts: {dict(counts)}")
    print(f"Grades saved to {GRADES}")

if __name__ == "__main__":
    main()
