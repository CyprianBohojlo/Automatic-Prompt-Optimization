**About the Project**

This project proposes the Automatic Prompt Optimization algorithm for generative financial Question-Answering (QA) tasks. Build on the ProTeGi framework by (Pryzant et al., 2023) it implements the concept of textual Gradient Descent and beam search. Also, it incorporates Reinforcement Learning methods such as Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO). The system targets financial questions grounded in real-world financial documents/filings across three datasets: **FinanceBench**, **DocFinQA**, and **FinDoc-RAG**. The goal of the algorithm is to optimize user's prompt for the LLM. It aims to increase LLM's performance with minimal resources (in terms of cost and computational power).

**Datasets**

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| FinanceBench | [patronus-ai/financebench](https://github.com/patronus-ai/financebench) | Open-source financial QA grounded in SEC filings (10-K, 10-Q, etc.). |
| DocFinQA | [kensho/DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA) | Document-grounded financial QA from HuggingFace. |
| FinDoc-RAG | [IDSIA/FinDoc-RAG](https://gitlab-core.supsi.ch/dti-idsia/ai-finance-papers/findoc-rag) | Financial document QA benchmark from IDSIA/SUPSI. |

The pipeline currently runs end-to-end on FinanceBench. Support for DocFinQA and FinDoc-RAG is being added separately.

**Example Use Cases**

Prompt optimization with neural RAG (default):
```bash
main.py --data_dir "C:\Users\Desktop\data" --prompts "C:\Users\Desktop\seed_prompt.txt"  --out "C:\Users\experimental_run_logs" --rounds 6 --beam_size 4 --eval_rounds 3 --eval_prompts_per_round 6 --samples_per_eval 36 --evaluator ppo --temperature 0.3 --top_k 2
```

Prompt optimization with Sturdy RAG:
```bash
main.py --data_dir "C:\Users\Desktop\data" --prompts "C:\Users\Desktop\seed_prompt.txt" --out "run_log_sturdy" --rag_method sturdy --sturdy_manifest ./output/trained_indices_manifest.csv --rounds 6 --beam_size 4
```

Compare neural vs Sturdy RAG on the same prompt:
```bash
evaluate.py --compare_rag --data_dir "C:\Users\Desktop\data" --prompt_file best_prompt.md --sturdy_manifest ./output/trained_indices_manifest.csv --metric bem
```

**The Layout**

| Script            | What it does                                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------------------- |
| `prepare_data.py` | Merge dataset JSONs, return `dataset_prepared.parquet`. Currently supports FinanceBench; DocFinQA and FinDoc-RAG support forthcoming. |
| `vectorize.py`    | Downloads source documents and creates Chroma vector stores. Currently supports FinanceBench PDFs; DocFinQA and FinDoc-RAG support forthcoming. |
| `optimizers.py`   | `ProTeGi` class: textual-gradient generation, MC paraphrases, beam search     .                                 |
| `evaluators.py`   | UCB, Successive Rejects, Successive Halving, brute-force, PPO evaluators.                                       |
| `PPO.py`          | PPO evaluator that can be used in evaluators.py                                                                 |
| `DPO.py`          | DPO evaluator that can be used in evaluators.py                                                                 |
| `predictors.py`   | `QA_Generator`: retrieves k chunks via neural (Chroma/e5) or Sturdy (hierarchical Bayesian LM) RAG, fills the prompt, calls OpenAI chat. |
| `scorers.py`      | `BEMScorer` and cache system.                                                                                   |
| `tasks.py`        | Dataset splits into training/testing, BEM-driven evaluation helper. Currently implements FinanceBench; DocFinQA and FinDoc-RAG support forthcoming. |
| `utils.py`        | OpenAI REST helper and prompt-section parser.                                                                   |
| `generate.py`     | Use the best prompt produced by the ProTeGi search to answer dataset questions.                                 |
| `evaluate.py`     | Grades answers with LLM-based judge or BEM; `--compare_rag` mode runs both RAG backends and reports side-by-side accuracy. |
| `main.py`         | Main file that launches the ProTeGi search loop.                                                                |
| `test_seed.txt`   | Seed prompt that must be used as one of the arguemnts when running main.py. Can be replaced with other prompt.  |


**Workflow**
1. Data preparation – run prepare_data.py once.

2. Embeddings – run vectorize.py to build Chroma vector stores (neural RAG).

3. *(Optional)* Sturdy RAG setup – provide a `trained_indices_manifest.csv` that maps each document to its pre-trained Sturdy Statistics index. Requires `STURDY_API_KEY`.

4. Prompt search – run main.py with your seed prompt. Add `--rag_method sturdy --sturdy_manifest <path>` to use the Sturdy backend instead of neural.

5. Answer generation – use generate.py with the best prompt returned by main.py.

6. Evaluation – run evaluate.py --metric bem (or gpt) to score answers.

7. *(Optional)* RAG comparison – run evaluate.py --compare_rag to evaluate both RAG backends on the same dataset and prompt.


**Arguments** 
| Argument for main.py       | Description                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--data_dir`               | Path to the folder containing the prepared dataset (e.g., `data`).                                           |
| `--prompts`                | Path to the seed prompt file (e.g., `seed_prompt.txt`).                                                      |
| `--out`                    | Output file to log the run (e.g., `run_log.txt`). The final best prompt is saved as `run_log.txt.prompt.md`. |
| `--rounds`                 | Number of optimization rounds (each round improves the prompts).                                             |
| `--beam_size`              | Number of prompt variants kept per round (used in beam search).                                              |
| `--eval_rounds`            | How many rounds of evaluation to perform during prompt search.                                               |
| `--eval_prompts_per_round` | Number of prompt candidates to evaluate per round.                                                           |
| `--samples_per_eval`       | Number of QA examples to sample for each evaluation.                                                         |
| '--evaluator'              | Evaluation strategy. Defaults to bandit-based search is ucb. Options are: ucb (Upper Confidence Bound), sr (Successive Rejects), sh (Successive Halving), brute (Brute Force) |
| '--temperature'            | Sampling temperature for OpenAI model (e.g., 0.0).                                                           |
| `--top_k`                  | How many top-scoring prompts to consider when selecting the best one.                                        |
| `--n_test_exs`             | Limits the number of test examples used during development (useful for faster debugging).                    |
| `--max_threads`            | Maximum number of things (e.g. prompts or examples) processed at the same time. Higher is faster (e.g. for cluster). |
| `--rag_method`             | RAG retrieval backend: `neural` (Chroma/e5 embeddings, default) or `sturdy` (Sturdy Statistics hierarchical Bayesian LM). |
| `--sturdy_manifest`        | Path to `trained_indices_manifest.csv`. Required when `--rag_method=sturdy`.                                 |

**Arguments for evaluate.py (RAG comparison mode)**
| Argument                   | Description                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--compare_rag`            | Run both neural and Sturdy RAG on the dataset and compare BEM accuracy side-by-side.                         |
| `--data_dir`               | Folder with `dataset_prepared.parquet` (required for `--compare_rag`).                                       |
| `--prompt_file`            | Prompt template file (required for `--compare_rag`).                                                         |
| `--sturdy_manifest`        | Path to `trained_indices_manifest.csv` (required for `--compare_rag`).                                       |
| `--top_k`                  | Number of retrieved chunks/results per query (default: 3).                                                   |
| `--n_examples`             | Limit comparison to first N examples (useful for quick tests).                                               |

**References**
 - Pryzant, R., Iter, D., Li, J., & et al. (2023). Automatic prompt optimization with "gradient descent" and beam search. arXiv preprint arXiv:2305.03495 
