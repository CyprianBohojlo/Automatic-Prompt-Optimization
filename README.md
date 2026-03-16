# Automatic Prompt Optimization for Financial QA

## About the Project

This project implements the Automatic Prompt Optimization algorithm for generative financial Question-Answering (QA) tasks. Built on the ProTeGi framework by (Pryzant et al., 2023), it implements the concept of Textual Gradient Descent and Beam Search. It also incorporates Reinforcement Learning methods such as Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO). The system targets financial questions grounded in real-world financial documents and filings. The goal of the algorithm is to optimize the user's prompt for the LLM, aiming to increase performance with minimal resources (in terms of cost and computational power).

## Retrieval Methods

The system supports two retrieval approaches, selectable via `--rag_method`:

- **Neural** (`--rag_method neural`, default): Dense embeddings (E5-small-v2) with per-filing Chroma vector stores. Each filing is chunked (1024 chars, 30 overlap) and embedded; at query time the top-K chunks are retrieved via MIPS.
- **Sturdy** (`--rag_method sturdy`): A single pre-trained hierarchical Bayesian language model (`bulk_train_all`) hosted on Sturdy Statistics. All documents from all three datasets (DocFinQA, FinDoc-RAG, FinanceBench) are in one index; at query time the index is filtered to the relevant document using `doc_id_mapping.csv`.

### Sturdy Setup

The Sturdy method requires:

1. A `STURDY_API_KEY` environment variable.
2. A `doc_id_mapping.csv` file that maps dataset document identifiers to Sturdy `doc_id` values. Place it in your `--data_dir` folder (auto-detected) or specify the path explicitly with `--doc_id_mapping`.
3. The `doc_id_mapping.csv` is produced by `build_doc_id_mapping.py`, which queries the trained Sturdy index and matches documents from all three datasets using exact phrase matching.

## Example Use Case

```bash
# Neural retrieval (default)
python main.py \
  --data_dir ./data \
  --prompts test_seed.txt \
  --out run_log.txt \
  --rounds 6 --beam_size 4 \
  --eval_rounds 3 --eval_prompts_per_round 6 --samples_per_eval 36 \
  --evaluator ppo --temperature 0.3 --top_k 2

# Sturdy retrieval (uses bulk index filtered by doc_id)
python main.py \
  --data_dir ./data \
  --prompts test_seed.txt \
  --out run_log_sturdy.txt \
  --rag_method sturdy \
  --rounds 6 --beam_size 4 \
  --eval_rounds 3 --eval_prompts_per_round 6 --samples_per_eval 36 \
  --evaluator ppo --temperature 0.3 --top_k 2
```

## File Layout

| Script               | What it does                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| `main.py`            | Launches the ProTeGi search loop.                                                                |
| `predictors.py`      | `QA_Generator`: retrieves K chunks (neural or Sturdy), fills the prompt, calls OpenAI chat.      |
| `tasks.py`           | `FinanceBenchTask`: dataset splits into training/testing, BEM-driven evaluation helper.           |
| `optimizers.py`      | `ProTeGi` class: textual-gradient generation, MC paraphrases, beam search.                       |
| `evaluators.py`      | UCB, Successive Rejects, Successive Halving, brute-force, PPO, and DPO evaluators.               |
| `PPO.py`             | PPO evaluator used by `evaluators.py`.                                                           |
| `DPO.py`             | DPO evaluator used by `evaluators.py`.                                                           |
| `scorers.py`         | `BEMScorer` and cache system.                                                                    |
| `vectorize.py`       | Downloads FinanceBench PDFs and creates per-filing Chroma vector stores.                         |
| `prepare_data.py`    | Merges FinanceBench JSONs, produces `dataset_prepared.parquet`.                                  |
| `build_doc_id_mapping.py` | Builds `doc_id_mapping.csv` by matching dataset documents to Sturdy `doc_id` values.        |
| `utils.py`           | OpenAI REST helper and prompt-section parser.                                                    |
| `generate.py`        | Uses the best prompt from ProTeGi to answer FinanceBench questions.                              |
| `evaluate.py`        | Grades answers with either LLM-based judge or BEM.                                               |
| `test_seed.txt`      | Seed prompt for `main.py`. Can be replaced with a custom prompt.                                 |

## Workflow

1. **Data preparation** -- run `prepare_data.py` once.
2. **Embeddings** -- run `vectorize.py` (neural method only).
3. **Prompt search** -- run `main.py` with your seed prompt and chosen `--rag_method`.
4. **Answer generation** -- use `generate.py` with the best prompt returned by `main.py`.
5. **Evaluation** -- run `evaluate.py --metric bem` (or `gpt`) to score answers.

## Arguments

### Core Arguments

| Argument                   | Description                                                                                        | Default      |
| -------------------------- | -------------------------------------------------------------------------------------------------- | ------------ |
| `--data_dir`               | Path to the folder containing `dataset_prepared.parquet` (and optionally `doc_id_mapping.csv`).    | *(required)* |
| `--prompts`                | Comma-separated list of seed prompt file paths.                                                    | *(required)* |
| `--out`                    | Output log file. The final best prompt is saved as `<out>.prompt.md`.                              | `run_log.txt`|
| `--rounds`                 | Number of optimization rounds.                                                                     | `6`          |
| `--beam_size`              | Number of prompt variants kept per round (beam search).                                            | `4`          |
| `--eval_rounds`            | Rounds of evaluation during prompt search.                                                         | `6`          |
| `--eval_prompts_per_round` | Number of prompt candidates to evaluate per round.                                                 | `10`         |
| `--samples_per_eval`       | Number of QA examples sampled per evaluation.                                                      | `5`          |
| `--evaluator`              | Evaluation strategy: `ucb`, `ucb-e`, `sr`, `s-sr`, `sh`, `bf`, `ppo`, `dpo`.                      | `ucb`        |
| `--temperature`            | Sampling temperature for the OpenAI model.                                                         | `0.0`        |
| `--top_k`                  | Number of retrieved chunks passed as context.                                                      | `3`          |
| `--max_threads`            | Max parallel workers for inference and evaluation.                                                 | `4`          |
| `--n_test_exs`             | Limit test examples (useful for debugging).                                                        | `None`       |

### RAG Method Arguments

| Argument              | Description                                                                                         | Default           |
| --------------------- | --------------------------------------------------------------------------------------------------- | ----------------- |
| `--rag_method`        | Retrieval method: `neural` (E5 + Chroma) or `sturdy` (bulk Sturdy index).                          | `neural`          |
| `--doc_id_mapping`    | Path to `doc_id_mapping.csv`. Defaults to `<data_dir>/doc_id_mapping.csv` when `--rag_method sturdy`. | *(auto-detected)* |
| `--sturdy_index_name` | Name of the Sturdy bulk index.                                                                      | `bulk_train_all`  |

### PPO Arguments (used with `--evaluator ppo`)

| Argument             | Description                                        | Default |
| -------------------- | -------------------------------------------------- | ------- |
| `--ppo_hidden`       | Hidden units in PPO value network.                 | `64`    |
| `--ppo_lr`           | Learning rate for PPO.                             | `2e-3`  |
| `--ppo_gamma`        | Discount factor for PPO.                           | `0.99`  |
| `--ppo_log_history`  | Log per-mini-batch rewards for plotting.           | `False` |

### DPO Arguments (used with `--evaluator dpo`)

| Argument               | Description                                      | Default |
| ---------------------- | ------------------------------------------------ | ------- |
| `--dpo_beta`           | Temperature for DPO loss.                        | `0.1`   |
| `--dpo_lr`             | Learning rate for DPO policy head.               | `3e-4`  |
| `--dpo_hidden`         | Hidden units in DPO policy MLP.                  | `128`   |
| `--dpo_margin`         | Min BEM gap to form a preference pair.           | `0.0`   |
| `--dpo_reference_free` | Run IPO (no reference-model KL term).            | `False` |
| `--dpo_gpt_judge`      | Use GPT judge on near-ties.                      | `False` |

## References

- Pryzant, R., Iter, D., Li, J., & et al. (2023). Automatic prompt optimization with "gradient descent" and beam search. *arXiv preprint arXiv:2305.03495*
- Islam, S. et al. (2023). FinanceBench: A New Benchmark for Financial Question Answering. *arXiv preprint arXiv:2311.11944*
