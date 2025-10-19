**About the Project**

This project proposes the Automatic Prompt Optimization algorithm for generative financial Question-Answering (QA) tasks. Build on the ProTeGi framework by (Pryzant et al., 2023) it implements the concept of textual Gradient Descent and beam search. Also, it incorporates Reinforcement Learning methods such as Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO). The system targets financial questions grounded in real-world financial documents/filings. The goal of the algorithm is to optimize user's prompt for the LLM. It aims to increase LLM’s performance with minimal resources (in terms of cost and computational power).

**Example Use Case**
main.py --data_dir "C:\Users\Desktop\data" --prompts "C:\Users\Desktop\seed_prompt.txt"  --out "C:\Users\experimental_run_logs" --rounds 6 --beam_size 4 --eval_rounds 3 --eval_prompts_per_round 6 --samples_per_eval 36 --evaluator ppo --temperature 0.3 --top_k 2 


**The Layout**

| Script            | What it does                                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------------------- |
| `prepare_data.py` | Merge FinanceBench JSONs, return `dataset_prepared.parquet`.                                                    |
| `vectorize.py`    | Downloads the PDFs from the FinanceBench github repo and creates Chroma vector stores .                         |
| `optimizers.py`   | `ProTeGi` class: textual-gradient generation, MC paraphrases, beam search     .                                 |
| `evaluators.py`   | UCB, Successive Rejects, Successive Halving, brute-force, PPO evaluators.                                       |
| `PPO.py`          | PPO evaluator that can be used in evaluators.py                                                                 |
| `DPO.py`          | DPO evaluator that can be used in evaluators.py                                                                 |
| `predictors.py`   | `QA_Generator`: retrieves k chunks, fills the prompt, calls OpenAI chat.                                        |
| `scorers.py`      | `BEMScorer` and cache system.                                                                                   |
| `tasks.py`        | `FinanceBenchTask`: dataset splits into training/testing, BEM-driven evaluation helper.                         |
| `utils.py`        | OpenAI REST helper and prompt-section parser.                                                                   |
| `generate.py`     | Use the best prompt produced by the ProTeGi search to answer FinanceBench questions                             |
| `evaluate.py`     | Grades the answers with either LLM-based judge or BEM                                                           |
| `main.py`         | Main file that launches the ProTeGi search loop.                                                                |
| `test_seed.txt`   | Seed prompt that must be used as one of the arguemnts when running main.py. Can be replaced with other prompt.  |


**Workflow**
1. Data preparation – run prepare_data.py once.

2. Embeddings – run vectorize.py

3. Prompt search – run main.py with your seed prompt.

4. Answer generation – use generate.py with the best prompt returned by main.py.

5. Evaluation – run evaluate.py --metric bem (or gpt) to score answers.


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
| '--max_hreads'             | Maximum number of things (e.g. prompts or examples) processed at the same time. Higher is faster (e.g. for cluster). |

**References**
 - Pryzant, R., Iter, D., Li, J., & et al. (2023). Automatic prompt optimization with "gradient descent" and beam search. arXiv preprint arXiv:2305.03495 
