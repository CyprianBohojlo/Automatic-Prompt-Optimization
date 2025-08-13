import argparse, json, os, random, time, pathlib
from tqdm import tqdm

import optimizers
from tasks      import get_task
from predictors import QA_Generator
from scorers    import BEMScorer
from evaluators import get_evaluator, PPOEvaluator, DPOEvaluator


def parse_args():
    p = argparse.ArgumentParser("ProTeGi prompt search (FinanceBench)")
    p.add_argument("--data_dir", required=True,
                   help="Folder with dataset_prepared.parquet")
    p.add_argument("--prompts", required=True,
                   help="Comma-separated list of seed prompt files")
    p.add_argument("--out", default="run_log.txt")
    # optimiser hyper-params 
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--eval_rounds", type=int, default=6)
    p.add_argument("--eval_prompts_per_round", type=int, default=10)
    p.add_argument("--samples_per_eval", type=int, default=5)
    p.add_argument("--evaluator",
                   choices=["ucb", "ucb-e", "sr", "s-sr", "sh", "bf", "ppo", "dpo"],
                   default="ucb")
    # model options
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_threads", type=int, default=4)
    p.add_argument("--n_test_exs", type=int, default=None)

    # PPO-specific options (ignored unless --evaluator ppo)
    p.add_argument("--ppo_hidden", type=int,   default=64)
    p.add_argument("--ppo_lr",     type=float, default=2e-3)
    p.add_argument("--ppo_gamma",  type=float, default=0.99)
    p.add_argument("--ppo_log_history", action="store_true",
                    help="Keep a list of per-mini-batch rewards inside PPOEvaluator for later plotting.")
    
    # DPO-specific options (ignored unless --evaluator dpo)
    p.add_argument("--dpo_beta", type=float, default=0.1,
                   help="β temperature for DPO loss")
    p.add_argument("--dpo_lr", type=float, default=3e-4,
                   help="learning rate for DPO policy head")
    p.add_argument("--dpo_hidden", type=int, default=128,
                   help="hidden units in DPO policy MLP")
    p.add_argument("--dpo_margin", type=float, default=0.0,
                   help="min BEM gap to form a preference pair")
    p.add_argument("--dpo_reference_free", action="store_true",
                   help="run IPO (no ref‐model KL term)")
    p.add_argument("--dpo_gpt_judge", action="store_true",
                   help="use predictor.judge_is_better() on near‐ties")

    return p.parse_args()


def load_prompts(prompt_files: str):
    paths = [pf.strip() for pf in prompt_files.split(",")]
    return [pathlib.Path(p).read_text() for p in paths]


def fill_defaults(cfg: dict):
    # Ensures optimiser keys exist so ProTeGi never raises KeyError
    defaults = dict(
        n_gradients=2,
        errors_per_gradient=4,
        gradients_per_error=5,
        steps_per_gradient=3,
        mc_samples_per_step=2,
        minibatch_size=16,
        max_expansion_factor=32,
        c=1.0)
    for k, v in defaults.items():
        cfg.setdefault(k, v)


def main() -> None:
    args   = parse_args()
    random.seed(42)

    config = vars(args).copy()
    config["task"]   = "financebench"
    config["scorer"] = "bem"
    config["eval_budget"] = (args.samples_per_eval * args.eval_rounds * args.eval_prompts_per_round)
    fill_defaults(config)

    task       = get_task("financebench", args.data_dir,
                          max_threads=args.max_threads)
    predictor  = QA_Generator({"temperature": args.temperature,
                               "top_k": args.top_k})
    scorer     = BEMScorer(predictor)

    if args.evaluator == "ppo":
        evaluator = PPOEvaluator(eval_rounds = args.eval_rounds,
                                 samples_per_eval = args.samples_per_eval,
                                 ppo_hidden = args.ppo_hidden,
                                 ppo_lr = args.ppo_lr,
                                 ppo_gamma = args.ppo_gamma,
                                 log_history = args.ppo_log_history)
        
    elif args.evaluator == "dpo":                      
        evaluator = DPOEvaluator(eval_rounds = args.eval_rounds,
                                 samples_per_eval = args.samples_per_eval,
                                 dpo_hidden = args.dpo_hidden,
                                 dpo_lr = args.dpo_lr,
                                 dpo_beta = args.dpo_beta,
                                 dpo_margin = args.dpo_margin,
                                 reference_free = args.dpo_reference_free)

    else:
        evaluator  = get_evaluator(args.evaluator)(config)
    bf_eval    = get_evaluator("bf")(config)

    optimiser  = optimizers.ProTeGi(config, evaluator, scorer, max_threads=args.max_threads, bf_eval=bf_eval)

    train_exs  = task.get_train_examples()
    test_exs   = task.get_test_examples()

    # prepare output file 
    if os.path.exists(args.out):
        os.remove(args.out)
    with open(args.out, "a") as f:
        f.write(json.dumps(config) + "\n")

    # seed prompts 
    candidates = load_prompts(args.prompts)

    # optimisation loop 
    for round in tqdm(range(config["rounds"] + 1), desc="round"):
        start = time.time()

        # expand
        if round > 0:
            candidates = optimiser.expand_candidates(candidates, task, predictor, train_exs)

        # score
        scores = optimiser.score_candidates(candidates, task, predictor, train_exs)
        scores, candidates = zip(*sorted(zip(scores, candidates), reverse=True))
        scores, candidates = list(scores), list(candidates)

        # select candidates
        candidates = candidates[: config["beam_size"]]
        scores = scores[: config["beam_size"]]

        #  record candidates, estimated scores, and true scores
        with open(args.out, "a") as f:
            f.write(f"======== ROUND {round}\n")
            f.write(f"{time.time() - start:.2f}s\n")
            f.write(json.dumps(scores) + "\n")

        metrics = []
        for candidate in candidates:
            accuracy = task.evaluate(predictor, candidate, test_exs, n=args.n_test_exs)
            metrics.append(accuracy)
        with open(args.out, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    print("\nSearch finished. Best prompt:\n")
    print(candidates[0])

    # save the top prompt
    pathlib.Path(args.out + ".prompt.md").write_text(candidates[0])
    print(f"\nSaved to {args.out}.prompt.md")


if __name__ == "__main__":
    main()
