import random, numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
import utils




# LOGGER

import logging
import os

# create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ensure output directory exists
out_dir = os.path.dirname(os.environ.get("PROTEGI_DEBUG_LOG", "")) or "."
os.makedirs(out_dir, exist_ok=True)

# file handler writes debug logs here
log_path = os.path.join(out_dir, "expand_debug.log")
fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)

# nice timestamped format
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
fh.setFormatter(formatter)

# attach to our logger
logger.addHandler(fh)


class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval

    @abstractmethod
    def expand_candidates(self, prompts, task, predictor, train_exs):
        pass


class ProTeGi(PromptOptimizer):
    #ProTeGi: Prompt Optimization with Textual Gradients  

    def _sample_error_str(self, examples, preds, n=4):
        
        # Just like in the ooriginal, sample up to n error cases where the BEM scorer says the model answer is not equivalent to gold answer
        
        error_idxs = []
        for i, (ex, pred) in enumerate(zip(examples, preds)):
            if not self.scorer.pair_equivalent(pred, ex["answer"], ex["question"]):
                error_idxs.append(i)

        if not error_idxs:
            return ""  # nothing to show

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        error_string = ""
        error_idx = 0

        for i in sample_idxs:
            ex = examples[i]
            pred = preds[i]
            error_string += f"## Example {error_idx + 1}\n"
            error_string += f'Question: "{ex["question"].strip()}"\n'
            error_string += f'Gold answer: "{ex["answer"].strip()}"\n'
            error_string += f'Prediction: "{pred.strip()}"\n\n'
            error_idx += 1
        return error_string.strip()

    def parse_tagged_text(self, text, start_tag, end_tag):
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index + len(end_tag) :]
        return texts

    def _get_gradients(self, prompt, error_string, num_feedbacks=5, n=1):
        gradient_prompt = f"""
        You are designing a retrieval-augmented question-answering prompt.

        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        Wrap each reason with <START> and <END>
        """
        gradient_prompt = "\n".join(line.lstrip() for line in gradient_prompt.split("\n"))
        res = utils.chatgpt(gradient_prompt, n=n)
        feedbacks = []
        for r in res:
            feedbacks += self.parse_tagged_text(r, "<START>", "<END>")
        return feedbacks

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1):
        transformation_prompt = f"""
        You are designing a retrieval-augmented question-answering prompt.

        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.
        """
        transformation_prompt = "\n".join(
            line.lstrip() for line in transformation_prompt.split("\n")
        )
        res = utils.chatgpt(transformation_prompt, n=n)
        new_prompts = []
        for r in res:
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
        return new_prompts

    def generate_synonyms(self, prompt_section, n=3):
        rewriter_prompt = (
            "Generate a variation of the following instruction while keeping the "
            "semantic meaning.\n\n"
            f"Input: {prompt_section}\n\nOutput:"
        )
        out = utils.chatgpt(rewriter_prompt, n=n)
        return [x for x in out if x]

    def get_gradients(self, prompt, task_section, examples, preds):
        prompt_feedbacks = []
        for _ in tqdm(range(self.opt["n_gradients"]), desc="gradients.."):
            error_string = self._sample_error_str(
                examples, preds, n=self.opt["errors_per_gradient"]
            )
            if not error_string:
                continue
            gradients = self._get_gradients(task_section, error_string, self.opt["gradients_per_error"], n=1)
            prompt_feedbacks += [(g, error_string) for g in gradients]
        return prompt_feedbacks

    def expand_candidates(self, prompts, task, predictor, train_exs):
        minibatch = random.sample(train_exs, k=self.opt["minibatch_size"])
        new_prompts = []

        for prompt in tqdm(prompts, desc=f"expanding {len(prompts)} prompts"):
            sections = utils.parse_sectioned_prompt(prompt)
            task_section = sections["task"].strip()

            preds = predictor.batch_inference(minibatch, prompt)

            # get gradeints
            new_task_sections = []
            if self.opt["n_gradients"] > 0:
                gradients = self.get_gradients(prompt, task_section, minibatch, preds)
                    # ---- DEBUG #1: print out what the LLM said went wrong ----
                logger.debug(">>> GRADIENTS RETURNED:")
                for idx, (feedback, err_str) in enumerate(gradients, start=1):
                    # precompute the escaped version
                    fb = feedback.replace('\n', '\\n')
                    es = err_str.replace('\n', '\\n')
                    logger.debug(f"Gradient {idx} feedback={fb}")
                    logger.debug(f"Gradient {idx} err_str ={es}")
                logger.debug(">>> end gradients")
                for feedback, err_str in gradients:
                    tmp = self.apply_gradient(task_section, err_str, feedback, self.opt["steps_per_gradient"])
                    new_task_sections += tmp

            # generate MC synonyms 
            mc_sampled = []
            if self.opt["mc_samples_per_step"] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc='mc samples'):
                    #mc_sampled += self.generate_synonyms(sect, n=self.opt["mc_samples_per_step"])

                    syns = self.generate_synonyms(sect, n=self.opt["mc_samples_per_step"])
                    mc_sampled += syns

                    # ---- DEBUG #2: print out the paraphrases you got ----
                    sec = sect.replace('\n','\\n')
                    logger.debug(f"<<< MC PARAPHRASES for section: {sec}")
                    for i, s in enumerate(syns,1):
                        syn = s.replace('\n','\\n')
                        logger.debug(f"   Synonym {i}: {syn}")
                    logger.debug(">>> end MC paraphrases")

            # combine
            new_sections = list(set(new_task_sections + mc_sampled))
            # tmp_prompts = [prompt.replace(task_section, s) for s in new_sections]   ########### OLD VERSION

            # --- robust replacement of Task section by slicing on headers ---
            def rebuild(prompt, new_task):
                lines = prompt.split("\n")
                # find the "# Task" header
                for i, line in enumerate(lines):
                    if line.strip().lower() == "# task":
                        start = i + 1
                        break
                else:
                    return prompt
                # find the next header (line starting with "# ")
                for j in range(start, len(lines)):
                    if lines[j].startswith("# "):
                        end = j
                        break
                else:
                    end = len(lines)

                # splice in new_task (split into lines), preserving exactly the rest
                new_task_lines = new_task.split("\n")
                return "\n".join(lines[:start] + new_task_lines + lines[end:])

            tmp_prompts = [ rebuild(prompt, s) for s in new_sections ]

            # ---- DEBUG: dump out each new candidate’s Task section ----
            logger.debug(f"=== expand_candidates (prompt #{prompts.index(prompt)+1}) ===")
            ts_escaped = task_section.replace('\n','\\n')
            logger.debug(f"Seed Task section: {ts_escaped}")
            logger.debug(f"Produced {len(tmp_prompts)} candidates.")
            for i, cand in enumerate(tmp_prompts, start=1):
                sects = utils.parse_sectioned_prompt(cand)
                ts    = sects.get("task","").replace("\n","\\n")
                logger.debug(f"[{i:2d}] TASK: {ts}")
            logger.debug("=== end expand debug ===")

            # optional filtering (the original much longer code but with BEM it woul me the call much more costly)
            if len(tmp_prompts) > self.opt["max_expansion_factor"]:
                tmp_prompts = random.sample(tmp_prompts, self.opt["max_expansion_factor"])

            new_prompts += tmp_prompts

        new_prompts += prompts
        return list(set(new_prompts))

    def score_candidates(self, prompts, task, predictor, train_exs):
        # if len(prompts) == 1:
        #     return [1.0]
        if len(prompts) == 1:
           # instead of perfect 1.0, actually score on a small sample
           sample = random.sample(train_exs, self.opt["samples_per_eval"])
           return [ self.scorer(sample, prompts[0]) ]

        return self.evaluator_fn(
            prompts,
            train_exs,
            task,
            predictor,
            scorer=self.scorer,
            rounds=self.opt["eval_rounds"],
            num_prompts_per_round=self.opt["eval_prompts_per_round"],
            samples_per_eval=self.opt["samples_per_eval"],
            max_threads=self.max_threads,
        )
