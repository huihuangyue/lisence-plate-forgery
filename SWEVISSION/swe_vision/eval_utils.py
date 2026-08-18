"""
Evaluation utilities for BabyVision / VLM benchmarks.
"""

import regex


LLM_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Process (internal reasoning):**

1. Read and understand the Question, Ground Truth Answer, and Model Output.
2. Ignore small wording differences, formatting, or synonyms.
3. If all factual content matches, conclude `1`. Otherwise, conclude `0`.

**Important:**

* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Output format:**

True

or

False

**Input:**

Question: {question},
Ground Truth Answer: {groundtruth},
Model Output: {modeloutput}
"""


def format_choices(choices):
    """Format multiple choice options as (A), (B), (C), etc."""
    if len(choices) == 0:
        return ""
    formatted = ""
    for idx, choice in enumerate(choices):
        formatted += f"({chr(65 + idx)}) {choice}\n"
    return formatted.strip()


def extract_boxed_answer(text):
    """
    Extract the content from the last \\boxed{} pattern.

    Also supports alternative format: <|begin_of_box|>...<|end_of_box|>

    Returns None if no pattern found.
    """
    if text is None:
        return None

    pattern = r'\\boxed\{((?:[^{}]|{(?:[^{}]|{.*})*})*)\}'
    matches = regex.findall(pattern, text)

    if matches:
        return matches[-1]

    pattern_alt = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
    matches_alt = regex.findall(pattern_alt, text)
    if matches_alt:
        return matches_alt[-1].strip()

    return None
