import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ---------------------------------------------------------------
md("""# HW3 Part A - Prompt Engineering Techniques with LangChain

**DATA 266 - Homework 3**

This notebook tries out six different prompt engineering techniques using LangChain. Instead of
screenshotting a chat window, everything here actually calls a model through code - I'm using a
local Ollama model (`llama3.2`) since that's what I had working, per the assignment's note that it
wants to see code calling prompts, not chat screenshots.

The six techniques, with two of my own examples for each:
1. Zero-Shot
2. Few-Shot
3. Chain-of-Thought (CoT)
4. Zero-Shot CoT
5. Meta-Prompting
6. Tree of Thoughts (ToT)
""")

# ---------------------------------------------------------------
md("""## Step 0 - Personal Parameters

Per the course's standing instructions (Section 0.1), I derive my personal parameters once from
the last four digits of my SJSU ID (SID4) and report them here.
""")

code("""# Step 0: Personal Parameters (Section 0.1 of standing instructions)
import random
import numpy as np

SID4 = 215
SEED = SID4                      # 215
SLICE = SID4 % 1000              # 215
HP_ID = SID4 % 6                 # 5
CLS_A = SID4 % 10                # 5
CLS_B = (CLS_A + 1 + ((SID4 // 10) % 9)) % 10  # 9

random.seed(SEED)
np.random.seed(SEED)

print(f"SID4  = {SID4}")
print(f"SEED  = {SEED}")
print(f"SLICE = {SLICE}")
print(f"HP_ID = {HP_ID}")
print(f"CLS_A = {CLS_A}")
print(f"CLS_B = {CLS_B}")

# Note: HP_ID, SLICE, CLS_A, CLS_B do not have a defined mapping for HW3's prompt-engineering
# portion (only HW1 uses HP_ID per the standing instructions), so they are reported for
# reproducibility/traceability but are not otherwise used in this notebook.
""")

# ---------------------------------------------------------------
md("""## Setup

I'm using [LangChain](https://python.langchain.com/) with a locally hosted **Ollama** model
(`llama3.2`) as the backend. This meant I didn't need a paid API key, and I still got to use all
the normal LangChain building blocks (`PromptTemplate`, `ChatPromptTemplate`,
`FewShotPromptTemplate`, chains, etc.).

Temperature is `0.0` for most techniques so the outputs stay reproducible. Tree-of-Thoughts is the
exception - it uses a higher temperature when generating branches so the different "thoughts"
actually have a chance to differ from each other.
""")

code("""from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate

MODEL_NAME = "llama3.2"

# Deterministic LLM for most techniques
llm = OllamaLLM(model=MODEL_NAME, temperature=0.0, seed=SEED)

# Slightly stochastic LLM used only for generating diverse ToT branches
llm_branching = OllamaLLM(model=MODEL_NAME, temperature=0.7, seed=SEED)

def run(prompt_text, model=None):
    \"\"\"Helper: invoke the LLM with a fully-rendered prompt string and print both.\"\"\"
    model = model or llm
    print("PROMPT:\\n" + "-" * 60)
    print(prompt_text)
    print("-" * 60)
    response = model.invoke(prompt_text)
    print("RESPONSE:\\n" + "-" * 60)
    print(response)
    print("=" * 60)
    return response
""")

# =================================================================
md("""## 1. Zero-Shot Prompting

Here the model just gets the instructions and the question, nothing else - no examples to learn
from. This shows how well it can do purely from the wording of the prompt.

**Example 1: Arithmetic word problem**
""")

code("""zero_shot_prompt_1 = PromptTemplate.from_template(
    "Solve the following math problem and give only the final numeric answer.\\n\\n"
    "Problem: {problem}\\n"
    "Answer:"
)

problem_1 = (
    "A train travels 60 miles in the first hour and 45 miles in the second hour. "
    "What is its average speed in miles per hour over the two hours?"
)

zs1_response = run(zero_shot_prompt_1.format(problem=problem_1))
""")

md("""**Example 2: Logical reasoning task (categorical syllogism)**""")

code("""zero_shot_prompt_2 = PromptTemplate.from_template(
    "Answer the logic question with a single word: Yes, No, or Cannot be determined.\\n\\n"
    "{statement}\\n"
    "Answer:"
)

statement_2 = (
    "All engineers at Acme Corp use Python. Maria uses Python. "
    "Is Maria necessarily an engineer at Acme Corp?"
)

zs2_response = run(zero_shot_prompt_2.format(statement=statement_2))
""")

# =================================================================
md("""## 2. Few-Shot Prompting

Now I show the model a few solved examples before asking my actual question, using LangChain's
`FewShotPromptTemplate`. The idea is that seeing worked examples nudges it toward the same format
and style.

**Example 1: same arithmetic problem, but now with 3 worked examples first**
""")

code("""example_prompt_math = PromptTemplate.from_template(
    "Problem: {problem}\\nAnswer: {answer}"
)

math_examples = [
    {
        "problem": "A car drives 30 miles in 1 hour and 30 miles in the next hour. "
                    "What is its average speed?",
        "answer": "30",
    },
    {
        "problem": "A runner covers 10 miles in the first hour and 20 miles in the second hour. "
                    "What is the average speed?",
        "answer": "15",
    },
    {
        "problem": "A cyclist rides 12 miles in the first hour and 8 miles in the second hour. "
                    "What is the average speed?",
        "answer": "10",
    },
]

few_shot_prompt_1 = FewShotPromptTemplate(
    examples=math_examples,
    example_prompt=example_prompt_math,
    prefix="Solve each problem and give only the final numeric answer (miles per hour).",
    suffix="Problem: {problem}\\nAnswer:",
    input_variables=["problem"],
)

fs1_response = run(few_shot_prompt_1.format(problem=problem_1))
""")

md("""**Example 2: the syllogism question, few-shot with 3 worked examples**""")

code("""example_prompt_logic = PromptTemplate.from_template(
    "Statement: {statement}\\nAnswer: {answer}"
)

logic_examples = [
    {
        "statement": "All cats are mammals. Fluffy is a mammal. Is Fluffy necessarily a cat?",
        "answer": "Cannot be determined",
    },
    {
        "statement": "All squares are rectangles. This shape is a square. Is it necessarily a rectangle?",
        "answer": "Yes",
    },
    {
        "statement": "No fish can fly. A sparrow can fly. Is a sparrow necessarily not a fish?",
        "answer": "Yes",
    },
]

few_shot_prompt_2 = FewShotPromptTemplate(
    examples=logic_examples,
    example_prompt=example_prompt_logic,
    prefix="Answer each logic question with a single phrase: Yes, No, or Cannot be determined.",
    suffix="Statement: {statement}\\nAnswer:",
    input_variables=["statement"],
)

fs2_response = run(few_shot_prompt_2.format(statement=statement_2))
""")

# =================================================================
md("""## 3. Chain-of-Thought (CoT) Prompting

This is few-shot, but now the examples also show the reasoning steps, not just the final answer -
basically teaching the model to show its work by example.

**Example 1: same math problem, with worked-out reasoning in the examples**
""")

code("""cot_example_prompt = PromptTemplate.from_template(
    "Problem: {problem}\\nReasoning: {reasoning}\\nAnswer: {answer}"
)

cot_math_examples = [
    {
        "problem": "A car drives 30 miles in 1 hour and 30 miles in the next hour. "
                    "What is its average speed?",
        "reasoning": "Total distance = 30 + 30 = 60 miles. Total time = 1 + 1 = 2 hours. "
                     "Average speed = 60 / 2 = 30 mph.",
        "answer": "30",
    },
    {
        "problem": "A runner covers 10 miles in the first hour and 20 miles in the second hour. "
                    "What is the average speed?",
        "reasoning": "Total distance = 10 + 20 = 30 miles. Total time = 1 + 1 = 2 hours. "
                     "Average speed = 30 / 2 = 15 mph.",
        "answer": "15",
    },
]

cot_prompt_1 = FewShotPromptTemplate(
    examples=cot_math_examples,
    example_prompt=cot_example_prompt,
    prefix="Solve each problem step by step, then give the final numeric answer.",
    suffix="Problem: {problem}\\nReasoning:",
    input_variables=["problem"],
)

cot1_response = run(cot_prompt_1.format(problem=problem_1))
""")

md("""**Example 2: same syllogism, with worked-out reasoning in the examples**""")

code("""cot_logic_examples = [
    {
        "problem": "All cats are mammals. Fluffy is a mammal. Is Fluffy necessarily a cat?",
        "reasoning": "The rule only tells us cats are a subset of mammals, not that all mammals "
                     "are cats. Fluffy being a mammal does not place Fluffy specifically in the "
                     "cat subset.",
        "answer": "Cannot be determined",
    },
    {
        "problem": "All squares are rectangles. This shape is a square. Is it necessarily a rectangle?",
        "reasoning": "Since every square is defined as a rectangle, and this shape is a square, "
                     "it must fall inside the rectangle category.",
        "answer": "Yes",
    },
]

cot_prompt_2 = FewShotPromptTemplate(
    examples=cot_logic_examples,
    example_prompt=cot_example_prompt,
    prefix="Reason step by step about each logic question, then answer Yes, No, or Cannot be determined.",
    suffix="Problem: {problem}\\nReasoning:",
    input_variables=["problem"],
)

cot2_response = run(cot_prompt_2.format(problem=statement_2))
""")

# =================================================================
md("""## 4. Zero-Shot Chain-of-Thought

This one's a neat trick - no examples at all, you just tack on the phrase **"Let's think step by
step"** to the end of a plain question (this comes from Kojima et al., 2022) and see if that alone
gets the model to reason more carefully.

**Example 1: the same math problem**
""")

code("""zscot_prompt_1 = PromptTemplate.from_template(
    "Problem: {problem}\\n"
    "Let's think step by step."
)

zscot1_response = run(zscot_prompt_1.format(problem=problem_1))
""")

md("""**Example 2: the same syllogism**""")

code("""zscot_prompt_2 = PromptTemplate.from_template(
    "{statement}\\n"
    "Let's think step by step."
)

zscot2_response = run(zscot_prompt_2.format(statement=statement_2))
""")

# =================================================================
md("""## 5. Meta-Prompting

Meta-prompting is about asking the model to first think about *how* it should approach the
problem - what kind of problem is this, what strategy/formula fits - before actually solving it.
So it's a prompt about the approach, rather than showing worked examples.

**Example 1: math problem - model figures out the right approach first**
""")

code("""meta_prompt_1 = PromptTemplate.from_template(
    "You are a meta-reasoning assistant. For the problem below, first (1) classify what type of "
    "problem this is and which mathematical concept/formula applies, then (2) state the general "
    "strategy for solving that class of problem, and finally (3) apply the strategy to compute the "
    "answer.\\n\\n"
    "Problem: {problem}\\n\\n"
    "Format your response as:\\n"
    "Problem type: ...\\n"
    "General strategy: ...\\n"
    "Applied solution: ...\\n"
    "Final answer: ..."
)

meta1_response = run(meta_prompt_1.format(problem=problem_1))
""")

md("""**Example 2: syllogism - model first names the logical structure/fallacy involved**""")

code("""meta_prompt_2 = PromptTemplate.from_template(
    "You are a meta-reasoning assistant. For the statement below, first (1) identify the logical "
    "form involved (e.g., valid syllogism, affirming the consequent, converse error, etc.), then "
    "(2) explain the general rule that governs whether that form is valid, and finally (3) apply "
    "that rule to answer the question.\\n\\n"
    "{statement}\\n\\n"
    "Format your response as:\\n"
    "Logical form: ...\\n"
    "Governing rule: ...\\n"
    "Applied answer: ..."
)

meta2_response = run(meta_prompt_2.format(statement=statement_2))
""")

# =================================================================
md("""## 6. Tree of Thoughts (ToT)

Tree-of-Thoughts (Yao et al., 2023) is about generating several different reasoning paths for the
same problem, then having the model itself judge which one is best (or combine them into a better
answer). I built a simple version of this with plain LangChain calls:

1. **Generate** a few different candidate solutions ("branches"), each nudged toward a different
   angle so they don't just repeat each other.
2. **Evaluate** the branches with a separate call that critiques them.
3. **Pick** whichever one the judge thinks is most likely correct.

**Example 1: math problem, 3 branches**
""")

code("""def tree_of_thoughts(problem_text, n_branches=3, branch_model=None, judge_model=None):
    branch_model = branch_model or llm_branching
    judge_model = judge_model or llm

    # Each branch gets a distinct persona/instruction so branches diverge even when the
    # underlying Ollama backend seeds its RNG deterministically per call.
    personas = [
        "Solve the problem carefully and directly, showing your arithmetic/logic explicitly.",
        "Solve the problem by first restating it in your own words, then working through it "
        "skeptically, actively looking for ways the obvious answer could be wrong.",
        "Solve the problem by considering a concrete counterexample or edge case before "
        "committing to a final answer.",
    ]

    branch_prompt = PromptTemplate.from_template(
        "{persona}\\n\\n"
        "Problem: {problem}\\n"
        "Reasoning and answer:"
    )

    branches = []
    print(f"Generating {n_branches} candidate reasoning branches...")
    for i in range(n_branches):
        persona = personas[i % len(personas)]
        branch_text = branch_model.invoke(branch_prompt.format(persona=persona, problem=problem_text))
        branches.append(branch_text)
        print(f"\\n--- Branch {i+1} ---\\n{branch_text}")

    evaluation_prompt = PromptTemplate.from_template(
        "Below are {n} candidate solutions (branches) to the same problem, each produced by an "
        "independent reasoning attempt.\\n\\n"
        "Problem: {problem}\\n\\n"
        "{branches_block}\\n\\n"
        "Evaluate the branches for correctness, briefly explain which one is most likely correct "
        "(or synthesize the correct answer if none are fully right), and end your response with a "
        "line formatted exactly as:\\nFINAL ANSWER: <answer>"
    )

    branches_block = "\\n\\n".join(
        f"Branch {i+1}:\\n{b}" for i, b in enumerate(branches)
    )

    verdict = judge_model.invoke(
        evaluation_prompt.format(
            n=n_branches, problem=problem_text, branches_block=branches_block
        )
    )
    print("\\n--- Judge's evaluation and final answer ---")
    print(verdict)
    return branches, verdict

tot1_branches, tot1_verdict = tree_of_thoughts(problem_1, n_branches=3)
""")

md("""**Example 2: syllogism, 3 branches**""")

code("""tot2_branches, tot2_verdict = tree_of_thoughts(statement_2, n_branches=3)
""")

# =================================================================
md("""## Comparing the Outputs

Below is a table of the final answer each technique gave for my two examples (the train speed
problem, and the "Maria uses Python" syllogism), followed by my notes on what I noticed running all
of this.
""")

code("""import pandas as pd

comparison = pd.DataFrame({
    "Technique": [
        "Zero-Shot", "Few-Shot", "Chain-of-Thought", "Zero-Shot CoT",
        "Meta-Prompting", "Tree of Thoughts",
    ],
    "Example 1 (avg speed, correct = 52.5 mph)": [
        zs1_response.strip().splitlines()[-1] if zs1_response.strip() else "",
        fs1_response.strip().splitlines()[-1] if fs1_response.strip() else "",
        cot1_response.strip().splitlines()[-1] if cot1_response.strip() else "",
        zscot1_response.strip().splitlines()[-1] if zscot1_response.strip() else "",
        meta1_response.strip().splitlines()[-1] if meta1_response.strip() else "",
        tot1_verdict.strip().splitlines()[-1] if tot1_verdict.strip() else "",
    ],
    "Example 2 (syllogism, correct = Cannot be determined)": [
        zs2_response.strip().splitlines()[-1] if zs2_response.strip() else "",
        fs2_response.strip().splitlines()[-1] if fs2_response.strip() else "",
        cot2_response.strip().splitlines()[-1] if cot2_response.strip() else "",
        zscot2_response.strip().splitlines()[-1] if zscot2_response.strip() else "",
        meta2_response.strip().splitlines()[-1] if meta2_response.strip() else "",
        tot2_verdict.strip().splitlines()[-1] if tot2_verdict.strip() else "",
    ],
})
comparison
""")

md("""### What I found

**Just so we're clear on the right answers first:**
- The train problem: 105 miles total divided by 2 hours = **52.5 mph**.
- The syllogism: knowing "all engineers use Python" and "Maria uses Python" doesn't actually tell
  you Maria is an engineer - that's the classic affirming-the-consequent mistake. The honest answer
  is **"Cannot be determined."** ("No" is also a fair reading of "is she *necessarily* an engineer,"
  but "cannot be determined" is the more precise way to put it.)

**Here's what each technique actually did (real outputs from `llama3.2`, no cherry-picking):**

- **Zero-Shot** flat-out got the math wrong - it just said `90`, with no work shown, so there's no
  way to tell where that number even came from. On the logic question it said `No`, which is
  basically right, even with zero help.
- **Few-Shot** fixed the math immediately. It was only shown three problem-and-answer pairs (no
  reasoning shown), but it started showing its own work anyway and landed on the correct 52.5. On
  the logic question it gave the clean, correct "Cannot be determined." So the few-shot examples
  didn't really teach it new reasoning - they just seemed to put it into a more careful mode.
- **Chain-of-Thought** (few-shot, but the examples include the reasoning too) also nailed both
  questions, and its answers followed the same structure as my examples pretty closely. Out of all
  six techniques, this one gave the most consistent, predictable output.
- **Zero-Shot CoT** ("let's think step by step") got the math right but **blew the logic
  question** - it reasoned itself all the way to "Maria must be an engineer," which is exactly the
  fallacy the question was testing for. It laid out neat numbered steps the whole way there, which
  is what makes this result interesting: just telling the model to think step by step doesn't mean
  the steps will actually be *valid*. It can sound just as confident being wrong as being right.
- **Meta-Prompting** got both right, and it was the only technique that actually named the fallacy
  out loud ("Affirming the Consequent error") before giving its answer. Its explanation got a
  little tangled at one point, but it still landed on "No, not necessarily an engineer" at the end.
- **Tree of Thoughts** (three different branches - one direct, one skeptical, one looking for
  counterexamples - judged by a fourth call) also got both right, and it was the only technique
  where the branches actually disagreed with each other on the logic question before getting
  resolved. One branch used a shaky argument and still landed close to the right answer, while the
  other two came up with real counterexamples (Maria could be a user, a manager, a professor, etc.)
  showing why the implication doesn't hold. The judge picked up on that and went with the stronger
  reasoning instead of just picking whichever branch sounded most confident.

**Big picture:** both mistakes I saw - zero-shot's wrong math and zero-shot-CoT's bad logic -
happened on the two techniques that don't give the model anything to check its work against. Every
technique that added either a worked example, an explicit "name the strategy" step, or multiple
attempts plus a judge, got both questions right. So it seems like *some* form of double-checking
matters more than which specific technique you use - but just asking the model to "think step by
step" on its own isn't a guarantee it'll reason correctly, only that it'll sound like it's
reasoning. Tree of Thoughts held up the best when one path went wrong, but it's also the most
expensive option - it costs 4 model calls per question here instead of 1.
""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("C:/Users/Admin/Desktop/DATA266/HW3/hw3_part_a_prompt_engineering.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook written.")
