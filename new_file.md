# Evaluation System Walkthrough
## 30-Minute Principal Presentation Guide

# Table of Contents

1. [Evaluation System Walkthrough](#evaluation-system-walkthrough)  
   1.1. [30-Minute Principal Presentation Guide](#30-minute-principal-presentation-guide)  
   1.2. [The Big Picture](#the-big-picture)

2. [Part 3: The How (10 minutes)](#part-3-the-how-10-minutes)  
   2.1. [Two Types of Evaluators](#two-types-of-evaluators)  
      - [Programmatic Evaluators (Fast & Deterministic)](#1-programmatic-evaluators-fast--deterministic)  
      - [LLM Judges (Nuanced & Expensive)](#2-llm-judges-nuanced--expensive)  
   2.2. [The Calibration Story (The Secret Sauce)](#the-calibration-story-the-secret-sauce)

3. [Auxiliary: Module Deep Dives](#auxiliary-module-deep-dives)  
   3.1. [`programmaticpy` — Rule-Based Evaluators](#programmaticpy---rule-based-evaluators)  
      - [B2 (Tentative Language) Evaluator Workflow](#how-a-programmatic-evaluator-works-b2-tentative-language)  
      - [C5 (Tone/Style) Evaluator Workflow](#how-another-one-works-c5-tonestyle)  
   3.2. [`judgespy` — LLM Tribunal Architecture](#judgespy---llm-tribunal-architecture)
   3.3. [`run_evalpy` — The Main Orchestrator](#runevalpy---the-main-orchestrator)  
   3.4. [`programmatic_calibrationpy` — Measuring Evaluator Accuracy](#programmatic_calibrationpy---measuring-evaluator-accuracy)  
   3.5. [`judge_calibrationpy` — Measuring Judge Accuracy](#judge_calibrationpy---measuring-judge-accuracy)  
   3.6. [`datasetpy` — The Data Foundation](#datasetpy---the-data-foundation)  
   3.7. [`recall_taskpy` — InspectAI Integration](#recall_taskpy---inspectai-integration)  
   3.8. [`turn_by_turnpy` — Generative Mode Support](#turn_by_turnpy---generative-mode-support)

4. [Quick Reference: Running Common Tasks](#quick-reference-running-common-tasks)

5. [Buffer / Q&A (5 minutes)](#buffer--qa-5-minutes)  
   5.1. [Anticipated Questions](#anticipated-questions)

6. [Part 4: The Iterative Workflow](#part-4-the-iterative-workflow)  
   6.1. [How to Actually Use This System](#how-to-actually-use-this-system)  
   6.2. [Workflow 1: No Calibration Needed](#workflow-1-no-calibration-needed)  
      - [Step-by-Step Prompt Iteration](#step-1-make-your-prompt-change)  
      - [Validation Without Overfitting](#step-5-validate-on-test-split)  
      - [Detecting and Handling Regressions](#step-6-handle-regressions)  
   6.3. [Workflow 2: With Calibration](#workflow-2-with-calibration)  
      - [When Calibration Is Required](#when-you-need-this-workflow)  
      - [Editing Evaluators](#step-1-edit-the-evaluator-not-the-ai-prompt)  
      - [Running Calibration](#step-2-run-calibration)  
      - [Error Analysis and Threshold Tuning](#step-3-analyze-the-errors)  
      - [Synthetic Data Expansion](#step-5-need-more-data-generate-it)  
      - [Saving & Using Calibration Output](#step-6-save-calibration-results)  
   6.4. [Calibration + Iteration: How They Interlock](#the-full-picture-calibration--iteration)  
   6.5. [Directory Structure & Artifact Locations](#where-files-live)

7. [Quick Decision Tree](#quick-decision-tree)


---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  DynamoDB ─→ dataset.py ─→ Normalized Items                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION LAYER                             │
│  run_eval.py orchestrates:                                       │
│    ├── programmatic.py (fast, deterministic)                    │
│    └── judges.py (nuanced, expensive)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CALIBRATION LAYER                             │
│  programmatic_calibration.py + judge_calibration.py             │
│  ─→ TPR/TNR metrics ─→ calibration_summary.json                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                │
│  metrics.py (Rogan-Gladen correction)                           │
│  visualize.py (HTML reports)                                    │
│  ─→ Corrected failure rates with confidence intervals           │
└─────────────────────────────────────────────────────────────────┘
```

The continuous loop: conversations come in → evaluators check them → calibration measures evaluator quality → results feed back into better evaluations.

---

# Part 3: The How (10 minutes)

## Two Types of Evaluators

### 1. Programmatic Evaluators (Fast & Deterministic)

These are code-based rules using spaCy for NLP pattern matching. They're cheap, fast, and reproducible.

**How they work:** We define patterns that indicate specific behaviors. For example, the B2 evaluator looks for "hedge words" in customer language:

- "maybe I could come in" → hedging detected
- "I'll be there Tuesday at 10" → no hedging

We calculate a "hedge ratio" - how many hedge-type tokens divided by total tokens. If it's above our threshold (0.30), we flag it as tentative language.

### 2. LLM Judges (Nuanced & Expensive)

For semantic judgments that rules can't capture, we use a "tribunal" architecture:

**Layer 1:** Two analyst LLMs examine the conversation from different angles
- MultiTurnAnalyst focuses on the AI's internal reasoning
- ConversationAnalyst focuses on the conversation flow

**Layer 2:** A MapUnit combines both analyses into a debate transcript

**Layer 3:** A MetaJudge reads both perspectives and makes a final call

**Layer 4:** A VerdictExtractor pulls out structured JSON from the verdict

This costs more tokens but catches subtle issues - like when consent was technically given but the context makes it ambiguous.

## The Calibration Story (The Secret Sauce)

Here's where it gets interesting. Neither evaluator type is perfect:
- Programmatic evaluators miss edge cases
- LLM judges hallucinate sometimes

So we measure *how wrong* each evaluator is:

1. **Start with labeled data** - Humans mark conversations as pass/fail
2. **Run our evaluators** - See how they compare to human labels
3. **Calculate TPR/TNR** - True Positive Rate (catching real failures) and True Negative Rate (not flagging good conversations)

Now here's the key move: **Rogan-Gladen correction**.

When we evaluate *unlabeled* production data, we can't just trust raw evaluator outputs. If our B3 evaluator has 85% TPR and 90% TNR, and it flags 100 conversations as failures, we know statistically that:
- Some of those 100 are false positives
- Some real failures weren't flagged

Rogan-Gladen math corrects for this, giving us a more accurate *true* failure rate estimate with confidence intervals.

**The 4-step recipe:**
1. Calibrate evaluators on labeled data → get TPR/TNR
2. Run evaluators on unlabeled production data → get raw counts
3. Apply Rogan-Gladen correction → get corrected failure rates
4. Bootstrap for confidence intervals → know how sure we are

---

# Auxiliary: Module Deep Dives

## `programmatic.py` - Rule-Based Evaluators

This is where the deterministic logic lives. About 10,000 lines of pattern matching, threshold comparisons, and evidence collection.

### How a Programmatic Evaluator Works: B2 (Tentative Language)

The B2 evaluator detects when a customer uses tentative language ("maybe Thursday could work") but the AI treats it as firm commitment.

**Step 1: Extract customer text**
```python
# Find customer turn immediately before booking (skip short acknowledgments)
customer_turn_before_booking, customer_turn_index = _find_relevant_customer_turn_before(
    conv,
    found_booking_turn - 1
)
```

**Step 2: Run spaCy pattern matching**

We've defined "hedge patterns" in spaCy's Matcher:

```python
# "I think/guess" patterns (low certainty)
commitment_matcher.add("THINKING_PATTERN", [[
    {"LOWER": "i"},
    {"LEMMA": {"IN": ["think", "guess", "suppose"]}},
]])

# Modal verbs indicating ability/availability uncertainty (strong hedging)
commitment_matcher.add("MODAL_ABILITY", [
    [
        {"LOWER": {"IN": ["might", "could", "may", "can"]}},
        {"LOWER": "be"},
        {"OP": "*", "LENGTH": {"<=": 2}},
        {"LOWER": {"IN": ["able", "available"]}},
        {"LOWER": "to", "OP": "?"},
        {"LEMMA": {"IN": ["come", "bring", "do", "make", "attend", "get", "schedule", "book", "visit"]}, "OP": "?"}
    ],
    [
        {"LOWER": {"IN": ["might", "could", "may", "can"]}},
        {"LEMMA": {"IN": ["do", "come", "bring", "make", "attend", "get", "have", "work"]}}
    ]
])
```

**Step 3: Calculate hedge ratio**

```python
# Build hedge-only pattern set (excluding no_hedging patterns)
hedge_only_patterns = set(strong_hedging + medium_hedging + weak_hedging)

# Track unique token indices in hedge patterns
hedged_token_indices = set()
for match_id, start, end in matches:
    rule_name = nlp.vocab.strings[match_id]
    if rule_name in hedge_only_patterns:
        hedged_token_indices.update(range(start, end))

# Calculate hedge_hits from unique hedged token indices
hedge_hits = len(hedged_token_indices)

# Calculate token count (non-punctuation, non-space)
token_count = len([t for t in doc if not t.is_punct and not t.is_space])
hedge_ratio = hedge_hits / token_count if token_count > 0 else 0.0
```

**Step 4: Compare to threshold**

```python
# Threshold of 0.30 means 30% of tokens must be hedge words
B2_HEDGE_RATIO_THRESHOLD = 0.30

# Use combined signals: is_tentative flag, hedge_ratio threshold, OR modal dependency
is_tentative = (
    commitment_analysis['is_tentative'] or 
    commitment_analysis['hedge_ratio'] > B2_HEDGE_RATIO_THRESHOLD or
    (modal_dependency_analysis['has_modal_dependency'] and 
     modal_dependency_analysis['modal_strength'] <= MODAL_DEPENDENCY_THRESHOLD)
)
```

- Each evaluator packages spaCy-derived text spans into the evidence payload so downstream analysis can trace why a rule triggered, while fallback logic uses regex patterns if NLP resources are unavailable.
- Example (from `_init_spacy_resources`):
    - Pattern matcher:
    - Component: spaCy `Matcher`, which uses token attribute dictionaries (`LOWER`, `LEMMA`, etc.) and evaluates them in sequence with optional or repeated tokens.
    - How it works: `commitment_matcher.add("CHECKING_PATTERN", [{"LOWER": {"IN": ["let","need","have"]}}, {"LOWER": "me", "OP": "?"}, {"LOWER": "to", "OP": "?"}, {"LEMMA": "check"}])` matches "let me check the calendar" because spaCy normalizes "checking" to "check" and optional tokens handle "me" and "to".
    - Phrase matcher:
    - Component: spaCy `PhraseMatcher`, which creates hash lookups for pre-built phrase documents and performs exact, case-normalized span matching efficiently.
    - How it works: `hedge_phrase_matcher.add("HEDGE_PHRASE", [nlp.make_doc("let me see if"), ...])` matches multi-word hedges like "let me see if that works" when the span matches the stored phrase.
    - Key token attributes:
    - `LOWER`: lowercase version of the token, enabling case-insensitive matching.
    - `LEMMA`: dictionary form of the token, so "checking" and "checked" both map to `check`.
    - `IN`: match anything in this little list. It’s basically spaCy’s way of letting you check if a token’s attribute is one of a few options, without repeating yourself.

**Step 5: Check AI response**

If customer was tentative but AI booked anyway without seeking confirmation, that's a B2 failure.

### How Another One Works: C5 (Tone/Style)

C5 checks if the AI's response has appropriate professional tone.

**What it checks:**
- Message length (>600 chars is too long for SMS)
- Jargon detection ("utilize", "facilitate", "notwithstanding")
- Excessive punctuation (multiple !!! or ???)
- ALL CAPS words (except known acronyms)
- Blacklisted phrases
- Sentiment analysis (negative polarity < -0.3 is problematic)

**The fuzzy part - good tone indicators:**
```python
# LESS RESTRICTIVE: Check for good tone indicators
good_tone_patterns = [
    r'\bplease\b',
    r'\bthank\s+you\b',
    r'\bhappy\s+to\b',
    r'\bI\'d\s+be\s+(?:happy|glad)\b',
    r'\blet\s+me\s+help\b',
    r'\bI\'m\s+here\s+to\s+help\b',
]

has_good_tone = any(
    re.search(pattern, ai_message, re.IGNORECASE)
    for pattern in good_tone_patterns
)
```

If there are minor issues but the AI used friendly language, it still passes. The evaluation is trying to catch *problematic* tone, not enforce perfection.

---

## `judges.py` - LLM Tribunal Architecture

When pattern matching isn't enough, we bring in the tribunal. This is expensive but catches subtle semantic issues.

### The C3 Tribunal: Escalation Detection

C3 checks: when a customer triggers an escalation (wants a manager, asks about pricing, mentions legal action), does the AI handle it properly?

**The architecture:**

```
Layer 1: Two Analysts (run in parallel via outer='broadcast')
├── MultiTurnAnalyst: Examines AI's internal reasoning
└── ConversationAnalyst: Examines conversation flow

Layer 2: MapUnit
└── Combines both analyses into debate transcript via _capture_and_log_debate

Layer 3: MetaJudge  
└── Reads debate, makes final pass/fail verdict

Layer 4: VerdictExtractor (Haiku)
└── Parses explanation into structured JSON
```

**Why two analysts?** Different perspectives catch different things. One might notice the AI's reasoning was flawed. The other might notice the conversation context made escalation necessary. The meta-judge synthesizes both views.

**The prompts are detailed.** The MultiTurnAnalyst prompt includes:
- Business rules the AI should follow
- Examples of proper escalation handling
- Specific patterns to look for (pricing requests, complaints, legal mentions)
- Instructions for structured output

**What the meta-judge sees:**
```
ANALYST A's FINDINGS:
[Evidence about AI reasoning, specific quotes, rule violations]

ANALYST B's FINDINGS:  
[Evidence about conversation flow, customer signals, response appropriateness]

Your task: Synthesize both analyses and determine if escalation 
handling meets business standards. Provide structured verdict.
```

**Cost tracking:** Each tribunal call tracks:
- Sonnet tokens (analysts + meta-judge)
- Haiku tokens (extractor)
- Estimated cost (~$0.03-0.05 per conversation)

This gets logged so you know exactly what the evaluation budget is.

---

## `run_eval.py` - The Main Orchestrator

Think of this as the command center. You tell it what dataset to use, which failure modes to check, and it coordinates everything.

**What happens when you run it:**

1. **Load the data** - Either from DynamoDB (live) or a JSONL snapshot
2. **Set up the evaluators** - Load programmatic evaluators + LLM judges based on your config
3. **Process each conversation** - For each item:
   - Run the programmatic evaluator (fast)
   - Run the judge (if enabled)
   - Both must agree for a pass
4. **Aggregate results** - Compute per-mode metrics, apply calibration corrections
5. **Generate report** - HTML dashboard with charts and tables

The key command-line options:
```bash
# Run both code + judges on dev split
python run_eval.py --dataset dynamodb --modes B1 B2 B3 --split dev --viz --open

# Programmatic only (fast iteration)
python run_eval.py --dataset dynamodb --modes B1 --no-judge

# Judge only (semantic deep dive)  
python run_eval.py --dataset dynamodb --modes C3_judge --judge-only

# Parallel execution with custom worker count
python run_eval.py --dataset dynamodb --modes B1 B2 B3 --split dev --parallel --workers 16
```

Parallel execution is built in. With `--parallel --workers 16`, it processes multiple conversations simultaneously using a thread pool. There's also a nested pool for running programmatic + judge evaluators concurrently on each conversation.

---

## `programmatic_calibration.py` - Measuring Evaluator Accuracy

Before you trust an evaluator, you need to know how accurate it is. This script measures that.

**What it does:**

1. **Load labeled data** - Conversations with human-assigned pass/fail labels
2. **Run evaluators** - Get predictions for each conversation
3. **Compare to ground truth** - Build confusion matrix
4. **Calculate metrics** - TPR (sensitivity), TNR (specificity), accuracy

**The output:**
```
Mode: B3
Split: dev
TPR: 0.85 (85% of real failures detected)
TNR: 0.92 (92% of real passes correctly identified)
Accuracy: 0.88
Confusion Matrix:
  Predicted Pass  Predicted Fail
  Actual Pass    46              4
  Actual Fail     8             42
```

**Why TPR and TNR matter:**
- High TPR = you catch real problems
- High TNR = you don't cry wolf
- You want both, but there's usually a tradeoff

**The threshold tuning loop:**
1. Run calibration on dev set
2. See TPR/TNR numbers
3. Adjust threshold (e.g., B2_HEDGE_RATIO_THRESHOLD)
4. Re-run calibration
5. Repeat until balanced

**Usage examples from the docstring:**
```bash
# Basic single-mode calibration on dev split
python programmatic_calibration.py --modes C3 --split dev --output c3_dev_cal

# Multi-mode calibration on test split
python programmatic_calibration.py --modes A1 A2 A3 --split test --output a_modes_test

# Primary-only filtering for clean calibration
python programmatic_calibration.py --modes B1 --split dev --output b1_primary --primary-only

# Sequential execution for debugging
python programmatic_calibration.py --modes C3 --split dev --output c3_debug --workers 1 --verbose
```

---

## `judge_calibration.py` - Measuring Judge Accuracy

Same idea as programmatic calibration, but for LLM judges.

**Key differences:**
- Much slower (API calls instead of local code)
- More expensive (tokens cost money)
- Bootstrap parallelism for confidence intervals

**The workflow:**
```bash
python judge_calibration.py --judge C3_judge --split dev --verbose --bootstrap-workers 8
```

This:
1. Loads dev set conversations with ground truth labels
2. Runs the C3 tribunal judge on each
3. Compares verdicts to human labels
4. Calculates TPR/TNR with bootstrap confidence intervals

**Output gets saved to `calibration_summary.json`** which `run_eval.py` loads to apply Rogan-Gladen correction during evaluation.

**Usage examples from the docstring:**
```bash
# Recovery mode (default) - evaluates test data conversations
python judge_calibration.py --judge C3_judge --train-size 20 --dev-size 10 --output cal_run_1 --verbose

# Generative mode - evaluates model-generated responses
python judge_calibration.py --judge B_judge --output b_tribunal_cal --eval-mode generative

# Filter to primary failure mode only (recommended for clean calibration)
python judge_calibration.py --judge C3_judge --output c3_primary_only --primary-only --verbose --override

# Bootstrap parallelization control
python judge_calibration.py --judge C3_judge --dev-size 50 --output c3_cal --bootstrap-workers 8
```

---

## `dataset.py` - The Data Foundation

Everything flows from how we load and format data. This module handles it.

**The DynamoDB schema:**

Each conversation fans out into multiple items:
- **TRACE item:** The actual conversation with all fields
- **MODE items:** Index by failure mode (A1, B2, etc.)
- **CATEGORY items:** Index by failure category
- **SPLIT items:** Index by train/dev/test

This "fan-out" pattern enables efficient queries: "Give me all B3 conversations from dev split" is fast.

**Key function: `trace_to_inspect_format`**

Converts raw DynamoDB data into the format evaluators expect:
```python
# Build input for model - structure must match curated dataset format
input_data = {
    "conversation": normalized_conversation,
    **mock_context  # Include all mock customer/vehicle/dealership/recall data
}

# Extract mode-specific rubric fields for each targeted mode
mode_rubrics = {}
for mode in targets:
    mode_rubrics[mode] = {
        'rubric': trace.get(f'{mode}_rubric', ''),
        'pass_if': trace.get(f'{mode}_pass_if', ''),
        'fail_if': trace.get(f'{mode}_fail_if', ''),
        'boundary_notes': trace.get(f'{mode}_boundary_notes', ''),
        'failure_mode_definition': trace.get(f'{mode}_failure_mode_definition', '')
    }

# Preserve mode-specific binary column values for ground truth extraction
# IMPORTANT: mode_specific_labels ALWAYS use BASE MODE keys (e.g., "C3", not "C3_tribunal")
mode_specific_labels = {}
for col in mode_columns:
    if col in trace:
        mode_code = col.split('_')[0]  # Extract base mode from column name
        if mode_code in requested_modes:
            binary_value = trace[col]
            if binary_value in (0, 1):
                mode_specific_labels[mode_code] = int(binary_value)
```

**The sampling helpers:**

When you run `--sample 30 --modes B1 B2`, it needs to:
1. Get conversations tagged with B1 or B2
2. Balance between them (stratified sampling option)
3. Return the right number

Functions like `get_traces_by_mode()` and `get_traces_without_mode()` (for negative examples) handle this.

---

## `recall_task.py` - InspectAI Integration

This bridges our evaluation system to InspectAI's framework (if you're using that).

**What it provides:**
- `PROMPT_TEMPLATE` - How to format conversations for the model
- `build_prompt()` - Constructs the actual prompt string
- `recall_code_scorer` - Wraps our programmatic evaluators
- `recall_judge_scorer` - Wraps our LLM judges

**Why it exists:** InspectAI has its own conventions for tasks, datasets, and scorers. This module translates our format to theirs, so you can run evaluations through either interface.

---

## `turn_by_turn.py` - Generative Mode Support

In "recovery mode," we replay logged conversations and check for violations.

In "generative mode," we only have the customer side - then call the model to generate fresh AI responses.

**What this module does:**
1. Takes a partial conversation (customer messages only)
2. Constructs prompts for each turn
3. Calls the model to generate AI response
4. Stitches responses back into full conversation
5. Hands complete conversation to evaluators

**The tricky parts:**
- Token accounting (track input/output tokens)
- Stop conditions (know when conversation should end)
- Message ordering (maintain proper turn sequence)
- System prompt management (model-specific handling)

This enables testing how a model *would* behave, not just how it *did* behave in historical data.

---

## Quick Reference: Running Common Tasks

**Iterate on a programmatic evaluator:**
```bash
python run_eval.py --dataset dynamodb --modes B3 --split dev --no-judge --viz --open
```

**Test a tribunal judge:**
```bash
python run_eval.py --dataset dynamodb --modes C3_judge --split dev --judge-only --viz
```

**Calibrate programmatic evaluators:**
```bash
python programmatic_calibration.py --modes B1 B2 B3 --split dev
```

**Calibrate tribunal judges:**
```bash
python judge_calibration.py --judge B_judge --split dev --bootstrap-workers 8
```

**Full evaluation with calibration:**
```bash
python run_eval.py --dataset dynamodb --suite recall_core --split test \
    --calibration-dir runs/calibration_latest --viz --open
```

---

# Buffer / Q&A (5 minutes)

## Anticipated Questions

**"How do you know the judges are actually right?"**
We calibrate against human-labeled ground truth. The calibration runs tell us exactly how accurate each judge is. Wide confidence intervals mean the judge is behaving inconsistently - essentially flipping a weighted coin on borderline cases.

**"What's the cost?"**
Programmatic evaluators are nearly free. Tribunal judges cost roughly $0.03-0.05 per conversation (3 Sonnet calls + 1 Haiku call).

**"Can this run in production?"**
Yes, that's the goal. Programmatic evaluators can run synchronously. Tribunal judges would be async guardrails - flag suspicious responses for human review.

**"What if external reviewers flag things that aren't actually errors?"**
This happened! External analysis flagged customers saying "sounds good" as insufficient consent. But in automotive recall scheduling, that *is* how people consent. Domain expertise matters - evaluations encode business rules, not abstract standards.

---

# Part 4: The Iterative Workflow

## How to Actually Use This System

The evaluation system isn't just for measuring - it's for *improving*. Here's how you use it to make smart, iterative changes to prompts while keeping everything stable.

### First: Which Workflow Do You Need?

**Trust your evaluators?** Use the "No Calibration" workflow. This is faster - you're just running evals and iterating on prompts.

**Seeing weird results?** If the logs show false positives or false negatives that don't make sense - like the evaluator flagging obviously good conversations or missing obvious failures - your evaluator needs recalibrating first. The TPR/TNR numbers in your report can't be trusted until you fix the evaluator.

**How to tell:** Look at the `traces.jsonl` output. Find cases where `code_pass` or `judge_pass` disagrees with what you'd expect. A few edge cases are normal. Systematic patterns mean the evaluator is broken.

---

## Workflow 1: No Calibration Needed

Use this when your evaluators are already well-calibrated and you're focused on improving the AI's behavior.

### Step 1: Make Your Prompt Change

Edit the system prompt, few-shot examples, or business rules that govern the AI's behavior. Keep track of what you changed - you'll want to know what worked.

### Step 2: Run Eval on Dev Split (Target Mode Only)

```bash
python run_eval.py --dataset dynamodb --modes B3 --split dev --viz --open
```

This gives you fast feedback on just the failure mode you're optimizing. The HTML report shows:
- How many conversations had the B3 scenario present
- How many failed the B3 check
- The failure rate with confidence interval

**Pro tip:** Use `--no-judge` for even faster iteration if you're tuning something the programmatic evaluator can catch. Add judges back in when you need semantic depth.

### Step 3: Dig Into Disagreements

Open `runs/[timestamp]/traces.jsonl` and look for patterns:

**What to look for:**
- Cases where evaluator said FAIL but conversation looks fine → potential false positive, or your intuition is wrong
- Cases where evaluator said PASS but conversation has issues → potential false negative, or the evaluator is too lenient
- Edge cases that are genuinely ambiguous → these inform boundary decisions

**The key question:** Is the evaluator wrong, or is the AI wrong? If the evaluator is wrong, you need calibration workflow. If the AI is wrong, keep iterating on the prompt.

### Step 4: Iterate Until Satisfied

Keep looping through steps 1-3:
- Change prompt → run eval → check results → repeat

**When are you "done"?**
- Failure rate is below your SLO threshold
- Confidence interval upper bound is below threshold (you're statistically confident)
- You've looked at failing cases and they're genuinely hard edge cases, not systematic issues

**Warning signs you're overfitting to dev:**
- You've made 10+ tweaks targeting specific conversations
- Your prompt has grown significantly with special cases
- You're achieving "perfect" results on dev (suspicious)

### Step 5: Validate on Test Split

Once you're happy with dev results, run the real test:

```bash
# Quick check: just your target mode on test
python run_eval.py --dataset dynamodb --modes B3 --split test --viz --open

# Full check: all modes to catch regressions
python run_eval.py --dataset dynamodb --suite recall_core --split test --viz --open
```

**Why run all modes?** Prompt changes can have unintended effects. Tightening B3 (availability verification) might accidentally make B1 (consent) worse if your new language confuses the consent detection. The full suite catches this.

### Step 6: Handle Regressions

If other failure modes got worse:

1. **Check if it's real** - Look at the actual failing conversations. Sometimes it's noise, especially with small sample sizes.

2. **Understand the tradeoff** - Did you make B3 better at the cost of B2? Sometimes that's acceptable. Sometimes it's not.

3. **Go back to dev** - Return to steps 1-3, but now you're balancing multiple modes. This is where prompt engineering gets hard.

**The golden rule:** Never ship a change that improves one mode but tanks another, unless you've explicitly decided that tradeoff is worth it.

---

## Workflow 2: With Calibration

Use this when your evaluators themselves need work - either because they're new, or because you've discovered they're not catching what they should.

### When You Need This Workflow

Signs your evaluator needs calibration:
- TPR is low (missing real failures)
- TNR is low (flagging good conversations as failures)
- You're seeing systematic patterns in FPs or FNs
- You just built a new evaluator and haven't calibrated it yet

### Step 1: Edit the Evaluator (Not the AI Prompt)

For programmatic evaluators, this means:
- Adjusting thresholds (e.g., `B2_HEDGE_RATIO_THRESHOLD`)
- Adding/removing pattern matches
- Changing the logic for pass/fail determination

For judges, this means:
- Editing the analyst prompts
- Adjusting the meta-judge prompt
- Adding calibration examples to the few-shot cache

### Step 2: Run Calibration

```bash
# For programmatic evaluators
python programmatic_calibration.py --modes B2 --split dev

# For judges
python judge_calibration.py --judge B_judge --split dev --verbose
```

This compares evaluator predictions against ground truth labels and gives you:
- TPR: Are you catching real failures?
- TNR: Are you leaving good conversations alone?
- Confusion matrix: Where exactly are the errors?

### Step 3: Analyze the Errors

Look at the false positives and false negatives:

**False Positives (evaluator said FAIL, ground truth says PASS):**
- Is your pattern too aggressive?
- Are you catching legitimate language that looks like the failure mode?
- Example: "Thursday could work" flagged as tentative when customer meant it definitively

**False Negatives (evaluator said PASS, ground truth says FAIL):**
- Is your pattern too narrow?
- Are there failure variations you're not catching?
- Example: "I might be able to swing by" not recognized as tentative

### Step 4: Iterate on the Evaluator

Adjust thresholds, add patterns, refine prompts. Re-run calibration. Repeat until:
- TPR ≥ 0.80 (catching 80%+ of real failures)
- TNR ≥ 0.85 (not flagging 85%+ of good conversations)
- Or whatever targets make sense for your use case

**The tradeoff:** Higher TPR usually means lower TNR and vice versa. Decide which errors are more costly for your application.

### Step 5: Need More Data? Generate It

If your calibration set is too small or missing important cases:

```bash
# Generate synthetic passing conversations
python targeted_synthetic_expansion.py --mode B3 --type pass --count 50

# Generate synthetic failing conversations  
python targeted_synthetic_expansion.py --mode B3 --type fail --count 50
```

Then transform and upload:

```bash
# Convert to DynamoDB format
python dynamodb_transform.py --input transformed_data/synthetic_b3.csv --output transformed_data/dynamodb_b3.json

# Upload to DynamoDB (if you have write access)
python dynamodb_transform.py --input transformed_data/synthetic_b3.csv --write-dynamodb
```

Now re-run calibration with the expanded dataset.

### Step 6: Save Calibration Results

Once calibration is good, the results live in `calibration_summary.json`. When you run evaluations, point to this:

```bash
python run_eval.py --dataset dynamodb --suite recall_core --split test \
    --calibration-dir runs/calibration_20241115 --viz --open
```

The evaluation will load TPR/TNR from calibration and apply Rogan-Gladen correction to give you *true* failure rate estimates, not raw (biased) counts.

---

## The Full Picture: Calibration + Iteration

In practice, you often do both:

1. **Calibrate evaluators** → Get trustworthy TPR/TNR
2. **Iterate on AI prompts** → Improve failure rates
3. **Discover evaluator gaps** → Go back to calibration
4. **Repeat**

It's a cycle. As you push the AI to handle harder cases, you discover edge cases the evaluator doesn't handle. As you improve the evaluator, you find AI behaviors you hadn't noticed before.

**Where files live:**
- Generated synthetic data: `transformed_data/` or `error_analysis/`
- Calibration results: `runs/[calibration_run]/calibration_summary.json`
- Evaluation results: `runs/[eval_run]/`
- HTML reports: `runs/[run]/report.html`

All of these feed back into `dataset.py` - it can load from DynamoDB (live), JSONL (snapshots), or the calibration outputs directly.

---

## Quick Decision Tree

```
Want to improve AI behavior?
├── Do you trust your evaluators?
│   ├── YES → Workflow 1 (No Calibration)
│   │         Edit prompt → eval dev → check disagreements → iterate → eval test
│   │
│   └── NO → Workflow 2 (With Calibration) first
│            Edit evaluator → calibrate → analyze errors → iterate → then Workflow 1

Seeing regressions in other modes?
├── Is it noise (small sample, wide CI)?
│   └── YES → Probably fine, monitor it
│
└── Is it systematic?
    └── YES → Go back to dev, balance the tradeoff
```
