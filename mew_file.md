# Evaluation System Code Reference - Line Numbers

This document maps each concept described in the walkthrough to specific line numbers in the codebase.

## Table of Contents

- [Evaluation System Code Reference - Line Numbers](#evaluation-system-code-reference---line-numbers)
  - [1. `programmatic.py` (~10,961 lines)](#1-programmaticpy-10961-lines)
    - [Overview & Imports](#overview--imports)
    - [Global Matchers (initialized at module load)](#global-matchers-initialized-at-module-load)
    - [Threshold Constants](#threshold-constants)
    - [B2 Tentative Language Detection](#b2-tentative-language-detection)
    - [B1 Premature Booking Detection](#b1-premature-booking-detection)
    - [B3 Availability Verification](#b3-availability-verification)
    - [C5 Tone/Style Detection](#c5-tonestyle-detection)
    - [C2 Out-of-Scope Detection](#c2-out-of-scope-detection)
    - [C3 Escalation Detection](#c3-escalation-detection)
    - [A1/A2/A3 Legitimacy & Provenance](#a1a2a3-legitimacy--provenance)
    - [EVALUATORS Registry](#evaluators-registry)
  - [2. `judges.py` (~4,287 lines)](#2-judgespy-4287-lines)
    - [Tribunal Architecture - B Modes](#tribunal-architecture---b-modes)
    - [Tribunal Architecture - C3](#tribunal-architecture---c3)
    - [Logging & Debug](#logging--debug)
    - [JUDGES Registry](#judges-registry)
    - [Cost Tracking](#cost-tracking)
  - [3. `programmatic_calibration.py` (~1,943 lines)](#3-programmatic_calibrationpy-1943-lines)
    - [Module Documentation](#module-documentation)
    - [Configuration & Constants](#configuration--constants)
    - [Core Functions](#core-functions)
    - [Main Execution Flow](#main-execution-flow)
    - [Statistics Computation](#statistics-computation)
  - [4. `judge_calibration.py` (~1,840 lines)](#4-judge_calibrationpy-1840-lines)
    - [Module Documentation](#module-documentation-1)
    - [Configuration & Constants](#configuration--constants-1)
    - [Thread Safety](#thread-safety)
    - [Core Functions](#core-functions-1)
    - [Main Execution Flow](#main-execution-flow-1)
    - [Statistics & Output](#statistics--output)
  - [5. `dataset.py` (~1,594 lines)](#5-datasetpy-1594-lines)
    - [Module Constants](#module-constants)
    - [Helper Functions](#helper-functions)
    - [DynamoDBDatasetLoader Class](#dynamodbdatasetloader-class)
    - [DynamoDB Schema (Fan-out Pattern)](#dynamodb-schema-fan-out-pattern)
    - [`trace_to_inspect_format()`](#trace_to_inspect_format)
  - [6. `recall_task.py` (~567 lines)](#6-recall_taskpy-567-lines)
    - [InspectAI Integration](#inspectai-integration)
    - [PROMPT_TEMPLATE](#prompt_template)
    - [`build_prompt()`](#build_prompt)
    - [Scorers](#scorers)
    - [Task Definitions](#task-definitions)
  - [7. `turn_by_turn.py` (~280 lines)](#7-turn_by_turnpy-280-lines)
    - [Core Functions](#core-functions-2)
    - [`evaluate_turn_by_turn()` Details](#evaluate_turn_by_turn-details)
    - [`run_item_generative()` Details](#run_item_generative-details)
  - [Quick Reference: Key Thresholds to Tune](#quick-reference-key-thresholds-to-tune)
  - [Quick Reference: Key Entry Points](#quick-reference-key-entry-points)
  - [Quick Reference: Evaluator Function Lines](#quick-reference-evaluator-function-lines)
  - [Quick Reference: Judge Function Lines](#quick-reference-judge-function-lines)


---

## 1. `programmatic.py` (~10,961 lines)

### Overview & Imports
| Concept | Lines | Description |
|---------|-------|-------------|
| Module docstring | 1-10 | Explains evaluator purpose and modes (recovery/generative) |
| spaCy imports | 19-25 | Try/except for spaCy, Matcher, PhraseMatcher |
| spaCy pipeline loading | 37-69 | `_load_spacy_pipeline()` tries models in order: trf → lg → md → sm |
| Pipeline initialization | 72-73 | `nlp, SPACY_MODEL_NAME = _load_spacy_pipeline()` |
| SPACY_AVAILABLE flag | 73 | Boolean set after pipeline load attempt |

### Global Matchers (initialized at module load)
| Matcher | Lines | Purpose |
|---------|-------|---------|
| commitment_matcher | 76 | B1/B2 hedge pattern detection |
| confirmation_matcher | 77 | Confirmation-seeking detection |
| consent_matcher | 78 | Consent language detection |
| handoff_matcher | 79 | Handoff language detection |
| hedge_phrase_matcher | 80 | Multi-word hedge phrases |
| oos_topic_matcher | 81 | C2 out-of-scope topics |
| escalation_phrase_matcher | 82 | C2/C3 escalation language |
| contact_marker_matcher | 83 | C2 phone/email detection |
| tone_issue_matcher | 84 | C5 tone problems |
| tone_blacklist_matcher | 85 | C5 blacklisted phrases |
| escalation_behavior_matcher | 86 | Escalation behavior patterns |
| url_context_matcher | 87 | A2 URL context phrases |
| url_provision_matcher | 88 | A2 URL provision patterns |
| info_request_matcher | 89 | C4 info-gathering patterns |
| verification_link_matcher | 90 | C4 verification URLs |
| info_completeness_matcher | 91 | C4 info completeness |

### Threshold Constants
| Constant | Line | Default | Purpose |
|----------|------|---------|---------|
| C4_INFO_GATHERING_THRESHOLD | 94 | 0.67 | Require 2 of 3 info types |
| B2_HEDGE_RATIO_THRESHOLD | 357 | 0.30 | Hedge words as % of tokens |
| C5_TONE_ISSUE_THRESHOLD | 363 | 0.5 | Tone issue score threshold |
| A2_VERIFICATION_SCORE_THRESHOLD | 369 | 0.5 | URL verification score |
| SEMANTIC_SIMILARITY_THRESHOLD | 412 | 0.7 | A1/A2 pattern matching |
| A2_CONCERN_INTENSITY_THRESHOLD | 442 | 0.75 | Concern intensity for A2 |
| A2_REGEX_CONCERN_PATTERNS | 444-457 | List | Scam/fraud/phishing patterns with intensity |
| MODAL_DEPENDENCY_THRESHOLD | 462 | 0.5 | Modal strength for B2 |
| SENTIMENT_ESCALATION_THRESHOLD | 467 | -0.5 | C3 negative sentiment |
| CONVERSATION_AGGREGATION_METHOD | 471 | "max" | Multi-turn aggregation method |

### B2 Tentative Language Detection
| Function/Pattern | Lines | Description |
|------------------|-------|-------------|
| Hedge ratio documentation | 96-129 | Explains hedge_ratio metric calculation |
| `_analyze_commitment_strength_spacy_B2()` | 1947-2079 | Main spaCy analysis for B2 |
| Strong hedging patterns list | 1995-1999 | HEDGE_ADVERB_MAYBE, MODAL_ABILITY, etc. |
| Medium hedging patterns list | 2000-2003 | QUESTION_TIME, THINKING_PATTERN, etc. |
| Weak hedging patterns list | 2004-2006 | WEAK_AGREEMENT, MODAL_PREFERENCE |
| No hedging patterns list | 2007-2009 | STRONG_AFFIRMATIVE, WORKS_FOR_ME, etc. |
| hedge_only_patterns set | 2012 | Combines strong + medium + weak hedging |
| hedged_token_indices set | 2016 | Track unique token indices |
| hedge_ratio calculation | 2032-2037 | `hedge_hits / token_count` |
| is_tentative logic | 2066 | `has_strong_hedge` only |
| `_analyze_modal_dependency_structure_B2()` | 2173-2261 | Dependency parsing for modal verbs |
| Modal strengths mapping | 2203-2211 | might=0.2, could=0.3, may=0.35, would=0.5, etc. |
| `evaluate_B2()` function | 6055+ | Main B2 evaluator entry point |
| Regex tentative_pattern | 6312-6342 | Fallback when spaCy unavailable |
| Combined signals check | 6389-6394 | is_tentative OR hedge_ratio > threshold OR modal_dependency |

### B1 Premature Booking Detection
| Function/Pattern | Lines | Description |
|------------------|-------|-------------|
| `_analyze_commitment_strength_spacy_B1()` | 1814-1944 | spaCy hedge analysis for B1 |
| `_analyze_modal_dependency_structure_B1()` | 2082-2170 | B1 modal dependency analysis |
| `_COMPLETED_BOOKING_CUE_PATTERN` | 3805-3826 | Regex for definitive booking |
| `_find_booking_language()` | 3829-3933 | Detects booking statements |
| `_find_consent_language()` | 3935-3959 | Detects consent in prior turn |
| Consent pattern (expanded) | 3947-3958 | book, yes, yeah, sure, sounds good, etc. |
| `_is_hypothetical_statement()` | 4040-4048 | Filters hypothetical/conditional |
| `_is_availability_only()` | 4050-4060 | Filters availability-only statements |
| `evaluate_B1()` | 5498+ | Main B1 evaluator entry point |
| `_evaluate_B1_recovery()` | 5604+ | B1 recovery mode logic |
| `_evaluate_B1_generative()` | 5733+ | B1 generative mode logic |

### B3 Availability Verification
| Function/Pattern | Lines | Description |
|------------------|-------|-------------|
| _B3_INCOMPLETE_RESPONSE_PATTERNS | 6579-6585 | Patterns for incomplete responses |
| _B3_COMPLETION_PATTERNS | 6587-6592 | Patterns for completed bookings |
| _B3_TENTATIVE_BOOKING_PATTERNS | 6594-6610 | Patterns for tentative booking language |
| _B3_CHECKING_AVAILABILITY_PATTERNS | 6612-6618 | Patterns for checking availability |
| _B3_VERIFIED_AVAILABILITY_PATTERNS | 6620-6628 | Patterns for verified availability |
| _B3_ESCALATION_FILTER | 6630 | Filter for escalation language |
| _B3_CUSTOMER_CONFIRM_FILTER | 6631 | Filter for customer confirmation |
| _B3_DAY_PATTERN | 6633 | Day of week pattern |
| _B3_MONTH_PATTERN | 6634 | Month name pattern |
| _B3_RELATIVE_PATTERN | 6635-6642 | Relative date patterns (today, tomorrow, etc.) |
| _B3_TIME_PATTERN | 6643 | Time pattern (HH:MM am/pm) |
| _B3_YEAR_PATTERN | 6644 | Year pattern (2025-2027) |
| _B3_ORDINAL_PATTERN | 6645 | Ordinal date pattern (1st, 2nd, etc.) |
| _B3_DAYPART_PATTERN | 6646 | Day part pattern (morning, afternoon, etc.) |
| _B3_PENDING_ENDINGS | 6655-6664 | Phrases indicating pending confirmation |
| _B3_ACCEPTANCE_TERMS | 6666-6670 | yes, yeah, book it, works, perfect, etc. |
| _B3_RELATIVE_PHRASES | 6672-6676 | Relative time phrases |
| _B3_AVAILABILITY_SYSTEM_PHRASES | 6678-6682 | system shows, calendar shows, etc. |
| _B3_AVAILABILITY_CONFIRM_PHRASES | 6684-6688 | you're all set, appointment is set, etc. |
| `evaluate_B3()` | 7475+ | Main B3 evaluator entry point |
| `_evaluate_B3_recovery()` | 7492+ | B3 recovery mode logic |
| `_evaluate_B3_generative()` | 7876+ | B3 generative mode logic |

### C5 Tone/Style Detection
| Function/Pattern | Lines | Description |
|------------------|-------|-------------|
| C5 configuration docs | 166-213 | Explains matcher types and scoring |
| Tone issue score calculation | 206-213 | Blacklisted=0.4, punctuation=0.3, caps=0.2, etc. |
| `_analyze_tone_issues_spacy()` | 2813+ | Main spaCy tone analysis |
| `evaluate_C5()` | 10620+ | Main C5 evaluator entry point |
| `_evaluate_C5_recovery()` | 10637+ | C5 recovery mode logic |
| `_evaluate_C5_generative()` | 10784+ | C5 generative mode logic |

### C2 Out-of-Scope Detection
| Function/Pattern | Lines | Description |
|------------------|-------|-------------|
| C2 configuration docs | 131-164 | Explains detection strategy |
| `_detect_oos_handling_spacy()` | 2694+ | spaCy-based OOS detection |

### C3 Escalation Detection
| Function/Pattern | Lines | Description |
|------------------|-------|-------------|
| `_has_escalation_keywords()` | 4063-4081 | Detects pricing, complaint, legal, legitimacy |
| `_has_handoff_language()` | 4084-4093 | Detects AI handoff phrases |

### A1/A2/A3 Legitimacy & Provenance
| Function/Pattern | Lines | Description |
|------------------|-------|-------------|
| A1 configuration docs | 215-282 | Legitimacy concern detection |
| A2 configuration docs | 284-352 | URL context detection |
| A2_REGEX_CONCERN_PATTERNS | 444-457 | Scam/fraud/phishing patterns with intensity |
| NOT_SURE_LEGITIMACY_KEYWORDS | 415-440 | Keywords for legitimacy concern detection |
| `_extract_positive_indicators()` | 9199-9232 | A3 positive provenance |
| `_extract_negative_indicators()` | 9235-9269 | A3 negative provenance (factual errors) |
| `_analyze_allstate_provenance_regex()` | 9272+ | Fallback regex for A3 |

### EVALUATORS Registry
| Item | Lines | Description |
|------|-------|-------------|
| EVALUATORS dict | 10937-10951 | Maps mode codes to evaluator functions |
| `evaluate_all_targeted()` | 10954-10961 | Runs all evaluators for item's targets |

---

## 2. `judges.py` (~4,287 lines)

### Tribunal Architecture - B Modes
| Component | Lines | Description |
|-----------|-------|-------------|
| `build_b_tribunal_judge()` | 2487-2507 | Factory function with architecture docs |
| Cost estimate comment | 2506 | ~$0.015-0.025/eval |
| `_log_b_debate_minimal()` | 2513-2591 | Thread-safe logging for parallel runs |
| BMultiTurnAnalyst unit | 2594-2599 | Layer 1 analyst (Sonnet) |
| BConversationAnalyst unit | 2601-2606 | Layer 1 analyst (Sonnet) |
| debate_layer_b | 2609 | Layer combining 2 analysts (broadcast) |
| _B_META_JUDGE_PROMPT | 2612+ | Full MetaJudge prompt with business rules |

### Tribunal Architecture - C3
| Component | Lines | Description |
|-----------|-------|-------------|
| C3 tribunal similar structure | ~1800-2200 | (Similar pattern to B tribunal) |
| `judge_c3_tribunal()` | varies | Main C3 tribunal entry point |

### Logging & Debug
| Function | Lines | Description |
|----------|-------|-------------|
| `_write_to_analysis_log()` | ~200-300 | Thread-local log buffer writes |
| `set_judge_console_logging()` | ~150-180 | Controls console output during parallel |

### JUDGES Registry
| Item | Lines | Description |
|------|-------|-------------|
| JUDGES comment | 4233 | # JUDGE REGISTRY |
| `judge_b1_tribunal()` wrapper | 4235-4237 | B1 explicit mode wrapper |
| `judge_b2_tribunal()` wrapper | 4239-4241 | B2 explicit mode wrapper |
| `judge_b3_tribunal()` wrapper | 4243-4245 | B3 explicit mode wrapper |
| JUDGES dict | 4247-4258 | Maps judge names to functions |
| `judge_all_targeted()` | 4261-4268 | Runs all judges for item's targets |
| `mock_judge()` | 4272-4287 | Testing mock (80% pass rate) |

### Cost Tracking
| Location | Lines | Description |
|----------|-------|-------------|
| Usage dict in return | ~4200-4211 | input_tokens, output_tokens, estimated_cost |
| Model breakdown | ~4206-4209 | sonnet_analyst_calls, metajudge_calls, haiku_calls |

---

## 3. `programmatic_calibration.py` (~1,943 lines)

### Module Documentation
| Section | Lines | Description |
|---------|-------|-------------|
| Module docstring | 1-96 | Full usage examples, workflow guidance |
| Performance optimizations | 9-15 | Parallel execution, vectorized metrics |
| Key differences from judge cal | 49-58 | Deterministic, no bootstrap CI needed |

### Configuration & Constants
| Item | Lines | Description |
|------|-------|-------------|
| AVAILABLE_MODES list | 128 | A1, A2, A3, B1, B2, B3, B4, C2, C3, C4, C5 |
| MODE_TO_BINARY_COLUMN | 131-143 | Maps mode to ground truth column name |

### Core Functions
| Function | Lines | Description |
|----------|-------|-------------|
| `_should_use_thread_pool()` | 146-156 | Decides threads vs processes |
| USE_THREAD_POOL | 158 | Boolean result of detection |
| `convert_decimal()` | 161-186 | DynamoDB Decimal → Python int/float |
| `convert_decimals_recursive()` | 188-200 | Recursive decimal conversion |

### Main Execution Flow
| Function/Section | Lines | Description |
|------------------|-------|-------------|
| `main()` | 1580-1936 | Entry point with arg parsing |
| DynamoDB init | ~1680-1690 | Creates loader instance |
| Split processing loop | ~1700-1880 | Processes train/dev/test |
| `compute_per_mode_statistics()` | ~1500-1580 | TPR/TNR/accuracy per mode |
| `save_split_summary()` | ~1400-1450 | Writes summary.json |
| `save_samples_jsonl()` | ~1450-1500 | Writes results.jsonl |
| `save_overall_summary()` | ~1500-1550 | Writes calibration_summary.json |

### Statistics Computation
| Function | Lines | Description |
|----------|-------|-------------|
| Confusion matrix | ~1300-1350 | TP, FP, TN, FN counts |
| TPR calculation | ~1350-1360 | TP / (TP + FN) |
| TNR calculation | ~1360-1370 | TN / (TN + FP) |
| `format_mode_summary()` | ~1370-1400 | Pretty-print mode stats |
| `format_confusion_matrix()` | ~1400-1420 | Pretty-print confusion matrix |

---

## 4. `judge_calibration.py` (~1,840 lines)

### Module Documentation
| Section | Lines | Description |
|---------|-------|-------------|
| Module docstring | 1-73 | Configuration options, workflow, examples |
| Trace filtering docs | 8-11 | --primary-only explanation |
| Label source docs | 13-17 | Mode-specific vs trace_pass_or_fail |
| Usage examples | 36-68 | Recovery, generative, bootstrap examples |

### Configuration & Constants
| Item | Lines | Description |
|------|-------|-------------|
| JUDGE_TO_MODE | 149-158 | Maps judge name to mode(s) |
| JUDGE_TO_BINARY_COLUMN | 160-173 | Maps judge to ground truth column |

### Thread Safety
| Function | Lines | Description |
|----------|-------|-------------|
| _SUPPRESS_LOCK | 175 | Threading lock for suppression |
| `_silent_print()` | 179-181 | No-op print replacement |
| `suppress_stdout()` context manager | 185-200 | Thread-safe stdout suppression |

### Core Functions
| Function | Lines | Description |
|----------|-------|-------------|
| `convert_decimal()` | 108-123 | DynamoDB Decimal conversion |
| `convert_decimals_recursive()` | 126-146 | Recursive conversion |

### Main Execution Flow
| Function/Section | Lines | Description |
|------------------|-------|-------------|
| `main()` | 1446-1832 | Entry point |
| Loader init | ~1540-1550 | DynamoDB connection |
| Judge function lookup | ~1550-1560 | Gets judge from JUDGES registry |
| Split processing | ~1570-1750 | Loop over train/dev/test |
| `compute_split_statistics()` | ~1000-1200 | TPR/TNR/CI calculation |
| Bootstrap CI | ~1100-1150 | --bootstrap-workers parallelism |

### Statistics & Output
| Function | Lines | Description |
|----------|-------|-------------|
| TPR/TNR calculation | 1778-1784 | With 95% confidence intervals |
| `format_split_summary()` | ~1300-1350 | Pretty-print split stats |
| `format_confusion_matrix()` | ~1350-1400 | Pretty-print confusion matrix |
| `save_split_summary()` | ~1200-1250 | Writes split summary.json |
| `save_samples_jsonl()` | ~1250-1300 | Writes results.jsonl |
| `save_overall_summary()` | 1818 | Writes calibration_summary.json |

---

## 5. `dataset.py` (~1,594 lines)

### Module Constants
| Item | Lines | Description |
|------|-------|-------------|
| TABLE | 23 | "as-evaluation-dataset" |
| REGION | 24 | "us-east-1" |
| VERSION | 26 | "v0.01" |

### Helper Functions
| Function | Lines | Description |
|----------|-------|-------------|
| `load_jsonl()` | 29-36 | Load JSONL file to list |
| `convert_decimals()` | 39-53 | Recursive Decimal conversion |
| `get_base_mode()` | 56-90 | Strips _tribunal/_judge suffixes |
| `expand_tribunal_modes()` | 93-133 | B_judge → [B1, B2, B3] |
| `is_primary_mode()` | 136-150 | Checks primary_category match |
| `save_jsonl()` | 153-159 | Save list to JSONL file |

### DynamoDBDatasetLoader Class
| Method | Lines | Description |
|--------|-------|-------------|
| `__init__()` | 165-168 | Creates boto3 resource, table ref |
| `batch_get_traces()` | 170-193 | Batch get by trace IDs (100 limit) |
| `get_traces_by_mode()` | 195-237 | Query by mode with sampling |
| `get_traces_without_mode()` | 239-450 | Negative examples for mode |
| `get_traces_by_split()` | 452-694 | All traces in a split |
| `get_traces_by_split_batched()` | 696-819 | Memory-efficient batched loading |
| `get_synthetic_traces()` | 821-886 | Fetch silver-labeled synthetic data |

### DynamoDB Schema (Fan-out Pattern)
| Key Pattern | Lines | Description |
|-------------|-------|-------------|
| TRACE#{id} | ~173 | Primary key for trace items |
| V#{version}#SPLIT#{split} | ~173 | Sort key with version and split |
| MODE#{split}#{mode} | ~220 | GSI for mode-based queries |
| RAND#{nn} | ~225 | Random sampling prefix |

### trace_to_inspect_format()
| Section | Lines | Description |
|---------|-------|-------------|
| Function signature | 889-907 | Main conversion function |
| Conversation parsing | 909-958 | JSON string → list with Unicode cleanup |
| Sender normalization | 960-972 | "system" → "ai" |
| Alternative format handling | 978-1039 | messages/history/traces arrays |
| Mode columns list | 1046-1059 | All binary column names |
| targets extraction | 1061-1070 | From binary columns |
| mode_specific_labels | 1077-1100 | Preserves ground truth for lookup |
| Mock context generation | 1106 | `generate_mock_context()` |
| Input data construction | 1126-1129 | Conversation + mock context |
| Mode rubrics extraction | 1132-1142 | Per-mode rubric fields |

---

## 6. `recall_task.py` (~567 lines)

### InspectAI Integration
| Item | Lines | Description |
|------|-------|-------------|
| InspectAI imports | 10-14 | Task, Dataset, Sample, Scorer, etc. |
| Evaluator imports | 23-24 | programmatic.evaluate_all_targeted, judges |

### PROMPT_TEMPLATE
| Section | Lines | Description |
|---------|-------|-------------|
| Template start | 28-29 | System role definition |
| conversation_context | 32-34 | {conversation} placeholder |
| customer_context | 36-39 | firstName, lastName, vehicle |
| dealership_context | 41-46 | name, phone, address, time_zone |
| recall_context | 48-51 | recall_details, appointment_length |
| additional_offerings | 53-56 | Dealership extra services |
| upstream_processing | 58-61 | extracted_data, scheduling_decision |
| Business guardrails | 88-147 | SCAM, OOS, ESCALATION, BOOKING GATE |
| Booking gate rules | 125-133 | 4 requirements for appointmentConfirmed |
| Response types | 176-180 | Clarification, Confirmation, Schedule, Alternative, Stop |

### build_prompt()
| Section | Lines | Description |
|---------|-------|-------------|
| Function signature | 273 | Takes item dict |
| Conversation extraction | ~280-290 | From item['input']['conversation'] |
| Context extraction | ~290-310 | Customer, vehicle, dealership, recall |
| Template formatting | ~310-340 | .format() with placeholders |

### Scorers
| Function | Lines | Description |
|----------|-------|-------------|
| `recall_code_scorer()` | 350-395 | Programmatic scorer wrapper |
| `recall_judge_scorer()` | 398-444 | LLM judge scorer wrapper |

### Task Definitions
| Function | Lines | Description |
|----------|-------|-------------|
| `recall_eval()` | 448-508 | Main task factory |
| `recall_eval_full()` | 512-518 | Full test with judges |
| `recall_eval_fast()` | 521-528 | Quick code-only (50 samples) |
| `recall_eval_critical()` | 531-538 | B1, B2, C3 only |

---

## 7. `turn_by_turn.py` (~280 lines)

### Core Functions
| Function | Lines | Description |
|----------|-------|-------------|
| `parse_test_conversation()` | 30-47 | Extract turns from item |
| `build_conversation_for_turn()` | 49-82 | Build item for prompt generation |
| `evaluate_turn_by_turn()` | 85-187 | Main turn-by-turn evaluation loop |
| `run_item_generative()` | 190-279 | Entry point for generative mode |

### evaluate_turn_by_turn() Details
| Section | Lines | Description |
|---------|-------|-------------|
| Test conversation parsing | 105-108 | Get turns from item |
| Customer turn handling | 119-121 | Add customer turn to conversation |
| Last turn check | 124-129 | Skip AI gen if ends with customer |
| Prompt building | 135 | `build_prompt_fn(turn_item)` |
| Model generation | 136-137 | `model.generate(prompt)` |
| AI message extraction | 140-142 | `extract_ai_message()`, `extract_json_from_output()` |
| Turn scoring | 153-156 | Score each mode for this turn |
| Turn result accumulation | 158-166 | turn_index, customer_message, ai_response, scores |
| Test AI turn skip | 168-170 | `sender == 'ai'` → continue |
| Score aggregation | 173-186 | All turns must pass for mode to pass |

### run_item_generative() Details
| Section | Lines | Description |
|---------|-------|-------------|
| Call evaluate_turn_by_turn | 202-204 | Get turn_results, final_conversation, full_output |
| Empty results handling | 206-221 | Return minimal result if no customer turns |
| Result aggregation | 224-246 | Aggregate per-mode results across turns |
| Token aggregation | 249-250 | Sum input/output tokens |
| Turn analyses collection | 253-261 | Collect analysis per turn |
| Final result construction | 263-279 | id, output, results, metadata, usage |
| eval_mode setting | 269 | CRITICAL: Set 'generative' explicitly |

---

## Quick Reference: Key Thresholds to Tune

| Threshold | File | Line | Current Value | Impact |
|-----------|------|------|---------------|--------|
| B2_HEDGE_RATIO_THRESHOLD | programmatic.py | 357 | 0.30 | ↑ = fewer FPs, ↓ TPR |
| C5_TONE_ISSUE_THRESHOLD | programmatic.py | 363 | 0.5 | ↑ = only severe issues |
| A2_VERIFICATION_SCORE_THRESHOLD | programmatic.py | 369 | 0.5 | ↑ = stricter URL context |
| C4_INFO_GATHERING_THRESHOLD | programmatic.py | 94 | 0.67 | Require 2/3 info types |
| MODAL_DEPENDENCY_THRESHOLD | programmatic.py | 462 | 0.5 | Modal verb strength cutoff |
| SEMANTIC_SIMILARITY_THRESHOLD | programmatic.py | 412 | 0.7 | 70% cosine similarity |
| A2_CONCERN_INTENSITY_THRESHOLD | programmatic.py | 442 | 0.75 | A2 concern intensity cutoff |
| SENTIMENT_ESCALATION_THRESHOLD | programmatic.py | 467 | -0.5 | C3 negative sentiment |

---

## Quick Reference: Key Entry Points

| Purpose | File | Function | Line |
|---------|------|----------|------|
| Run programmatic evaluators | programmatic.py | `EVALUATORS[mode](output, item)` | 10938 |
| Run all programmatic evaluators | programmatic.py | `evaluate_all_targeted()` | 10954 |
| Run LLM judges | judges.py | `JUDGES[judge_name](output, item)` | 4247 |
| Run all LLM judges | judges.py | `judge_all_targeted()` | 4261 |
| Calibrate programmatic | programmatic_calibration.py | `main()` | 1580 |
| Calibrate judges | judge_calibration.py | `main()` | 1446 |
| Convert trace to eval format | dataset.py | `trace_to_inspect_format()` | 889 |
| Build model prompt | recall_task.py | `build_prompt()` | 273 |
| Run generative evaluation | turn_by_turn.py | `run_item_generative()` | 190 |

---

## Quick Reference: Evaluator Function Lines

| Mode | Main Function | Line | Recovery Mode | Generative Mode |
|------|---------------|------|---------------|-----------------|
| A1 | `evaluate_A1()` | 4511 | - | - |
| A2 | `evaluate_A2()` | 4863 | - | - |
| A3 | `evaluate_A3()` | 5199 | - | - |
| B1 | `evaluate_B1()` | 5498 | `_evaluate_B1_recovery()` @ 5604 | `_evaluate_B1_generative()` @ 5733 |
| B2 | `evaluate_B2()` | 6055 | `_evaluate_B2_recovery()` @ 6072 | `_evaluate_B2_generative()` @ 6278 |
| B3 | `evaluate_B3()` | 7475 | `_evaluate_B3_recovery()` @ 7492 | `_evaluate_B3_generative()` @ 7876 |
| C2 | `evaluate_C2()` | 9639 | - | - |
| C3 | `evaluate_C3()` | 9985 | - | - |
| C4 | `evaluate_C4()` | 10246 | - | - |
| C5 | `evaluate_C5()` | 10620 | `_evaluate_C5_recovery()` @ 10637 | `_evaluate_C5_generative()` @ 10784 |

---

## Quick Reference: Judge Function Lines

| Judge | Main Function | Line | JUDGES Entry |
|-------|---------------|------|--------------|
| A3_judge | `judge_a3_tribunal()` | 2851 | 4249 |
| C3_judge | `judge_c3_tribunal()` | 3314 | 4252 |
| B_judge | `judge_b_tribunal()` | 3781 | 4254 |
| B1_judge | `judge_b1_tribunal()` | 4235 | 4255 |
| B2_judge | `judge_b2_tribunal()` | 4239 | 4256 |
| B3_judge | `judge_b3_tribunal()` | 4243 | 4257 |
