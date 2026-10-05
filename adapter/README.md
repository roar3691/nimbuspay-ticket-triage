---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- transformers
---

# NimbusPay ticket-triage adapter (run C)

Standard PEFT LoRA adapter for the fictional NimbusPay assessment. Load the base
model at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` before loading this adapter.

Trained on a free Colab Tesla T4 with FP16 base weights. LoRA rank 16, alpha 32,
dropout 0.05, attention and MLP projection targets, 18,464,768 trainable parameters.
Training used 1,333 cleaned training rows, seed 42, 160 optimizer updates,
learning rate 2e-4, microbatch 1 and accumulation 16. Complete assistant JSON and
its end token were supervised; prompt/header/padding/separator tokens were masked.

On all 200 dev rows: exact match 62.5%, mean field accuracy 93.9375%, parseable
JSON 100%. The unchanged supplied scorer was rerun locally on the saved raw
predictions. JSON parsing does not imply schema-valid field values. Priority is
weakest at 64.5%; the evaluation uses one seed and a small dev set.

Use original system/user messages and the native chat template. Generate greedily
with 200 new tokens; decode the generated suffix, omit model special tokens and
strip whitespace. Apply no output repair or field filling.

The original dataset and base weights are not in this repository. See the main
README, results directory and status manifest for reproduction and outstanding
GPU checks. The adapter reload/sample check and test predictions remain pending.

Codex assisted implementation and analysis; submitted model predictions are
produced solely by Qwen and its adapters. This is an assessment model, not a
validated production payments-triage service.
