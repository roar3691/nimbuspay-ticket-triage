# NimbusPay ticket triage

VIMA3YA ML intern assessment. Work in progress; no GPU results have been measured yet.

Base model: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Planned precision: 4-bit NF4 QLoRA, with a controlled FP16 LoRA comparison.
Best result: **pending actual Colab T4 execution**.

The supplied dataset and base-model weights are deliberately excluded from Git.
Upload the original assessment archive privately into Colab when running the notebook.
All supervised data preparation uses training rows and the schema only. Dev is evaluation-only.

The final repository will contain a PEFT adapter, executed Colab notebook, cleaning log,
raw test predictions, three-page report, and saved experiment metrics.

Deadline used: October 6, 2026, 17:06 IST, following the assignment email's 24-hour rule.
Keep this repository private until the deadline. No commits or pushes after that time.
The candidate will make the repository public and email the link during the following hour.

AI assistance: OpenAI Codex helped inspect the pack and implement the audit, training,
evaluation, and reporting code. No model/API is used to generate training labels or test predictions
apart from the assigned base model and its fine-tuned adapters.
