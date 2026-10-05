# NimbusPay ticket triage

VIMA3YA ML intern assessment. Work in progress: model loading and 4-bit forward inference
passed on a free Colab Tesla T4; controlled training is running. Dev metrics are pending.

Base model: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Planned precision: 4-bit NF4 QLoRA, with a controlled FP16 LoRA comparison.
Best result: **pending actual Colab T4 execution**.

The supplied dataset and base-model weights are deliberately excluded from Git.
Upload the original assessment archive privately into Colab when running the notebook.
All supervised data preparation uses training rows and the schema only. Dev is evaluation-only.

The final repository will contain a PEFT adapter, executed Colab notebook, cleaning log,
raw test predictions, three-page report, and saved experiment metrics.

## Reproduction

1. Open `notebook.ipynb` in Google Colab and choose the free T4 runtime.
2. Run all cells and upload the original `training_assessment.zip` when prompted.
3. Allow private Drive checkpoint storage, or set `USE_DRIVE_BACKUP=False` and download
   outputs before disconnecting. The default path trains a new selected adapter.
4. Set `RUN_ALL_EXPERIMENTS=True` only to rerun both baselines and all five experiments.
   The under-three-hour check applies to the default path.

The notebook embeds the source files and pins dependencies from `requirements.txt`.
PyTorch/CUDA versions supplied by Colab are recorded in `environment.json`.
Inference is greedy, one complete ticket at a time, using the native chat template,
`max_new_tokens=200`, generated-token decoding and whitespace stripping.

Local audit checks require only Python's standard library:

```sh
python3 -m unittest -v test_audit.py
python3 audit.py --train private/candidate_pack/data/train.jsonl --out artifacts
python3 score.py --gold private/candidate_pack/data/dev.jsonl --pred results/B/predictions_dev.jsonl
python3 verify_submission.py --data private/candidate_pack/data
```

The audit retains 1,333 of 1,445 rows: 194 fixed and 112 dropped original rows.
The training maximum is 2,215 complete tokens, rounded to 2,304. No ticket is truncated.
All 40 long tickets remain intact. Prompt, header, padding and the template separator
are masked; assistant JSON and its end token are supervised.

Deadline used: October 6, 2026, 17:06 IST, following the assignment email's 24-hour rule.
Keep this repository private until the deadline. No commits or pushes after that time.
The candidate will make the repository public and email the link during the following hour.

AI assistance: OpenAI Codex helped inspect the pack and implement the audit, training,
evaluation, and reporting code. No model/API is used to generate training labels or test predictions
apart from the assigned base model and its fine-tuned adapters.
