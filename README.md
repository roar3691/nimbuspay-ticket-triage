# NimbusPay ticket triage

VIMA3YA ML intern assessment. Five controlled training/dev runs completed on a free
Colab Tesla T4. The rules baseline, all 400 test predictions, and the adapter
reload check have since completed on a T4. The fresh-training reproduction check
and final report are still incomplete. Saved dev predictions were independently
rescored with the supplied, unchanged scorer.

Base model: `Qwen/Qwen2.5-1.5B-Instruct`, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Selected run: **C, FP16 LoRA**; dev exact match **62.5%**, mean field accuracy
**93.9375%**, parseable JSON **100%**. C wins the tie with E on mean field accuracy.
Its adapter contains 18,464,768 trainable parameters and is 73,911,112 bytes.

| Run | Change | Dev exact | Mean fields | Training minutes |
|---|---|---:|---:|---:|
| Fixed baseline | Untuned, original prompt | 0.0% | 0.0% | — |
| Rules baseline | Untuned, schema added | 0.0% | 0.0% | — |
| A | Raw labels, QLoRA | 58.5% | 92.9375% | 31.52 |
| B | Cleaned, QLoRA | 61.5% | 93.3750% | 31.01 |
| C | Cleaned, FP16 LoRA | 62.5% | 93.9375% | 22.90 |
| D | B with LR 1e-4 | 54.5% | 92.0625% | 32.07 |
| E | B with notation variants | 62.5% | 93.5625% | 31.65 |

Both untuned baselines returned Markdown-fenced output on all 200 dev rows; the
unchanged scorer accepts only bare JSON, so both measured 0% JSON validity and
accuracy. No outputs were repaired. The rules baseline's fenced bodies resembled
the target schema, but the fences alone make them invalid to the scorer. The
selected C adapter generated 400 test records with unique, complete IDs, and its
seeded 10-row reload check reproduced every output. A local schema audit found
10 test outputs with out-of-spec values (8 transaction-ID formats and 2 channel
values); these raw generations are retained unchanged. Test accuracy is unknown
because test labels are withheld.

Pending: fresh default notebook verification on a new T4 runtime (under three
hours) and the final report of at most three pages. See
`results/submission_status.json` for the current state. Ten reviewed actual dev
failures are saved in `results/error_analysis.json`.

The supplied dataset, original archive, derived training data and base-model
weights are excluded from Git. Upload the assessment archive privately into Colab.
All supervised preparation uses training rows and the schema only; dev is
used for evaluation and run selection.

## Reproduction

1. Open `notebook.ipynb` in Google Colab and choose a free T4 runtime.
2. Run all cells and upload the original `training_assessment.zip` when prompted.
3. Allow private Drive checkpoint storage, or set `USE_DRIVE_BACKUP=False`.
   The default path trains a new C adapter in a fresh run directory.
4. Set `RUN_ALL_EXPERIMENTS=True` only to rerun both baselines and all experiments.
   The under-three-hour assertion applies to the default path; it has not yet
   been verified by a complete fresh run.

The notebook embeds source and pins dependencies from `requirements.txt`.
PyTorch is pinned to 2.11.0; the supplied CUDA build and measured versions are
recorded in `results/environment.json`. Inference is greedy, one complete ticket
at a time, with the native chat template, `max_new_tokens=200`, generated-token
decoding and whitespace stripping. No field repair or constrained decoding occurs.

## Continue after GPU access returns

Open `resume.ipynb` on a free T4. It uses the existing private Drive folder,
rescoring the completed fixed baseline and generating only missing predictions.
It never retrains A-E. Prediction records close individually to reduce loss on
Drive after interruptions. Resume validates the ID prefix and adapter provenance;
it rejects torn writes instead of repairing model outputs.

With the same dependencies installed, the corresponding command is:

```sh
python3 resume_submission.py --data private/candidate_pack/data \
  --schema SCHEMA.md --out /content/drive/MyDrive/nimbuspay-ticket-triage-private/artifacts \
  --submission /content/drive/MyDrive/nimbuspay-ticket-triage-private/submission
```

For a fresh runtime without access to the original Drive, create a private local
transfer bundle with `build_portable_backup.py --archive /path/to/training_assessment.zip
--output /path/to/new-backup-folder`. Upload its `portable_resume.ipynb` to Colab,
then upload the generated ZIP when prompted. Checksums verify the restore; the
notebook skips completed training and resumes inference. The transfer ZIP contains
assessment data and must stay private. Only the selected adapter is needed to
resume; optional other adapters are collected from `private/all_adapters`.

Local audit/recovery checks need only Python's standard library:

```sh
python3 -m unittest -v test_audit.py test_recovery.py
python3 audit.py --train private/candidate_pack/data/train.jsonl --out artifacts
python3 score.py --gold private/candidate_pack/data/dev.jsonl --pred results/C/predictions_dev.jsonl
```

Final acceptance, once the pending artifacts are produced:

```sh
python3 verify_submission.py --data private/candidate_pack/data
```

The audit retains 1,333 of 1,445 rows: 194 fixed and 112 dropped original rows.
Maximum complete sequence length is 2,215 tokens, rounded to 2,304. Dynamic
padding preserves complete tickets. All 40 original long rows were measured;
36 unique long tickets remain after deduplication. Prompt/header/padding/template
separator tokens are masked; assistant JSON and its end token are supervised.

Deadline: October 6, 2026, 17:06 IST, following the email's 24-hour rule.
Keep the repository private until then, and make no later commits or pushes.
The candidate makes it public and manually emails the link during 17:06-18:06 IST.

AI assistance: OpenAI Codex assisted archive inspection, deterministic audit code,
training/evaluation implementation, Colab browser operation, repository setup,
verification and report preparation. No external model/API generated training
labels or submitted predictions. Predictions come from Qwen and its PEFT adapters.
The candidate must review the implementation and be able to explain it.
