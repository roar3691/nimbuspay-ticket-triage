"""Build a self-contained Colab notebook from the reviewed source files."""
import argparse
import json
from pathlib import Path


def build(root,final_run='B',experiments=False):
    cells=[]
    def markdown(text):
        cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):
        cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(keepends=True)})
    markdown('# NimbusPay: ticket triage on a free Colab T4\n\nUse **Runtime → Change runtime type → T4 GPU**. Upload the original assessment archive when prompted. This notebook contains all training/evaluation code; it never requires public dataset hosting. The default path trains the selected final configuration and regenerates predictions. Other experiments are optional. All displayed scores come from execution.')
    saved=root/'results'
    if any((saved/name/'result.json').exists() for name in ['baseline_fixed','baseline_rules','A','B','C','D','E']):
        lines=['## Saved measured experiment results','', '| Run | Dev exact | Mean fields | Training min | Peak allocated GiB |','|---|---:|---:|---:|---:|']
        for name in ['baseline_fixed','baseline_rules','A','B','C','D','E']:
            if not (saved/name/'result.json').exists():
                lines.append(f'| {name} (pending GPU access) | - | - | - | - |')
                continue
            r=json.loads((saved/name/'result.json').read_text())
            minutes=f"{r['training_minutes']:.2f}" if 'training_minutes' in r else '-'
            memory=f"{r['peak_gpu_allocated_bytes']/2**30:.2f}" if 'peak_gpu_allocated_bytes' in r else '-'
            lines.append(f"| {name} | {r['metrics']['exact_match']:.1%} | {r['metrics']['mean_field_acc']:.1%} | {minutes} | {memory} |")
        lines.extend(['','These are saved measurements; the default path below performs a fresh selected run. Full logs, configurations and raw dev outputs are in the repository results directory.'])
        markdown('\n'.join(lines))
    code("import os, time\nos.environ['USE_TF']='0'\nos.environ['USE_FLAX']='0'\nos.environ['TOKENIZERS_PARALLELISM']='false'\nNOTEBOOK_STARTED=time.perf_counter()\n%pip install -q "+' '.join((root/'requirements.txt').read_text().splitlines())+'\n')
    markdown('## Runtime and input setup\n\nThe training stack, including PyTorch, is pinned. The CUDA build supplied by Colab and exact package versions are recorded in `environment.json`. The archive and all derived training files remain private. Drive stores checkpoints between sessions; review its account prompt if requested.')
    code(f"RUN_ALL_EXPERIMENTS = {experiments!r}\nFINAL_RUN = {final_run!r}\nUSE_DRIVE_BACKUP = True\n\nimport json, shutil, zipfile, sys\nfrom pathlib import Path\nimport torch\nfrom google.colab import files, drive\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Free Colab T4 required'\nprint('GPU:',torch.cuda.get_device_name(0),'PyTorch:',torch.__version__,'CUDA:',torch.version.cuda)\nROOT=Path('/content/nimbuspay')\nROOT.mkdir(exist_ok=True)\nos.chdir(ROOT)\nif not Path('private/candidate_pack/data/train.jsonl').exists():\n    uploaded=files.upload()\n    archive=next(name for name in uploaded if name.endswith('.zip'))\n    with zipfile.ZipFile(archive) as z:\n        expected={{'candidate_pack/ASSIGNMENT.md','candidate_pack/SCHEMA.md','candidate_pack/score.py','candidate_pack/data/train.jsonl','candidate_pack/data/dev.jsonl','candidate_pack/data/test_inputs.jsonl'}}\n        assert expected.issubset(set(z.namelist()))\n        for name in expected:\n            path=ROOT/'private'/name\n            path.parent.mkdir(parents=True,exist_ok=True)\n            path.write_bytes(z.read(name))\n    del uploaded\nDATA=ROOT/'private/candidate_pack/data'\nOUT=ROOT/'artifacts'\nif USE_DRIVE_BACKUP:\n    drive.mount('/content/drive')\n    BACKUP=Path('/content/drive/MyDrive/nimbuspay-ticket-triage-private')\n    BACKUP.mkdir(exist_ok=True)\n    OUT=BACKUP/'artifacts'\nelse:\n    BACKUP=None\nSUBMISSION=ROOT/'submission'\nOUT.mkdir(parents=True,exist_ok=True)\n")
    markdown('## Source: auditable training-only cleaning\n\nLabels are corrected by deterministic code and the supplied schema. The repair function is never used for inference. Ambiguous or truncated rows are dropped. Original category/language labels are retained for substantive tickets; the audit does not claim to independently relabel every category.')
    for filename in ['audit.py','score.py','training.py','test_audit.py','test_recovery.py']:
        code(f'%%writefile {filename}\n'+(root/filename).read_text())
    code("import unittest\nfrom test_audit import AuditTests\nresult=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))\nassert result.wasSuccessful()\n")
    markdown('## Audit, sequence lengths, and loss masking\n\nAll examples retain their complete native chat template. The sequence bound is the observed maximum rounded to 128, with dynamic padding. Only assistant-response/end tokens contribute loss. Dev/test are never passed to the training dataset.')
    code("from training import *\ntokenizer, datasets, maximum = prepare(DATA,OUT)\nDEV=load(DATA/'dev.jsonl')\nTEST=load(DATA/'test_inputs.jsonl')\nassert len(DEV)==200 and len(TEST)==400\nassert all(row['id'].startswith('tr-') for rows in datasets.values() for row in rows)\nexample=encode_row(tokenizer,datasets['clean'][0])\nprint('Mask verification:',{'prompt_tokens':example['labels'].count(-100),'supervised_tokens':sum(x!=-100 for x in example['labels']),'sequence_tokens':len(example['input_ids'])})\n")
    markdown('## Baselines and controlled experiments\n\nA is raw QLoRA, B is cleaned QLoRA, C changes precision to FP16 LoRA, D changes only learning rate, and E changes only equivalent input notation. Fixed 160 updates keep optimizer exposure consistent; effective epochs differ when cleaning changes row count. Every run starts from the original base weights. Greedy dev evaluation uses the supplied scorer. A rules-in-prompt baseline is a reference only.')
    code("if RUN_ALL_EXPERIMENTS:\n    if not (OUT/'baseline_rules/result.json').exists():\n        baseline(tokenizer,DEV,OUT,(ROOT/'private/candidate_pack/SCHEMA.md').read_text())\n    for name in ['B','A','C','D','E']:\n        train_run(name,tokenizer,datasets[RUNS[name]['data']],maximum,OUT,DEV)\nelse:\n    # Default reproduction starts a new run directory, never silently reuses an old adapter.\n    OUT=(BACKUP if BACKUP is not None else ROOT)/f'reproduction_{time.time_ns()}'\n    OUT.mkdir(exist_ok=True)\n    tokenizer,datasets,maximum=prepare(DATA,OUT)\n    train_run(FINAL_RUN,tokenizer,datasets[RUNS[FINAL_RUN]['data']],maximum,OUT,DEV)\n")
    markdown('## Export and reproducibility\n\nInference uses original system/user messages, greedy `model.generate(max_new_tokens=200)`, and whitespace stripping only after normal tokenizer decoding. No fields are repaired. A seeded test sample is regenerated after reloading the adapter. Failed/non-JSON model outputs remain unchanged in the prediction file.')
    code("best=export_best(tokenizer,DEV,TEST,OUT,SUBMISSION)\nprint('Selected run:',json.dumps(best,indent=2))\nprint('Notebook elapsed minutes:',(time.perf_counter()-NOTEBOOK_STARTED)/60)\nassert (time.perf_counter()-NOTEBOOK_STARTED)<3*60*60 or RUN_ALL_EXPERIMENTS, 'Default reproduction exceeded 3 hours'\n# Only code-safe summaries, dev predictions and adapters are exported. Training rows remain private.\nPUBLIC_RESULTS=SUBMISSION/'results'\nPUBLIC_RESULTS.mkdir(exist_ok=True)\nfor path in OUT.rglob('*.json'):\n    if 'checkpoints' not in path.parts and 'adapter' not in path.parts:\n        target=PUBLIC_RESULTS/path.relative_to(OUT)\n        target.parent.mkdir(parents=True,exist_ok=True)\n        shutil.copy2(path,target)\nfor name in list(RUNS)+['baseline_fixed','baseline_rules']:\n    source=OUT/name/'predictions_dev.jsonl'\n    if source.exists():\n        target=PUBLIC_RESULTS/name/source.name\n        target.parent.mkdir(parents=True,exist_ok=True)\n        shutil.copy2(source,target)\nif BACKUP is not None:\n    shutil.copytree(SUBMISSION,BACKUP/'submission',dirs_exist_ok=True)\narchive=shutil.make_archive('/content/nimbuspay_submission','zip',SUBMISSION)\nfiles.download(archive)\n")
    markdown('## Interpretation and AI disclosure\n\nUse the saved actual dev metrics to analyze weak fields/categories and at least ten specific failures. Peak memory is measured with CUDA allocator counters; training minutes exclude model loading and dev generation. Single-seed comparisons are exploratory. Codex assisted code, audit checks, and report preparation; no external model generated training labels or submitted predictions. The completed report summarizes limitations and a one-week follow-up plan.')
    for index,item in enumerate(cells):item['id']=f'cell-{index}'
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','colab':{'name':'notebook.ipynb','gpuType':'T4'},'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}},'cells':cells}
    (root/'notebook.ipynb').write_text(json.dumps(nb,indent=1))


def build_recovery(root):
    """Separate inference-only continuation; never trains any adapter."""
    cells=[]
    def cell(kind,text):
        result={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':result.update(execution_count=None,outputs=[])
        cells.append(result)
    cell('markdown','# Resume saved NimbusPay evaluation\n\nRun only when **free Colab T4** access is available. This notebook loads completed Drive artifacts, finishes the baseline and exports test predictions. It never calls training. Completed predictions are preserved after checking IDs; incomplete writes are rejected for inspection. The separate notebook.ipynb remains the fresh-training reproduction path.')
    cell('code',"import os\nos.environ['USE_TF']='0'\nos.environ['USE_FLAX']='0'\nos.environ['TOKENIZERS_PARALLELISM']='false'\n%pip install -q "+' '.join((root/'requirements.txt').read_text().splitlines())+'\n')
    cell('code',"from pathlib import Path\nimport torch, sys, zipfile, shutil\nfrom google.colab import drive, files\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Free T4 required; do not use the CPU recovery runtime for inference'\nROOT=Path('/content/nimbuspay_resume')\nROOT.mkdir(exist_ok=True)\nos.chdir(ROOT);sys.path.insert(0,str(ROOT))\ndrive.mount('/content/drive')\nBACKUP=Path('/content/drive/MyDrive/nimbuspay-ticket-triage-private')\nOUT=BACKUP/'artifacts'\nassert OUT.exists()\nuploaded=files.upload()\narchive=next(name for name in uploaded if name.endswith('.zip'))\nwith zipfile.ZipFile(archive) as z:\n    for name in ['candidate_pack/SCHEMA.md','candidate_pack/data/dev.jsonl','candidate_pack/data/test_inputs.jsonl']:\n        path=ROOT/'private'/name\n        path.parent.mkdir(parents=True,exist_ok=True)\n        path.write_bytes(z.read(name))\ndel uploaded\nDATA=ROOT/'private/candidate_pack/data'\nSUBMISSION=BACKUP/'submission'\n")
    for filename in ['audit.py','score.py','training.py','resume_submission.py']:
        cell('code',f'%%writefile {filename}\n'+(root/filename).read_text())
    cell('code',"from resume_submission import resume\nbest=resume(DATA,OUT,SUBMISSION,ROOT/'private/candidate_pack/SCHEMA.md')\nshutil.copy2(ROOT/'training.py',SUBMISSION/'executed_training.py')\narchive=shutil.make_archive('/content/nimbuspay_resumed_submission','zip',SUBMISSION)\nprint('Saved private submission:',SUBMISSION)\nfiles.download(archive)\n")
    for index,item in enumerate(cells):item['id']=f'resume-{index}'
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','colab':{'name':'resume.ipynb','gpuType':'T4'},'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}},'cells':cells}
    (root/'resume.ipynb').write_text(json.dumps(nb,indent=1))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--final-run',default='C');ap.add_argument('--experiments',action='store_true');ap.add_argument('--recovery',action='store_true')
    args=ap.parse_args()
    if args.recovery:build_recovery(Path(__file__).parent)
    else:build(Path(__file__).parent,args.final_run,args.experiments)
