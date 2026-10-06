"""Create a PRIVATE, checksummed transfer bundle from saved NimbusPay progress.

The generated ZIP includes the assessment data. Never commit or publish it.
Only the selected adapter is required for inference; other completed adapters
are included when present in private/all_adapters.
"""
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def notebook(root):
    cells=[]
    def add(kind,text):
        cell={'id':f'portable-{len(cells)}','cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown', '# Resume NimbusPay from a local backup\n\nSelect **Runtime → Change runtime type → T4 GPU**. Run cells in order and upload `nimbuspay_private_transfer.zip` when prompted. This notebook restores completed experiments, finishes the schema-prompt baseline, and exports the selected C adapter’s 400 test predictions. It never trains. Keep the transfer ZIP and this notebook private: the ZIP contains assessment data. The original account’s Drive is not needed.\n\nThe separate `notebook.ipynb` is the fresh-training reproduction path; its under-three-hour verification remains outstanding.')
    add('code', "import os, time\nos.environ['USE_TF']='0'\nos.environ['USE_FLAX']='0'\nos.environ['TOKENIZERS_PARALLELISM']='false'\nRESUME_STARTED=time.perf_counter()\n%pip install -q "+' '.join((root/'requirements.txt').read_text().splitlines())+'\n')
    add('code', '''import hashlib, json, shutil, sys, zipfile
from pathlib import Path
import torch
from google.colab import files, drive
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Free Colab T4 required'
ROOT=Path('/content/nimbuspay_portable')
assert not ROOT.exists(), 'For an interrupted session, use the existing restore and rerun later cells; do not overwrite progress'
uploaded=files.upload()
assert len(uploaded)==1, 'Upload only nimbuspay_private_transfer.zip'
archive=Path(next(iter(uploaded)))
del uploaded
with zipfile.ZipFile(archive) as z:
    names=z.namelist()
    assert len(names)==len(set(names)), 'Duplicate archive entries'
    manifest=json.loads(z.read('MANIFEST.json'))
    assert set(names)==set(manifest['files'])|{'MANIFEST.json'}, 'Unexpected or missing files'
    for name,record in manifest['files'].items():
        p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts, 'Unsafe archive path'
        assert z.getinfo(name).file_size==record['bytes'], name
    ROOT.mkdir()
    z.extractall(ROOT)
for name,record in manifest['files'].items():
    h=hashlib.sha256()
    with (ROOT/name).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==record['sha256'], 'Checksum failed: '+name
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
DATA=ROOT/'private/candidate_pack/data'
OUT=ROOT/'artifacts'
SUBMISSION=ROOT/'submission'
assert (ROOT/'score.py').read_bytes()==(ROOT/'private/candidate_pack/score.py').read_bytes()
print('Private backup verified:',len(manifest['files']),'files')
print('Saved progress:',json.dumps(manifest['progress'],indent=2))
''')
    add('markdown', '## Private progress storage\n\nDrive backup is enabled by default and uses the account connected to this runtime. Review the permission prompt when mounting Drive. Set `USE_DRIVE_BACKUP=False` to work entirely inside the runtime; then use the private progress-download cell regularly because runtime files disappear on disconnect. Existing Drive progress is preserved only when its backup ID matches this bundle.')
    add('code', '''USE_DRIVE_BACKUP=True
BACKUP=None
if USE_DRIVE_BACKUP:
    drive.mount('/content/drive')
    BACKUP=Path('/content/drive/MyDrive/nimbuspay-portable-private')
    marker=BACKUP/'backup_id.txt'
    backup_id=manifest['backup_id']
    if BACKUP.exists():
        assert marker.exists() and marker.read_text()==backup_id, 'Different backup at Drive destination; choose a new folder'
    else:
        BACKUP.mkdir()
        shutil.copytree(OUT,BACKUP/'artifacts')
        marker.write_text(backup_id)
    OUT=BACKUP/'artifacts'
    SUBMISSION=BACKUP/'submission'
SUBMISSION.mkdir(exist_ok=True)
print('Resume artifact destination:',OUT)
''')
    add('code', '''import unittest
checks=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromName('test_recovery'))
assert checks.wasSuccessful()
from resume_submission import resume
from training import environment
(OUT/'resume_environment.json').write_text(json.dumps(environment(),indent=2))
try:
    best=resume(DATA,OUT,SUBMISSION,ROOT/'private/candidate_pack/SCHEMA.md')
finally:
    print('Progress files persist at:',OUT, 'and',SUBMISSION,flush=True)
print('Resume elapsed minutes:',(time.perf_counter()-RESUME_STARTED)/60)
''')
    add('markdown','## Download completed submission\n\nRun after the previous cell succeeds. This export contains predictions, selected adapter, logs and cleaning log. It excludes training data. Download this notebook through File → Download → Download .ipynb to preserve real execution outputs. Fresh training verification and the final report still require separate completion.')
    add('code', '''shutil.copy2(ROOT/'training.py',SUBMISSION/'executed_training.py')
archive=shutil.make_archive('/content/nimbuspay_resumed_submission','zip',SUBMISSION)
print('Completed submission export:',Path(archive).stat().st_size,'bytes')
files.download(archive)
''')
    add('markdown', '## Optional private progress export after an interruption\n\nThis ZIP contains private assessment data. Run manually after stopping inference, including when inference failed. It captures the latest closed prediction records for a later restart. The latest matching Drive folder is used automatically by the setup cell on subsequent sessions. Keep this ZIP local and private.')
    add('code', '''EXPORT_PRIVATE_PROGRESS=False
if EXPORT_PRIVATE_PROGRESS:
    import importlib.util
    spec=importlib.util.spec_from_file_location('portable_backup',ROOT/'build_portable_backup.py')
    backup=importlib.util.module_from_spec(spec);spec.loader.exec_module(backup)
    backup.refresh_progress(ROOT,OUT,SUBMISSION)
    backup.write_manifest(ROOT,manifest['progress'])
    archive=shutil.make_archive('/content/nimbuspay_private_transfer','zip',ROOT)
    files.download(archive)
''')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','colab':{'name':'portable_resume.ipynb','gpuType':'T4'},'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}},'cells':cells}
    for cell in cells:
        if cell['cell_type']=='code':
            ast.parse(''.join(line for line in cell['source'] if not line.startswith('%pip ')))
    return nb


def copytree(source,target):
    if source.exists():shutil.copytree(source,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.zip'))


def refresh_progress(root,out,submission):
    root,out,submission=map(Path,(root,out,submission))
    if out.resolve()!=(root/'artifacts').resolve():copytree(out,root/'artifacts')
    if submission.resolve()!=(root/'submission').resolve():copytree(submission,root/'submission')


def write_manifest(stage,progress):
    # Derive resume counts from actual files, not the original stale status.
    progress=dict(progress)
    for name in ['baseline_fixed','baseline_rules']:
        path=stage/'artifacts'/name/'predictions_dev.jsonl'
        progress[name+'_saved_rows']=len(path.read_text().splitlines()) if path.exists() else 0
    path=stage/'submission/predictions_test.jsonl'
    progress['test_saved_rows']=len(path.read_text().splitlines()) if path.exists() else 0
    files={str(p.relative_to(stage)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(stage.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
    # Keep the restore lineage stable when only inference progress changes.
    backup_id=hashlib.sha256((digest(stage/'private/training_assessment.zip')+digest(stage/'artifacts/C/adapter/adapter_model.safetensors')).encode()).hexdigest()
    (stage/'MANIFEST.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'backup_id':backup_id,'private_contains_assessment_data':True,'progress':progress,'files':files},indent=2))
    return files


def build(root,archive,output):
    root,archive,output=map(Path,(root,archive,output))
    output.mkdir(parents=True,exist_ok=True)
    stage=output/'restore'
    assert not stage.exists(), 'Choose a new output folder to avoid overwriting a previous backup'
    stage.mkdir()
    # Explicit project file selection excludes credentials and local caches.
    for pattern in ['*.py','*.md','*.ipynb','requirements.txt','cleaning_log.csv','submission_email.txt']:
        for source in root.glob(pattern):
            if source.is_file():shutil.copy2(source,stage/source.name)
    copytree(root/'results',stage/'artifacts')
    shutil.copy2(root/'cleaning_log.csv',stage/'artifacts/cleaning_log.csv')
    copytree(root/'private/all_adapters',stage/'private/all_adapters')
    all_adapters=stage/'private/all_adapters'
    available=[]
    for name in 'ABCDE':
        source=all_adapters/name/'adapter'
        if source.exists():
            shutil.move(str(source),stage/'artifacts'/name/'adapter')
            available.append(name)
    if all_adapters.exists():shutil.rmtree(all_adapters)
    if 'C' not in available:
        copytree(root/'adapter',stage/'artifacts/C/adapter')
        available.append('C')
    assert (stage/'artifacts/C/adapter/adapter_model.safetensors').exists()
    private=stage/'private';private.mkdir(exist_ok=True)
    shutil.copy2(archive,private/'training_assessment.zip')
    with zipfile.ZipFile(archive) as z:
        for name in ['candidate_pack/ASSIGNMENT.md','candidate_pack/SCHEMA.md','candidate_pack/score.py','candidate_pack/data/train.jsonl','candidate_pack/data/dev.jsonl','candidate_pack/data/test_inputs.jsonl']:
            target=private/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    assert (stage/'score.py').read_bytes()==(private/'candidate_pack/score.py').read_bytes()
    for name in ['WORK_STATUS.md','executed_preflight.ipynb','executed_backup.ipynb','report_draft.pdf','gpu_quota_blocked.jpg','recovered_artifacts.jpg']:
        if (root/'private'/name).exists():shutil.copy2(root/'private'/name,private/name)
    for name in ['audit_repeat','pasted_results']:
        copytree(root/'private'/name,private/name)
    if (root/'private/recovered/review_dev_failures.json').exists():shutil.copy2(root/'private/recovered/review_dev_failures.json',private/'review_dev_failures.json')
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    # Git history contains another copy of the selected weights. Preserve it
    # locally beside the ZIP so a Colab upload need not include those twice.
    history=output/'repository_history.bundle'
    subprocess.run(['git','bundle','create',str(history),'--all'],cwd=root,check=True)
    (private/'repository_state.json').write_text(json.dumps({'head':git_commit,'working_tree':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)},indent=2))
    progress=json.loads((root/'results/submission_status.json').read_text())
    progress['locally_preserved_adapters']=sorted(available)
    progress['optimizer_checkpoints_included']=False
    progress['base_weights_included']=False
    (stage/'portable_resume.ipynb').write_text(json.dumps(notebook(root),indent=1))
    guide='''# Private NimbusPay local transfer

This folder and ZIP contain the original assessment data. Keep them private; do not commit or publish them.

1. Open Colab → File → Upload notebook and choose `portable_resume.ipynb` from this folder.
2. Select Runtime → Change runtime type → T4 GPU.
3. Run cells in order. Upload `nimbuspay_private_transfer.zip` when the notebook asks.
4. The manifest verifies every file. Completed A–E training is skipped. The fixed baseline's 200 saved outputs are preserved. The schema baseline must generate all 200 dev rows; its interrupted rows were not recovered.
5. Drive backup defaults to the currently connected account's `nimbuspay-portable-private` folder; it never depends on the original Drive. Approve its permission prompt yourself, or set USE_DRIVE_BACKUP=False and download private progress regularly.
6. On success download `nimbuspay_resumed_submission.zip` and the executed notebook. Keep both for final local verification.

Saved: selected C adapter (62.5% dev exact match, 93.9375% mean field accuracy), all experiment scores/logs/raw dev outputs, data audit/evidence, original archive, code, and actual interrupted notebook outputs. See MANIFEST.json for the exact adapter inventory and checksums.

Still unfinished: schema baseline; 400 test predictions; independent adapter reload check; fresh default notebook under-three-hour verification; final report and submission checks. Resume inference does not satisfy fresh-training verification. Use `notebook.ipynb` separately for that requirement, uploading the preserved original archive when prompted.

The pinned public Qwen base weights are downloaded again in Colab; no Hugging Face token is required. Completed-run optimizer checkpoints are omitted because no training continuation is required. The original account's saved files are not deleted. Git history is preserved separately in `repository_history.bundle` beside this ZIP; it is not needed for the Colab upload.

Expected next outputs: backup verification, resume environment, schema baseline progress in batches of 25/200, dev metrics, test progress in batches of 25/400, a 10-row reload check, then RESUMED EVALUATION AND TEST EXPORT COMPLETE.

Inference continuation is estimated at 55–80 minutes after GPU connection and upload, based on the previous T4 timings. Fresh reproduction and final packaging add roughly 75–110 minutes; GPU availability can add an unknown delay.
'''
    (stage/'TRANSFER_README.md').write_text(guide)
    manifest=write_manifest(stage,progress)
    zip_path=output/'nimbuspay_private_transfer.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(stage.rglob('*')):
            if path.is_file():z.write(path,str(path.relative_to(stage)))
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        saved=json.loads(z.read('MANIFEST.json'))
        assert set(z.namelist())==set(saved['files'])|{'MANIFEST.json'}
        for name,record in saved['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==record['sha256'],name
    shutil.copy2(stage/'portable_resume.ipynb',output/'portable_resume.ipynb')
    shutil.copy2(stage/'TRANSFER_README.md',output/'START_HERE.md')
    shutil.copy2(stage/'MANIFEST.json',output/'MANIFEST.json')
    (output/'SHA256SUMS.txt').write_text(f'{digest(zip_path)}  {zip_path.name}\n{digest(output/"portable_resume.ipynb")}  portable_resume.ipynb\n{digest(history)}  repository_history.bundle\n')
    print(json.dumps({'zip':str(zip_path),'bytes':zip_path.stat().st_size,'files_verified':len(manifest),'adapters':sorted(available),'selected_run':'C'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    build(Path(__file__).parent,args.archive,args.output)
