"""Verify saved artifacts without modifying predictions or requiring a GPU."""
import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

from audit import clean, load
from score import score


def verify(root, data):
    root, data = Path(root), Path(data)
    required = ['README.md','notebook.ipynb','report.pdf','cleaning_log.csv',
                'predictions_test.jsonl','adapter/adapter_config.json',
                'adapter/adapter_model.safetensors']
    assert all((root/name).is_file() for name in required), 'Missing required artifact'
    assert all((root/name).stat().st_size > 0 for name in required)
    assert (root/'adapter/adapter_model.safetensors').stat().st_size < 100_000_000, 'Use Git LFS for large weights'
    source_scorer = data.parent/'score.py'
    assert (root/'score.py').read_bytes() == source_scorer.read_bytes(), 'Provided scorer changed'
    raw = load(data/'train.jsonl')
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        clean(raw,Path(first)); clean(raw,Path(second))
        for filename in ['cleaned_train.jsonl','cleaning_log.csv','audit_summary.json']:
            assert (Path(first)/filename).read_bytes() == (Path(second)/filename).read_bytes(), filename
        assert (root/'cleaning_log.csv').read_bytes() == (Path(first)/'cleaning_log.csv').read_bytes()
    changes = list(csv.DictReader((root/'cleaning_log.csv').open()))
    assert len({r['id'] for r in changes}) == len(changes)
    assert all(r['action'] in {'fix','drop'} and r['reason'] for r in changes)
    test, predictions = load(data/'test_inputs.jsonl'), load(root/'predictions_test.jsonl')
    assert len(test) == len(predictions) == 400
    assert len({r['id'] for r in predictions}) == 400
    assert {r['id'] for r in predictions} == {r['id'] for r in test}
    assert all(set(r) == {'id','output'} and isinstance(r['output'],str) for r in predictions)
    results = root/'results'
    dev = load(data/'dev.jsonl')
    assert len(dev) == 200
    completed = []
    for name in ['baseline_fixed','baseline_rules','A','B','C','D','E']:
        run = results/name
        saved = json.loads((run/'result.json').read_text())
        pred = load(run/'predictions_dev.jsonl')
        assert len(pred) == 200 and len({r['id'] for r in pred}) == 200
        computed = score(dev,pred)
        assert computed == json.loads((run/'dev_metrics.json').read_text()), name
        assert computed['all'] == saved['metrics'], name
        if name in 'ABCDE':
            logs = json.loads((run/'training_log.json').read_text())
            assert saved['max_steps'] == 160 and max(r.get('step',0) for r in logs) == 160
            assert saved['trainable_parameters'] < 50_000_000
            assert saved['training_minutes'] > 0
            assert saved['peak_gpu_reserved_bytes'] >= saved['peak_gpu_allocated_bytes'] > 0
            completed.append(saved)
    best = max(completed,key=lambda r:(r['metrics']['exact_match'],r['metrics']['mean_field_acc'],-r['peak_gpu_allocated_bytes']))
    assert json.loads((results/'best_run.json').read_text()) == best
    reproduction = json.loads((results/'reproduction_check.json').read_text())
    assert reproduction == {'sample_size':10,'all_outputs_match':True,'seed':42}
    nb = json.loads((root/'notebook.ipynb').read_text())
    cells = [c for c in nb['cells'] if c['cell_type']=='code']
    assert all(c.get('execution_count') is not None for c in cells), 'Notebook must be executed top to bottom'
    assert not any(o.get('output_type')=='error' for c in cells for o in c.get('outputs',[])), 'Notebook contains execution error'
    assert any(c.get('outputs') for c in cells), 'Notebook lacks actual outputs'
    # Publication should never include the original/derived training rows.
    import subprocess
    tracked = subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
    assert not any(name.startswith(('private/','data/','model_cache/')) or name.endswith('.zip') or Path(name).name in {'train.jsonl','cleaned_train.jsonl','augmented_train.jsonl','raw_training_evidence.jsonl'} for name in tracked)
    manifest = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in required}
    result = {'required_artifacts':required,'test_predictions':400,'dev_rows_per_run':200,'completed_runs':7,'selected_run':best['run'],'deterministic_cleaning':True,'artifact_sha256':manifest}
    (results/'acceptance_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--root',type=Path,default=Path(__file__).parent)
    args=ap.parse_args();verify(args.root,args.data)
