"""Resume baseline inference and test export from completed Drive runs.

Run only after free Colab T4 access returns. This entry point never calls training.
Existing baseline JSONL records must form an intact prefix of the dev IDs.
"""
import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer
from training import MODEL_ID, REVISION, baseline, environment, export_best, load


def resume(data, out, submission, schema):
    print('Resume environment:', json.dumps(environment()), flush=True)
    for name in 'ABCDE':
        result=json.loads((out/name/'result.json').read_text())
        assert result['max_steps']==160 and result['metrics']['n']==200
        assert result['model_id']==MODEL_ID and result['revision']==REVISION
        assert (out/name/'adapter/adapter_model.safetensors').exists()
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,trust_remote_code=False)
    dev=load(data/'dev.jsonl');test=load(data/'test_inputs.jsonl')
    assert len(dev)==200 and len(test)==400
    # Complete saved baselines; the fixed baseline's 200 outputs are rescored,
    # while only missing schema-baseline IDs are generated.
    baseline(tokenizer,dev,out,schema.read_text())
    best=export_best(tokenizer,dev,test,out,submission,resume_test=True)
    results=submission/'results'
    results.mkdir(exist_ok=True)
    for path in out.rglob('*.json'):
        if 'checkpoints' not in path.parts and 'adapter' not in path.parts:
            target=results/path.relative_to(out)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    for name in ['A','B','C','D','E','baseline_fixed','baseline_rules']:
        target=results/name/'predictions_dev.jsonl'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(out/name/'predictions_dev.jsonl',target)
    print('RESUMED EVALUATION AND TEST EXPORT COMPLETE',json.dumps(best),flush=True)
    return best


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--submission',type=Path,required=True)
    ap.add_argument('--schema',type=Path,required=True)
    args=ap.parse_args()
    resume(args.data,args.out,args.submission,args.schema)
