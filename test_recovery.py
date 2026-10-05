"""CPU-only interruption tests for prediction-file recovery, without ML imports."""
import ast
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


class Tensor:
    shape=(1,3)
    def to(self,device):return self


class Tokens:
    def __init__(self,output):self.output=output
    def __getitem__(self,key):return self.output


class Tokenizer:
    pad_token_id=0
    def apply_chat_template(self,messages,**kwargs):return Tensor()
    def decode(self,tokens,**kwargs):return tokens


class Model:
    device='cpu'
    config=SimpleNamespace(use_cache=False,max_position_embeddings=16384)
    generation_config=SimpleNamespace(eos_token_id=1)
    def __init__(self,output='raw non-JSON output',fail_on=None):
        self.calls=0;self.output=output;self.fail_on=fail_on
    def eval(self):pass
    def generate(self,**kwargs):
        self.calls+=1
        if self.calls==self.fail_on:raise RuntimeError('simulated runtime interruption')
        return Tokens(self.output)


tree=ast.parse(Path(__file__).with_name('training.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_rows')
function.decorator_list=[]
namespace={'copy':copy,'json':json,'time':time,'Path':Path,
           'GenerationConfig':lambda **kwargs:SimpleNamespace(**kwargs),
           'torch':SimpleNamespace(ones_like=lambda value:value)}
exec(compile(ast.Module(body=[function],type_ignores=[]),'training.py','exec'),namespace)
generate_rows=namespace['generate_rows']


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'predictions.jsonl'
        self.rows=[{'id':f'dv-{i}','messages':[{'role':'system','content':'fixed'},{'role':'user','content':'full ticket'}]} for i in range(2)]
    def test_interrupted_records_survive_and_resume_only_missing_ids(self):
        first=Model(output='not valid JSON',fail_on=2)
        with self.assertRaisesRegex(RuntimeError,'interruption'):
            generate_rows(first,Tokenizer(),self.rows,self.path)
        completed=json.loads(self.path.read_text())
        self.assertEqual(completed,{'id':'dv-0','output':'not valid JSON'})
        second=Model(output='second raw answer')
        predictions=generate_rows(second,Tokenizer(),self.rows,self.path,resume=True)
        self.assertEqual(second.calls,1)
        self.assertEqual(predictions[0],completed)
        self.assertEqual([p['id'] for p in predictions],['dv-0','dv-1'])
    def test_completed_file_is_preserved_without_generation(self):
        original=[{'id':r['id'],'output':'  unchanged non-JSON\n  '} for r in self.rows]
        self.path.write_text(''.join(json.dumps(p)+'\n' for p in original))
        before=self.path.read_bytes();model=Model()
        self.assertEqual(generate_rows(model,Tokenizer(),self.rows,self.path,resume=True),original)
        self.assertEqual(model.calls,0);self.assertEqual(self.path.read_bytes(),before)
    def test_wrong_id_prefix_is_rejected_without_writing(self):
        self.path.write_text(json.dumps({'id':'wrong','output':'raw'})+'\n')
        before=self.path.read_bytes()
        with self.assertRaises(AssertionError):generate_rows(Model(),Tokenizer(),self.rows,self.path,resume=True)
        self.assertEqual(self.path.read_bytes(),before)
    def test_torn_write_is_rejected_without_repair(self):
        self.path.write_text('{"id":"dv-0","output":"unfinished')
        before=self.path.read_bytes()
        with self.assertRaisesRegex(AssertionError,'Incomplete JSONL'):
            generate_rows(Model(),Tokenizer(),self.rows,self.path,resume=True)
        self.assertEqual(self.path.read_bytes(),before)


if __name__=='__main__':unittest.main()
