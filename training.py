"""Transparent PEFT training, evaluation and export for a free Colab T4."""
import copy
import gc
import importlib.metadata
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, Trainer, TrainingArguments, set_seed
from audit import augment, clean, dump_jsonl, load
from score import score

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
RUNS = {
    'A': {'data':'raw', 'four_bit':True, 'lr':2e-4},
    'B': {'data':'clean', 'four_bit':True, 'lr':2e-4},
    'C': {'data':'clean', 'four_bit':False, 'lr':2e-4},
    'D': {'data':'clean', 'four_bit':True, 'lr':1e-4},
    'E': {'data':'augmented', 'four_bit':True, 'lr':2e-4},
}


def environment():
    assert torch.cuda.is_available(), 'Training requires Colab GPU'
    assert 'T4' in torch.cuda.get_device_name(0), 'Only free Colab T4 is allowed'
    packages = {p:importlib.metadata.version(p) for p in ['torch','transformers','peft','accelerate','bitsandbytes','datasets','huggingface-hub','tokenizers','safetensors']}
    return {'gpu':torch.cuda.get_device_name(0),'gpu_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda':torch.version.cuda,'packages':packages,'model_id':MODEL_ID,'revision':REVISION}


def load_base(four_bit):
    set_seed(42)
    kwargs = {'revision':REVISION,'torch_dtype':torch.float16,'device_map':{'':0},'attn_implementation':'sdpa','trust_remote_code':False}
    if four_bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(MODEL_ID,**kwargs)


def encode_row(tokenizer,row):
    assert row['id'].startswith('tr-'), 'Dev/test must never enter tokenization for training'
    prefix = tokenizer.apply_chat_template(row['messages'][:2],tokenize=True,add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False)
    assert full[:len(prefix)] == prefix, 'Chat template must preserve assistant-header prefix'
    end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    end_position = full.index(end_id,len(prefix))
    # The template adds a separator newline after the assistant end token.
    # Supervise the complete response and end token, mask that separator too.
    labels = [-100]*len(prefix) + full[len(prefix):end_position+1] + [-100]*(len(full)-end_position-1)
    supervised = tokenizer.decode(full[len(prefix):],skip_special_tokens=False)
    assert supervised.startswith(row['messages'][-1]['content'])
    assert full[end_position] == end_id
    assert labels[end_position] == end_id
    assert all(value == -100 for value in labels[end_position+1:])
    return {'input_ids':full,'attention_mask':[1]*len(full),'labels':labels}


class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, rows, maximum):
        self.items = [encode_row(tokenizer,r) for r in rows]
        assert all(len(item['input_ids'])<=maximum for item in self.items), 'No truncation allowed'
    def __len__(self):
        return len(self.items)
    def __getitem__(self,index):
        return self.items[index]


class AssistantCollator:
    def __init__(self,tokenizer):
        self.pad = tokenizer.pad_token_id
    def __call__(self,items):
        size=math.ceil(max(len(item['input_ids']) for item in items)/8)*8
        batch={key:[] for key in ('input_ids','attention_mask','labels')}
        for item in items:
            padding=size-len(item['input_ids'])
            batch['input_ids'].append(item['input_ids']+[self.pad]*padding)
            batch['attention_mask'].append(item['attention_mask']+[0]*padding)
            batch['labels'].append(item['labels']+[-100]*padding)
        return {key:torch.tensor(values,dtype=torch.long) for key,values in batch.items()}


def prepare(data_dir,out):
    data_dir,out=Path(data_dir),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    raw=load(data_dir/'train.jsonl')
    cleaned,summary=clean(raw,out)
    augmented,n=augment(cleaned)
    dump_jsonl(out/'augmented_train.jsonl',augmented)
    summary['augmented_rows']=n
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2))
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,trust_remote_code=False)
    assert tokenizer.pad_token_id is not None
    lengths=[len(encode_row(tokenizer,row)['input_ids']) for row in raw+cleaned+augmented]
    maximum=math.ceil(max(lengths)/128)*128
    stats={'p50':int(np.percentile(lengths,50)),'p95':int(np.percentile(lengths,95)),'p99':int(np.percentile(lengths,99)),'maximum_observed':max(lengths),'max_sequence_length':maximum,'no_truncation':True}
    (out/'token_lengths.json').write_text(json.dumps(stats,indent=2))
    (out/'environment.json').write_text(json.dumps(environment(),indent=2))
    print('Audit:',json.dumps(summary));print('Token lengths:',json.dumps(stats))
    return tokenizer,{'raw':raw,'clean':cleaned,'augmented':augmented},maximum


@torch.inference_mode()
def generate_rows(model,tokenizer,rows,path,rules=None):
    """Only generation/decode/whitespace strip. No audit helpers or output repair."""
    model.eval()
    model.config.use_cache=True
    eos=model.generation_config.eos_token_id
    generation=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=200,eos_token_id=eos,pad_token_id=tokenizer.pad_token_id,use_cache=True)
    predictions=[]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    with path.open('w') as f:
        for index,row in enumerate(rows):
            messages=copy.deepcopy([m for m in row['messages'] if m['role'] in {'system','user'}])
            assert [m['role'] for m in messages] == ['system','user']
            if rules is not None:
                messages[0]['content'] += '\n\n'+rules
            inputs=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt').to(model.device)
            assert inputs.shape[1]+200 <= model.config.max_position_embeddings, 'Input exceeds model context; never truncate'
            ids=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),generation_config=generation)
            output=tokenizer.decode(ids[0,inputs.shape[1]:],skip_special_tokens=True).strip()
            prediction={'id':row['id'],'output':output}
            predictions.append(prediction);f.write(json.dumps(prediction,ensure_ascii=False)+'\n');f.flush()
            if (index+1)%25==0:print(f'{path.name}: {index+1}/{len(rows)}; {(time.perf_counter()-started)/60:.1f} min',flush=True)
    return predictions


def evaluate(model,tokenizer,dev,out,rules=None):
    out=Path(out)
    predictions=generate_rows(model,tokenizer,dev,out/'predictions_dev.jsonl',rules=rules)
    metrics=score(dev,predictions)
    (out/'dev_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Dev:',json.dumps(metrics['all']),flush=True)
    return metrics


def release(model):
    # Caller must discard its own reference before emptying the cache.
    del model
    gc.collect()
    torch.cuda.empty_cache()


def baseline(tokenizer,dev,out,rules):
    root=Path(out)
    model=load_base(False)
    first=evaluate(model,tokenizer,dev,root/'baseline_fixed')
    second=evaluate(model,tokenizer,dev,root/'baseline_rules',rules=rules)
    for name,metrics in [('baseline_fixed',first),('baseline_rules',second)]:
        (root/name/'result.json').write_text(json.dumps({'run':name,'four_bit':False,'metrics':metrics['all']},indent=2))
    del model;gc.collect();torch.cuda.empty_cache()


def train_run(name,tokenizer,rows,maximum,out,dev,max_steps=160):
    config=RUNS[name]
    destination=Path(out)/name
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'result.json').exists():
        print('Using saved completed run',name)
        return json.loads((destination/'result.json').read_text())
    set_seed(42)
    model=load_base(config['four_bit'])
    if config['four_bit']:
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    else:
        model.requires_grad_(False)
        model.enable_input_require_grads()
    model.config.use_cache=False
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    model=get_peft_model(model,lora)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < trainable <= 50_000_000
    assert all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
    dataset=TicketDataset(tokenizer,rows,maximum)
    kwargs=dict(output_dir=str(destination/'checkpoints'),max_steps=max_steps,per_device_train_batch_size=1,gradient_accumulation_steps=16,learning_rate=config['lr'],warmup_steps=8,lr_scheduler_type='cosine',optim='adamw_torch',weight_decay=0,fp16=True,bf16=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},max_grad_norm=1.0,logging_steps=10,save_strategy='steps',save_steps=50,save_total_limit=1,report_to='none',seed=42,data_seed=42,group_by_length=True,dataloader_num_workers=0,remove_unused_columns=False)
    args=TrainingArguments(**kwargs)
    (destination/'configuration.json').write_text(json.dumps({'run':name,'model_id':MODEL_ID,'revision':REVISION,'training_arguments':args.to_dict(),'lora':lora.to_dict(),'four_bit':config['four_bit'],'quantization':{'type':'nf4','double_quantization':True,'compute_dtype':'float16'} if config['four_bit'] else None,'trainable_parameters':trainable},indent=2,default=lambda value: sorted(value) if isinstance(value,set) else str(value)))
    trainer=Trainer(model=model,args=args,train_dataset=dataset,data_collator=AssistantCollator(tokenizer),processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started=time.perf_counter()
    candidates=sorted((destination/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    trainer.train(resume_from_checkpoint=str(candidates[-1]) if candidates else None)
    assert trainer.state.global_step == max_steps
    assert all(math.isfinite(step['loss']) for step in trainer.state.log_history if 'loss' in step), 'Non-finite training loss'
    torch.cuda.synchronize()
    minutes=(time.perf_counter()-started)/60
    peak_allocated=torch.cuda.max_memory_allocated()
    peak_reserved=torch.cuda.max_memory_reserved()
    model.save_pretrained(destination/'adapter',safe_serialization=True)
    tokenizer.save_pretrained(destination/'adapter')
    (destination/'training_log.json').write_text(json.dumps(trainer.state.log_history,indent=2))
    model.gradient_checkpointing_disable()
    metrics=evaluate(model,tokenizer,dev,destination)
    result={'run':name,**config,'seed':42,'train_rows':len(rows),'trainable_parameters':trainable,'max_steps':max_steps,'effective_batch_size':16,'max_sequence_length':maximum,'training_minutes':minutes,'peak_gpu_allocated_bytes':peak_allocated,'peak_gpu_reserved_bytes':peak_reserved,'metrics':metrics['all'],'model_id':MODEL_ID,'revision':REVISION}
    (destination/'result.json').write_text(json.dumps(result,indent=2))
    print('RESULT:',json.dumps(result),flush=True)
    del trainer,model,dataset;gc.collect();torch.cuda.empty_cache()
    return result


def select_best(out):
    results=[json.loads(path.read_text()) for path in Path(out).glob('[A-E]/result.json')]
    assert results,'No completed training runs'
    best=max(results,key=lambda r:(r['metrics']['exact_match'],r['metrics']['mean_field_acc'],-r['peak_gpu_allocated_bytes']))
    (Path(out)/'best_run.json').write_text(json.dumps(best,indent=2))
    return best


def export_best(tokenizer,dev,test,out,submission):
    out,submission=Path(out),Path(submission)
    best=select_best(out)
    submission.mkdir(parents=True,exist_ok=True)
    shutil.copytree(out/best['run']/'adapter',submission/'adapter',dirs_exist_ok=True)
    shutil.copy2(out/'cleaning_log.csv',submission/'cleaning_log.csv')
    model=PeftModel.from_pretrained(load_base(best['four_bit']),submission/'adapter')
    predictions=generate_rows(model,tokenizer,test,submission/'predictions_test.jsonl')
    assert len(predictions)==len(test)==400
    assert {p['id'] for p in predictions}=={r['id'] for r in test}
    del model;gc.collect();torch.cuda.empty_cache()
    model=PeftModel.from_pretrained(load_base(best['four_bit']),submission/'adapter')
    chosen=random.Random(42).sample(test,10)
    rerun=generate_rows(model,tokenizer,chosen,out/'reproduction_sample.jsonl')
    lookup={p['id']:p['output'] for p in predictions}
    assert all(p['output']==lookup[p['id']] for p in rerun),'Adapter reload/sample reproduction mismatch'
    (out/'reproduction_check.json').write_text(json.dumps({'sample_size':10,'all_outputs_match':True,'seed':42},indent=2))
    del model;gc.collect();torch.cuda.empty_cache()
    return best
