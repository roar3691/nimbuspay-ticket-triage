"""Build a three-page report from measured scores and reviewed dev failures.

No placeholder scores or invented examples are accepted. Review failure notes
in results/error_analysis.json before calling this script.
"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from score import KEYS, load, parse, same, score


def collect(root,data):
    results=root/'results'
    runs={name:json.loads((results/name/'result.json').read_text()) for name in ['baseline_fixed','baseline_rules','A','B','C','D','E']}
    best=json.loads((results/'best_run.json').read_text())
    dev=load(data/'dev.jsonl');predictions=load(results/best['run']/'predictions_dev.jsonl')
    metrics=score(dev,predictions)
    assert metrics==json.loads((results/best['run']/'dev_metrics.json').read_text())
    lookup={p['id']:p['output'] for p in predictions}
    failures=[]
    for row in dev:
        gold=json.loads(row['messages'][-1]['content']);pred=parse(lookup[row['id']])
        wrong=[k for k in KEYS if pred is None or k not in pred or not same(pred[k],gold[k])]
        if wrong or set(pred or {})!=set(KEYS):
            failures.append({'id':row['id'],'category':gold['category'],'wrong_fields':wrong,
                             'gold':gold,'prediction':pred,'raw_output':lookup[row['id']]})
    return runs,best,metrics,failures


def build(root,data):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
    from pypdf import PdfReader
    runs,best,metrics,failures=collect(root,data)
    notes=json.loads((root/'results/error_analysis.json').read_text())
    assert len(notes)>=10 and len({n['id'] for n in notes})==len(notes)
    actual={f['id']:f for f in failures}
    assert all(n['id'] in actual and n.get('explanation') for n in notes)
    audit=json.loads((root/'results/audit_summary.json').read_text())
    lengths=json.loads((root/'results/token_lengths.json').read_text())
    env=json.loads((root/'results/environment.json').read_text())
    body=ParagraphStyle('body',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=5,textColor=colors.HexColor('#223047'))
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=10,spaceAfter=3)
    title=ParagraphStyle('title',parent=body,fontName='Helvetica-Bold',fontSize=19,leading=23,spaceAfter=10)
    heading=ParagraphStyle('heading',parent=body,fontName='Helvetica-Bold',fontSize=11,leading=14,spaceBefore=7,spaceAfter=5)
    story=[]
    def p(text,style=body): return Paragraph(text,style)
    def add(text,style=body): story.append(p(text,style))
    def table(rows,widths):
        t=Table([[p(str(v),small) for v in row] for row in rows],colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#9ba9bf')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fb')]),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.append(t)
    add('NimbusPay ticket triage',title)
    add(f"Selected run {best['run']} | Dev exact match {best['metrics']['exact_match']:.1%} | Mean field accuracy {best['metrics']['mean_field_acc']:.1%}")
    add('Training-only audit and evidence',heading)
    f=audit['findings']
    add(f"Parsed all {audit['input_rows']:,} training rows and validated JSON keys, field types, schema invariants and source evidence. Retained {audit['retained_rows']:,}; fixed {audit['actions']['fix']} original rows and dropped {audit['actions']['drop']}. Finding counts overlap across fixes. The original archive is preserved privately.")
    add(f"Dropped {f['truncated_or_invalid_json']} incomplete labels, {f['empty_or_test_ticket_unrelated_label']} empty/test tickets, and {f['duplicate']} duplicate tickets (60 byte-identical plus 17 whitespace-equivalent). Kept the earliest ID after corrected-label agreement. Converted {f['legacy_schema_prio_to_priority']} legacy labels. Ticket/schema checks repaired {f['ticket_evidence_amount_paise']} amounts with Decimal, {f['ticket_evidence_txn_date']} dates, {f['ticket_evidence_txn_id']} IDs, {f['schema_priority_order_and_thresholds']} priorities and {f['schema_needs_human_repeat_or_p1']} human-escalation fields.")
    add('Quoted-history lines are excluded only from correction evidence, while full input text is retained. Amount distractors, scratch-card mentions and prior ticket numbers are checked explicitly. Category/language labels for substantive tickets are retained; this is a targeted audit, not exhaustive relabelling. Every changed original ID appears once in cleaning_log.csv. Private evidence records before/after values and source text; nine audit tests and repeat-run byte comparisons verify determinism.')
    add('Measured baselines and controlled comparisons',heading)
    rows=[['Run / change','Exact','Fields','Peak GiB A/R','Train min']]
    labels={'baseline_fixed':'Base, fixed prompt (FP16)','baseline_rules':'Base + schema (FP16)','A':'A: raw labels, 4-bit','B':'B: cleaned, 4-bit','C':'C: B with FP16 LoRA','D':'D: B with LR 1e-4','E':'E: B with notation variants'}
    for name,r in runs.items():
        rows.append([labels[name],f"{r['metrics']['exact_match']:.1%}",f"{r['metrics']['mean_field_acc']:.1%}",f"{r['peak_gpu_allocated_bytes']/2**30:.2f}/{r['peak_gpu_reserved_bytes']/2**30:.2f}" if name in 'ABCDE' else '-',f"{r['training_minutes']:.1f}" if name in 'ABCDE' else '-'])
    table(rows,[200,48,48,92,58])
    add('A/R = peak CUDA allocated/reserved during training; elapsed excludes model load and dev generation. A vs B isolates cleaning, B vs C precision, B vs D learning rate, and B vs E input notation. All use one seed and 160 updates; cleaning changes effective epochs. Treat differences as exploratory, without statistical confidence intervals.',small)
    add('Pipeline and selection',heading)
    add(f"Qwen2.5-1.5B-Instruct (1.54B), pinned revision {best['revision'][:12]}, on {escape(env['gpu'])}. PEFT trains {best['trainable_parameters']:,} parameters. Rank 16, alpha 32, dropout .05 on attention/MLP projections; AdamW, batch 1 x accumulation 16, cosine LR 2e-4 (D: 1e-4), eight warmup steps, FP16 compute and gradient checkpointing. QLoRA uses NF4/double quantization.")
    add(f"Native chat template; prompt/header/padding/separator tokens masked, assistant JSON plus end token supervised. Observed p50/p95/p99/max: {lengths['p50']}/{lengths['p95']}/{lengths['p99']}/{lengths['maximum_observed']} tokens; bound {lengths['max_sequence_length']}, dynamic padding, no truncation/packing, all 40 long tickets retained. E replaces {audit['augmented_rows']} eligible tickets (20% of eligible) with deterministic equivalent amount/ID notation, preserving labels and size.",small)
    story.append(PageBreak())
    add('Error analysis: selected adapter',title)
    add(f"All 200 dev rows, supplied scorer unchanged. JSON valid {metrics['all']['json_valid']:.1%}; {len(failures)} rows fail exact match. The explanations below are hypotheses grounded in the observed field differences and ticket evidence, not causal proof.")
    fields=sorted(metrics['all']['fields'].items(),key=lambda pair:pair[1])
    table([['Field (weakest first)','Accuracy']]+[[k,f'{v:.1%}'] for k,v in fields],[220,80])
    categories=sorted(((name.removeprefix('category='),v) for name,v in metrics.items() if name.startswith('category=')),key=lambda pair:pair[1]['exact_match'])
    table([['Category','n','Exact','Fields']]+[[k,str(v['n']),f"{v['exact_match']:.1%}",f"{v['mean_field_acc']:.1%}"] for k,v in categories],[200,50,65,75])
    add('Actual dev failures 1-5',heading)
    def failure(number,n):
        a=actual[n['id']]
        diffs=[]
        for key in a['wrong_fields']:
            value=(a['prediction'] or {}).get(key,'<missing>')
            diffs.append(f"{key}: {json.dumps(value,ensure_ascii=True)} -> {json.dumps(a['gold'][key],ensure_ascii=True)}")
        if not diffs: diffs=['Unexpected extra JSON keys']
        # Notes are reviewed separately rather than generating stock explanations.
        return KeepTogether([p(f"<b>{number}. {escape(n['id'])} ({escape(a['category'])})</b> | prediction -> gold",small),p(escape('; '.join(diffs)),small),p(escape(n['explanation']),body)])
    for i,n in enumerate(notes[:5],1):story.append(failure(i,n))
    story.append(PageBreak())
    add('Failures, next steps and reproducibility',title)
    for i,n in enumerate(notes[5:10],6):story.append(failure(i,n))
    add('One more week',heading)
    add('Days 1-2: manually audit remaining ambiguous training categories and language labels; expand training-only coverage around measured weak rules, threshold boundaries, quoted history, relative dates and competing amounts. Days 3-4: compare balanced sampling and focused amount/date notation augmentation under the same generation contract. Days 5-6: repeat the strongest configuration across three seeds and test a small rank/step budget. Day 7: lock the selected configuration, validate a fresh T4 run and independently reload the adapter. Keep dev as evaluation-only and avoid tuning against test outputs.')
    add('Inference and practical limitations',heading)
    add('Inference uses exactly original system/user messages, native chat formatting, greedy model.generate and 200 new tokens. Decode only generated tokens with normal special-token omission, then strip whitespace; no repairs or rules. The selected PEFT adapter is reloaded independently and a seeded sample of ten test outputs must match. Complete inputs and failed outputs are preserved. Only one seed and 200 dev rows limit conclusions; hidden test may be harder. Colab package/runtime drift is recorded in environment.json.')
    add('AI assistance disclosure',heading)
    add('OpenAI Codex assisted archive inspection, deterministic audit code, training/evaluation implementation, browser operation in Colab, repository setup, verification and report drafting. Codex did not call another model/API to generate labels or submitted predictions. Training corrections are deterministic schema-based code; all predictions are raw outputs from Qwen and its PEFT adapters. Scores and examples are read from saved outputs. The candidate must review the code and explain the choices in the live defense.')
    output=root/'report.pdf'
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#69758a'))
        canvas.drawString(36,22,'NimbusPay | measured T4 experiment report')
        canvas.drawRightString(576,22,f'{doc.page}/3')
    SimpleDocTemplate(str(output),pagesize=(612,792),rightMargin=36,leftMargin=36,topMargin=32,bottomMargin=35).build(story,onFirstPage=footer,onLaterPages=footer)
    assert len(PdfReader(output).pages)==3, 'Report exceeds page budget; revise layout'
    print(output)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--root',type=Path,default=Path(__file__).parent)
    args=ap.parse_args();build(args.root,args.data)
