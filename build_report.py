"""Build a three-page report from measured scores and reviewed dev failures.

No placeholder scores or invented examples are accepted. Review failure notes
in results/error_analysis.json before calling this script.
"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from score import KEYS, load, parse, same, score


def collect(root,data,allow_incomplete=False):
    results=root/'results'
    runs={}
    for name in ['baseline_fixed','baseline_rules','A','B','C','D','E']:
        path=results/name/'result.json'
        assert path.exists() or allow_incomplete, f'Missing measured result: {name}'
        runs[name]=json.loads(path.read_text()) if path.exists() else None
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


def build(root,data,allow_incomplete=False):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
    from pypdf import PdfReader
    runs,best,metrics,failures=collect(root,data,allow_incomplete)
    status=json.loads((root/'results/submission_status.json').read_text()) if (root/'results/submission_status.json').exists() else {}
    if not allow_incomplete:
        assert status.get('test_predictions_complete') and status.get('fresh_free_t4_notebook_verified'), 'Final report requires completed GPU acceptance work'
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
    if allow_incomplete:
        add('<b>DRAFT - GPU quota blocker.</b> Rules baseline, test predictions, adapter reload and fresh notebook verification remain incomplete. All A-E dev scores below were independently rescored from recovered outputs.',small)
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
        if r is None:
            rows.append([labels[name],'Pending','Pending','-','-'])
            continue
        rows.append([labels[name],f"{r['metrics']['exact_match']:.1%}",f"{r['metrics']['mean_field_acc']:.1%}",f"{r['peak_gpu_allocated_bytes']/2**30:.2f}/{r['peak_gpu_reserved_bytes']/2**30:.2f}" if name in 'ABCDE' else '-',f"{r['training_minutes']:.1f}" if name in 'ABCDE' else '-'])
    table(rows,[200,48,48,92,58])
    add('A/R = peak CUDA allocated/reserved during training; elapsed excludes model load and dev generation. A vs B isolates cleaning, B vs C precision, B vs D learning rate, and B vs E input notation. All use one seed and 160 updates; cleaning changes effective epochs. Treat differences as exploratory, without statistical confidence intervals.',small)
    add('The fixed baseline wraps all 200 responses in Markdown fences, giving 0% under the unchanged strict scorer. Its first output also uses a different ticket schema. JSON validity in the table means parseable JSON object, not complete schema compliance.',small)
    add('Pipeline and selection',heading)
    add(f"Qwen2.5-1.5B-Instruct (1.54B), pinned revision {best['revision'][:12]}, on {escape(env['gpu'])}. PEFT trains {best['trainable_parameters']:,} parameters. Rank 16, alpha 32, dropout .05 on attention/MLP projections; AdamW, batch 1 x accumulation 16, cosine LR 2e-4 (D: 1e-4), eight warmup steps, FP16 compute and gradient checkpointing. QLoRA uses NF4/double quantization.")
    add(f"Native chat template; prompt/header/padding/separator tokens masked, assistant JSON plus end token supervised. Across raw, clean and augmented sequences, p50/p95/p99/max: {lengths['p50']}/{lengths['p95']}/{lengths['p99']}/{lengths['maximum_observed']} tokens; bound {lengths['max_sequence_length']}, dynamic padding, no truncation/packing. All 40 original long rows were measured; 36 unique long tickets remain after deduplication. E replaces {audit['augmented_rows']} eligible tickets (20% of eligible) with deterministic equivalent amount/ID notation, preserving labels and size.",small)
    story.append(PageBreak())
    add('Error analysis: selected adapter',title)
    add(f"All 200 dev rows, supplied scorer unchanged. JSON valid {metrics['all']['json_valid']:.1%}; {len(failures)} rows fail exact match. The explanations below are hypotheses grounded in the observed field differences and ticket evidence, not causal proof.")
    fields=sorted(metrics['all']['fields'].items(),key=lambda pair:pair[1])
    categories=sorted(((name.removeprefix('category='),v) for name,v in metrics.items() if name.startswith('category=')),key=lambda pair:pair[1]['exact_match'])
    rows=[['Field (weakest first)','Accuracy','Category','n','Exact','Fields']]
    for index in range(max(len(fields),len(categories))):
        field=[fields[index][0],f'{fields[index][1]:.1%}'] if index<len(fields) else ['','']
        if index<len(categories):
            category,value=categories[index]
            group=[category,str(value['n']),f"{value['exact_match']:.1%}",f"{value['mean_field_acc']:.1%}"]
        else:
            group=['','','','']
        rows.append(field+group)
    table(rows,[115,55,140,35,55,55])
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
    add('Inference uses original system/user messages, native chat formatting, greedy model.generate and 200 new tokens. Decode generated tokens with normal special-token omission and strip whitespace; no repairs or rules. Acceptance requires reloading the adapter and matching ten seeded test outputs. One seed and 200 dev rows limit conclusions; hidden test may be harder. Package/runtime drift is recorded in environment.json.')
    if allow_incomplete:
        add('At quota exhaustion, Colab refused another GPU connection. All A-E adapters/results and fixed-baseline outputs survived in private Drive; the partial rules-baseline file did not persist. CPU recovery downloaded the winning 73.9 MB adapter and verified all completed scores. Recovery code now closes each prediction record. GPU inference and the fresh free-T4 check remain pending; no alternate hardware or invented metrics are substituted.',small)
    add('AI assistance disclosure',heading)
    add('OpenAI Codex assisted archive inspection, deterministic audit code, training/evaluation implementation, browser operation in Colab, repository setup, verification and report drafting. Codex did not call another model/API to generate labels or submitted predictions. Training corrections are deterministic schema-based code; all predictions are raw outputs from Qwen and its PEFT adapters. Scores and examples are read from saved outputs. The candidate must review the code and explain the choices in the live defense.')
    output=root/('private/report_draft.pdf' if allow_incomplete else 'report.pdf')
    output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#69758a'))
        canvas.drawString(36,22,'NimbusPay | '+('DRAFT | ' if allow_incomplete else '')+'measured T4 experiment report')
        canvas.drawRightString(576,22,f'{doc.page}/3')
    SimpleDocTemplate(str(output),pagesize=(612,792),rightMargin=36,leftMargin=36,topMargin=32,bottomMargin=35).build(story,onFirstPage=footer,onLaterPages=footer)
    assert len(PdfReader(output).pages)==3, 'Report exceeds page budget; revise layout'
    print(output)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--root',type=Path,default=Path(__file__).parent);ap.add_argument('--allow-incomplete',action='store_true')
    args=ap.parse_args();build(args.root,args.data,args.allow_incomplete)
