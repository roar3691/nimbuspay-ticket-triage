"""Training-only, evidence-based label repair. Never imported by inference."""
import argparse
import copy
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

KEYS = ('category', 'priority', 'amount_paise', 'txn_date', 'txn_id', 'channel', 'language', 'needs_human')
CATEGORIES = {'payment_failed', 'refund', 'fraud', 'account_access', 'kyc', 'offers', 'other'}
NO_TRANSACTION = {'account_access', 'kyc', 'other'}
LEGAL = re.compile(r'\b(RBI|ombudsman|consumer court|lawyer|legal notice|legal action)\b', re.I)
REPEAT = re.compile(r'third time|already written \d+ times|contacted support about this before|again and again|teesri baar|pehle bhi complaint|baar baar complaint', re.I)
ID_PATTERN = re.compile(r'\bnp[\s\-]*\d(?:[\s\-]*\d){9}(?!\d)', re.I)
NUM = r'\d[\d,]*(?:\.\d+)?'
MONEY = re.compile(rf'(?:₹|\bINR\b|\bRs\.?)\s*(?P<a>{NUM})\s*(?P<as>lakhs?|k\b)?|(?<![\w.])(?P<b>{NUM})\s*(?P<bs>lakhs?\b|k\b|rupees\b|rupaye\b)', re.I)
DISTRACTOR = re.compile(r'account balance|daily limit|balance is|account mein abhi|fee(?:s)?\b|limit is', re.I)
MONTHS = {m: i for i, m in enumerate(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'), 1)}
MONTH = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def body(text):
    """Evidence ignores Received/Source metadata and earlier quoted emails."""
    return '\n'.join(line for line in text.splitlines()[2:] if not line.lstrip().startswith('>'))


def amount_evidence(text):
    candidates = []
    for match in MONEY.finditer(text):
        prefix = text[max(0, match.start()-70):match.start()].rsplit('. ', 1)[-1].rsplit('\n', 1)[-1]
        if DISTRACTOR.search(prefix):
            continue
        value = Decimal((match['a'] or match['b']).replace(',', ''))
        suffix = (match['as'] or match['bs'] or '').lower()
        multiplier = 100000 if suffix.startswith('lakh') else 1000 if suffix == 'k' else 1
        paise = value * multiplier * 100
        if paise != paise.to_integral_value():
            raise ValueError('amount_has_fractional_paise')
        candidates.append((int(paise), match.group(), match.span()))
    values = {v for v, _, _ in candidates}
    if len(values) > 1:
        raise ValueError('ambiguous_disputed_amount')
    return (next(iter(values)) if values else None), candidates


def resolved_date(text, received):
    text = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)\b', r'\1', text, flags=re.I)
    found = []
    for m in re.finditer(r'(?<![\d/\-])(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})(?!\d)', text):
        day, month, year = map(int, m.groups())
        found.append(date(year if year >= 100 else 2000+year, month, day))
    for pattern, order in [(rf'\b(\d{{1,2}})\s+({MONTH})(?:\s+(\d{{4}}))?\b', 'dm'), (rf'\b({MONTH})\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b', 'md')]:
        for m in re.finditer(pattern, text, re.I):
            day = int(m[1] if order == 'dm' else m[2])
            month = MONTHS[(m[2] if order == 'dm' else m[1])[:3].lower()]
            year = int(m[3]) if m[3] else received.year
            parsed = date(year, month, day)
            if not m[3] and parsed > received:
                parsed = date(year-1, month, day)
            found.append(parsed)
    # Longer relative phrases take precedence over substrings such as yesterday.
    relative = text.lower()
    if re.search(r'day before yesterday|\bparso\b', relative):
        found.append(received-timedelta(days=2))
        relative = re.sub(r'day before yesterday|\bparso\b', '', relative)
    for m in re.finditer(r'\b(\d+)\s+(?:days? ago|din pehle)\b', relative):
        found.append(received-timedelta(days=int(m[1])))
    if re.search(r'\byesterday\b|\bkal\b', relative):
        found.append(received-timedelta(days=1))
    if re.search(r'\btoday\b|\baaj\b', relative):
        found.append(received)
    values = {d.isoformat() for d in found}
    if len(values) > 1:
        raise ValueError('ambiguous_transaction_date')
    return next(iter(values)) if values else None


def priority(category, amount, text):
    if category == 'fraud' or (amount is not None and amount >= 5000000) or LEGAL.search(text):
        return 'P1'
    if category == 'account_access' or (category in {'payment_failed', 'refund'} and amount is not None and amount >= 500000):
        return 'P2'
    return 'P3'


def validate(label):
    assert set(label) == set(KEYS)
    assert label['category'] in CATEGORIES
    assert label['priority'] in {'P1','P2','P3'}
    assert label['language'] in {'en','hi-en'}
    assert type(label['needs_human']) is bool
    assert label['amount_paise'] is None or type(label['amount_paise']) is int
    assert label['channel'] in {None,'upi','card','netbanking','wallet'}
    assert label['txn_id'] is None or re.fullmatch(r'NP\d{10}', label['txn_id'])
    if label['txn_date'] is not None:
        assert date.fromisoformat(label['txn_date']).isoformat() == label['txn_date']
    if label['category'] in NO_TRANSACTION:
        assert all(label[k] is None for k in ('amount_paise','txn_date','txn_id','channel'))
    assert label['priority'] != 'P1' or label['needs_human'] is True


def repair(row):
    original = json.loads(row['messages'][-1]['content'])
    label = copy.deepcopy(original)
    text = body(row['messages'][1]['content'])
    if re.fullmatch(r'\s*(?:test(?: test)?(?: 123)?|asdf asdf|hi|hello|hey|[.!?]*)\s*', text, re.I):
        raise ValueError('empty_or_test_ticket_unrelated_label')
    reasons = []
    if 'prio' in label:
        label['priority'] = {'low':'P3','medium':'P2','high':'P1'}[label.pop('prio')]
        reasons.append('legacy_schema_prio_to_priority')
    if label['category'] not in CATEGORIES:
        raise ValueError('unknown_category')
    if label['category'] in NO_TRANSACTION:
        for k in ('amount_paise','txn_date','txn_id','channel'):
            if label[k] is not None:
                reasons.append('non_transaction_category_' + k)
            label[k] = None
    else:
        received = date.fromisoformat(row['messages'][1]['content'].splitlines()[0].split()[1])
        amount, _ = amount_evidence(text)
        parsed_date = resolved_date(text, received)
        ids = {re.sub(r'[^A-Z0-9]', '', m.group().upper()) for m in ID_PATTERN.finditer(text)}
        if len(ids) > 1:
            raise ValueError('ambiguous_nimbuspay_transaction_id')
        mode_matches = []
        mode_text = re.sub(r'\b(?:scratch|srcatch)\s+card\b', 'reward', text, flags=re.I)
        for mode, pattern in [('netbanking',r'\bnetbanking\b|\binternet banking\b'),('upi',r'\bupi\b'),('wallet',r'\bwallet\b'),('card',r'\bcard\b|\bvisa\b|\bmastercard\b')]:
            if re.search(pattern, mode_text, re.I):
                mode_matches.append(mode)
        # Multiple modes require semantic review rather than guessing.
        if len(mode_matches) > 1:
            raise ValueError('ambiguous_transaction_channel')
        values = {'amount_paise':amount,'txn_date':parsed_date,'txn_id':next(iter(ids)) if ids else None,'channel':mode_matches[0] if mode_matches else None}
        for key, value in values.items():
            if type(label[key]) is not type(value) or label[key] != value:
                reasons.append('ticket_evidence_' + key)
                label[key] = value
    expected = priority(label['category'], label['amount_paise'], text)
    if label['priority'] != expected:
        reasons.append('schema_priority_order_and_thresholds')
        label['priority'] = expected
    human = expected == 'P1' or bool(REPEAT.search(text))
    if label['needs_human'] is not human:
        reasons.append('schema_needs_human_repeat_or_p1')
        label['needs_human'] = human
    validate(label)
    return {k:label[k] for k in KEYS}, reasons


def clean(rows, out):
    assert all(row['id'].startswith('tr-') for row in rows), 'Only original training rows are accepted'
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    retained, log, evidence = [], [], []
    counters = Counter()
    for row in sorted(rows, key=lambda r:r['id']):
        user = row['messages'][1]['content']
        normalized = re.sub(r'\s+', ' ', user).strip()
        try:
            label, reasons = repair(row)
            if normalized in groups:
                prior_id, prior_label = groups[normalized]
                if label != prior_label:
                    raise ValueError('conflicting_normalized_duplicate')
                reason = 'duplicate_ticket_keep_' + prior_id
                log.append({'id':row['id'],'action':'drop','reason':reason})
                counters['duplicate'] += 1
                evidence.append({'id':row['id'],'drop_reason':reason,'kept_id':prior_id,'ticket':user,'before':row['messages'][-1]['content'],'corrected_label_agrees':True})
                continue
            groups[normalized] = (row['id'], label)
            cleaned = copy.deepcopy(row)
            cleaned['messages'][-1]['content'] = json.dumps(label, ensure_ascii=False)
            retained.append(cleaned)
            if reasons:
                log.append({'id':row['id'],'action':'fix','reason':';'.join(reasons)})
                counters.update(reasons)
                evidence.append({'id':row['id'],'before':row['messages'][-1]['content'],'after':label,'ticket':user,'rules':reasons})
        except (json.JSONDecodeError, ValueError, AssertionError, KeyError) as exc:
            reason = 'truncated_or_invalid_json' if isinstance(exc,json.JSONDecodeError) else str(exc) or 'schema_validation_failure'
            log.append({'id':row['id'],'action':'drop','reason':reason})
            counters[reason] += 1
            evidence.append({'id':row['id'],'drop_reason':reason,'ticket':user,'before':row['messages'][-1]['content']})
    dump_jsonl(out/'cleaned_train.jsonl',retained)
    dump_jsonl(out/'raw_training_evidence.jsonl',evidence)
    with (out/'cleaning_log.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','action','reason']);writer.writeheader();writer.writerows(log)
    summary={'input_rows':len(rows),'retained_rows':len(retained),'actions':dict(Counter(r['action'] for r in log)),'findings':dict(counters),'long_ticket_count':sum(len(r['messages'][1]['content'])>1500 for r in rows),'retained_long_ticket_count':sum(len(r['messages'][1]['content'])>1500 for r in retained),'training_sha256':hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2))
    return retained,summary


def augment(rows, seed=42):
    """Equivalent money/id notation observed in train; no new labels."""
    import random
    rng = random.Random(seed)
    result = copy.deepcopy(rows)
    eligible = [i for i,r in enumerate(rows) if any(json.loads(r['messages'][-1]['content'])[key] is not None for key in ('txn_id','amount_paise'))]
    rng.shuffle(eligible)
    selected = eligible[:round(.2*len(eligible))]
    for i in selected:
        row = result[i]
        label = json.loads(row['messages'][-1]['content'])
        text = row['messages'][1]['content']
        if label['txn_id'] is not None:
            digits = label['txn_id'][2:]
            variants = ['np'+digits, 'NP-'+digits, 'NP '+digits[:5]+' '+digits[5:]]
            match = next(m for m in ID_PATTERN.finditer(text) if re.sub(r'[^A-Z0-9]','',m.group().upper()) == label['txn_id'])
            replacement = next(v for v in variants if v != match.group())
            row['messages'][1]['content'] = text[:match.start()] + replacement + text[match.end():]
        else:
            _, candidates = amount_evidence(body(text))
            original = candidates[0][1].rstrip()
            value = Decimal(label['amount_paise'])/100
            variants = [f'INR {value:.2f}', f'₹{value:.2f}', f'Rs {value:.2f}']
            replacement = next(v for v in variants if v != original)
            row['messages'][1]['content'] = text.replace(original,replacement,1)
        repaired,_ = repair(row)
        assert repaired == label, 'Augmentation must preserve every label field'
    assert len(result) == len(rows)
    return result, len(selected)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--out',default='artifacts')
    args=ap.parse_args();rows,summary=clean(load(args.train),args.out)
    augmented,n=augment(rows);dump_jsonl(Path(args.out)/'augmented_train.jsonl',augmented)
    summary['augmented_rows']=n
    print(json.dumps(summary,indent=2))
