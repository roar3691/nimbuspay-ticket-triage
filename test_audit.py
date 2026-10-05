import copy
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from audit import KEYS, amount_evidence, augment, clean, priority, repair, resolved_date, validate
from score import score


def row(text, **fields):
    label=dict(zip(KEYS,('refund','P3',None,None,None,None,'en',False)))
    label.update(fields)
    return {'id':'tr-00001','messages':[{'role':'system','content':"You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."},{'role':'user','content':'Received: 2026-01-02 (Fri)\nSource: email\n\n'+text},{'role':'assistant','content':json.dumps(label)}]}


class AuditTests(unittest.TestCase):
    def test_decimal_scale_and_distractors(self):
        self.assertEqual(amount_evidence('My account balance is Rs 200. Total Rs 594.50.')[0],59450)
        self.assertEqual(amount_evidence('FYI my daily limit is INR 80,000. It was for 1.3k.')[0],130000)
        self.assertEqual(amount_evidence('Account mein abhi ₹900 pade hain. Total 2.2 lakh.')[0],22000000)

    def test_threshold_precedence(self):
        self.assertEqual(priority('refund',499999,''),'P3')
        self.assertEqual(priority('refund',500000,''),'P2')
        self.assertEqual(priority('offers',4999999,''),'P3')
        self.assertEqual(priority('offers',5000000,''),'P1')
        self.assertEqual(priority('account_access',None,''),'P2')
        self.assertEqual(priority('kyc',None,'My lawyer will send a legal notice.'),'P1')

    def test_dates_and_quoted_history(self):
        self.assertEqual(resolved_date('That was day before yesterday.',date(2026,1,2)),'2025-12-31')
        self.assertEqual(resolved_date('Transaction date Dec 31.',date(2026,1,2)),'2025-12-31')
        self.assertEqual(resolved_date('This happened on 2nd August 2026.',date(2026,8,14)),'2026-08-02')
        fixed,_=repair(row('My refund was yesterday.\n> On 30 Dec 2025, support wrote: UPI INR 90,000'))
        self.assertEqual(fixed['txn_date'],'2026-01-01')
        self.assertIsNone(fixed['channel'])
        self.assertIsNone(fixed['amount_paise'])

    def test_id_scratch_card_and_repeated_contact(self):
        fixed,_=repair(row('My scratch card reward is missing. Amount INR 200. np 12345 67890. This is the third time I am contacting you.',category='offers'))
        self.assertIsNone(fixed['channel'])
        self.assertEqual(fixed['txn_id'],'NP1234567890')
        self.assertTrue(fixed['needs_human'])

    def test_prior_ticket_number_alone_is_not_repeat(self):
        fixed,_=repair(row('My refund is missing. My earlier ticket number is #12345.'))
        self.assertFalse(fixed['needs_human'])

    def test_no_transaction_categories(self):
        fixed,_=repair(row('I forgot my password. My balance is INR 100000. My debit card number is blocked.',category='account_access',amount_paise=10000000,channel='card'))
        self.assertTrue(all(fixed[k] is None for k in ('amount_paise','txn_date','txn_id','channel')))
        self.assertEqual(fixed['priority'],'P2')

    def test_invalid_truncated_duplicate_and_no_leakage(self):
        first=row('My refund is missing. INR 200.')
        duplicate=copy.deepcopy(first);duplicate['id']='tr-00002';duplicate['messages'][1]['content']+='  '
        truncated=copy.deepcopy(first);truncated['id']='tr-00003';truncated['messages'][-1]['content']='{"category":'
        with tempfile.TemporaryDirectory() as d:
            retained,summary=clean([first,duplicate,truncated],d)
            self.assertEqual(len(retained),1)
            self.assertEqual(summary['actions']['drop'],2)
            self.assertTrue((Path(d)/'cleaning_log.csv').exists())
            dev=copy.deepcopy(first);dev['id']='dv-00001'
            with self.assertRaises(AssertionError):clean([dev],d)

    def test_augmentation_is_deterministic_and_label_preserving(self):
        rows=[]
        for i in range(10):
            r=row('My refund is missing. INR 200. NP1234567890.')
            r['id']=f'tr-{i:05d}';fixed,_=repair(r);r['messages'][-1]['content']=json.dumps(fixed);rows.append(r)
        first,n=augment(rows)
        self.assertEqual(n,2)
        self.assertEqual((first,n),augment(rows))
        self.assertEqual([r['messages'][-1]['content'] for r in first],[r['messages'][-1]['content'] for r in rows])
        self.assertEqual(sum(a['messages'][1]['content']!=b['messages'][1]['content'] for a,b in zip(first,rows)),2)

    def test_strict_scorer_and_validator(self):
        r=row('My refund is missing.')
        label=json.loads(r['messages'][-1]['content']);validate(label)
        label['needs_human']=0
        metrics=score([r],[{'id':r['id'],'output':json.dumps(label)}])
        self.assertEqual(metrics['all']['exact_match'],0)
        self.assertEqual(metrics['all']['fields']['needs_human'],0)


if __name__=='__main__':unittest.main()
