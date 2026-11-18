import sys
from pathlib import Path
import unittest
import random
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,read_json,read_jsonl,sha256
from benchmark_data import features,sample,canonical,MANUAL
from inference import load,generate,score_tail
from apply_edit import apply_edit

class Contracts(unittest.TestCase):
    def test_real_checkpoint(self):
        m,b=load(ROOT/'checkpoints/base.pt')
        self.assertEqual(sum(p.numel() for p in m.parameters()),1644)
        self.assertIs(m.lm_head.weight,m.token_emb.weight)
        self.assertFalse((ROOT/'checkpoints/base.pt').read_bytes().startswith(b'version https://git-lfs'))

    def test_strict_scoring_does_not_hide_format_errors(self):
        good=[0]*11+[112]
        self.assertTrue(score_tail(good,0)['strict_exact'])
        bad=[0]*12
        s=score_tail(bad,0)
        self.assertFalse(s['strict_exact']);self.assertFalse(s['format_valid'])
        self.assertTrue(s['digits_exact']);self.assertTrue(s['numeric_correct'])
        self.assertFalse(score_tail([112]*12,0)['strict_exact'])

    def test_carry_construction(self):
        rng=random.Random(90)
        for c,lo,hi in [('single_carry',1,1),('chain_short',2,3),('chain_medium',4,6),('chain_long',7,10)]:
            for _ in range(100):
                a,b=sample(rng,c);f=features(a,b)
                self.assertLess(a,10**10);self.assertLess(b,10**10)
                self.assertLessEqual(lo,f['max_carry_chain']);self.assertLessEqual(f['max_carry_chain'],hi)
                self.assertEqual(f['max_carry_chain'],f['carry_count'])

    def test_split_hashes_and_no_leakage(self):
        p=read_json(ROOT/'protocol.json');seen={canonical(a,b) for a,b in MANUAL}
        sizes={'diagnostic':1024,'validation':2048,'test_uniform':5000,'test_stress':2000,'test_exhaustive_00_99':10000}
        # Do not load any sealed predictions or final metrics.
        for split,meta in p['datasets'].items():
            path=ROOT/'data'/f'{split}.jsonl'
            self.assertEqual(sha256(path),meta['sha256'])
            rows=read_jsonl(path);self.assertEqual(len(rows),sizes[split])
            self.assertEqual(len({r['id'] for r in rows}),len(rows))
            if split=='test_exhaustive_00_99':
                self.assertEqual({(r['a'],r['b']) for r in rows},{(a,b) for a in range(100) for b in range(100)})
                continue
            for r in rows:
                self.assertEqual(r['expected'],r['a']+r['b'])
                self.assertFalse(r['a']<100 and r['b']<100)
                key=canonical(r['a'],r['b']);self.assertNotIn(key,seen);seen.add(key)
                self.assertEqual(r['max_carry_chain'],features(r['a'],r['b'])['max_carry_chain'])

    def test_edit_alias_mask_rollback_and_reload(self):
        m,blob=load(ROOT/'checkpoints/base.pt');base=blob['model_state']
        spec={'operations':[{'kind':'scale','tensor':'lm_head.weight','slices':[[0,1],[0,2]],'factor':1.01}]}
        edited,diff,masks=apply_edit(base,spec)
        self.assertGreater(diff['changed_scalars'],0)
        self.assertIs(edited['lm_head.weight'],edited['token_emb.weight'])
        self.assertTrue(torch.equal(edited['token_emb.weight'][1:],base['token_emb.weight'][1:]))
        pairs=[(r['a'],r['b']) for r in read_jsonl(ROOT/'data/diagnostic.jsonl')[:10]]
        answer=generate(m,pairs);m.load_state_dict(edited,strict=True)
        m.load_state_dict(base,strict=True);self.assertEqual(answer,generate(m,pairs))
        # This isolated unit-test edit is never evaluated on validation or used for candidate selection.
        self.assertTrue(torch.equal(base['token_emb.weight'],blob['model_state']['token_emb.weight']))

if __name__=='__main__':unittest.main()
