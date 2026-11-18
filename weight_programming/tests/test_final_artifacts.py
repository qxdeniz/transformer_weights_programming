"""Audit frozen results after the final opening; never changes model selection."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,read_json,read_jsonl,sha256
from inference import load,generate
from apply_edit import apply_edit

@unittest.skipUnless((ROOT/'results/final/access_log.json').exists(),'Final stage not complete')
class FinalAudit(unittest.TestCase):
    def test_frozen_hashes_and_budget(self):
        f=read_json(ROOT/'freeze.json')
        self.assertEqual(sha256(ROOT/'checkpoints/selected.pt'),f['selected_sha256'])
        self.assertEqual(sha256(ROOT/'edit_spec.json'),f['edit_spec_sha256'])
        self.assertEqual(sha256(ROOT/'protocol.json'),f['protocol_sha256'])
        self.assertEqual(sha256(ROOT/'run_experiment.py'),f['evaluator_sha256'])
        self.assertEqual(sha256(ROOT/'checkpoints/base.pt'),read_json(ROOT/'sources.json')['checkpoint_sha256'])
        attempts=read_json(ROOT/'results/attempts.json')
        self.assertEqual(len(attempts),12)
        self.assertTrue(all(r['status']=='complete' for r in attempts))
        best=min(attempts,key=lambda r:(-r['validation_correct'],r['right_to_wrong'],r['changed_scalars'],r['delta_l2']))
        self.assertEqual(best['id'],f['selected']['id'])

    def test_edit_reconstruction_and_controls(self):
        _,b=load(ROOT/'checkpoints/base.pt');_,e=load(ROOT/'checkpoints/selected.pt')
        weights,diff,masks=apply_edit(b['model_state'],read_json(ROOT/'edit_spec.json'))
        self.assertEqual(diff['changed_scalars'],52)
        for k,v in weights.items():self.assertTrue(torch.equal(v,e['model_state'][k]),k)
        f=read_json(ROOT/'freeze.json');support={k:(weights[k]!=b['model_state'][k]) for k in masks}
        for c in f['random_controls']:
            p=ROOT/'checkpoints'/c['checkpoint'];self.assertEqual(sha256(p),c['sha256'])
            _,control=load(p);total2=0;count=0
            for k,mask in support.items():
                v=control['model_state'][k];base=b['model_state'][k]
                self.assertTrue(torch.equal(v[~mask],base[~mask]))
                delta=v.double()-base.double();total2+=float(delta.square().sum());count+=int(torch.count_nonzero(delta))
            self.assertEqual(count,52)
            self.assertAlmostEqual(total2**.5,diff['delta_l2'],places=5)

    def test_full_uniform_predictions_reproduce(self):
        data=read_jsonl(ROOT/'data/test_uniform.jsonl')
        pairs=[(r['a'],r['b']) for r in data]
        for name,path in [('baseline',ROOT/'checkpoints/base.pt'),('selected',ROOT/'checkpoints/selected.pt')]:
            model,_=load(path);prompts,tails=generate(model,pairs)
            saved=read_jsonl(ROOT/'results/final'/name/'test_uniform.jsonl')
            self.assertEqual(prompts,[r['input_tokens'] for r in saved])
            self.assertEqual(tails,[r['generated_tokens'] for r in saved])

if __name__=='__main__':unittest.main()
