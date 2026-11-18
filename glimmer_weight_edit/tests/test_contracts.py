import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from utils import ROOT,read_json,read_rows,sha
from language_model import load,log_likelihoods,encode
from edit_weights import apply_edit,EMBED,ALIAS

class Contracts(unittest.TestCase):
    def test_original_checkpoint_and_ties(self):
        m,t=load()
        self.assertEqual(sum(p.numel() for p in m.parameters()),11856)
        self.assertIs(m.lm_head.weight,m.model.embed_tokens.weight)
        self.assertEqual(sha(ROOT/'upstream/model/model.safetensors'),'e61d0b7487386d59d6bf6d62bb9cab28e43b3be18c678f00a9a27cdf41e86c01')

    def test_no_shared_sentence_and_fixed_hashes(self):
        seen=set();ids=set();p=read_json(ROOT/'protocol.json')
        for split,meta in p['datasets'].items():
            path=ROOT/'data'/f'{split}.jsonl';self.assertEqual(sha(path),meta['sha256'])
            rows=read_rows(path);self.assertEqual(len(rows),meta['n'])
            for r in rows:
                self.assertNotIn(r['id'],ids);ids.add(r['id'])
                for text in [r['sentence_good'],r['sentence_bad']]:
                    self.assertNotIn(text,seen);seen.add(text)

    def test_likelihood_matches_incremental_definition(self):
        m,t=load();sentences=['The cat sleeps.','The cats sleep in the house.']
        ids,scores=log_likelihoods(m,t,sentences)
        for tokens,score in zip(ids,scores):
            total=0.
            for i in range(1,len(tokens)):
                logits=m(torch.tensor([tokens[:i]]),use_cache=False).logits[0,-1].float().log_softmax(-1)
                total+=float(logits[tokens[i]])
            self.assertAlmostEqual(score,total,places=4)
        single=log_likelihoods(m,t,sentences,batch_size=1)[1]
        self.assertTrue(all(abs(a-b)<1e-4 for a,b in zip(scores,single)))

    def test_pure_edit_mask_and_rollback(self):
        m,t=load();base={k:v.clone() for k,v in m.state_dict().items()};base[ALIAS]=base[EMBED]
        names={n for n,_ in m.named_parameters()}
        state,diff,masks=apply_edit(base,names,{'operations':[{'kind':'scale','tensor':'model.norm.weight','factor':1.01}]})
        self.assertEqual(diff['changed_scalars'],16)
        self.assertIs(state[ALIAS],state[EMBED])
        self.assertTrue(torch.equal(state[EMBED],base[EMBED]))
        original=log_likelihoods(m,t,['The cat sleeps.']);m.load_state_dict(state,strict=True)
        m.load_state_dict(base,strict=True);self.assertEqual(original,log_likelihoods(m,t,['The cat sleeps.']))

    @unittest.skipUnless((ROOT/'results/final/access_log.json').exists(),'Final stage not opened')
    def test_frozen_artifacts_and_fresh_scores(self):
        f=read_json(ROOT/'freeze.json');self.assertLessEqual(f['attempts_used'],12)
        for path,digest in [(ROOT/'checkpoints/selected.pt',f['selected_sha256']),(ROOT/'protocol.json',f['protocol_sha256']),(ROOT/'edit_spec.json',f['edit_spec_sha256']),(ROOT/'experiment.py',f['evaluator_sha256'])]:self.assertEqual(sha(path),digest)
        m,t=load();base={k:v.clone() for k,v in m.state_dict().items()};base[ALIAS]=base[EMBED]
        names={n for n,_ in m.named_parameters()};state,diff,_=apply_edit(base,names,read_json(ROOT/'edit_spec.json'))
        selected,st=load(ROOT/'checkpoints/selected.pt')
        for k,v in selected.state_dict().items():self.assertTrue(torch.equal(v,state[k]),k)
        rows=read_rows(ROOT/'results/final/selected/test.jsonl');sentences=[s for r in rows for s in [r['sentence_good'],r['sentence_bad']]]
        _,scores=log_likelihoods(selected,st,sentences)
        self.assertEqual(scores,[s for r in rows for s in [r['good_loglikelihood'],r['bad_loglikelihood']]])
        for c in f['controls']:
            path=ROOT/'checkpoints'/c['checkpoint'];self.assertEqual(sha(path),c['sha256'])
            control,_=load(path);cs=control.state_dict();d2=0.;n=0
            for name in names:
                support=state[name]!=base[name];self.assertTrue(torch.equal(cs[name][~support],base[name][~support]))
                delta=cs[name].double()-base[name].double();d2+=float(delta.square().sum());n+=int(torch.count_nonzero(delta))
            self.assertEqual(n,diff['changed_scalars']);self.assertAlmostEqual(d2**.5,diff['delta_l2'],places=5)

if __name__=='__main__':unittest.main()
