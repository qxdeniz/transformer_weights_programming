import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from utils import ROOT,read_json,read_rows,sha
from language_model import load,log_likelihoods,PARAMETERS
from edit_weights import apply_edit,EMBED,ALIAS

class Contracts(unittest.TestCase):
    def test_original_checkpoint_and_tied_head(self):
        m,t=load();self.assertEqual(sum(p.numel() for p in m.parameters()),PARAMETERS)
        self.assertEqual(sum(p.numel() for n,p in m.named_parameters() if n.startswith('transformer.h.')),398336)
        self.assertIs(m.lm_head.weight,m.transformer.wte.weight)
        self.assertEqual(sha(ROOT/'upstream/model/pytorch_model.bin'),'07f9609ea882b8163ff3b23d40e2b82cb715d409631beb15c84b164f3877dae7')

    def test_disjoint_new_data_and_fixed_code(self):
        p=read_json(ROOT/'protocol.json');seen=set();ids=set()
        for name,digest in p['code_sha256'].items():self.assertEqual(sha(ROOT/name),digest)
        old={'anaphor_gender_agreement','determiner_noun_agreement_1','regular_plural_subject_verb_agreement_1','irregular_plural_subject_verb_agreement_1','principle_A_domain_1','npi_present_1','wh_vs_that_no_gap','irregular_past_participle_adjectives'}
        self.assertFalse(old.intersection(p['paradigms']))
        for split,meta in p['datasets'].items():
            path=ROOT/'data'/f'{split}.jsonl';self.assertEqual(sha(path),meta['sha256'])
            rows=read_rows(path);self.assertEqual(len(rows),meta['n'])
            for row in rows:
                self.assertNotIn(row['id'],ids);ids.add(row['id'])
                for sentence in [row['sentence_good'],row['sentence_bad']]:self.assertNotIn(sentence,seen);seen.add(sentence)

    def test_raw_likelihood_incremental_and_batch(self):
        m,t=load();sentences=['The cat sleeps.','The little cats sleep in the house.']
        ids,scores=log_likelihoods(m,t,sentences)
        for tokens,score in zip(ids,scores):
            total=0.
            for i in range(1,len(tokens)):
                logits=m(torch.tensor([tokens[:i]]),use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1)
                total+=float(logits[tokens[i]])
            self.assertAlmostEqual(total,score,places=4)
        single=log_likelihoods(m,t,sentences,batch_size=1)[1]
        self.assertTrue(all(abs(a-b)<1e-4 for a,b in zip(scores,single)))

    def test_broad_mask_and_rollback(self):
        m,t=load();base={k:v.clone() for k,v in m.state_dict().items()};base[ALIAS]=base[EMBED]
        names={n for n,_ in m.named_parameters()}
        operations=[{'kind':'scale','tensor':f'transformer.h.{i}.attn.attention.q_proj.weight','factor':1.1} for i in range(8)]
        state,diff,masks=apply_edit(base,names,{'operations':operations})
        self.assertEqual(diff['changed_scalars'],32768);self.assertTrue(torch.equal(state[EMBED],base[EMBED]))
        self.assertIs(state[ALIAS],state[EMBED])
        buffers={n:b.clone() for n,b in m.named_buffers()}
        answer=log_likelihoods(m,t,['The cat sleeps.']);m.load_state_dict(state,strict=True);m.load_state_dict(base,strict=True)
        self.assertEqual(answer,log_likelihoods(m,t,['The cat sleeps.']))
        self.assertTrue(all(torch.equal(buffers[n],b) for n,b in m.named_buffers()))

    @unittest.skipUnless((ROOT/'results/final/access_log.json').exists(),'Final results not open yet')
    def test_selected_reconstruction_and_freeze(self):
        f=read_json(ROOT/'freeze.json')
        for path,digest in [(ROOT/'checkpoints/selected.pt',f['selected_sha256']),(ROOT/'protocol.json',f['protocol_sha256']),(ROOT/'edit_spec.json',f['edit_spec_sha256']),(ROOT/'experiment.py',f['evaluator_sha256'])]:self.assertEqual(sha(path),digest)
        self.assertEqual(f['attempts_used'],12)
        m,t=load();base={k:v.clone() for k,v in m.state_dict().items()};base[ALIAS]=base[EMBED];names={n for n,_ in m.named_parameters()}
        state,diff,_=apply_edit(base,names,read_json(ROOT/'edit_spec.json'));self.assertGreaterEqual(diff['changed_scalars'],10000)
        selected,tokenizer=load(ROOT/'checkpoints/selected.pt')
        for name,v in selected.state_dict().items():self.assertTrue(torch.equal(v,state[name]),name)
        # Full diagnostic scores are reproduced with the same batching; final
        # outcomes are not used for selecting or changing weights.
        saved=read_rows(ROOT/'results'/f['selected']['id']/'diagnostic.jsonl')
        sentences=[s for r in saved for s in [r['sentence_good'],r['sentence_bad']]]
        _,scores=log_likelihoods(selected,tokenizer,sentences)
        self.assertEqual(scores,[s for r in saved for s in [r['good_loglikelihood'],r['bad_loglikelihood']]])

    @unittest.skipUnless((ROOT/'results/final/access_log.json').exists(),'Final results not open yet')
    def test_controls_mask_and_l2(self):
        f=read_json(ROOT/'freeze.json');m,_=load();base=m.state_dict();selected,_=load(ROOT/'checkpoints/selected.pt');chosen=selected.state_dict();names={n for n,_ in m.named_parameters()}
        for record in f['controls']:
            path=ROOT/'checkpoints'/record['checkpoint'];self.assertEqual(sha(path),record['sha256'])
            control,_=load(path);state=control.state_dict();count=0;norm2=0.
            for name in names:
                mask=chosen[name]!=base[name];self.assertTrue(torch.equal(state[name][~mask],base[name][~mask]))
                delta=state[name].double()-base[name].double();count+=int(torch.count_nonzero(delta));norm2+=float(delta.square().sum())
            self.assertEqual(count,f['selected']['changed_scalars']);self.assertAlmostEqual(norm2**.5,f['selected']['delta_l2'],places=5)

if __name__=='__main__':unittest.main()
