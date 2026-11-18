"""Predeclared eight-paradigm BLiMP subset. Splits use no model outputs."""
from collections import Counter
import random
from utils import ROOT,read_rows,write_rows,write_json,sha

PARADIGMS=['anaphor_gender_agreement','determiner_noun_agreement_1',
           'regular_plural_subject_verb_agreement_1','irregular_plural_subject_verb_agreement_1',
           'principle_A_domain_1','npi_present_1','wh_vs_that_no_gap','irregular_past_participle_adjectives']

def prepare():
    assert not (ROOT/'protocol.json').exists(),'Data and protocol already fixed'
    result={k:[] for k in ['diagnostic','validation','test']};seen_sentences=set();dropped={}
    for j,paradigm in enumerate(PARADIGMS):
        rows=read_rows(ROOT/'upstream/blimp'/f'{paradigm}.jsonl')
        rng=random.Random(92101+j);rng.shuffle(rows);retained=[]
        for row in rows:
            good=' '.join(row['sentence_good'].split());bad=' '.join(row['sentence_bad'].split())
            if good==bad or good in seen_sentences or bad in seen_sentences:continue
            seen_sentences.update([good,bad]);retained.append({'id':f"{paradigm}:{row['pairID']}",
                  'paradigm':paradigm,'sentence_good':good,'sentence_bad':bad,'field':row['field'],
                  'linguistics_term':row['linguistics_term'],'source_pairID':row['pairID']})
        assert len(retained)>384,(paradigm,len(retained))
        dropped[paradigm]={'source_n':len(rows),'retained_n':len(retained),'duplicate_or_shared_sentence_removed':len(rows)-len(retained)}
        # Independent deterministic permutations after diagnostic and validation removal.
        d_rng=random.Random(92201+j);v_rng=random.Random(92301+j);t_rng=random.Random(92401+j)
        indices=list(range(len(retained)));chosen=d_rng.sample(indices,128);ds=set(chosen)
        result['diagnostic'].extend(retained[i] for i in chosen)
        left=[i for i in indices if i not in ds];chosen=v_rng.sample(left,256);vs=set(chosen)
        result['validation'].extend(retained[i] for i in chosen)
        left=[i for i in left if i not in vs];t_rng.shuffle(left);result['test'].extend(retained[i] for i in left)
    manifest={}
    for split,rows in result.items():
        random.Random({'diagnostic':92501,'validation':92502,'test':92503}[split]).shuffle(rows)
        path=ROOT/'data'/f'{split}.jsonl';write_rows(path,rows)
        manifest[split]={'n':len(rows),'sha256':sha(path),'by_paradigm':dict(Counter(r['paradigm'] for r in rows))}
    protocol={'benchmark':'BLiMP eight-paradigm editing study v1','paradigms':PARADIGMS,'datasets':manifest,'deduplication':dropped,
       'sampling':'Named paradigms selected for linguistic coverage before running this model; shuffle/dedup before any model outputs. Remove every exact shared sentence globally, even within a split.',
       'seeds':{'dedup_shuffle_base':92101,'diagnostic_base':92201,'validation_base':92301,'test_base':92401,'split_order':[92501,92502,92503]},
       'model':'Glint-Research/Glimmer-1-Base','independent_parameters':11856,
       'primary_metric':'pair accuracy: raw summed log P(sentence_good | BOS) > raw summed log P(sentence_bad | BOS); exact ties counted wrong',
       'secondary_metrics':['macro accuracy over paradigms','raw log-likelihood margin','diagnostic greedy generations (illustrative only)'],
       'scoring':'Official Transformers Llama forward, eager attention, causal likelihood; teacher-forced sentence scoring is standard for BLiMP, unlike the separate addition task. No EOS, length normalization, or response repair.',
       'inference':{'dtype':'float32','device':'cpu','threads':1,'seed':91101,'score_batch_size':64,'bos_id':1,'context_limit':512,'generation_tokens':32,'greedy':True},
       'validation_budget':12,'selection':['max validation correct','min right_to_wrong','min changed_scalars','min delta_l2'],
       'controls':{'zero_edit':True,'rollback':True,'random_seeds':[93101,93102,93103],'random_match':'same actual nonzero mask and L2 norm'},
       'statistics':{'method':'paired bootstrap stratified by paradigm, fixed observed test weights','replicates':20000,'seed':94101,'confidence':.95},
       'success':'test micro accuracy difference positive with stratified 95% paired interval lower bound > 0; per-paradigm effects and regressions reported separately',
       'scope':'8 of 67 BLiMP paradigms; not a reproduction of model-card full BLiMP 52.43%; no claim of general intelligence or improved free-form generation',
       'contamination':'Original FineWeb-Edu training sample unavailable. BLiMP held out from editing, not certified absent from original training. Templates and vocabulary are shared across splits.',
       'sealing':'Baseline test evaluated in separate process, outcomes sealed until checkpoint/spec/code/protocol freeze; development then closes',
       'code_sha256':{name:sha(ROOT/name) for name in ['language_model.py','benchmark.py','edit_weights.py','experiment.py']}}
    write_json(ROOT/'protocol.json',protocol);return protocol
