"""Predeclared eight-paradigm BLiMP subset. Splits use no model outputs."""
from collections import Counter
import random
from utils import ROOT,read_rows,write_rows,write_json,sha

PARADIGMS=['anaphor_number_agreement', 'determiner_noun_agreement_2', 'regular_plural_subject_verb_agreement_2', 'irregular_plural_subject_verb_agreement_2', 'irregular_past_participle_verbs', 'distractor_agreement_relational_noun', 'passive_1', 'intransitive']

def prepare():
    assert not (ROOT/'protocol.json').exists(),'Data and protocol already fixed'
    result={k:[] for k in ['diagnostic','validation','test']};seen_sentences=set();dropped={}
    for j,paradigm in enumerate(PARADIGMS):
        rows=read_rows(ROOT/'upstream/blimp'/f'{paradigm}.jsonl')
        rng=random.Random(102101+j);rng.shuffle(rows);retained=[]
        for row in rows:
            good=' '.join(row['sentence_good'].split());bad=' '.join(row['sentence_bad'].split())
            if good==bad or good in seen_sentences or bad in seen_sentences:continue
            seen_sentences.update([good,bad]);retained.append({'id':f"{paradigm}:{row['pairID']}",
                  'paradigm':paradigm,'sentence_good':good,'sentence_bad':bad,'field':row['field'],
                  'linguistics_term':row['linguistics_term'],'source_pairID':row['pairID']})
        assert len(retained)>384,(paradigm,len(retained))
        dropped[paradigm]={'source_n':len(rows),'retained_n':len(retained),'duplicate_or_shared_sentence_removed':len(rows)-len(retained)}
        # Independent deterministic permutations after diagnostic and validation removal.
        d_rng=random.Random(102201+j);v_rng=random.Random(102301+j);t_rng=random.Random(102401+j)
        indices=list(range(len(retained)));chosen=d_rng.sample(indices,128);ds=set(chosen)
        result['diagnostic'].extend(retained[i] for i in chosen)
        left=[i for i in indices if i not in ds];chosen=v_rng.sample(left,256);vs=set(chosen)
        result['validation'].extend(retained[i] for i in chosen)
        left=[i for i in left if i not in vs];t_rng.shuffle(left);result['test'].extend(retained[i] for i in left)
    manifest={}
    for split,rows in result.items():
        random.Random({'diagnostic':102501,'validation':102502,'test':102503}[split]).shuffle(rows)
        path=ROOT/'data'/f'{split}.jsonl';write_rows(path,rows)
        manifest[split]={'n':len(rows),'sha256':sha(path),'by_paradigm':dict(Counter(r['paradigm'] for r in rows))}
    protocol={'benchmark':'TinyStories broad-weight BLiMP study v1','paradigms':PARADIGMS,'datasets':manifest,'deduplication':dropped,
       'sampling':'Named paradigms selected for linguistic coverage before running this model; shuffle/dedup before any model outputs. Remove every exact shared sentence globally, even within a split.',
       'seeds':{'dedup_shuffle_base':102101,'diagnostic_base':102201,'validation_base':102301,'test_base':102401,'split_order':[102501,102502,102503]},
       'model':'roneneldan/TinyStories-1M','independent_parameters':3745984,
       'primary_metric':'pair accuracy: raw summed log P(sentence_good | BOS) > raw summed log P(sentence_bad | BOS); exact ties counted wrong',
       'secondary_metrics':['macro accuracy over paradigms','raw log-likelihood margin','same-token-length pair accuracy','diagnostic greedy generations (illustrative only)'],
       'scoring':'Official Transformers GPT-Neo forward, eager attention, causal likelihood; teacher-forced sentence scoring is standard for BLiMP, unlike the separate addition task. No EOS, length normalization, or response repair.',
       'inference':{'dtype':'float32','device':'cpu','threads':1,'seed':101101,'score_batch_size':16,'bos_id':50256,'context_limit':2048,'generation_tokens':48,'greedy':True},
       'validation_budget':12,'minimum_changed_candidate_scalars':10000,'edit_scope':'Existing full attention/FFN projection matrices across transformer blocks; no global output-temperature-only candidate','selection':['max validation correct','min right_to_wrong','min changed_scalars','min delta_l2'],
       'controls':{'zero_edit':True,'rollback':True,'random_seeds':[103101,103102,103103],'random_match':'same actual nonzero mask and L2 norm'},
       'statistics':{'method':'paired bootstrap stratified by paradigm, fixed observed test weights','replicates':20000,'seed':104101,'confidence':.95},
       'success':'Require positive test micro difference with stratified 95% paired CI lower > 0 AND positive same-token-length subset difference with stratified 95% paired CI lower > 0. Report all paradigm regressions. This intersection rule is fixed before edits.',
       'scope':'8 of 67 BLiMP paradigms, disjoint from the earlier Glimmer study; no comparison of raw accuracy across these different subsets; no claim of general intelligence',
       'contamination':'Original TinyStories training stream not audited for these benchmark sentences. BLiMP held out from editing, not certified absent from original training. Templates and vocabulary are shared across splits.',
       'sealing':'Baseline test evaluated in separate process, outcomes sealed until checkpoint/spec/code/protocol freeze; development then closes',
       'code_sha256':{name:sha(ROOT/name) for name in ['language_model.py','benchmark.py','edit_weights.py','experiment.py','reporting.py']}}
    write_json(ROOT/'protocol.json',protocol);return protocol
