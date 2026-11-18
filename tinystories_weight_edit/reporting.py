"""Frozen statistical reporting for the preregistered broad-edit study."""
import csv,math
from collections import defaultdict
import numpy as np
from utils import ROOT,read_json,read_rows,write_json

def transitions(base,after):
    assert [r['id'] for r in base]==[r['id'] for r in after]
    return {'wrong_to_right':sum(not b['correct'] and a['correct'] for b,a in zip(base,after)),
            'right_to_wrong':sum(b['correct'] and not a['correct'] for b,a in zip(base,after))}

def exact_p(w,r):
    n=w+r
    return min(1.,2*sum(math.comb(n,k) for k in range(min(w,r)+1))/2**n) if n else 1.

def compare(base,after,paradigms,replicates,seed):
    assert len(base)==len(after)>0
    counts=transitions(base,after);n=len(base)
    before=sum(r['correct'] for r in base);new=sum(r['correct'] for r in after)
    row={'n':n,'before_correct':before,'after_correct':new,'before_accuracy':before/n,'after_accuracy':new/n,
         'delta_pp':100*(new-before)/n,**counts,'exact_two_sided_p':exact_p(counts['wrong_to_right'],counts['right_to_wrong']),
         'by_paradigm':{}}
    rng=np.random.default_rng(seed);sample=np.zeros(replicates);macro=np.zeros(replicates);nonempty=[]
    for paradigm in paradigms:
        b=[r for r in base if r['paradigm']==paradigm];a=[r for r in after if r['paradigm']==paradigm]
        if not b:continue
        nn=len(b);cc=transitions(b,a);bc=sum(r['correct'] for r in b);ac=sum(r['correct'] for r in a)
        row['by_paradigm'][paradigm]={'n':nn,'before_correct':bc,'after_correct':ac,'delta_pp':100*(ac-bc)/nn,**cc}
        prob=[cc['right_to_wrong']/nn,1-(cc['right_to_wrong']+cc['wrong_to_right'])/nn,cc['wrong_to_right']/nn]
        draws=rng.multinomial(nn,prob,size=replicates);d=(draws[:,2]-draws[:,0])/nn
        sample+=100*nn/n*d;nonempty.append((bc/nn,ac/nn));macro+=100*d
    macro/=len(nonempty)
    row['before_macro_accuracy']=sum(x[0] for x in nonempty)/len(nonempty)
    row['after_macro_accuracy']=sum(x[1] for x in nonempty)/len(nonempty)
    row['paired_stratified_95ci_pp']=np.quantile(sample,[.025,.975]).tolist()
    row['paired_macro_95ci_pp']=np.quantile(macro,[.025,.975]).tolist()
    return row

def make_report():
    from experiment import verify
    assert (ROOT/'results/final/access_log.json').exists()
    protocol=verify();f=read_json(ROOT/'freeze.json')
    base=read_rows(ROOT/'results/final/baseline/test.jsonl');bm=read_json(ROOT/'results/final/baseline/test.metrics.json')
    comparisons=[]
    for name in ['selected']+[f"random_{c['seed']}" for c in f['controls']]:
        after=read_rows(ROOT/'results/final'/name/'test.jsonl');em=read_json(ROOT/'results/final'/name/'test.metrics.json')
        row={'model':name,**compare(base,after,protocol['paradigms'],protocol['statistics']['replicates'],protocol['statistics']['seed'])}
        same=[i for i,r in enumerate(base) if len(r['good_tokens'])==len(r['bad_tokens'])]
        if same:
            row['same_token_length']=compare([base[i] for i in same],[after[i] for i in same],protocol['paradigms'],protocol['statistics']['replicates'],protocol['statistics']['seed']+1)
        row['before_seconds']=bm['inference_seconds'];row['after_seconds']=em['inference_seconds'];comparisons.append(row)
    write_json(ROOT/'results/final/comparisons.json',comparisons)
    simple=[{k:v for k,v in r.items() if k not in ['by_paradigm','same_token_length']} for r in comparisons]
    with (ROOT/'results/final/comparisons.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(simple[0]));w.writeheader();w.writerows(simple)
    target=comparisons[0];ci=target['paired_stratified_95ci_pp'];equal=target.get('same_token_length')
    success=ci[0]>0 and bool(equal and equal['paired_stratified_95ci_pp'][0]>0)
    lines=['# TinyStories-1M: broad direct weight edits','',
      'Actual independent parameter count: 3,745,984, including embeddings; eight transformer blocks contain 398,336 parameters. Architecture: GPT-Neo, width 64, 16 heads, FFN 256, learned positions, alternating global/local attention, tied input/output. The source model was pretrained by its authors; our experiment performs no training.',
      '', '## Final held-out comparisons','',
      '| Condition | N | Before % | After % | Delta pp | Wrong→right | Right→wrong | 95% stratified paired CI pp |',
      '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in comparisons:
        lo,hi=r['paired_stratified_95ci_pp'];lines.append(f"| {r['model']} | {r['n']} | {100*r['before_accuracy']:.3f} | {100*r['after_accuracy']:.3f} | {r['delta_pp']:+.3f} | {r['wrong_to_right']} | {r['right_to_wrong']} | [{lo:+.3f}, {hi:+.3f}] |")
    lines+=['', 'Preregistered joint improvement criterion: '+('met on these fixed subsets.' if success else 'not met.'),
       f"Selected {f['selected']['id']} after {f['attempts_used']} validation trials. Changed {f['selected']['changed_scalars']} independent scalars ({100*f['selected']['fraction']:.3f}% of the whole model); L2 delta {f['selected']['delta_l2']:.9g}, relative L2 {f['selected']['relative_l2']:.9g}."]
    if equal:
        lo,hi=equal['paired_stratified_95ci_pp']
        lines+=['', '## Preregistered same-token-length subset','',
           f"N={equal['n']}; correct {equal['before_correct']} → {equal['after_correct']}; delta {equal['delta_pp']:+.3f} pp, 95% paired interval [{lo:+.3f}, {hi:+.3f}]. W→R {equal['wrong_to_right']}, R→W {equal['right_to_wrong']}."]
    lines+=['', '## By paradigm','',
       '| Paradigm | N | Before correct | After correct | W→R | R→W |','|---|---:|---:|---:|---:|---:|']
    for k,r in target['by_paradigm'].items():lines.append(f"| {k} | {r['n']} | {r['before_correct']} | {r['after_correct']} | {r['wrong_to_right']} | {r['right_to_wrong']} |")
    lines+=['','## Method and limits','',
       'Eight new BLiMP paradigms were selected before running this model; they differ from the earlier Glimmer study. Do not rank models by raw accuracy across these distinct subsets. Exact duplicate or shared sentences are removed before any model scores. Templates and vocabulary remain shared.',
       'The primary metric is the standard minimal-pair preference by summed causal sentence log likelihood conditioned on BOS, without EOS or length normalization; ties count wrong. This is teacher-forced likelihood scoring, not answer generation. The predeclared positive result requires positive bootstrap lower bounds on BOTH the full held-out test and its same-token-length subset. This intersection rule limits a pure length-preference interpretation; it does not prove general grammar competence.',
       'All candidates edit complete attention/FFN projections, with at least 10,000 actual changed independent scalars. No global-output-temperature-only candidate is allowed. Candidate ranking uses validation correct count, then fewer lost baseline successes, fewer changed values and smaller L2 norm. Budget 12, including coefficient variants. All attempts retained. No gradients, backward, optimizer, regression, SFT, LoRA, distillation or evolutionary search.',
       'Original checkpoint unchanged. Model keys/shapes/dtypes, tied embeddings, fixed buffers and values outside the declared mask are checked; zero-save/reload and rollback must reproduce predictions. Candidate evaluations run in fresh processes with no diagnostic activation interventions. Legacy 4.28 nonpersistent mask buffers are checked against the current constructor and omitted only for compatible strict parameter loading; missing tied head is restored from the original embedding. These pre-baseline compatibility operations are not weight improvements.',
       'The test baseline was saved by a separate process and kept sealed until checkpoint, spec, protocol and evaluator freeze. Selection does not read final outcomes. After opening, no further edits are selected using this test. This is organizational separation, not technical inaccessibility to the researcher.',
       'Three fixed-seed random controls use the same actual changed mask and L2 norm within float32 rounding tolerance. No best-seed selection or broad superiority over random editing is claimed.',
       'Bootstrap: 20,000 replicates, fixed seed, stratified by paradigm and observed stratum weights. Exact conditional two-sided paired tests are descriptive sensitivity checks, especially for few discordant pairs. Intervals are conditional on the selected edit and test examples, not fresh pretraining runs, independent templates, model searches or different editors. Per-paradigm results are descriptive secondary analyses.',
       'Model training stream was not audited against benchmark examples. One checkpoint, one editor, a fixed finite search and eight paradigms cannot establish broad improvement in language or free-form generation. Short generation samples are illustrative and do not establish improved coherence.',
       'CPU float32, one thread, official eager GPT-Neo, score batch 16. Single warmed inference timing excludes model loading and report/scoring overhead. Construction seconds measure edit application and serialization, not reasoning time.',
       '', '## Sources','',
       'TinyStories model: https://huggingface.co/roneneldan/TinyStories-1M ; revision 77f1b168e219585646439073245fe87e56b3023e. No explicit model license found in the pinned card/tree; redistribution permission is not asserted. Public artifact bundle uses a pinned downloader and declarative edit, omitting source weights.',
       'BLiMP: Warstadt et al. (2020), TACL 8:377–392, https://doi.org/10.1162/tacl_a_00321 ; https://github.com/alexwarstadt/blimp , CC-BY 4.0. Present subset/filtering/editing splits modify the original evaluation setup. Sources, revisions and SHA-256 are in sources.json.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Saved',ROOT/'REPORT.md',flush=True)
