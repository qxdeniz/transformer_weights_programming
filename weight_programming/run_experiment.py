"""Stage-based CarryBench-10 runner; final measurements stay sealed until freeze."""
import argparse
from collections import Counter,defaultdict
import copy
import datetime
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import torch
from common import ROOT,read_json,write_json,read_jsonl,write_jsonl,sha256
from inference import load,generate,score_tail,configure,verify_source_hashes
from apply_edit import apply_edit
from benchmark_data import build_datasets

FINAL=['test_uniform','test_stress','test_exhaustive_00_99']
BASE=ROOT/'checkpoints/base.pt'

def validate_protocol():
    protocol=read_json(ROOT/'protocol.json')
    verify_source_hashes()
    for name,meta in protocol['datasets'].items():
        assert sha256(ROOT/'data'/f'{name}.jsonl')==meta['sha256']
    for name,digest in protocol['code_sha256'].items():
        assert sha256(ROOT/name)==digest,f'Protocol code changed: {name}'
    return protocol

def summary(rows,seconds):
    n=len(rows)
    out={'n':n,'strict_correct':sum(r['strict_exact'] for r in rows),
         'strict_em':sum(r['strict_exact'] for r in rows)/n,
         'upstream_digits_em':sum(r['digits_exact'] for r in rows)/n,
         'numeric_accuracy':sum(r['numeric_correct'] for r in rows)/n,
         'format_validity':sum(r['format_valid'] for r in rows)/n,
         'per_digit_accuracy_lsd_first':[sum(r['digit_correct'][i] for r in rows)/n for i in range(11)],
         'inference_seconds':seconds,'examples_per_second':n/seconds,
         'termination_counts':dict(Counter(r['termination'] for r in rows))}
    strata=defaultdict(list)
    for r in rows:
        strata['source:'+r['source']].append(r)
        strata['max_chain:'+str(r['max_carry_chain'])].append(r)
        for label in r['labels']:strata['label:'+label].append(r)
    out['by_category']={k:{'n':len(v),'correct':sum(r['strict_exact'] for r in v),
                             'strict_em':sum(r['strict_exact'] for r in v)/len(v)} for k,v in sorted(strata.items())}
    return out

def measure(checkpoint,split,outdir):
    validate_protocol()
    data=read_jsonl(ROOT/'data'/f'{split}.jsonl')
    model,_=load(checkpoint)
    # Identical warmup and batches on every measurement; no timing model loading.
    generate(model,[(1234567890,9876543210)]*16)
    start=time.perf_counter()
    prompts,tails=generate(model,[(r['a'],r['b']) for r in data])
    elapsed=time.perf_counter()-start
    rows=[{**r,'input_tokens':p,'generated_tokens':t,**score_tail(t,r['expected'])}
          for r,p,t in zip(data,prompts,tails)]
    outdir=Path(outdir);write_jsonl(outdir/f'{split}.jsonl',rows)
    metrics=summary(rows,elapsed)
    metrics['checkpoint_sha256']=sha256(checkpoint)
    metrics['data_sha256']=sha256(ROOT/'data'/f'{split}.jsonl')
    write_json(outdir/f'{split}.metrics.json',metrics)
    return rows,metrics

def child_measure(checkpoint,split,outdir):
    # A separate process has no development state, hooks, or access to labels in generate().
    cmd=[sys.executable,str(ROOT/'run_experiment.py'),'--stage','worker','--checkpoint',str(checkpoint),
         '--split',split,'--out',str(outdir)]
    proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-4000:])

def prepare():
    if not (ROOT/'sources.json').exists():
        raise RuntimeError('sources.json required; obtain pinned upstream source first')
    verify_source_hashes()
    BASE.parent.mkdir(exist_ok=True)
    original=ROOT/'upstream/checkpoints/best.pt'
    if BASE.exists():assert sha256(BASE)==sha256(original)
    else:shutil.copyfile(original,BASE)
    m,blob=load(BASE)
    assert sha256(BASE)==sha256(original)
    inv=[]
    for name,p in m.named_parameters(remove_duplicate=False):
        inv.append({'name':name,'shape':list(p.shape),'dtype':str(p.dtype),'scalars':p.numel(),
                    'alias_of':'token_emb.weight' if name=='lm_head.weight' else None})
    buffers=[{'name':n,'shape':list(b.shape),'dtype':str(b.dtype),'persistent':n in m.state_dict()} for n,b in m.named_buffers()]
    write_json(ROOT/'parameter_inventory.json',{'parameters':inv,'buffers':buffers,'independent_scalars':sum(p.numel() for p in m.parameters()),'config':blob['model_config']})
    if not (ROOT/'protocol.json').exists():build_datasets()
    validate_protocol()
    import numpy
    env={'python':sys.version,'executable':sys.executable,'pytorch':torch.__version__,'numpy':numpy.__version__,
         'platform':platform.platform(),'machine':platform.machine(),'device':'cpu','dtype':'float32',
         'threads':torch.get_num_threads(),'seed':1776,'batch_size':256,'greedy_tokens':12,
         'venv':'Python 3.12 venv; existing system site packages are read only; no global packages changed',
         'cpu_count':os.cpu_count(),'torch_build':torch.__config__.show()}
    (ROOT/'environment.txt').write_text(json.dumps(env,indent=2)+'\n')
    (ROOT/'requirements.txt').write_text(f'torch=={torch.__version__}\nnumpy=={numpy.__version__}\n')
    print('Prepared CarryBench-10; original weights preserved.',flush=True)

def write_dev_report():
    lines=['# Baseline: open development sets only','',
           'Published author claim: 99.04% digit-only EM on their test. The measurements below are independent; strict EM includes EOS.','',
           '| Set | N | Strict EM | Digit EM | Format valid | Seconds |','|---|---:|---:|---:|---:|---:|']
    for split in ['diagnostic','validation']:
        m=read_json(ROOT/'results/baseline'/f'{split}.metrics.json')
        lines.append(f"| {split} | {m['n']} | {100*m['strict_em']:.4f}% | {100*m['upstream_digits_em']:.4f}% | {100*m['format_validity']:.4f}% | {m['inference_seconds']:.3f} |")
    lines+=['','## Diagnostic categories','','| Category | N | Correct | EM |','|---|---:|---:|---:|']
    m=read_json(ROOT/'results/baseline/diagnostic.metrics.json')
    for k,v in m['by_category'].items():
        lines.append(f"| {k} | {v['n']} | {v['correct']} | {100*v['strict_em']:.3f}% |")
    errors=[r for r in read_jsonl(ROOT/'results/baseline/diagnostic.jsonl') if not r['strict_exact']]
    lines+=['','## Diagnostic error examples','','| A | B | Expected | Predicted | Max carry run |','|---:|---:|---:|---:|---:|']
    for r in errors[:24]:lines.append(f"| {r['a']} | {r['b']} | {r['expected']} | {r['decoded']} | {r['max_carry_chain']} |")
    lines+=['','Observed errors do not by themselves identify a causal mechanism. Activation analysis and controlled weight edits are required.','Final test outcomes are sealed and were not used for this report.']
    (ROOT/'baseline_errors.md').write_text('\n'.join(lines)+'\n')

def baseline():
    if (ROOT/'results/baseline/complete.json').exists():
        print('Baseline already complete; not overwriting measurements.');return
    validate_protocol();m,blob=load(BASE)
    diagnostic=read_jsonl(ROOT/'data/diagnostic.jsonl')
    pairs=[(r['a'],r['b']) for r in diagnostic[:100]]
    prompts,tails=generate(m,pairs)
    assert (prompts,tails)==generate(m,pairs),'Repeated baseline changed'
    from src.data import preprocess,postprocess,EOS_ID
    from src.eval import evaluate_exact_match,predict_single
    # Reference single-example path loads upstream checkpoint independently.
    for i,(a,b) in enumerate(pairs):
        assert prompts[i]==preprocess(a,b)
        ref=predict_single(BASE,a,b,'cpu')
        assert ref['prediction']==postprocess(tails[i])
        one=m.generate(torch.tensor([preprocess(a,b)]),12)[0,-12:].tolist()
        assert one==tails[i],'Batch and individual greedy paths differ'
    aa=torch.tensor([p[0] for p in pairs]);bb=torch.tensor([p[1] for p in pairs])
    ref_em,ref_tok=evaluate_exact_match(m,aa,bb,256,torch.device('cpu'))
    scored=[score_tail(t,r['expected']) for t,r in zip(tails,diagnostic[:100])]
    assert ref_em==sum(r['digits_exact'] for r in scored)/100
    assert abs(ref_tok-sum(sum(r['digit_correct']) for r in scored)/1100)<1e-12
    zero=ROOT/'checkpoints/zero_edit.pt'
    torch.save({'model_config':blob['model_config'],'model_state':blob['model_state']},zero)
    zero_m,_=load(zero);assert generate(zero_m,pairs)==(prompts,tails)
    child_measure(zero,'manual',ROOT/'results/zero_edit')
    child_measure(BASE,'manual',ROOT/'results/baseline')
    assert [r['generated_tokens'] for r in read_jsonl(ROOT/'results/zero_edit/manual.jsonl')]==[r['generated_tokens'] for r in read_jsonl(ROOT/'results/baseline/manual.jsonl')]
    write_json(ROOT/'results/baseline/preflight.json',{'upstream_agreement_examples':100,'token_predictions_equal':True,
        'repeated_predictions_equal':True,'zero_edit_equal':True,'new_process_equal':True,'base_sha256':sha256(BASE)})
    for split in ['diagnostic','validation']:
        _,metrics=measure(BASE,split,ROOT/'results/baseline')
        print(split,json.dumps({k:metrics[k] for k in ['n','strict_em','upstream_digits_em','format_validity','inference_seconds']}),flush=True)
    write_dev_report()
    # Deliberately never load or print the resulting final files in development.
    for split in FINAL:
        print('Saving sealed baseline:',split,flush=True)
        child_measure(BASE,split,ROOT/'sealed_final/baseline')
    write_json(ROOT/'results/baseline/complete.json',{'status':'complete','base_sha256':sha256(BASE),'final_results_sealed':True})

def paired_counts(base,edited):
    assert [r['id'] for r in base]==[r['id'] for r in edited]
    return {'wrong_to_right':sum(not b['strict_exact'] and e['strict_exact'] for b,e in zip(base,edited)),
            'right_to_wrong':sum(b['strict_exact'] and not e['strict_exact'] for b,e in zip(base,edited))}

def evaluate_edit(spec_path):
    protocol=validate_protocol()
    assert (ROOT/'results/baseline/complete.json').exists()
    assert not (ROOT/'freeze.json').exists(),'Final checkpoint already frozen; development closed'
    ledger_path=ROOT/'results/attempts.json'
    ledger=read_json(ledger_path) if ledger_path.exists() else []
    assert len(ledger)<protocol['validation_budget'],'All validation attempts spent'
    spec=read_json(spec_path)
    assert spec.get('hypothesis') and spec.get('prediction') and spec.get('risk')
    attempt=f'candidate_{len(ledger)+1:02d}';folder=ROOT/'results'/attempt
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'edit_spec.json',spec)
    record={'id':attempt,'status':'started','spec_sha256':sha256(folder/'edit_spec.json')}
    ledger.append(record);write_json(ledger_path,ledger) # charge attempt before evaluating
    started=time.perf_counter();base_model,blob=load(BASE)
    original_buffers={n:t.clone() for n,t in base_model.named_buffers()}
    weights,diff,masks=apply_edit(blob['model_state'],spec)
    assert diff['changed_scalars']>0,'A candidate must be nonzero'
    path=ROOT/'checkpoints'/f'{attempt}.pt'
    torch.save({'model_config':blob['model_config'],'model_state':weights},path)
    preparation_seconds=time.perf_counter()-started
    edited,_=load(path)
    assert all(torch.equal(original_buffers[n],t) for n,t in edited.named_buffers())
    assert edited.lm_head.weight is edited.token_emb.weight
    pairs=[(r['a'],r['b']) for r in read_jsonl(ROOT/'data/diagnostic.jsonl')[:100]]
    base_answers=generate(base_model,pairs)
    base_model.load_state_dict(weights,strict=True)
    edited_answers=generate(base_model,pairs)
    assert edited_answers==generate(edited,pairs)
    base_model.load_state_dict(blob['model_state'],strict=True)
    assert generate(base_model,pairs)==base_answers,'Rollback failed'
    child_measure(path,'diagnostic',folder)
    child_measure(path,'validation',folder)
    # Saved/reloaded checkpoint was evaluated by fresh processes. Compare same-batch predictions.
    reloaded=read_jsonl(folder/'diagnostic.jsonl')[:100]
    assert [r['generated_tokens'] for r in reloaded]==edited_answers[1]
    val=read_jsonl(folder/'validation.jsonl');base_val=read_jsonl(ROOT/'results/baseline/validation.jsonl')
    metrics=read_json(folder/'validation.metrics.json')
    write_json(folder/'diff.json',diff)
    write_json(folder/'checks.json',{'shapes_dtypes_keys_unchanged':True,'buffers_unchanged':True,
               'aliases_preserved':True,'finite':True,'outside_mask_unchanged':True,'rollback_equal':True,
               'fresh_process_predictions_equal':True,'source_and_protocol_hashes_verified':True})
    record.update({'status':'complete','validation_correct':metrics['strict_correct'],**paired_counts(base_val,val),
                   **diff,'preparation_seconds':preparation_seconds,'checkpoint_sha256':sha256(path)})
    write_json(ledger_path,ledger)
    print(json.dumps(record,indent=2),flush=True)

def finalize():
    protocol=validate_protocol()
    ledger=read_json(ROOT/'results/attempts.json')
    completed=[r for r in ledger if r['status']=='complete']
    assert completed,'No nonzero candidates'
    best=min(completed,key=lambda r:(-r['validation_correct'],r['right_to_wrong'],r['changed_scalars'],r['delta_l2']))
    selected=ROOT/'checkpoints/selected.pt'
    freeze_path=ROOT/'freeze.json'
    if not freeze_path.exists():
        shutil.copyfile(ROOT/'checkpoints'/f"{best['id']}.pt",selected)
        shutil.copyfile(ROOT/'results'/best['id']/'edit_spec.json',ROOT/'edit_spec.json')
        _,base_blob=load(BASE);_,selected_blob=load(selected)
        base=base_blob['model_state'];edited=selected_blob['model_state']
        # Random controls use exactly the nonzero support and joint L2 norm of the chosen delta.
        support={k:(edited[k]!=base[k]) for k in base if k!='lm_head.weight'}
        count=sum(int(s.sum()) for s in support.values());assert count==best['changed_scalars']
        controls=[]
        for seed in protocol['controls']['random_seeds']:
            g=torch.Generator().manual_seed(seed)
            vector=torch.randn(count,generator=g,dtype=torch.float64)
            vector*=best['delta_l2']/float(vector.norm())
            state={k:v.clone() for k,v in base.items()};offset=0
            for name,mask in support.items():
                n=int(mask.sum());state[name][mask]+=vector[offset:offset+n].to(state[name].dtype);offset+=n
            state['lm_head.weight']=state['token_emb.weight']
            actual={k:state[k].double()-base[k].double() for k in support}
            actual_norm=math.sqrt(sum(float(d.square().sum()) for d in actual.values()))
            assert abs(actual_norm-best['delta_l2'])<max(1e-6,1e-5*best['delta_l2'])
            assert all(torch.equal(state[k][~mask],base[k][~mask]) for k,mask in support.items())
            assert sum(int(torch.count_nonzero(d)) for d in actual.values())==count
            p=ROOT/'checkpoints'/f'random_{seed}.pt'
            torch.save({'model_config':base_blob['model_config'],'model_state':state},p)
            controls.append({'seed':seed,'checkpoint':p.name,'sha256':sha256(p),'delta_l2':actual_norm,'changed_scalars':count})
        write_json(freeze_path,{'selected':best,'selected_sha256':sha256(selected),'edit_spec_sha256':sha256(ROOT/'edit_spec.json'),
                   'protocol_sha256':sha256(ROOT/'protocol.json'),'evaluator_sha256':sha256(ROOT/'run_experiment.py'),
                   'inference_sha256':sha256(ROOT/'inference.py'),'random_controls':controls,
                   'attempts_used':len(ledger),'final_metrics_accessed':False,
                   'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    frozen=read_json(freeze_path)
    assert sha256(selected)==frozen['selected_sha256']
    assert sha256(ROOT/'edit_spec.json')==frozen['edit_spec_sha256']
    assert sha256(ROOT/'protocol.json')==frozen['protocol_sha256']
    assert sha256(ROOT/'run_experiment.py')==frozen['evaluator_sha256']
    for name,path in [('selected',selected)]+[(f"random_{c['seed']}",ROOT/'checkpoints'/c['checkpoint']) for c in frozen['random_controls']]:
        for split in FINAL:
            print('Final evaluation:',name,split,flush=True)
            child_measure(path,split,ROOT/'results/final'/name)
    # First access to baseline final predictions after immutable freeze and all evaluations.
    shutil.copytree(ROOT/'sealed_final/baseline',ROOT/'results/final/baseline',dirs_exist_ok=True)
    write_json(ROOT/'results/final/access_log.json',{'first_opened_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'freeze_sha256':sha256(freeze_path),'development_closed':True})
    report()

def report():
    assert (ROOT/'freeze.json').exists() and (ROOT/'results/final/access_log.json').exists()
    import numpy as np
    protocol=validate_protocol();frozen=read_json(ROOT/'freeze.json')
    comparisons=[]
    models=['selected']+[f"random_{c['seed']}" for c in frozen['random_controls']]
    for split in FINAL:
        b=read_jsonl(ROOT/'results/final/baseline'/f'{split}.jsonl')
        bm=read_json(ROOT/'results/final/baseline'/f'{split}.metrics.json')
        for name in models:
            e=read_jsonl(ROOT/'results/final'/name/f'{split}.jsonl')
            em=read_json(ROOT/'results/final'/name/f'{split}.metrics.json')
            counts=paired_counts(b,e);n=len(b)
            row={'set':split,'model':name,'n':n,'before_correct':bm['strict_correct'],'after_correct':em['strict_correct'],
                 'before_em':bm['strict_em'],'after_em':em['strict_em'],
                 'delta_pp':100*(em['strict_em']-bm['strict_em']),**counts,
                 'before_seconds':bm['inference_seconds'],'after_seconds':em['inference_seconds'],
                 'before_format':bm['format_validity'],'after_format':em['format_validity']}
            if split!='test_exhaustive_00_99':
                # Per-example differences take values -1,0,+1. Their empirical multinomial
                # distribution gives exactly the same bootstrap as resampling the pairs.
                probs=[counts['right_to_wrong']/n,1-(counts['right_to_wrong']+counts['wrong_to_right'])/n,counts['wrong_to_right']/n]
                rng=np.random.default_rng(protocol['uncertainty']['seed'])
                samples=rng.multinomial(n,probs,size=protocol['uncertainty']['replicates'])
                lo,hi=np.quantile(100*(samples[:,2]-samples[:,0])/n,[.025,.975])
                row['paired_bootstrap_95ci_pp']=[float(lo),float(hi)]
            comparisons.append(row)
    write_json(ROOT/'results/final/comparisons.json',comparisons)
    import csv
    fields=['set','model','n','before_correct','after_correct','before_em','after_em','delta_pp','wrong_to_right','right_to_wrong','before_seconds','after_seconds','before_format','after_format','paired_bootstrap_95ci_pp']
    with (ROOT/'results/final/comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(comparisons)
    lines=['# Weight-space editing of a 1,644-parameter addition transformer','',
       '## Scope and provenance','',
       'The source model was trained by its author. This experiment performed inference and direct weight edits only; no gradients, optimizer, distillation, or new training. Architecture, tokenizer, positional encoding, forward, generation and decoding were unchanged.',
       'Source: https://github.com/anadim/smallest-addition-transformer-codex ; commit '+read_json(ROOT/'sources.json')['commit_sha']+'.',
       'No explicit license file was found in the pinned repository tree. Public availability is not treated as a redistribution license. Artifacts here are a private research workspace; sources can be fetched by the pinned hashes.',
       '', '## Results','',
       '| Set | Model | N | Before | After | Delta pp | Wrong→right | Right→wrong | 95% paired CI pp |',
       '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in comparisons:
        ci=r.get('paired_bootstrap_95ci_pp');cs=f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else 'finite exhaustive domain'
        lines.append(f"| {r['set']} | {r['model']} | {r['n']} | {100*r['before_em']:.3f}% | {100*r['after_em']:.3f}% | {r['delta_pp']:+.3f} | {r['wrong_to_right']} | {r['right_to_wrong']} | {cs} |")
    uniform=next(r for r in comparisons if r['set']=='test_uniform' and r['model']=='selected')
    lo,hi=uniform['paired_bootstrap_95ci_pp']
    outcome='confirmed positive change on test_uniform' if lo>0 else ('confirmed negative change on test_uniform' if hi<0 else 'uncertain change on test_uniform; interval includes zero')
    lines+=['','## Interpretation','',outcome+'.',
       f"Selected {frozen['selected']['id']} after {frozen['attempts_used']} validation attempts. Changed {frozen['selected']['changed_scalars']} independent scalars; L2 delta {frozen['selected']['delta_l2']:.9g}.",
       'All candidate outcomes are in results/attempts.json; selection used validation only. Random seeds were fixed in advance; no best-seed selection.',
       '', '## Encoding','',
       'Each operand is zero-padded to 10 decimal digits. Ten Pab digit-pair tokens appear least-significant column first between BOS and equals. The unmodified model greedily generates 12 tokens: 11 reversed sum digits and EOS. Strict EM requires all 12 tokens; author-compatible digits-only EM and the upstream numeric decoder are secondary metrics.',
       '', '## Limits','',
       'One model, checkpoint, arithmetic operation, and generation configuration. No conclusion about general mathematical reasoning, decimal multiplication, other models, or unseen implementations. The training stream is unavailable, so splits are held out from editing, not proven absent from original training. The 0…99 test intentionally includes both operand orders and four known manual examples (with 99+1 also represented in reverse). Bootstrap intervals summarize test-sampling uncertainty for a fixed selected edit, not variability across independent training runs. Stress strata are engineered distributions, not a natural usage distribution. Three random edits are descriptive controls and cannot establish superiority over all random editing.',
       '', '## Timing','',
       'CPU float32, one thread, batch 256, greedy 12-token generation, fixed warmup. Individual timings are one run per condition and are not a speed benchmark; loading and scoring are excluded.',
       '', '## Reproduction','',
       'See README.md. Protocol and code hashes, predictions, all metrics, unchanged buffers/ties, zero edit, rollback and fresh-process reload checks are retained. Final errors must not be used for further confirmatory edits on these same tests.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Report saved:',ROOT/'REPORT.md',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['prepare','baseline','edit','finalize','report','worker'],required=True)
    p.add_argument('--edit-spec',type=Path)
    p.add_argument('--max-candidates',type=int,default=12)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--split');p.add_argument('--out',type=Path)
    args=p.parse_args();configure()
    if args.stage=='prepare':prepare()
    elif args.stage=='baseline':baseline()
    elif args.stage=='edit':
        if not args.edit_spec:p.error('--edit-spec required; the agent must propose a documented meaningful edit')
        if args.max_candidates>12:p.error('Maximum budget is 12')
        if (ROOT/'results/attempts.json').exists() and len(read_json(ROOT/'results/attempts.json'))>=args.max_candidates:p.error('Requested budget exhausted')
        evaluate_edit(args.edit_spec)
    elif args.stage=='finalize':finalize()
    elif args.stage=='report':report()
    else:measure(args.checkpoint,args.split,args.out)

if __name__=='__main__':main()
