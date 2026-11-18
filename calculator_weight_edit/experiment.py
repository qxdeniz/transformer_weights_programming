"""Independent second-model experiment. No training; development closes at freeze."""
import argparse,copy,csv,datetime,json,math,os,platform,shutil,subprocess,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import torch
from data_protocol import ROOT,prepare,read_json,write_json,read_rows,write_rows,sha,key
from model_api import load,generate,output_text,reference_generate,configure
from edit_weights import apply_edit

BASE=ROOT/'upstream/model.pt'
FINAL=['test_uniform','test_exhaustive']

def verify():
    src=read_json(ROOT/'sources.json')
    for name,digest in src['files_sha256'].items():assert sha(ROOT/'upstream'/name)==digest,name
    p=read_json(ROOT/'protocol.json')
    for name,d in p['datasets'].items():assert sha(ROOT/'data'/f'{name}.jsonl')==d['sha256']
    for name,digest in p['code_sha256'].items():assert sha(ROOT/name)==digest,'Protocol code changed: '+name
    return p

def metrics(rows,seconds):
    n=len(rows);groups=defaultdict(list)
    for row in rows:groups[row['operation']].append(row)
    by={op:{'n':len(rs),'correct':sum(r['strict_exact'] for r in rs),'em':sum(r['strict_exact'] for r in rs)/len(rs)} for op,rs in groups.items()}
    return {'n':n,'strict_correct':sum(r['strict_exact'] for r in rows),'strict_em':sum(r['strict_exact'] for r in rows)/n,
            'text_em':sum(r['text_correct'] for r in rows)/n,'format_validity':sum(r['format_valid'] for r in rows)/n,
            'by_operation':by,'macro_em':sum(v['em'] for v in by.values())/len(by),
            'inference_seconds':seconds,'termination_counts':dict(Counter(r['termination'] for r in rows))}

def measure(checkpoint,split,folder):
    verify();data=read_rows(ROOT/'data'/f'{split}.jsonl');m,t,v=load(checkpoint)
    generate(m,t,['two plus three equals']*16)
    start=time.perf_counter();prompts,tails=generate(m,t,[r['prompt'] for r in data]);elapsed=time.perf_counter()-start
    rows=[]
    numeric_ids={v[w] for w in v if w not in ['[PAD]','[START]','[END]','[UNK]','plus','minus','times','equals']}
    for r,pred,prompt_ids in zip(data,tails,prompts):
        target=[v[w] for w in r['expected_text'].split()]+[v['[END]']]
        text=output_text(pred,v)
        valid=bool(pred and pred[-1]==v['[END]'] and all(x in numeric_ids for x in pred[:-1]) and 1<=len(pred[:-1])<=2)
        rows.append({**r,'input_tokens':prompt_ids,'target_tokens':target,'generated_tokens':pred,
                     'generated_text':text,'strict_exact':pred==target,'text_correct':text==r['expected_text'],
                     'format_valid':valid,'termination':'END' if pred and pred[-1]==v['[END]'] else 'token_limit',
                     'generated_token_count':len(pred)})
    folder=Path(folder);write_rows(folder/f'{split}.jsonl',rows)
    mm=metrics(rows,elapsed);mm['checkpoint_sha256']=sha(checkpoint);write_json(folder/f'{split}.metrics.json',mm)
    return rows,mm

def child(checkpoint,split,folder):
    r=subprocess.run([sys.executable,str(ROOT/'experiment.py'),'worker','--checkpoint',str(checkpoint),'--split',split,'--out',str(folder)],cwd=ROOT,capture_output=True,text=True)
    if r.returncode:raise RuntimeError(r.stdout[-1000:]+'\n'+r.stderr[-4000:])

def counts(base,after):
    assert [r['id'] for r in base]==[r['id'] for r in after]
    return {'wrong_to_right':sum(not a['strict_exact'] and b['strict_exact'] for a,b in zip(base,after)),
            'right_to_wrong':sum(a['strict_exact'] and not b['strict_exact'] for a,b in zip(base,after))}

def baseline():
    assert not (ROOT/'results/baseline/complete.json').exists(),'Baseline exists'
    if not (ROOT/'protocol.json').exists():prepare()
    verify();m,t,v=load(BASE)
    rows=read_rows(ROOT/'data/diagnostic.jsonl')[:100];prompts=[r['prompt'] for r in rows]
    encoded,tails=generate(m,t,prompts)
    assert (encoded,tails)==generate(m,t,prompts)
    for r,pred,ids in zip(rows,tails,encoded):
        reference=reference_generate(m,t,r['prompt'],max_new_tokens=8)
        rendered=t.decode(ids+[x for x in pred if x!=v['[END]']])
        assert reference==rendered,(r['prompt'],reference,rendered)
        one=generate(m,t,[r['prompt']])[1][0];assert one==pred
        assert v['[UNK]'] not in ids
    state={k:x.clone() for k,x in m.state_dict().items()}
    zero=ROOT/'checkpoints/zero.pt';zero.parent.mkdir(exist_ok=True);torch.save(state,zero)
    mz,tz,_=load(zero);assert generate(mz,tz,prompts)==(encoded,tails)
    child(zero,'diagnostic',ROOT/'results/zero_edit')
    import numpy
    write_json(ROOT/'environment.json',{'python':sys.version,'pytorch':torch.__version__,'numpy':numpy.__version__,
               'platform':platform.platform(),'cpu_count':os.cpu_count(),'device':'cpu','threads':1,'dtype':'float32','config':read_json(ROOT/'upstream/config.json')})
    write_json(ROOT/'parameter_inventory.json',{'parameters':[{'name':n,'shape':list(p.shape),'scalars':p.numel(),'dtype':str(p.dtype)} for n,p in m.named_parameters()],
                'buffers':[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in m.named_buffers()],
                'independent_scalars':104740,'shared_weight_groups':[]})
    for split in ['diagnostic','validation']:
        rr,mm=measure(BASE,split,ROOT/'results/baseline');print(split,json.dumps(mm),flush=True)
    assert [r['generated_tokens'] for r in read_rows(ROOT/'results/zero_edit/diagnostic.jsonl')]==[r['generated_tokens'] for r in read_rows(ROOT/'results/baseline/diagnostic.jsonl')]
    for split in FINAL:
        print('Saving sealed baseline:',split,flush=True);child(BASE,split,ROOT/'sealed_final/baseline')
    write_json(ROOT/'results/baseline/complete.json',{'status':'complete','reference_agreement':100,'single_batch_token_equality':100,'zero_reload_equal':True,'final_metrics_sealed':True})

def edit(spec_path):
    p=verify();assert not (ROOT/'freeze.json').exists(),'Development closed'
    assert (ROOT/'results/baseline/complete.json').exists()
    ledger_path=ROOT/'results/attempts.json';ledger=read_json(ledger_path) if ledger_path.exists() else []
    assert len(ledger)<p['validation_budget']
    spec=read_json(spec_path);assert all(spec.get(k) for k in ['hypothesis','prediction','risk','operations'])
    name=f'candidate_{len(ledger)+1:02d}';folder=ROOT/'results'/name;folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'edit_spec.json',spec)
    record={'id':name,'status':'started','spec_sha256':sha(folder/'edit_spec.json')};ledger.append(record);write_json(ledger_path,ledger)
    started=time.perf_counter();m,t,v=load(BASE);base={k:x.clone() for k,x in m.state_dict().items()}
    names={n for n,_ in m.named_parameters()};state,diff,masks=apply_edit(base,names,spec);assert diff['changed_scalars']>0
    checkpoint=ROOT/'checkpoints'/f'{name}.pt';torch.save(state,checkpoint)
    preparation=time.perf_counter()-started
    prompts=[r['prompt'] for r in read_rows(ROOT/'data/diagnostic.jsonl')[:100]]
    original=generate(m,t,prompts);m.load_state_dict(state,strict=True);after=generate(m,t,prompts)
    m.load_state_dict(base,strict=True);assert generate(m,t,prompts)==original,'Rollback mismatch'
    loaded,tl,_=load(checkpoint);assert generate(loaded,tl,prompts)==after
    child(checkpoint,'diagnostic',folder);child(checkpoint,'validation',folder)
    assert [r['generated_tokens'] for r in read_rows(folder/'diagnostic.jsonl')[:100]]==after[1]
    baseline_val=read_rows(ROOT/'results/baseline/validation.jsonl');val=read_rows(folder/'validation.jsonl')
    mm=read_json(folder/'validation.metrics.json')
    record.update({'status':'complete','validation_correct':mm['strict_correct'],'validation_macro_em':mm['macro_em'],
                   **counts(baseline_val,val),**diff,'preparation_seconds':preparation,'checkpoint_sha256':sha(checkpoint)})
    write_json(folder/'diff.json',diff);write_json(folder/'checks.json',{'rollback_equal':True,'zero_buffer_changes':True,
               'outside_mask_unchanged':True,'architecture_keys_shapes_dtypes_unchanged':True,'finite':True,'fresh_process_equal':True})
    write_json(ledger_path,ledger);print(json.dumps(record),flush=True)

def finalize():
    protocol=verify();ledger=read_json(ROOT/'results/attempts.json')
    winner=min([r for r in ledger if r['status']=='complete'],key=lambda r:(-r['validation_correct'],r['right_to_wrong'],r['changed_scalars'],r['delta_l2']))
    freeze=ROOT/'freeze.json';selected=ROOT/'checkpoints/selected.pt'
    if not freeze.exists():
        shutil.copyfile(ROOT/'checkpoints'/f"{winner['id']}.pt",selected)
        shutil.copyfile(ROOT/'results'/winner['id']/'edit_spec.json',ROOT/'edit_spec.json')
        m,_,_=load(BASE);base=m.state_dict();edited=torch.load(selected,map_location='cpu',weights_only=True)
        support={k:edited[k]!=base[k] for k,_ in m.named_parameters()};count=sum(int(x.sum()) for x in support.values())
        assert count==winner['changed_scalars'];controls=[]
        for seed in protocol['random_seeds']:
            g=torch.Generator().manual_seed(seed);vec=torch.randn(count,generator=g,dtype=torch.float64);vec*=winner['delta_l2']/float(vec.norm())
            state={k:x.clone() for k,x in base.items()};offset=0
            for name,mask in support.items():
                n=int(mask.sum());state[name][mask]+=vec[offset:offset+n].to(state[name].dtype);offset+=n
            actual=math.sqrt(sum(float((state[k].double()-base[k].double()).square().sum()) for k in support))
            assert abs(actual-winner['delta_l2'])<max(1e-6,1e-5*winner['delta_l2'])
            assert all(torch.equal(state[k][~mask],base[k][~mask]) for k,mask in support.items())
            assert sum(int(torch.count_nonzero(state[k]-base[k])) for k in support)==count
            path=ROOT/'checkpoints'/f'random_{seed}.pt';torch.save(state,path)
            controls.append({'seed':seed,'checkpoint':path.name,'sha256':sha(path),'delta_l2':actual,'changed_scalars':count})
        write_json(freeze,{'selected':winner,'selected_sha256':sha(selected),'edit_spec_sha256':sha(ROOT/'edit_spec.json'),
                  'protocol_sha256':sha(ROOT/'protocol.json'),'evaluator_sha256':sha(ROOT/'experiment.py'),
                  'controls':controls,'attempts_used':len(ledger),'final_metrics_accessed':False,
                  'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    frozen=read_json(freeze)
    for path,digest in [(selected,frozen['selected_sha256']),(ROOT/'edit_spec.json',frozen['edit_spec_sha256']),(ROOT/'protocol.json',frozen['protocol_sha256']),(ROOT/'experiment.py',frozen['evaluator_sha256'])]:assert sha(path)==digest
    for name,path in [('selected',selected)]+[(f"random_{c['seed']}",ROOT/'checkpoints'/c['checkpoint']) for c in frozen['controls']]:
        for split in FINAL:
            print('Final:',name,split,flush=True);child(path,split,ROOT/'results/final'/name)
    shutil.copytree(ROOT/'sealed_final/baseline',ROOT/'results/final/baseline',dirs_exist_ok=True)
    write_json(ROOT/'results/final/access_log.json',{'opened_after_freeze':True,'freeze_sha256':sha(freeze),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    report()

def report():
    assert (ROOT/'results/final/access_log.json').exists()
    import numpy as np
    p=verify();f=read_json(ROOT/'freeze.json');comparisons=[]
    for split in FINAL:
        base=read_rows(ROOT/'results/final/baseline'/f'{split}.jsonl');bm=read_json(ROOT/'results/final/baseline'/f'{split}.metrics.json')
        for name in ['selected']+[f"random_{c['seed']}" for c in f['controls']]:
            after=read_rows(ROOT/'results/final'/name/f'{split}.jsonl');em=read_json(ROOT/'results/final'/name/f'{split}.metrics.json')
            n=len(base);cc=counts(base,after);delta=100*(em['strict_em']-bm['strict_em'])
            row={'set':split,'model':name,'n':n,'before_correct':bm['strict_correct'],'after_correct':em['strict_correct'],
                 'before_em':bm['strict_em'],'after_em':em['strict_em'],'delta_pp':delta,**cc,
                 'before_macro_em':bm['macro_em'],'after_macro_em':em['macro_em'],
                 'before_seconds':bm['inference_seconds'],'after_seconds':em['inference_seconds'],
                 'before_format':bm['format_validity'],'after_format':em['format_validity'],'by_operation':{}}
            rng=np.random.default_rng(p['bootstrap']['seed']);bs=np.zeros(p['bootstrap']['replicates']);macro_bs=np.zeros_like(bs)
            for op in ['plus','minus','times']:
                b=[r for r in base if r['operation']==op];e=[r for r in after if r['operation']==op];c=counts(b,e);nn=len(b)
                row['by_operation'][op]={'n':nn,'before_correct':bm['by_operation'][op]['correct'],'after_correct':em['by_operation'][op]['correct'],**c}
                if split=='test_uniform':
                    prob=[c['right_to_wrong']/nn,1-(c['right_to_wrong']+c['wrong_to_right'])/nn,c['wrong_to_right']/nn]
                    draw=rng.multinomial(nn,prob,size=p['bootstrap']['replicates']);diff=(draw[:,2]-draw[:,0])/nn
                    bs+=100*(nn/n)*diff;macro_bs+=100/3*diff
            if split=='test_uniform':
                row['paired_stratified_95ci_pp']=np.quantile(bs,[.025,.975]).tolist()
                row['paired_macro_95ci_pp']=np.quantile(macro_bs,[.025,.975]).tolist()
            comparisons.append(row)
    write_json(ROOT/'results/final/comparisons.json',comparisons)
    simple=[{k:v for k,v in r.items() if k!='by_operation'} for r in comparisons]
    with (ROOT/'results/final/comparisons.csv').open('w',newline='') as fh:
        fields=list(dict.fromkeys(k for r in simple for k in r));w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(simple)
    lines=['# Second model: Calculator LLM (104,740 parameters)','',
       '## Method','',
       'The original training architecture and tokenizer were loaded from the pinned GitHub revision; released weights came from the pinned HF Space. The demo tokenizer has a letter-x replacement bug that corrupts six/sixty; using the original training tokenizer is a pre-baseline compatibility fix, not a weight improvement.',
       'Original training code repeats every valid task; no claim of novelty relative to original training is made. Splits are held out from weight editing, with commutative groups kept together.',
       'Task: word-form addition, nonnegative subtraction and multiplication, operands and answers 0…99. This is a distinct benchmark from decimal ten-digit addition; raw accuracy is not a direct model ranking.',
       '', '## Final results','',
       '| Set | Model | N | Before % | After % | Delta pp | W→R | R→W | 95% paired CI pp |',
       '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in comparisons:
        ci=r.get('paired_stratified_95ci_pp');cs=f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else 'complete finite domain'
        lines.append(f"| {r['set']} | {r['model']} | {r['n']} | {100*r['before_em']:.3f} | {100*r['after_em']:.3f} | {r['delta_pp']:+.3f} | {r['wrong_to_right']} | {r['right_to_wrong']} | {cs} |")
    target=next(r for r in comparisons if r['set']=='test_uniform' and r['model']=='selected');lo,hi=target['paired_stratified_95ci_pp']
    result='positive change confirmed under this editing protocol' if lo>0 else ('negative change confirmed' if hi<0 else 'change uncertain; interval includes zero')
    lines+=['',result+'.','',f"Selected {f['selected']['id']} after {f['attempts_used']} validation trials. Changed {f['selected']['changed_scalars']} / 104740 scalars; L2 delta {f['selected']['delta_l2']:.8f}.",'',
            '## Selected edit by operation','','| Operation | N | Before correct | After correct | W→R | R→W |','|---|---:|---:|---:|---:|---:|']
    for op,r in target['by_operation'].items():lines.append(f"| {op} | {r['n']} | {r['before_correct']} | {r['after_correct']} | {r['wrong_to_right']} | {r['right_to_wrong']} |")
    lines+=['','## Limits','',
       'All valid tasks may have been seen during original training. The final set is stratified by operation and canonical commutative pair; it is not uniform over ordered inputs. Multiplication is restricted to products at most 99. Decimal multiplication is not supported. Exhaustive evaluation includes development examples and both commutative orders; it is a finite-domain description, not an independent generalization test.',
       'One checkpoint per architecture and adaptive selection by validation cannot establish broad effects across models. Bootstrap uncertainty is conditional on the fixed selected edit and within-stratum sampling, not repeated model training. Three random seeds are descriptive controls only. Final errors must not be reused for further confirmatory selection on these same tests.',
       'No gradients, training, regression, general optimizer or evolutionary search were used. Buffers, architecture, tokenizer, forward and decoding stay fixed across all reported before/after conditions. Candidate construction time records code application and serialization; agent reasoning time was not separately measured.',
       'CPU float32, one thread, greedy 8-token limit, batch 128 grouped by input length; reference single-example agreement checked on 100 fixed diagnostic cases. Times exclude model loading and scoring, and are single-run observations.',
       '', 'Sources: https://huggingface.co/spaces/slahiri/small_calculator_model and https://github.com/slahiri/small_calculator_model . Revisions and SHA-256 in sources.json. MIT is claimed by the GitHub README, but its linked LICENSE file is absent; no separate Space weights license was supplied.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Saved',ROOT/'REPORT.md',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['baseline','edit','finalize','report','worker']);p.add_argument('--spec',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--split');p.add_argument('--out',type=Path)
    a=p.parse_args();configure()
    if a.stage=='baseline':baseline()
    elif a.stage=='edit':edit(a.spec)
    elif a.stage=='finalize':finalize()
    elif a.stage=='report':report()
    else:measure(a.checkpoint,a.split,a.out)

if __name__=='__main__':main()
