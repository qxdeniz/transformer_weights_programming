"""No-gradient editing study on a general text transformer and a held-out BLiMP subset."""
import argparse,datetime,json,math,os,platform,shutil,subprocess,sys,time,csv
from pathlib import Path
from collections import Counter,defaultdict
import torch
from utils import ROOT,read_json,write_json,read_rows,write_rows,sha
from benchmark import prepare
from language_model import load,log_likelihoods,greedy,configure
from edit_weights import apply_edit,EMBED,ALIAS

BASE=ROOT/'upstream/model/model.safetensors'

def verify():
    for name,digest in read_json(ROOT/'sources.json')['files_sha256'].items():assert sha(ROOT/'upstream'/name)==digest,name
    p=read_json(ROOT/'protocol.json')
    for split,meta in p['datasets'].items():assert sha(ROOT/'data'/f'{split}.jsonl')==meta['sha256']
    for name,digest in p['code_sha256'].items():assert sha(ROOT/name)==digest,'Protocol code changed: '+name
    return p

def load_ckpt(path):return load(None if Path(path)==BASE else path)

def summary(rows,elapsed):
    groups=defaultdict(list)
    for r in rows:groups[r['paradigm']].append(r)
    by={k:{'n':len(v),'correct':sum(r['correct'] for r in v),'accuracy':sum(r['correct'] for r in v)/len(v),
           'mean_margin':sum(r['margin'] for r in v)/len(v)} for k,v in groups.items()}
    return {'n':len(rows),'correct':sum(r['correct'] for r in rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),
            'macro_accuracy':sum(x['accuracy'] for x in by.values())/len(by),'mean_margin':sum(r['margin'] for r in rows)/len(rows),
            'ties':sum(r['margin']==0 for r in rows),'by_paradigm':by,'inference_seconds':elapsed}

def measure(checkpoint,split,folder):
    verify();data=read_rows(ROOT/'data'/f'{split}.jsonl');m,t=load_ckpt(checkpoint)
    sentences=[s for r in data for s in [r['sentence_good'],r['sentence_bad']]]
    log_likelihoods(m,t,['The cat sleeps.','The cat sleep.']*8)
    start=time.perf_counter();tokens,ll=log_likelihoods(m,t,sentences);elapsed=time.perf_counter()-start
    rows=[]
    for i,r in enumerate(data):
        good,bad=ll[2*i:2*i+2];margin=good-bad
        rows.append({**r,'good_tokens':tokens[2*i],'bad_tokens':tokens[2*i+1],
                     'good_loglikelihood':good,'bad_loglikelihood':bad,'margin':margin,'correct':margin>0})
    folder=Path(folder);write_rows(folder/f'{split}.jsonl',rows)
    mm=summary(rows,elapsed);mm['checkpoint_sha256']=sha(checkpoint);mm['dataset_sha256']=sha(ROOT/'data'/f'{split}.jsonl')
    write_json(folder/f'{split}.metrics.json',mm);return rows,mm

def child(checkpoint,split,folder):
    p=subprocess.run([sys.executable,str(ROOT/'experiment.py'),'worker','--checkpoint',str(checkpoint),'--split',split,'--out',str(folder)],cwd=ROOT,capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stdout[-500:]+'\n'+p.stderr[-5000:])

def transitions(a,b):
    assert [r['id'] for r in a]==[r['id'] for r in b]
    return {'wrong_to_right':sum(not x['correct'] and y['correct'] for x,y in zip(a,b)),
            'right_to_wrong':sum(x['correct'] and not y['correct'] for x,y in zip(a,b))}

def illustrative(model,tokenizer,folder):
    prompts=['The cat','Once upon a time','Today I learned that','The capital of France is','A good explanation']
    write_json(Path(folder)/'generations.json',[greedy(model,tokenizer,p) for p in prompts])

def baseline():
    assert not (ROOT/'results/baseline/complete.json').exists(),'Baseline already completed'
    if not (ROOT/'protocol.json').exists():prepare()
    protocol=verify();m,t=load()
    data=read_rows(ROOT/'data/diagnostic.jsonl')[:50];sentences=[s for r in data for s in [r['sentence_good'],r['sentence_bad']]]
    tokens,scores=log_likelihoods(m,t,sentences)
    assert (tokens,scores)==log_likelihoods(m,t,sentences)
    # Independent official Hugging Face loader; tied missing head is resolved by HF.
    from transformers import LlamaForCausalLM
    reference=LlamaForCausalLM.from_pretrained(ROOT/'upstream/model',local_files_only=True,dtype=torch.float32,attn_implementation='eager').cpu().eval()
    for name,param in m.state_dict().items():assert torch.equal(param,reference.state_dict()[name]),name
    rt,rs=log_likelihoods(reference,t,sentences,batch_size=1)
    assert rt==tokens and max(abs(a-b) for a,b in zip(scores,rs))<1e-4
    assert [scores[i]>scores[i+1] for i in range(0,100,2)]==[rs[i]>rs[i+1] for i in range(0,100,2)]
    state=m.state_dict();zero=ROOT/'checkpoints/zero.pt';zero.parent.mkdir(exist_ok=True);torch.save(state,zero)
    mz,tz=load(zero);assert log_likelihoods(mz,tz,sentences)==(tokens,scores)
    child(zero,'diagnostic',ROOT/'results/zero_edit')
    import transformers,tokenizers,numpy,safetensors
    write_json(ROOT/'environment.json',{'python':sys.version,'executable':sys.executable,'pytorch':torch.__version__,
              'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'numpy':numpy.__version__,'safetensors':safetensors.__version__,
              'platform':platform.platform(),'device':'cpu','dtype':'float32','threads':1,'attention':'official eager','score_batch':64,
              'dependencies':'Local venv with read-only system torch and compatible read-only cached packages; no global packages modified.'})
    (ROOT/'requirements.txt').write_text(f'torch=={torch.__version__}\ntransformers=={transformers.__version__}\ntokenizers=={tokenizers.__version__}\nsafetensors=={safetensors.__version__}\nnumpy=={numpy.__version__}\nmatplotlib\n')
    write_json(ROOT/'parameter_inventory.json',{'parameters':[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype),'scalars':p.numel()} for n,p in m.named_parameters()],
              'buffers':[{'name':n,'shape':list(b.shape),'dtype':str(b.dtype)} for n,b in m.named_buffers()],
              'independent_parameters':11856,'alias_groups':[[EMBED,ALIAS]],'config':m.config.to_dict()})
    for split in ['diagnostic','validation']:
        _,mm=measure(BASE,split,ROOT/'results/baseline');print(split,json.dumps(mm),flush=True)
    a=read_rows(ROOT/'results/zero_edit/diagnostic.jsonl');b=read_rows(ROOT/'results/baseline/diagnostic.jsonl')
    assert [r['margin'] for r in a]==[r['margin'] for r in b]
    illustrative(m,t,ROOT/'results/baseline')
    print('Saving sealed baseline test',flush=True);child(BASE,'test',ROOT/'sealed_final/baseline')
    write_json(ROOT/'results/baseline/complete.json',{'status':'complete','official_loader_equal':True,'single_batch_score_agreement_sentences':100,
                  'repeat_equal':True,'zero_edit_equal':True,'test_sealed':True,'base_sha256':sha(BASE)})

def edit(spec_path):
    protocol=verify();assert not (ROOT/'freeze.json').exists(),'Development closed'
    assert (ROOT/'results/baseline/complete.json').exists()
    ledger_path=ROOT/'results/attempts.json';ledger=read_json(ledger_path) if ledger_path.exists() else []
    assert len(ledger)<protocol['validation_budget']
    spec=read_json(spec_path);assert all(spec.get(k) for k in ['hypothesis','prediction','risk','operations'])
    name=f'candidate_{len(ledger)+1:02d}';folder=ROOT/'results'/name;folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'edit_spec.json',spec);record={'id':name,'status':'started','spec_sha256':sha(folder/'edit_spec.json')}
    ledger.append(record);write_json(ledger_path,ledger)
    started=time.perf_counter();m,t=load();base={k:v.clone() for k,v in m.state_dict().items()};base[ALIAS]=base[EMBED]
    buffer_copy={n:v.clone() for n,v in m.named_buffers()};names={n for n,_ in m.named_parameters()}
    state,diff,masks=apply_edit(base,names,spec);assert diff['changed_scalars']>0
    checkpoint=ROOT/'checkpoints'/f'{name}.pt';torch.save(state,checkpoint);preparation=time.perf_counter()-started
    sentences=[r['sentence_good'] for r in read_rows(ROOT/'data/diagnostic.jsonl')[:100]]
    original=log_likelihoods(m,t,sentences);m.load_state_dict(state,strict=True);edited=log_likelihoods(m,t,sentences)
    assert all(torch.equal(buffer_copy[n],b) for n,b in m.named_buffers())
    assert m.lm_head.weight is m.model.embed_tokens.weight
    m.load_state_dict(base,strict=True);assert log_likelihoods(m,t,sentences)==original,'Rollback failed'
    loaded,tl=load(checkpoint);assert log_likelihoods(loaded,tl,sentences)==edited
    child(checkpoint,'diagnostic',folder);child(checkpoint,'validation',folder)
    val=read_rows(folder/'validation.jsonl');baseline_val=read_rows(ROOT/'results/baseline/validation.jsonl');mm=read_json(folder/'validation.metrics.json')
    record.update({'status':'complete','validation_correct':mm['correct'],**transitions(baseline_val,val),**diff,'preparation_seconds':preparation,'checkpoint_sha256':sha(checkpoint)})
    write_json(folder/'diff.json',diff);write_json(folder/'checks.json',{'keys_shapes_dtypes_fixed':True,'buffers_fixed':True,'aliases_fixed':True,'finite':True,'outside_mask_fixed':True,'rollback_equal':True,'fresh_process_equal':True})
    write_json(ledger_path,ledger);print(json.dumps(record),flush=True)

def finalize():
    protocol=verify();ledger=read_json(ROOT/'results/attempts.json')
    winner=min([r for r in ledger if r['status']=='complete'],key=lambda r:(-r['validation_correct'],r['right_to_wrong'],r['changed_scalars'],r['delta_l2']))
    selected=ROOT/'checkpoints/selected.pt';freeze=ROOT/'freeze.json'
    if not freeze.exists():
        shutil.copyfile(ROOT/'checkpoints'/f"{winner['id']}.pt",selected)
        shutil.copyfile(ROOT/'results'/winner['id']/'edit_spec.json',ROOT/'edit_spec.json')
        model,_=load();base=model.state_dict();edited=torch.load(selected,map_location='cpu',weights_only=True)
        support={name:edited[name]!=base[name] for name,_ in model.named_parameters()};count=sum(int(m.sum()) for m in support.values())
        assert count==winner['changed_scalars'];controls=[]
        for seed in protocol['controls']['random_seeds']:
            generator=torch.Generator().manual_seed(seed);vec=torch.randn(count,generator=generator,dtype=torch.float64);vec*=winner['delta_l2']/float(vec.norm())
            state={k:v.clone() for k,v in base.items()};offset=0
            for name,mask in support.items():
                n=int(mask.sum());state[name][mask]+=vec[offset:offset+n].to(state[name].dtype);offset+=n
            state[ALIAS]=state[EMBED]
            actual=math.sqrt(sum(float((state[k].double()-base[k].double()).square().sum()) for k in support))
            assert abs(actual-winner['delta_l2'])<max(1e-6,winner['delta_l2']*1e-5)
            assert sum(int(torch.count_nonzero(state[k]-base[k])) for k in support)==count
            assert all(torch.equal(state[k][~mask],base[k][~mask]) for k,mask in support.items())
            path=ROOT/'checkpoints'/f'random_{seed}.pt';torch.save(state,path)
            controls.append({'seed':seed,'checkpoint':path.name,'sha256':sha(path),'delta_l2':actual,'changed_scalars':count})
        write_json(freeze,{'selected':winner,'selected_sha256':sha(selected),'edit_spec_sha256':sha(ROOT/'edit_spec.json'),
                   'protocol_sha256':sha(ROOT/'protocol.json'),'evaluator_sha256':sha(ROOT/'experiment.py'),
                   'controls':controls,'attempts_used':len(ledger),'test_opened':False,'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    f=read_json(freeze)
    for path,digest in [(selected,f['selected_sha256']),(ROOT/'edit_spec.json',f['edit_spec_sha256']),(ROOT/'protocol.json',f['protocol_sha256']),(ROOT/'experiment.py',f['evaluator_sha256'])]:assert sha(path)==digest
    for name,path in [('selected',selected)]+[(f"random_{c['seed']}",ROOT/'checkpoints'/c['checkpoint']) for c in f['controls']]:
        print('Final test:',name,flush=True);child(path,'test',ROOT/'results/final'/name)
    shutil.copytree(ROOT/'sealed_final/baseline',ROOT/'results/final/baseline',dirs_exist_ok=True)
    write_json(ROOT/'results/final/access_log.json',{'freeze_sha256':sha(freeze),'opened_after_freeze':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    model,tokenizer=load(selected);illustrative(model,tokenizer,ROOT/'results/final/selected');report()

def report():
    assert (ROOT/'results/final/access_log.json').exists()
    import numpy as np
    protocol=verify();f=read_json(ROOT/'freeze.json');baseline=read_rows(ROOT/'results/final/baseline/test.jsonl');bm=read_json(ROOT/'results/final/baseline/test.metrics.json')
    comparisons=[]
    for name in ['selected']+[f"random_{c['seed']}" for c in f['controls']]:
        edited=read_rows(ROOT/'results/final'/name/'test.jsonl');em=read_json(ROOT/'results/final'/name/'test.metrics.json');n=len(baseline)
        cc=transitions(baseline,edited);row={'model':name,'n':n,'before_correct':bm['correct'],'after_correct':em['correct'],
            'before_accuracy':bm['accuracy'],'after_accuracy':em['accuracy'],'delta_pp':100*(em['accuracy']-bm['accuracy']),**cc,
            'before_macro_accuracy':bm['macro_accuracy'],'after_macro_accuracy':em['macro_accuracy'],
            'before_seconds':bm['inference_seconds'],'after_seconds':em['inference_seconds'],'by_paradigm':{}}
        rng=np.random.default_rng(protocol['statistics']['seed']);rep=protocol['statistics']['replicates'];sample=np.zeros(rep);macro_sample=np.zeros(rep)
        for paradigm in protocol['paradigms']:
            b=[r for r in baseline if r['paradigm']==paradigm];e=[r for r in edited if r['paradigm']==paradigm];counts=transitions(b,e);nn=len(b)
            probs=[counts['right_to_wrong']/nn,1-(counts['right_to_wrong']+counts['wrong_to_right'])/nn,counts['wrong_to_right']/nn]
            draws=rng.multinomial(nn,probs,size=rep);d=(draws[:,2]-draws[:,0])/nn
            sample+=100*(nn/n)*d;macro_sample+=100/len(protocol['paradigms'])*d
            row['by_paradigm'][paradigm]={'n':nn,'before_correct':bm['by_paradigm'][paradigm]['correct'],'after_correct':em['by_paradigm'][paradigm]['correct'],**counts}
        row['paired_stratified_95ci_pp']=np.quantile(sample,[.025,.975]).tolist();row['paired_macro_95ci_pp']=np.quantile(macro_sample,[.025,.975]).tolist();comparisons.append(row)
    write_json(ROOT/'results/final/comparisons.json',comparisons)
    simple=[{k:v for k,v in r.items() if k!='by_paradigm'} for r in comparisons]
    with (ROOT/'results/final/comparisons.csv').open('w',newline='') as fh:
        fields=list(simple[0]);w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(simple)
    lines=['# General language model: Glimmer-1-Base weight editing','',
      'Model: 11,856 independent parameters, two Llama layers, hidden width 16, grouped-query attention, SiLU MLP width 24, tied 512-token embedding/output. The author pretrained on FineWeb-Edu; no arithmetic-specialist training is used here.',
      '', '## Frozen BLiMP subset results','',
      '| Condition | N | Before % | After % | Delta pp | Wrong→right | Right→wrong | 95% paired CI pp |',
      '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in comparisons:
        lo,hi=r['paired_stratified_95ci_pp'];lines.append(f"| {r['model']} | {r['n']} | {100*r['before_accuracy']:.3f} | {100*r['after_accuracy']:.3f} | {r['delta_pp']:+.3f} | {r['wrong_to_right']} | {r['right_to_wrong']} | [{lo:+.3f}, {hi:+.3f}] |")
    target=comparisons[0];lo,hi=target['paired_stratified_95ci_pp']
    conclusion='Positive change confirmed on this fixed BLiMP subset.' if lo>0 else ('Negative change confirmed on this subset.' if hi<0 else 'Improvement is not confirmed: the interval includes zero.')
    lines+=['',conclusion,'',f"Selected {f['selected']['id']} after {f['attempts_used']} validation evaluations. Changed {f['selected']['changed_scalars']} / 11856 independent scalars, L2 delta {f['selected']['delta_l2']:.9g}, relative L2 {f['selected']['relative_l2']:.9g}.",
       '', '## By paradigm: selected edit','','| Paradigm | N | Before correct | After correct | W→R | R→W |','|---|---:|---:|---:|---:|---:|']
    for k,r in target['by_paradigm'].items():lines.append(f"| {k} | {r['n']} | {r['before_correct']} | {r['after_correct']} | {r['wrong_to_right']} | {r['right_to_wrong']} |")
    lines+=['', '## Method and limits','',
      'Primary metric is the standard minimal-pair full-sentence method: summed causal log likelihood of the grammatical sentence must exceed the ungrammatical sentence. BOS is conditioned on; EOS is not scored; no length normalization. Ties count wrong. This scoring uses teacher-forced likelihood and is distinct from free-form task solving. Illustrative greedy generations cannot establish improved language quality.',
      'Eight BLiMP paradigms were selected for linguistic coverage before running this model. This is not full 67-paradigm BLiMP, and its results must not be equated to the model-card 52.43% figure. Exact duplicate/shared sentences were removed before splitting. Templates and vocabulary remain shared. Original training samples are unavailable, so absence of benchmark examples from pretraining is not certified.',
      'Fixed diagnostic and validation data were used for hypotheses and checkpoint selection, with a maximum of 12 validation evaluations. All attempted edits are retained. The final test baseline was evaluated in a separate process and not opened until the selected weights, edit specification, protocol and evaluator were frozen. After opening, no edits were selected from these test errors.',
      'Zero save/reload, rollback, strict architecture/shapes/dtypes, buffers and tied weights, finite values and unchanged values outside declared masks were checked. Saved candidates were evaluated in fresh processes, without activation-modifying hooks. Three fixed-seed random controls match the actual nonzero support and L2 norm within float32 tolerance. They provide descriptive controls only.',
      'Paired bootstrap is stratified by paradigm, with fixed observed stratum weights, 20,000 replicates and seed 94101. The interval describes test-example sampling for this selected edit, not independent pretraining runs, fresh models, novel grammar types, or adaptive model-search uncertainty. Per-paradigm changes are descriptive secondary results.',
      'CPU float32, one thread, official Transformers eager attention, batch size 64. One warmed inference measurement per condition; loading/scoring overhead excluded. Candidate construction time measures code application/serialization, not agent reasoning time.',
      'This study changes existing values only. No gradients, backward, optimizer steps, SFT, LoRA, regression, distillation, evolutionary search or retraining were used. The checkpoint remains the original pretrained architecture.',
      '', '## Attribution','',
      'Model: https://huggingface.co/Glint-Research/Glimmer-1-Base (MIT declared in card). Pinned revision and SHA-256 in sources.json.',
      'BLiMP: Warstadt et al. (2020), The Benchmark of Linguistic Minimal Pairs for English, TACL 8:377–392, https://doi.org/10.1162/tacl_a_00321 ; data https://github.com/alexwarstadt/blimp (CC-BY 4.0). The present subset and editing splits are modifications of the original evaluation setup.']
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
