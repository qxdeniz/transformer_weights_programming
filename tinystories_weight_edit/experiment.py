"""No-gradient editing study on TinyStories with broad block edits and a fresh BLiMP subset."""
import argparse,datetime,json,math,os,platform,shutil,subprocess,sys,time,csv
from pathlib import Path
from collections import Counter,defaultdict
import torch
from utils import ROOT,read_json,write_json,read_rows,write_rows,sha
from benchmark import prepare
from language_model import load,log_likelihoods,greedy,configure
from edit_weights import apply_edit,EMBED,ALIAS

BASE=ROOT/'upstream/model/pytorch_model.bin'

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
    prompts=['Once upon a time, there was a little girl','A little dog was playing in the park','Tom opened the door and saw','The two friends wanted to help','One rainy day, Lily found']
    write_json(Path(folder)/'generations.json',[greedy(model,tokenizer,p) for p in prompts])

def baseline():
    assert not (ROOT/'results/baseline/complete.json').exists(),'Baseline already completed'
    if not (ROOT/'protocol.json').exists():prepare()
    protocol=verify();m,t=load()
    data=read_rows(ROOT/'data/diagnostic.jsonl')[:50];sentences=[s for r in data for s in [r['sentence_good'],r['sentence_bad']]]
    tokens,scores=log_likelihoods(m,t,sentences)
    assert (tokens,scores)==log_likelihoods(m,t,sentences)
    # Independent official Hugging Face loader; tied missing head is resolved by HF.
    from transformers import GPTNeoForCausalLM
    reference=GPTNeoForCausalLM.from_pretrained(ROOT/'upstream/model',local_files_only=True,dtype=torch.float32,attn_implementation='eager').cpu().eval()
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
              'platform':platform.platform(),'device':'cpu','dtype':'float32','threads':1,'attention':'official GPT-Neo eager, alternating global/local','score_batch':16,
              'dependencies':'Local venv with read-only system torch and compatible read-only cached packages; no global packages modified.'})
    (ROOT/'requirements.txt').write_text(f'torch=={torch.__version__}\ntransformers=={transformers.__version__}\ntokenizers=={tokenizers.__version__}\nsafetensors=={safetensors.__version__}\nnumpy=={numpy.__version__}\nmatplotlib\n')
    write_json(ROOT/'parameter_inventory.json',{'parameters':[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype),'scalars':p.numel()} for n,p in m.named_parameters()],
              'buffers':[{'name':n,'shape':list(b.shape),'dtype':str(b.dtype)} for n,b in m.named_buffers()],
              'independent_parameters':3745984,'alias_groups':[[EMBED,ALIAS]],'config':m.config.to_dict()})
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
    state,diff,masks=apply_edit(base,names,spec);assert diff['changed_scalars']>=protocol['minimum_changed_candidate_scalars']
    checkpoint=ROOT/'checkpoints'/f'{name}.pt';torch.save(state,checkpoint);preparation=time.perf_counter()-started
    sentences=[r['sentence_good'] for r in read_rows(ROOT/'data/diagnostic.jsonl')[:100]]
    original=log_likelihoods(m,t,sentences);m.load_state_dict(state,strict=True);edited=log_likelihoods(m,t,sentences)
    assert all(torch.equal(buffer_copy[n],b) for n,b in m.named_buffers())
    assert m.lm_head.weight is m.transformer.wte.weight
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
    from reporting import make_report
    make_report()


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['baseline','edit','finalize','report','worker']);p.add_argument('--spec',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--split');p.add_argument('--out',type=Path)
    a=p.parse_args();configure()
    if a.stage=='baseline':baseline()
    elif a.stage=='edit':edit(a.spec)
    elif a.stage=='finalize':finalize()
    elif a.stage=='report':report()
    else:measure(a.checkpoint,a.split,a.out)

if __name__=='__main__':main()
