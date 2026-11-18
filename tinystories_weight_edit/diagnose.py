"""Read-only open-split analysis; no final-test access or activation changes."""
from collections import defaultdict
import torch
from utils import ROOT,read_rows,read_json,write_json
from language_model import load

def main():
    m,t=load();rows=read_rows(ROOT/'results/baseline/diagnostic.jsonl')
    norms=defaultdict(list);handles=[]
    def post(name):
        def capture(module,args,output):norms[name].append(float(output.detach().norm(dim=-1).mean()))
        return capture
    def pre(name):
        def capture(module,args):norms[name].append(float(args[0].detach().norm(dim=-1).mean()))
        return capture
    for i,block in enumerate(m.transformer.h):
        handles.extend([block.attn.attention.out_proj.register_forward_hook(post(f'{i}:attention')),
                        block.mlp.c_proj.register_forward_hook(post(f'{i}:ffn')),
                        block.ln_1.register_forward_pre_hook(pre(f'{i}:residual_before_attention')),
                        block.ln_2.register_forward_pre_hook(pre(f'{i}:residual_before_ffn'))])
    try:
        for row in rows[:64]:m(torch.tensor([row['good_tokens']]),use_cache=False,logits_to_keep=1)
    finally:
        for handle in handles:handle.remove()
    by_length=defaultdict(list)
    for r in rows:
        diff=len(r['good_tokens'])-len(r['bad_tokens'])
        by_length['same' if diff==0 else 'good_shorter' if diff<0 else 'good_longer'].append(r)
    result={'diagnostic_errors':[r for r in rows if not r['correct']][:32],
         'mean_branch_norms':{k:sum(v)/len(v) for k,v in norms.items()},
         'by_token_length':{k:{'n':len(v),'correct':sum(r['correct'] for r in v)} for k,v in by_length.items()},
         'note':'Activation norms and examples motivate hypotheses; they do not establish causality. All read-only hooks removed.'}
    write_json(ROOT/'results/diagnostics.json',result)
    print('Branch norms:',result['mean_branch_norms']);print('Length groups:',result['by_token_length'])

if __name__=='__main__':
    with torch.inference_mode():main()
