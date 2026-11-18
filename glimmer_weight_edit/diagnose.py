"""Read-only diagnostic analysis; never imports final test outcomes."""
from collections import defaultdict
import torch
from utils import ROOT,read_rows,write_json,read_json
from language_model import load,encode

def main():
    m,t=load();rows=read_rows(ROOT/'results/baseline/diagnostic.jsonl')
    weight=m.model.embed_tokens.weight.detach();norms=weight.norm(dim=1)
    largest=norms.topk(20)
    top=[{'id':int(i),'text':t.decode([int(i)]),'norm':float(n)} for n,i in zip(largest.values,largest.indices)]
    groups=defaultdict(list)
    for row in rows:
        difference=len(row['good_tokens'])-len(row['bad_tokens'])
        groups['good_shorter' if difference<0 else 'good_longer' if difference>0 else 'same_token_length'].append(row)
    lengths={k:{'n':len(v),'correct':sum(r['correct'] for r in v),'accuracy':sum(r['correct'] for r in v)/len(v)} for k,v in groups.items()}
    contributions=defaultdict(list);handles=[]
    def hook(name):
        def read(module,args,output):
            out=output[0] if isinstance(output,tuple) else output
            contributions[name].append(float(out.norm(dim=-1).mean()))
        return read
    for i,layer in enumerate(m.model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_hook(hook(f'layer{i}_attention_output')))
        handles.append(layer.mlp.down_proj.register_forward_hook(hook(f'layer{i}_mlp_output')))
    try:
        for row in rows[:64]:m(torch.tensor([row['good_tokens']]),use_cache=False)
    finally:
        for h in handles:h.remove()
    generations=read_json(ROOT/'results/baseline/generations.json')
    repeated=[]
    for g in generations:
        tail=g['generated_tokens']
        repeated.append({'prompt':g['prompt'],'tail':tail,'most_common_token':max(set(tail),key=tail.count),'max_token_count':max(tail.count(x) for x in tail)})
    result={'embedding_top_norms':top,'norm_final_weights':m.model.norm.weight.tolist(),
       'mean_output_module_l2':{k:sum(v)/len(v) for k,v in contributions.items()},'diagnostic_length_groups':lengths,
       'diagnostic_errors':[r for r in rows if not r['correct']][:24],'greedy_repetition':repeated,
       'note':'Norm comparisons and repeated tokens are observations; causal claims require independent validation. No diagnostic hooks remain.'}
    write_json(ROOT/'results/diagnostics.json',result)
    print({k:result[k] for k in ['embedding_top_norms','mean_output_module_l2','diagnostic_length_groups','greedy_repetition']})

if __name__=='__main__':
    with torch.inference_mode():main()
