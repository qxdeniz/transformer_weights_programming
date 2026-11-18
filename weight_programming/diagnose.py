"""Read-only analysis of open diagnostic examples; never reads validation or final outcomes."""
import math
import torch
from common import ROOT,read_jsonl,write_json
from inference import load
from src.data import ITOS

def main():
    model,_=load(ROOT/'checkpoints/base.pt')
    rows=read_jsonl(ROOT/'results/baseline/diagnostic.jsonl')
    weights={n:p.detach().tolist() for n,p in model.named_parameters()}
    spectra={n:{'norm':float(p.norm()),'min':float(p.min()),'max':float(p.max()),
                **({'singular_values':torch.linalg.svdvals(p).tolist()} if p.ndim==2 else {})}
             for n,p in model.named_parameters()}
    write_json(ROOT/'results/diagnostics/weights.json',weights)
    write_json(ROOT/'results/diagnostics/spectra.json',spectra)
    errors=[r for r in rows if not r['strict_exact']][:48]
    controls=[r for r in rows if r['strict_exact']][:24]
    traces=[]
    for row in errors+controls:
        mismatch=[i for i,(a,b) in enumerate(zip(row['generated_tokens'],row['target_tokens'])) if a!=b]
        steps=mismatch[:1] if mismatch else [min(5,row['max_carry_chain'])]
        for step in steps:
            prefix=row['input_tokens']+row['generated_tokens'][:step]
            captured={}
            def capture(module,args):
                captured['attn_input']=args[0].detach().clone()
            h=model.blocks[0].attn.register_forward_pre_hook(capture)
            try:
                logits,_=model(torch.tensor([prefix]));last=logits[0,-1]
            finally:h.remove()
            attention=model.blocks[0].attn
            q,k,v=attention.qkv(captured['attn_input']).chunk(3,dim=-1)
            length=len(prefix)
            q=q.view(1,length,2,4).transpose(1,2)
            k=k.view(1,length,2,4).transpose(1,2)
            scores=(q@k.transpose(-1,-2))/2
            mask=attention.mask[:length,:length]
            probs=scores.masked_fill(~mask,float('-inf')).softmax(-1)[0,:,-1]
            heads=[]
            for head in range(2):
                val,idx=probs[head].topk(min(5,length))
                heads.append([{'position':int(i),'token':ITOS[prefix[int(i)]],'probability':float(v)} for v,i in zip(val,idx)])
            top,indices=last.topk(5)
            traces.append({'id':row['id'],'a':row['a'],'b':row['b'],'source':row['source'],'max_chain':row['max_carry_chain'],
                 'correct_example':row['strict_exact'],'step':step,'prefix_tokens':prefix,
                 'expected_token':row['target_tokens'][step],'predicted_token':int(last.argmax()),
                 'margin_predicted_minus_expected':float(last.max()-last[row['target_tokens'][step]]),
                 'top_logits':[{'token':ITOS[int(i)],'logit':float(v)} for v,i in zip(top,indices)],
                 'head_attention_top5':heads,'head_attention':probs.tolist()})
    write_json(ROOT/'results/diagnostics/traces.json',traces)
    print(f'Read-only traces: {len(traces)} diagnostic examples. No persistent hooks.')

if __name__=='__main__':
    with torch.inference_mode():main()
