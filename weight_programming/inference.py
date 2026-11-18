"""Inference receives operand pairs only, never targets. Uses unchanged upstream code."""
import sys
import random
from pathlib import Path
import torch
from common import ROOT, sha256, read_json
sys.path.insert(0,str(ROOT/'upstream'))
from src.model import ModelConfig,TinyDecoderLM
from src.data import preprocess_batch,preprocess,postprocess,TARGET_LEN,SUM_DIGITS,EOS_ID

def configure():
    random.seed(1776);torch.manual_seed(1776)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(False)

def load(path):
    configure()
    blob=torch.load(path,map_location='cpu',weights_only=True)
    model=TinyDecoderLM(ModelConfig(**blob['model_config'])).cpu()
    state=blob['model_state']
    assert all(t.dtype==torch.float32 for t in state.values())
    assert torch.equal(state['lm_head.weight'],state['token_emb.weight'])
    model.load_state_dict(state,strict=True);model.eval()
    assert model.lm_head.weight is model.token_emb.weight
    assert sum(p.numel() for p in model.parameters())==1644
    return model,blob

@torch.inference_mode()
def generate(model,pairs,batch_size=256):
    tails=[];prompts=[]
    for start in range(0,len(pairs),batch_size):
        chunk=pairs[start:start+batch_size]
        a=torch.tensor([p[0] for p in chunk],dtype=torch.int64)
        b=torch.tensor([p[1] for p in chunk],dtype=torch.int64)
        assert bool(((a>=0)&(a<10**10)&(b>=0)&(b<10**10)).all())
        prompt=preprocess_batch(a,b)
        output=model.generate(prompt,max_new_tokens=TARGET_LEN)
        assert output.shape[1]==prompt.shape[1]+TARGET_LEN
        tails.extend(output[:,-TARGET_LEN:].tolist());prompts.extend(prompt.tolist())
    return prompts,tails

def score_tail(tail,expected):
    target=[int(x) for x in f'{expected:011d}'[::-1]]+[EOS_ID]
    valid=(len(tail)==12 and all(0<=x<=9 for x in tail[:11]) and tail[11]==EOS_ID)
    digits_correct=tail[:11]==target[:11]
    numeric=postprocess(tail) # exact upstream decoder, including its padding behaviour
    return {'target_tokens':target,'decoded':numeric,'strict_exact':tail==target,
            'digits_exact':digits_correct,'numeric_correct':numeric==expected,
            'format_valid':valid,'eos_present':EOS_ID in tail,
            'termination':'eos_at_expected_position' if valid else ('premature_eos' if EOS_ID in tail[:11] else 'malformed_or_missing_eos'),
            'generated_token_count':len(tail),
            'digit_correct':[int(x==y) for x,y in zip(tail[:11],target[:11])]}

def verify_source_hashes():
    sources=read_json(ROOT/'sources.json')
    for name,digest in sources['files_sha256'].items():
        assert sha256(ROOT/'upstream'/name)==digest,f'Upstream file changed: {name}'

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/base.pt');p.add_argument('--a',type=int,required=True);p.add_argument('--b',type=int,required=True)
    args=p.parse_args();verify_source_hashes();m,_=load(args.checkpoint)
    prompt,tail=generate(m,[(args.a,args.b)])
    print(json.dumps({'input_tokens':prompt[0],'generated_tokens':tail[0],'prediction':postprocess(tail[0])}))
