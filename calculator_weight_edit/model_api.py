"""Unmodified original training architecture/tokenizer; batch-equivalent greedy inference."""
import sys
from pathlib import Path
from collections import defaultdict
import torch
import json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream/github/src'))
from model import CalculatorLLM
from tokenizer import Tokenizer
from generate import generate as reference_generate

def configure():
    torch.manual_seed(82101);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32);torch.set_grad_enabled(False)

def load(checkpoint=None):
    configure();cfg=json.loads((ROOT/'upstream/config.json').read_text());vocab=json.loads((ROOT/'upstream/vocab.json').read_text())
    m=CalculatorLLM(**cfg);state=torch.load(checkpoint or ROOT/'upstream/model.pt',map_location='cpu',weights_only=True)
    m.load_state_dict(state,strict=True);m.eval();m.cpu()
    assert sum(p.numel() for p in m.parameters())==104740
    assert all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in m.parameters())
    return m,Tokenizer(vocab),vocab

@torch.inference_mode()
def generate(model,tokenizer,prompts,batch_size=128,max_new_tokens=8):
    """Group equal-length prefixes, remove finished rows; no padding, labels or answer repair."""
    encoded=[tokenizer.encode(p,add_special_tokens=True)[:-1] for p in prompts]
    tails=[[] for _ in prompts];groups=defaultdict(list)
    for i,ids in enumerate(encoded):groups[len(ids)].append(i)
    eos=tokenizer.vocab['[END]']
    for length,indices in sorted(groups.items()):
        assert length+max_new_tokens<=model.max_seq_len
        for start in range(0,len(indices),batch_size):
            active=indices[start:start+batch_size]
            x=torch.tensor([encoded[i] for i in active],dtype=torch.long)
            for _ in range(max_new_tokens):
                next_ids=model(x)[:,-1].argmax(-1)
                for i,t in zip(active,next_ids.tolist()):tails[i].append(t)
                keep=next_ids!=eos
                if not bool(keep.any()):break
                x=torch.cat([x,next_ids[:,None]],1)[keep]
                active=[i for i,k in zip(active,keep.tolist()) if k]
    return encoded,tails

def output_text(tail,vocab):
    inverse={v:k for k,v in vocab.items()}
    return ' '.join(inverse[t] for t in tail if t!=vocab['[END]'])

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('prompt');p.add_argument('--checkpoint',type=Path)
    args=p.parse_args();m,t,v=load(args.checkpoint);ids,tails=generate(m,t,[args.prompt])
    print(json.dumps({'prompt':args.prompt,'input_tokens':ids[0],'generated_tokens':tails[0],'output':output_text(tails[0],v)}))
