"""Pinned official Llama implementation; no labels, edits, or training in inference."""
import os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'.hf_cache'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from transformers import LlamaConfig,LlamaForCausalLM,AutoTokenizer
from safetensors.torch import load_file

def configure():
    torch.manual_seed(91101);torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)

def load(checkpoint=None):
    configure();cfg=json.loads((ROOT/'upstream/model/config.json').read_text())
    config=LlamaConfig(**cfg);config._attn_implementation='eager'
    model=LlamaForCausalLM(config).cpu().float()
    if checkpoint is None:state=load_file(str(ROOT/'upstream/model/model.safetensors'))
    else:state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert 'model.embed_tokens.weight' in state
    if 'lm_head.weight' in state:assert torch.equal(state['lm_head.weight'],state['model.embed_tokens.weight'])
    state['lm_head.weight']=state['model.embed_tokens.weight']
    model.load_state_dict(state,strict=True);model.tie_weights();model.eval()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert sum(p.numel() for p in model.parameters())==11856
    assert all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in model.parameters())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model',local_files_only=True)
    return model,tokenizer

def encode(tokenizer,text):
    ids=tokenizer.encode(text,add_special_tokens=False)
    if not ids or ids[0]!=1:ids=[1]+ids
    assert 1<len(ids)<=512
    return ids

@torch.inference_mode()
def log_likelihoods(model,tokenizer,sentences,batch_size=64):
    """Raw summed causal sentence likelihood, BOS conditioned; no EOS or length normalization."""
    encoded=[encode(tokenizer,s) for s in sentences]
    scores=[]
    for start in range(0,len(encoded),batch_size):
        chunk=encoded[start:start+batch_size];length=max(map(len,chunk))
        ids=torch.full((len(chunk),length),tokenizer.pad_token_id,dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,tokens in enumerate(chunk):ids[i,:len(tokens)]=torch.tensor(tokens);mask[i,:len(tokens)]=1
        logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
        logp=logits[:,:-1].float().log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)
        sums=(logp.double()*mask[:,1:]).sum(-1)
        scores.extend(sums.tolist())
    return encoded,scores

@torch.inference_mode()
def greedy(model,tokenizer,prompt,max_new_tokens=32):
    ids=encode(tokenizer,prompt);tokens=torch.tensor([ids]);tail=[]
    for _ in range(max_new_tokens):
        nxt=int(model(tokens,use_cache=False).logits[0,-1].argmax());tail.append(nxt)
        tokens=torch.cat([tokens,torch.tensor([[nxt]])],1)
        if nxt==model.config.eos_token_id:break
    return {'prompt':prompt,'input_tokens':ids,'generated_tokens':tail,
            'continuation':tokenizer.decode(tail,skip_special_tokens=True),
            'termination':'EOS' if tail and tail[-1]==model.config.eos_token_id else 'token_limit'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('prompt');p.add_argument('--checkpoint',type=Path)
    args=p.parse_args();m,t=load(args.checkpoint);print(json.dumps(greedy(m,t,args.prompt),ensure_ascii=False))
