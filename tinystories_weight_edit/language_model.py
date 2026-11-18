"""Official GPT-Neo inference with verified legacy-buffer compatibility handling."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'.hf_cache'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from transformers import GPTNeoConfig,GPTNeoForCausalLM,AutoTokenizer

PARAMETERS=3745984

def configure():
    torch.manual_seed(101101);torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True);torch.set_grad_enabled(False)

def load(checkpoint=None):
    configure();cfg=json.loads((ROOT/'upstream/model/config.json').read_text())
    config=GPTNeoConfig(**cfg);config._attn_implementation='eager'
    model=GPTNeoForCausalLM(config).cpu().float()
    path=checkpoint or ROOT/'upstream/model/pytorch_model.bin'
    state=torch.load(path,map_location='cpu',weights_only=True)
    # Transformers 4.28 serialized deterministic attention masks that current
    # Transformers registers as nonpersistent. Check values, then omit only these.
    for i,block in enumerate(model.transformer.h):
        name=f'transformer.h.{i}.attn.attention.bias'
        if name in state:
            assert torch.equal(state[name],block.attn.attention.bias),name
            del state[name]
        name=f'transformer.h.{i}.attn.attention.masked_bias'
        if name in state:
            assert state[name].numel()==1 and float(state[name])==-1e9,name
            del state[name]
    if 'lm_head.weight' in state:
        assert torch.equal(state['lm_head.weight'],state['transformer.wte.weight'])
    assert model.config.tie_word_embeddings
    state['lm_head.weight']=state['transformer.wte.weight']
    model.load_state_dict(state,strict=True);model.tie_weights();model.eval()
    assert model.lm_head.weight is model.transformer.wte.weight
    assert sum(p.numel() for p in model.parameters())==PARAMETERS
    assert all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in model.parameters())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'upstream/model',local_files_only=True)
    return model,tokenizer

def encode(tokenizer,text):
    ids=tokenizer.encode(text,add_special_tokens=False)
    if not ids or ids[0]!=tokenizer.bos_token_id:ids=[tokenizer.bos_token_id]+ids
    assert 1<len(ids)<=2048
    return ids

@torch.inference_mode()
def log_likelihoods(model,tokenizer,sentences,batch_size=16):
    """Full sentence raw likelihood, excluding BOS and EOS; deterministic right padding."""
    encoded=[encode(tokenizer,s) for s in sentences];scores=[]
    for start in range(0,len(encoded),batch_size):
        chunk=encoded[start:start+batch_size];length=max(map(len,chunk))
        ids=torch.full((len(chunk),length),tokenizer.eos_token_id,dtype=torch.long);mask=torch.zeros_like(ids)
        for i,tokens in enumerate(chunk):ids[i,:len(tokens)]=torch.tensor(tokens);mask[i,:len(tokens)]=1
        logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
        lp=logits[:,:-1].float().log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)
        scores.extend((lp.double()*mask[:,1:]).sum(-1).tolist())
    return encoded,scores

@torch.inference_mode()
def greedy(model,tokenizer,prompt,max_new_tokens=48):
    ids=encode(tokenizer,prompt);x=torch.tensor([ids]);tail=[]
    for _ in range(max_new_tokens):
        nxt=int(model(x,use_cache=False,logits_to_keep=1).logits[0,-1].argmax());tail.append(nxt)
        x=torch.cat([x,torch.tensor([[nxt]])],1)
        if nxt==tokenizer.eos_token_id:break
    return {'prompt':prompt,'input_tokens':ids,'generated_tokens':tail,'continuation':tokenizer.decode(tail,skip_special_tokens=True),
            'termination':'EOS' if tail and tail[-1]==tokenizer.eos_token_id else 'token_limit'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('prompt');p.add_argument('--checkpoint',type=Path)
    a=p.parse_args();m,t=load(a.checkpoint);print(json.dumps(greedy(m,t,a.prompt),ensure_ascii=False))
