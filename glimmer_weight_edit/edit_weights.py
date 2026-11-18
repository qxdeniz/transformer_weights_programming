"""Task-independent scalar and scaling edits, preserving tied embeddings."""
import torch
ALIAS='lm_head.weight'
EMBED='model.embed_tokens.weight'

def apply_edit(base,names,spec):
    state={k:v.clone() for k,v in base.items()};state[ALIAS]=state[EMBED]
    names=set(names)-{ALIAS};masks={k:torch.zeros_like(state[k],dtype=torch.bool) for k in names}
    for op in spec['operations']:
        name=EMBED if op['tensor']==ALIAS else op['tensor'];assert name in masks
        if op['kind']=='scale':
            index=tuple(slice(*v) if isinstance(v,list) else v for v in op.get('slices',[[None,None]]*state[name].ndim))
            state[name][index]*=op['factor'];masks[name][index]=True
        elif op['kind']=='add':
            index=tuple(op['index']);state[name][index]+=op['value'];masks[name][index]=True
        else:raise ValueError(op['kind'])
    changed=0;total=0;d2=0.;b2=0.;details={}
    for k,v in state.items():
        assert v.shape==base[k].shape and v.dtype==base[k].dtype and torch.isfinite(v).all()
        if k==ALIAS:continue
        if k not in names:assert torch.equal(v,base[k]);continue
        assert torch.equal(v[~masks[k]],base[k][~masks[k]])
        d=v.double()-base[k].double();n=int(torch.count_nonzero(d));changed+=n;total+=v.numel();d2+=float(d.square().sum());b2+=float(base[k].double().square().sum())
        if n:details[k]={'changed':n,'delta_l2':float(d.norm())}
    return state,{'changed_scalars':changed,'total_scalars':total,'fraction':changed/total,'delta_l2':d2**.5,'relative_l2':(d2/b2)**.5,'tensors':details},masks
