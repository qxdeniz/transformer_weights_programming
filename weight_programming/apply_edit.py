"""A pure, input-independent editor. No examples, labels, gradients or optimizers."""
import torch

ALIASES={'token_emb.weight':['token_emb.weight','lm_head.weight']}

def canonical_name(name):
    return 'token_emb.weight' if name=='lm_head.weight' else name

def apply_edit(base_state_dict,edit_spec):
    # Recreate weight aliases explicitly; state_dict contains both tied names.
    out={k:v.clone() for k,v in base_state_dict.items()}
    out['lm_head.weight']=out['token_emb.weight']
    masks={k:torch.zeros_like(v,dtype=torch.bool) for k,v in out.items() if k!='lm_head.weight'}
    for op in edit_spec['operations']:
        name=canonical_name(op['tensor']);t=out[name];mask=masks[name]
        kind=op['kind']
        if kind=='scale':
            index=tuple(slice(*v) if isinstance(v,list) else v for v in op.get('slices',[[None,None]]*t.ndim))
            t[index]*=float(op['factor']);mask[index]=True
        elif kind=='add':
            index=tuple(op['index']);t[index]+=float(op['value']);mask[index]=True
        elif kind=='rank_one':
            u=torch.tensor(op['u'],dtype=t.dtype);v=torch.tensor(op['v'],dtype=t.dtype)
            assert t.shape==(len(u),len(v))
            delta=float(op['strength'])*torch.outer(u,v);t.add_(delta);mask|=(delta!=0)
        elif kind=='set':
            index=tuple(op['index']);t[index]=float(op['value']);mask[index]=True
        else:raise ValueError(f'Unknown edit kind {kind}')
    changed=0;delta2=0.;base2=0.;total=0;details={}
    for k,v in out.items():
        if k=='lm_head.weight':continue
        base=base_state_dict[k]
        assert v.shape==base.shape and v.dtype==base.dtype
        assert torch.isfinite(v).all()
        assert torch.equal(v[~masks[k]],base[~masks[k]])
        diff=v.double()-base.double();n=int(torch.count_nonzero(diff));changed+=n
        delta2+=float(diff.square().sum());base2+=float(base.double().square().sum());total+=v.numel()
        if n:details[k]={'changed':n,'delta_l2':float(diff.norm()),'max_abs_delta':float(diff.abs().max())}
    assert out.keys()==base_state_dict.keys()
    return out,{'changed_scalars':changed,'total_independent_scalars':total,'fraction':changed/total,
                'delta_l2':delta2**.5,'relative_l2':(delta2/base2)**.5,'tensors':details},masks
