"""Pure declarative changes to existing parameter values; fixed buffers cannot be edited."""
import torch

def apply_edit(base,parameter_names,spec):
    state={k:v.clone() for k,v in base.items()};masks={k:torch.zeros_like(base[k],dtype=torch.bool) for k in parameter_names}
    for op in spec['operations']:
        name=op['tensor'];assert name in masks,'Editing a buffer or unknown name is forbidden'
        t=state[name];mask=masks[name]
        if op['kind']=='scale':
            index=tuple(slice(*v) if isinstance(v,list) else v for v in op.get('slices',[[None,None]]*t.ndim))
            t[index]*=op['factor'];mask[index]=True
        elif op['kind']=='add':
            idx=tuple(op['index']);t[idx]+=op['value'];mask[idx]=True
        else:raise ValueError(op['kind'])
    changed=0;total=0;delta2=0.;norm2=0.;details={}
    for k,v in state.items():
        assert v.shape==base[k].shape and v.dtype==base[k].dtype and torch.isfinite(v).all()
        if k not in masks:assert torch.equal(v,base[k]);continue
        assert torch.equal(v[~masks[k]],base[k][~masks[k]])
        d=v.double()-base[k].double();n=int(torch.count_nonzero(d));changed+=n;total+=v.numel()
        delta2+=float(d.square().sum());norm2+=float(base[k].double().square().sum())
        if n:details[k]={'changed_scalars':n,'delta_l2':float(d.norm())}
    return state,{'changed_scalars':changed,'total_scalars':total,'fraction':changed/total,'delta_l2':delta2**.5,'relative_l2':(delta2/norm2)**.5,'tensors':details},masks
