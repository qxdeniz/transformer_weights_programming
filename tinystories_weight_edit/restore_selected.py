"""Recreate the frozen edit from pinned source, without opening test or tuning again."""
import argparse,hashlib,json
from pathlib import Path
import torch
from utils import ROOT,read_json,write_json,sha
from language_model import load
from edit_weights import apply_edit,EMBED,ALIAS

def tensor_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if name==ALIAS:continue
        h.update(name.encode()+b'\0');h.update(str(value.dtype).encode()+b'\0');h.update(json.dumps(list(value.shape)).encode()+b'\0')
        h.update(value.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'checkpoints/selected_reconstructed.pt');p.add_argument('--write-manifest',action='store_true');a=p.parse_args()
    assert not a.out.exists(),'Choose an unused --out file'
    model,_=load();base={k:v.clone() for k,v in model.state_dict().items()};base[ALIAS]=base[EMBED]
    names={n for n,_ in model.named_parameters()};state,diff,_=apply_edit(base,names,read_json(ROOT/'edit_spec.json'))
    digest=tensor_sha(state);manifest=ROOT/'artifact_manifest.json'
    if manifest.exists():assert digest==read_json(manifest)['selected_tensor_sha256']
    a.out.parent.mkdir(exist_ok=True,parents=True);torch.save(state,a.out);reloaded,_=load(a.out)
    assert tensor_sha(reloaded.state_dict())==digest
    if a.write_manifest:
        frozen,_=load(ROOT/'checkpoints/selected.pt');assert tensor_sha(frozen.state_dict())==digest
        write_json(manifest,{'original_checkpoint_sha256':sha(ROOT/'upstream/model/pytorch_model.bin'),
                 'original_tensor_sha256':tensor_sha(base),'selected_checkpoint_sha256':sha(ROOT/'checkpoints/selected.pt'),
                 'selected_tensor_sha256':digest,'diff':diff,
                 'note':'Canonical independent float32 tensor hash is serialization-independent on the recorded little-endian platform. Serialized checkpoint bytes may vary with filename/PyTorch version.'})
    print('Restored fixed edit:',a.out,'tensor SHA-256:',digest)

if __name__=='__main__':main()
