"""Reconstruct the fixed published edit, without validation selection or retraining."""
import argparse
from pathlib import Path
import hashlib
import json
import torch
from common import ROOT,read_json,write_json,sha256
from inference import load,verify_source_hashes
from apply_edit import apply_edit

def tensor_digest(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        if name=='lm_head.weight':continue
        h.update(name.encode()+b'\0');h.update(str(t.dtype).encode()+b'\0')
        h.update(json.dumps(list(t.shape)).encode()+b'\0')
        h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'checkpoints/selected_reconstructed.pt');p.add_argument('--write-manifest',action='store_true')
    args=p.parse_args();verify_source_hashes()
    original=ROOT/'upstream/checkpoints/best.pt';_,blob=load(original)
    state,diff,_=apply_edit(blob['model_state'],read_json(ROOT/'edit_spec.json'))
    digest=tensor_digest(state)
    manifest=ROOT/'artifact_manifest.json'
    if manifest.exists():assert digest==read_json(manifest)['selected_tensor_sha256']
    if args.out.exists():raise RuntimeError('Output already exists; choose a new --out')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model_config':blob['model_config'],'model_state':state},args.out)
    _,reloaded=load(args.out);assert tensor_digest(reloaded['model_state'])==digest
    if args.write_manifest:
        _,selected=load(ROOT/'checkpoints/selected.pt')
        assert digest==tensor_digest(selected['model_state'])
        write_json(manifest,{'base_tensor_sha256':tensor_digest(blob['model_state']),
                    'selected_tensor_sha256':digest,'original_checkpoint_sha256':sha256(original),
                    'selected_checkpoint_sha256':sha256(ROOT/'checkpoints/selected.pt'),
                    'diff':diff,'note':'Canonical independent tensor hash is serialization-independent on the recorded little-endian float32 platform. Checkpoint file bytes can differ across torch serialization versions.'})
    print('Restored fixed edit:',args.out,'tensor SHA-256:',digest)

if __name__=='__main__':main()
