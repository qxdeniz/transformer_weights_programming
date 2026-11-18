"""Pinned original model and BLiMP data downloads with exact hash verification."""
import urllib.request,hashlib
from utils import ROOT,read_json,sha

def main():
    sources=read_json(ROOT/'sources.json')
    for name,digest in sources['files_sha256'].items():
        path=ROOT/'upstream'/name
        if path.exists():assert sha(path)==digest,name;continue
        if name.startswith('model/'):
            remote=sources.get('remote_paths',{}).get(name,name[6:])
            url=f"https://huggingface.co/roneneldan/TinyStories-1M/resolve/{sources['model_revision']}/"+remote
        else:url=f"https://raw.githubusercontent.com/alexwarstadt/blimp/{sources['blimp_commit']}/data/"+name[6:]
        req=urllib.request.Request(url,headers={'User-Agent':'TinyStories-broad-edit-reproduction'})
        with urllib.request.urlopen(req,timeout=180) as response:raw=response.read()
        assert hashlib.sha256(raw).hexdigest()==digest,name
        assert not raw.startswith(b'version https://git-lfs.github.com/spec/v1'),name
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);print('Verified',name)

if __name__=='__main__':main()
