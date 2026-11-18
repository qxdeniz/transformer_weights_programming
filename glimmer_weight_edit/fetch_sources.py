"""Download pinned model and attributed BLiMP source files; verify SHA-256."""
import urllib.request
from utils import ROOT,read_json,sha

def main():
    src=read_json(ROOT/'sources.json')
    for name,digest in src['files_sha256'].items():
        path=ROOT/'upstream'/name
        if path.exists():assert sha(path)==digest;continue
        if name.startswith('model/'):
            url=f"https://huggingface.co/Glint-Research/Glimmer-1-Base/resolve/{src['model_revision']}/"+name[6:]
        else:url=f"https://raw.githubusercontent.com/alexwarstadt/blimp/{src['blimp_commit']}/data/"+name[6:]
        req=urllib.request.Request(url,headers={'User-Agent':'Glimmer-BLiMP-editing-reproduction'})
        with urllib.request.urlopen(req,timeout=60) as response:raw=response.read()
        import hashlib
        assert hashlib.sha256(raw).hexdigest()==digest,name
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);print('Verified',name)

if __name__=='__main__':main()
