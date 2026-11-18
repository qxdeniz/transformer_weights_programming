"""Fetch original upstream files at the recorded commit and verify every SHA-256."""
import urllib.request
from common import ROOT,read_json,sha256

def main():
    src=read_json(ROOT/'sources.json')
    prefix=f"https://raw.githubusercontent.com/anadim/smallest-addition-transformer-codex/{src['commit_sha']}/"
    for name,digest in src['files_sha256'].items():
        path=ROOT/'upstream'/name
        if path.exists():
            assert sha256(path)==digest,f'Existing file differs: {name}'
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        req=urllib.request.Request(prefix+name,headers={'User-Agent':'CarryBench-10-reproduction'})
        with urllib.request.urlopen(req,timeout=60) as response:raw=response.read()
        import hashlib
        assert hashlib.sha256(raw).hexdigest()==digest,f'Unexpected download: {name}'
        assert not raw.startswith(b'version https://git-lfs.github.com/spec/v1')
        path.write_bytes(raw)
        print('Verified',name)
    print('Original source and checkpoint verified. License status remains unspecified; no redistribution permission is asserted.')

if __name__=='__main__':main()
