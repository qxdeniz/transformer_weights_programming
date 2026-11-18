"""Package own code, protocols, predictions and paper artifacts; source weights stay local."""
from pathlib import Path
import hashlib,json,zipfile,argparse
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='research_results.zip')
    args=parser.parse_args();archive=ROOT/args.out;included=[]
    for project in ['weight_programming','glimmer_weight_edit','tinystories_weight_edit']:
        for path in (ROOT/project).rglob('*'):
            if not path.is_file():continue
            rel=path.relative_to(ROOT);parts=rel.parts
            if any(p.startswith('.') or p=='__pycache__' for p in parts):continue
            if 'upstream' in parts or 'checkpoints' in parts or 'sealed_final' in parts:continue
            if path.name=='weights.json':continue
            if path.suffix in {'.py','.md','.json','.jsonl','.csv','.tex','.pdf','.svg','.png','.txt'}:included.append(path)
    for path in (ROOT/'paper').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:included.append(path)
    included.extend([ROOT/'RESEARCH_OVERVIEW_RU.md',ROOT/'make_research_bundle.py',ROOT/'calculator_weight_edit/PILOT_STATUS.md'])
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(included))}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(set(included)):z.write(path,str(path.relative_to(ROOT)))
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        z.writestr('BUNDLE_README.txt',
            'Private research deliverable: own code, protocols, all candidate/final predictions and paper drafts.\n'
            'Original/edited checkpoints, upstream code/data copies, virtual environments and unopened pilot results are omitted.\n'
            'For clean reproduction copy project .py, tests, sources.json and requirements.txt into a new directory, then follow README.md.\n'
            'Fetch scripts verify original source SHA-256; fixed edits can be restored with restore_selected.py.\n'
            'All original and edited weights remain available in the local project checkpoints folders.\n'
            'BLiMP-derived data attribution: Warstadt et al. (2020), https://doi.org/10.1162/tacl_a_00321 , CC-BY 4.0.\n'
            'Selected subset, filtering and editing splits modify the original evaluation setup; no endorsement is claimed.\n')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('Archive:',archive,'bytes:',archive.stat().st_size,'files:',len(manifest),'SHA-256:',hashlib.sha256(archive.read_bytes()).hexdigest())

if __name__=='__main__':main()
