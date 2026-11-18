"""CarryBench-10 v1: deterministic, integer-exact arithmetic benchmark."""
import random
from collections import Counter
from common import ROOT, write_json, write_jsonl, sha256

LIMIT = 10**10
MANUAL = [(0, 0), (7, 8), (47, 38), (99, 1), (LIMIT-1, 1), (LIMIT-1, LIMIT-1)]
DEV_MIX = {"uniform":256, "single_carry":128, "chain_short":128,
           "chain_medium":128, "chain_long":128, "zero":64,
           "leading_zeros":64, "power10":64, "overflow11":64}
STRESS_MIX = {"single_carry":300, "chain_short":300, "chain_medium":400,
              "chain_long":400, "zero":150, "leading_zeros":150,
              "power10":150, "overflow11":150}
SEEDS = {"diagnostic":41001, "validation":41002, "test_uniform":41003,
         "test_stress":41004, "test_exhaustive_00_99":41005}

def canonical(a, b):
    return (min(a,b), max(a,b))

def features(a, b):
    ad = [(a//10**i)%10 for i in range(10)]
    bd = [(b//10**i)%10 for i in range(10)]
    carry = 0
    carry_out = []
    digits = []
    for x, y in zip(ad,bd):
        t = x+y+carry
        digits.append(t%10)
        carry = t//10
        carry_out.append(carry)
    digits.append(carry)
    assert sum(d*10**i for i,d in enumerate(digits)) == a+b
    longest = run = 0
    for c in carry_out:
        run = run+1 if c else 0
        longest = max(longest,run)
    labels = []
    if sum(carry_out)==0: labels.append("no_carry")
    if sum(carry_out)==1: labels.append("single_carry")
    if longest>=2: labels.append("carry_chain")
    if a==0 or b==0: labels.append("zero_operand")
    if a<10**9 or b<10**9: labels.append("leading_zeros_10_digit_input")
    if a in {10**k for k in range(10)} or b in {10**k for k in range(10)}: labels.append("power10")
    if carry_out[-1]: labels.append("overflow11")
    return {"labels":labels, "carry_out_lsd_first":carry_out,
            "carry_count":sum(carry_out), "max_carry_chain":longest}

def make_row(split, index, a, b, source):
    return {"id":f"{split}:{index:05d}", "a":a, "b":b, "expected":a+b,
            "source":source, **features(a,b)}

def sample(rng, category):
    if category=="uniform": return rng.randrange(LIMIT),rng.randrange(LIMIT)
    if category=="zero": return (0,rng.randrange(100,LIMIT))
    if category=="leading_zeros": return rng.randrange(10**9),rng.randrange(10**9)
    if category=="power10": return 10**rng.randrange(10),rng.randrange(100,LIMIT)
    if category=="overflow11": return rng.randrange(LIMIT//2,LIMIT),rng.randrange(LIMIT//2,LIMIT)
    # Start with columns that cannot carry, then insert a proven carry run.
    ad=[];bd=[]
    for _ in range(10):
        x=rng.randrange(9); y=rng.randrange(9-x)
        ad.append(x);bd.append(y)
    lengths={"single_carry":(1,1),"chain_short":(2,3),"chain_medium":(4,6),"chain_long":(7,10)}
    lo,hi=lengths[category]; length=rng.randint(lo,hi); start=rng.randrange(11-length)
    x=rng.randint(1,9);y=rng.randint(10-x,9)
    ad[start]=x;bd[start]=y
    for i in range(start+1,start+length):
        x=rng.randrange(10);ad[i]=x;bd[i]=9-x
    a=sum(x*10**i for i,x in enumerate(ad));b=sum(x*10**i for i,x in enumerate(bd))
    f=features(a,b)
    assert f['max_carry_chain']==length and f['carry_count']==length
    return a,b

def build_datasets():
    if (ROOT/'protocol.json').exists():
        raise RuntimeError("Protocol already exists; refusing to regenerate committed data")
    seen={canonical(a,b) for a,b in MANUAL}
    manifests={}
    for split,counts in [
        ('diagnostic',DEV_MIX), ('validation',{k:v*2 for k,v in DEV_MIX.items()}),
        ('test_uniform',{'uniform':5000}), ('test_stress',STRESS_MIX)]:
        rng=random.Random(SEEDS[split]);rows=[]
        for category,count in counts.items():
            for _ in range(count):
                for attempt in range(100000):
                    a,b=sample(rng,category)
                    if rng.randrange(2):a,b=b,a
                    key=canonical(a,b)
                    if (a<100 and b<100) or key in seen:continue
                    seen.add(key);break
                else:raise RuntimeError('Sampling exhausted')
                rows.append(make_row(split,len(rows),a,b,category))
        rng.shuffle(rows)
        # ID depends on final deterministic order, not class ordering.
        for i,row in enumerate(rows):row['id']=f'{split}:{i:05d}'
        path=ROOT/'data'/f'{split}.jsonl';write_jsonl(path,rows)
        manifests[split]={'size':len(rows),'seed':SEEDS[split],'sha256':sha256(path),
                          'distribution':counts}
    split='test_exhaustive_00_99'
    rows=[make_row(split,100*a+b,a,b,'exhaustive_ordered') for a in range(100) for b in range(100)]
    path=ROOT/'data'/f'{split}.jsonl';write_jsonl(path,rows)
    manifests[split]={'size':len(rows),'sha256':sha256(path),'distribution':'All ordered pairs in [0,99]^2; includes known manual pairs'}
    write_jsonl(ROOT/'data/manual.jsonl',[make_row('manual',i,a,b,'known_manual') for i,(a,b) in enumerate(MANUAL)])
    write_json(ROOT/'data/manifest.json',manifests)
    protocol={
        'benchmark':'CarryBench-10', 'version':1,'operand_range':[0,LIMIT-1],
        'created_before_edits':True,'datasets':manifests,'manual_pairs':MANUAL,
        'deduplication':'unordered pairs across splits and within random splits; reserve a<100 and b<100 for exhaustive',
        'mixture_note':'Disjoint source strata; measured labels may overlap. Digit features use exact schoolbook carry arithmetic.',
        'chain_generation':'one sum>=10 starts run, length-1 sums=9 propagate it; other column sums<=8; verified exactly',
        'primary_metric':'Exact 12-token target: 11 reversed zero-padded decimal digits followed by EOS',
        'secondary_metrics':['upstream_digits_only_exact_match','numeric_accuracy','format_validity','per_digit_accuracy'],
        'generation':{'device':'cpu','dtype':'float32','threads':1,'seed':1776,'batch_size':256,'max_new_tokens':12,'algorithm':'unmodified upstream greedy generate'},
        'validation_budget':12,
        'selection':['max validation strict exact match','min right_to_wrong','min changed_independent_scalars','min delta_L2'],
        'controls':{'zero_edit':True,'rollback':True,'random_seeds':[71001,71002,71003],'random_mask_and_norm':'same as selected meaningful edit'},
        'uncertainty':{'method':'paired multinomial bootstrap of per-example differences','replicates':20000,'seed':61001,'confidence':0.95},
        'success':'test_uniform after-minus-before strict EM CI lower bound > 0; stress/exhaustive reported separately',
        'final_access':'Baseline final evaluations in separate subprocess, sealed_final; no development access before freeze.json',
        'training_overlap':'No claim that inputs were absent from original training; author training stream is unavailable',
        'code_sha256':{f:sha256(ROOT/f) for f in ['benchmark_data.py','inference.py','apply_edit.py','run_experiment.py']}}
    write_json(ROOT/'protocol.json',protocol)
    return protocol
