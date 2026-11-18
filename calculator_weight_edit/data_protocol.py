"""CalcWords-99: fixed source-stratified splits, with commutative duplicates grouped."""
import random,json,hashlib
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ONES=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
TENS=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']

def words(n):
    assert 0<=n<=99
    return ONES[n] if n<20 else TENS[n//10]+(' '+ONES[n%10] if n%10 else '')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def read_json(p):return json.loads(Path(p).read_text())
def write_rows(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def read_rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]

def key(row):
    a,b=row['a'],row['b']
    return (row['operation'],min(a,b),max(a,b)) if row['operation']!='minus' else (row['operation'],a,b)

def all_rows():
    rows=[]
    for op in ['plus','minus','times']:
        for a in range(100):
            for b in range(100):
                result=a+b if op=='plus' else a-b if op=='minus' else a*b
                if not 0<=result<=99:continue
                rows.append({'id':f'{op}:{a:02d}:{b:02d}','operation':op,'a':a,'b':b,'expected':result,
                             'prompt':f'{words(a)} {op} {words(b)} equals','expected_text':words(result),
                             'zero_operand':a==0 or b==0,'identity':(b==0 if op!='times' else b==1),
                             'two_word_answer':len(words(result).split())==2})
    return rows

def prepare():
    assert not (ROOT/'protocol.json').exists(),'Refuse to regenerate fixed protocol'
    domain=all_rows();groups=defaultdict(list)
    for row in domain:groups[key(row)].append(row)
    # Canonical orientation is sampled without reading model outputs. A random split
    # over groups prevents a+b and b+a from crossing development/final boundaries.
    splits={n:[] for n in ['diagnostic','validation','test_uniform']}
    seeds={'diagnostic':83101,'validation':83102,'test_uniform':83103}
    fixed_counts={'diagnostic':{'plus':200,'minus':200,'times':112},
                  'validation':{'plus':400,'minus':400,'times':224},
                  'test_uniform':{'plus':1000,'minus':1000,'times':160}}
    # The task has only 672 valid multiplication orders, so its canonical domain
    # is much smaller. Allocate all stages from distinct canonical groups below.
    # Explicit feasible counts are checked before writing any protocol.
    by_op=defaultdict(list)
    for k,v in groups.items():by_op[k[0]].append(k)
    # Choose a feasible 10%/20%/remaining allocation for multiplication only.
    mult_n=len(by_op['times'])
    fixed_counts['diagnostic']['times']=mult_n//10
    fixed_counts['validation']['times']=mult_n//5
    fixed_counts['test_uniform']['times']=mult_n-fixed_counts['diagnostic']['times']-fixed_counts['validation']['times']
    remaining={op:sorted(keys) for op,keys in by_op.items()}
    for split in ['diagnostic','validation','test_uniform']:
        rng=random.Random(seeds[split])
        for op,count in fixed_counts[split].items():
            available=remaining[op];chosen=rng.sample(available,count);chosen_set=set(chosen)
            remaining[op]=[k for k in available if k not in chosen_set]
            for k in chosen:splits[split].append(dict(rng.choice(groups[k])))
        rng.shuffle(splits[split])
    manifest={}
    for name,rows in {**splits,'test_exhaustive':domain}.items():
        path=ROOT/'data'/f'{name}.jsonl';write_rows(path,rows)
        manifest[name]={'n':len(rows),'sha256':sha(path),'counts_by_operation':dict(Counter(r['operation'] for r in rows)),'seed':seeds.get(name)}
    protocol={'benchmark':'CalcWords-99 v1','model_parameters':104740,
       'primary_test':'test_uniform','primary_metric':'strict exact answer token sequence including END',
       'datasets':manifest,'deduplication':'operation plus unordered operands for plus/times; ordered operands for minus; one random orientation per canonical group in random splits',
       'macro_metric':'unweighted mean of per-operation strict EM; primary alongside stratified micro EM (sample weights fixed in advance)',
       'source_training_overlap':'Original GitHub generator repeats ALL valid tasks in training; no absence-from-model-training claim. Only editing splits are held out.',
       'technical_preflight':'Use original GitHub training tokenizer. Space demo tokenizer corrupts x inside six/sixty. This is a pre-baseline compatibility fix, never a weight improvement.',
       'sampling_note':'Stratified sample, not uniform over operations or ordered pairs. One orientation per commutative group; exhaustive includes all ordered pairs and development examples.',
       'inference':{'device':'cpu','dtype':'float32','threads':1,'seed':82101,'batch_size':128,'max_new_tokens':8,'algorithm':'original greedy argmax with END stop, equal-length batching verified against original single-example generate'},
       'validation_budget':12,'selection':['max validation strict_correct','min right_to_wrong','min changed_scalars','min delta_l2'],
       'random_seeds':[84101,84102,84103],
       'bootstrap':{'method':'stratified paired bootstrap within operation; fixed test stratum weights','replicates':20000,'seed':85101,'confidence':.95},
       'success':'Positive primary-test micro difference with lower 95% stratified paired CI > 0; all operation changes and exhaustive errors reported',
       'final_access':'Separate-process baseline sealed until selected weights, spec and evaluator frozen',
       'code_sha256':{name:sha(ROOT/name) for name in ['data_protocol.py','model_api.py','edit_weights.py','experiment.py']}}
    write_json(ROOT/'protocol.json',protocol);return protocol

if __name__=='__main__':prepare()
