"""Broad explicit block edits: three declared finite hypotheses, twelve trials total."""
import subprocess,sys
from utils import ROOT,write_json,read_json

HYPOTHESES={
 'H1':{'observation':'Diagnostic distractor-agreement accuracy is 57/128; correct main-subject agreement often loses to a nearby plural distractor. Existing attention branches mix context, with layer-output norms 0.13–0.69.',
       'hypothesis':'Change the gain of contextual attention relative to the residual stream throughout the existing blocks.',
       'formula':'For all eight blocks, scale attention.out_proj.weight and out_proj.bias together.',
       'coefficients':[.9,1.1,1.2],
       'prediction':'A stronger contextual branch may improve agreement decisions; a weaker counter-direction tests that interpretation. Observe all regressions.',
       'risk':'All attention computations are rescaled, including useful ones. Stronger incorrect attention may worsen grammar and story generation.'},
 'H2':{'observation':'Read-only diagnostic mean FFN output norm is 0.877 in block 0 and 0.222 in block 7; block-7 residual before FFN is 0.173. Many mistakes favour familiar lexical alternatives or distractor agreement.',
       'hypothesis':'Rebalance FFN updates that can overwrite contextual evidence; reduced gain may preserve useful attention/residual information.',
       'formula':'For all eight blocks, scale mlp.c_proj.weight and c_proj.bias together. Do not change activation, width, forward, embeddings or readout.',
       'coefficients':[.8,.9,1.1],
       'prediction':'Modest reduction can help contextual agreement, while an increase is an explicit counter-direction. Norms alone are not causal proof.',
       'risk':'FFN updates are co-adapted to residual and attention paths. Even modest global rescaling can destroy useful cancellations or lexical knowledge.'},
 'H3':{'observation':'Diagnostic agreement failures can involve competition between source tokens. This tests attention concentration separately from H1 output amplitude; it does not assume the correct token always ranks first.',
       'hypothesis':'Change the concentration of existing attention distributions without replacing their learned directions.',
       'formula':'For all eight blocks, scale attention.q_proj.weight. Query-key scores scale by the same factor; K, V and the causal/local mask remain fixed.',
       'coefficients':[.8,1.2,1.5],
       'prediction':'Sharper or softer selection can alter agreement evidence where context sources compete; validation decides whether the predicted benefit exists.',
       'risk':'Sharpening a wrong maximum reinforces it; softening can dilute a useful precise source.'},
 'combinations':{'rule':'Choose best singleton in each family by the registered validation ranking; evaluate H1+H2, H1+H3, H1+H2+H3 with those exact coefficients. No new coefficient fitting.', 'count':3},
 'budget':12,'minimum_changed_scalars':10000,'test_access':False,
 'degrees_of_freedom':'Many scalar values change, but the search has only explicit group coefficients; this is not evidence of many independently designed computations.',
 'forbidden_methods':'No gradients, optimizer, regression, evolutionary method, random candidate search, training or output-temperature-only edit.'}

def rank(r):return (-r['validation_correct'],r['right_to_wrong'],r['changed_scalars'],r['delta_l2'])

def run(spec,number):
    path=ROOT/'hypotheses'/f'edit_{number:02d}.json';write_json(path,spec)
    result=subprocess.run([sys.executable,str(ROOT/'experiment.py'),'edit','--spec',str(path)],cwd=ROOT,capture_output=True,text=True)
    if result.returncode:print(result.stdout,result.stderr);raise RuntimeError(f'Candidate {number} failed')
    record=read_json(ROOT/'results/attempts.json')[-1]
    print(f"{record['id']}: validation {record['validation_correct']}/2048; +{record['wrong_to_right']} -{record['right_to_wrong']}; changed {record['changed_scalars']}",flush=True)
    return record

def main():
    assert not (ROOT/'results/attempts.json').exists(),'Refuse to rerun selection'
    write_json(ROOT/'hypotheses/preregistered_hypotheses.json',HYPOTHESES)
    number=0;best={};specs={}
    for family in ['H1','H2','H3']:
        records=[]
        for coefficient in HYPOTHESES[family]['coefficients']:
            number+=1;operations=[]
            for layer in range(8):
                if family=='H1':names=[f'transformer.h.{layer}.attn.attention.out_proj.{part}' for part in ['weight','bias']]
                elif family=='H2':names=[f'transformer.h.{layer}.mlp.c_proj.{part}' for part in ['weight','bias']]
                else:names=[f'transformer.h.{layer}.attn.attention.q_proj.weight']
                operations.extend({'kind':'scale','tensor':name,'factor':coefficient} for name in names)
            spec={'family':family,'coefficient':coefficient,'hypothesis':HYPOTHESES[family]['hypothesis'],
                  'evidence':HYPOTHESES[family]['observation'],'formula':HYPOTHESES[family]['formula'],
                  'prediction':HYPOTHESES[family]['prediction'],'risk':HYPOTHESES[family]['risk'],'operations':operations}
            record=run(spec,number);records.append(record);specs[record['id']]=spec
        best[family]=specs[min(records,key=rank)['id']]
    for families in [('H1','H2'),('H1','H3'),('H1','H2','H3')]:
        number+=1
        spec={'family':'+'.join(families),'hypothesis':'Compose the disjoint declared attention/FFN/query changes with best singleton coefficients.',
              'prediction':'Combine the registered mechanisms; final confirmation is independent and includes same-length pairs.',
              'risk':'Co-adapted branch interactions may create more regressions than singleton trials.',
              'components':[{k:v for k,v in best[family].items() if k!='operations'} for family in families],
              'operations':[op for family in families for op in best[family]['operations']]}
        run(spec,number)
    assert number==12

if __name__=='__main__':main()
