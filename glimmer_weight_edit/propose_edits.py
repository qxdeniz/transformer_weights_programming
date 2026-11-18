"""Three declared finite hypotheses plus three combinations; no optimizer or training."""
import sys,subprocess
from utils import ROOT,read_json,write_json

HYPOTHESES={
 'H1':{'observation':'Diagnostic greedy prompts repeat comma token 15 (30–32 times), token 69 (24 times), and token 165 (32 times). All evidence is open diagnostic output.',
       'hypothesis':'Tied embedding/readout self-copy paths contribute to repeated-token attractors; modestly shrink the three diagnosed rows.',
       'formula':'Scale rows 15,69,165 of model.embed_tokens.weight, preserving lm_head tie.',
       'coefficients':[.95,.85,.70],
       'prediction':'May reduce repetition on known diagnostic continuations and shift sentence likelihoods; it may only replace one repeated token with another.',
       'risk':'Those tokens also encode legitimate text. Tied input/output changes can harm grammar preferences and other tokens may become new attractors.'},
 'H2':{'observation':'Every diagnostic pair with fewer good-sentence tokens is correct (286/286), every pair with more good-sentence tokens is wrong (0/309); same-length pairs are 261/429. Raw sentence scoring is strongly length dominated.',
       'hypothesis':'Increase the gain of existing output evidence relative to the nearly uniform per-token penalty via the final RMSNorm scale.',
       'formula':'model.norm.weight *= coefficient; logits for each fixed prefix scale uniformly. This is probability calibration, not new computation.',
       'coefficients':[1.25,2.0,4.0],
       'prediction':'Sentence likelihood ordering can change even though exact greedy argmax ordering should remain unchanged. Any benefit is calibration of this metric, not evidence of improved free generation.',
       'risk':'Overconfidence in wrong existing predictions can lower likelihood quality and pair accuracy.'},
 'H3':{'observation':'Read-only diagnostic activation norms: mean attention output 0.052 and 0.100 versus MLP 0.034 and 0.078 in layers 0 and 1; tied-embedding greedy output often copies a token repeatedly.',
       'hypothesis':'Strengthen existing last-layer contextual attention relative to the residual token self-copy path.',
       'formula':'model.layers.1.self_attn.o_proj.weight *= coefficient; no other architecture or attention rule changes.',
       'coefficients':[1.2,1.5,2.0],
       'prediction':'Contextual attention can change which token evidence dominates, affecting grammatical likelihood preferences and repetition.',
       'risk':'Activation norms are not causal proof; strengthening an incorrect attention computation can worsen both measures.'},
 'combinations':{'rule':'Choose singleton winner of each family by registered validation rank; try H1+H2, H2+H3 and H1+H2+H3. No new coefficient tuning.', 'count':3},
 'budget':12,'final_test_access':False,
 'method':'Explicit formulas reasoned from diagnostic observations; no gradients, optimizer, regression, evolutionary search, or stochastic candidate search.'}

def rank(r):return (-r['validation_correct'],r['right_to_wrong'],r['changed_scalars'],r['delta_l2'])

def run(spec,number):
    path=ROOT/'hypotheses'/f'edit_{number:02d}.json';write_json(path,spec)
    p=subprocess.run([sys.executable,str(ROOT/'experiment.py'),'edit','--spec',str(path)],cwd=ROOT,capture_output=True,text=True)
    if p.returncode:print(p.stdout,p.stderr);raise RuntimeError(f'Candidate {number} failed')
    record=read_json(ROOT/'results/attempts.json')[-1]
    print(f"{record['id']}: validation {record['validation_correct']}/2048; +{record['wrong_to_right']} -{record['right_to_wrong']}; changed {record['changed_scalars']}",flush=True)
    return record

def main():
    assert not (ROOT/'results/attempts.json').exists(),'Refusing to repeat adaptive selection'
    write_json(ROOT/'hypotheses/preregistered_hypotheses.json',HYPOTHESES)
    number=0;best={};specs={}
    for family in ['H1','H2','H3']:
        records=[]
        for coefficient in HYPOTHESES[family]['coefficients']:
            number+=1
            if family=='H1':
                ops=[{'kind':'scale','tensor':'model.embed_tokens.weight','slices':[[i,i+1],[None,None]],'factor':coefficient} for i in [15,69,165]]
            elif family=='H2':ops=[{'kind':'scale','tensor':'model.norm.weight','factor':coefficient}]
            else:ops=[{'kind':'scale','tensor':'model.layers.1.self_attn.o_proj.weight','factor':coefficient}]
            spec={'family':family,'coefficient':coefficient,'hypothesis':HYPOTHESES[family]['hypothesis'],
                  'evidence':HYPOTHESES[family]['observation'],'formula':HYPOTHESES[family]['formula'],
                  'prediction':HYPOTHESES[family]['prediction'],'risk':HYPOTHESES[family]['risk'],'operations':ops}
            record=run(spec,number);records.append(record);specs[record['id']]=spec
        best[family]=specs[min(records,key=rank)['id']]
    for families in [('H1','H2'),('H2','H3'),('H1','H2','H3')]:
        number+=1
        spec={'family':'+'.join(families),'hypothesis':'Compose disjoint best singleton hypotheses, with fixed coefficients.',
              'prediction':'Combine the declared changes; inspect test only after freeze. Calibration can influence metric without changing computation.',
              'risk':'Autoregressive and normalization interactions can cause additional regressions.',
              'components':[{k:v for k,v in best[family].items() if k!='operations'} for family in families],
              'operations':[op for family in families for op in best[family]['operations']]}
        run(spec,number)
    assert number==12

if __name__=='__main__':main()
