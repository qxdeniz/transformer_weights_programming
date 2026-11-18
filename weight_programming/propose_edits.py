"""Explicit finite hypotheses, not an optimizer. All coefficients declared before trials."""
import subprocess
import sys
from common import ROOT,write_json,read_json
from inference import load

HYPOTHESES={
 'H1':{'observation':'Diagnostic 00897: head 1 puts 0.855 on current P52 but 0.129 on previous P00; 00218 and 00366 also show current-column contamination.',
       'mechanism':'Head 1 is usually a near-one-hot current-column reader; sharpen the existing attention scores by scaling its query affine map.',
       'formula':'Wq[4:8,:] *= factor; bq[4:8] *= factor. QK scores for head 1 scale by factor; K,V,projection and all architecture stay fixed.',
       'factors':[1.05,1.15,1.30],
       'prediction':'Reduced contamination where the correct source already ranks first; middle-digit and final-carry errors may decrease.',
       'risk':'Sharpening a wrong first-ranked location can reinforce it and introduce regressions.'},
 'H2':{'observation':'Diagnostic 00015 and 00141 generate all 11 sum digits correctly but digit 5 outranks EOS on final step by 1.55 and 1.05 logits.',
       'mechanism':'A small EOS readout margin can be corrected locally. The EOS embedding is tied, but EOS never appears in a correctly formatted input prefix.',
       'formula':'token_emb.weight[112,:] *= factor; preserve alias with lm_head.weight.',
       'factors':[1.10,1.15,1.20],
       'prediction':'Correct missing termination on already correct sums; digit-only accuracy need not improve.',
       'risk':'EOS may overtake a digit early. Such cases remain errors under strict EM.'},
 'H3':{'observation':'Leading-zero diagnostic errors 00228,00433,00606,00718,00436,00827,00915 favour 1 over 0 by small margins; 00003 has the opposite confusion.',
       'mechanism':'Correct a limited global 0-versus-1 readout bias analytically, while accepting the risk to genuine 1 answers.',
       'formula':'ln_f.bias += margin * (embedding[0]-embedding[1]) / ||embedding[0]-embedding[1]||^2. This increases logit0-logit1 by exactly margin for a fixed hidden state.',
       'margins':[0.15,0.30,0.60],
       'prediction':'Some low-margin false carries into padded zero positions become zero; regressions to true ones may offset gains.',
       'risk':'The same fixed bias acts at every output position and also changes other digit margins.'},
 'combinations':{'rule':'Choose the best singleton in each family by the preregistered validation ranking; try H1+H2, H1+H3, H1+H2+H3. No new coefficient fitting.',
                 'count':3},
 'budget':12,'selection_inputs':'diagnostic observations; validation ranks; never final predictions or metrics',
 'baseline_unchanged':True,
 'no_optimization':'No backward, gradients, optimizer, regression, evolutionary method, or stochastic candidate search. Random controls are generated only after selection.'}

def rank(record):
    return (-record['validation_correct'],record['right_to_wrong'],record['changed_scalars'],record['delta_l2'])

def run(spec,number):
    path=ROOT/'hypotheses'/f'edit_{number:02d}.json';write_json(path,spec)
    result=subprocess.run([sys.executable,str(ROOT/'run_experiment.py'),'--stage','edit','--edit-spec',str(path)],
                          cwd=ROOT,capture_output=True,text=True)
    if result.returncode:
        print(result.stdout);print(result.stderr);raise RuntimeError(f'Candidate {number} failed')
    record=read_json(ROOT/'results/attempts.json')[-1]
    print(f"{record['id']}: validation {record['validation_correct']}/2048; +{record['wrong_to_right']} -{record['right_to_wrong']}; changed={record['changed_scalars']}",flush=True)
    return record

def main():
    assert not (ROOT/'results/attempts.json').exists(),'Refuse to repeat trial selection'
    write_json(ROOT/'hypotheses/preregistered_hypotheses.json',HYPOTHESES)
    _,blob=load(ROOT/'checkpoints/base.pt')
    diff=blob['model_state']['token_emb.weight'][0].double()-blob['model_state']['token_emb.weight'][1].double()
    denom=float(diff.square().sum())
    best={};specs={};number=0
    for family in ['H1','H2','H3']:
        records=[]
        factors=HYPOTHESES[family].get('factors',HYPOTHESES[family].get('margins'))
        for value in factors:
            number+=1
            if family=='H1':
                ops=[{'kind':'scale','tensor':'blocks.0.attn.qkv.weight','slices':[[4,8],[None,None]],'factor':value},
                     {'kind':'scale','tensor':'blocks.0.attn.qkv.bias','slices':[[4,8]],'factor':value}]
            elif family=='H2':
                ops=[{'kind':'scale','tensor':'token_emb.weight','slices':[[112,113],[None,None]],'factor':value}]
            else:
                ops=[{'kind':'add','tensor':'ln_f.bias','index':[i],'value':float(value*x/denom)} for i,x in enumerate(diff)]
            spec={'family':family,'coefficient':value,'hypothesis':HYPOTHESES[family]['mechanism'],
                  'evidence':HYPOTHESES[family]['observation'],'formula':HYPOTHESES[family]['formula'],
                  'prediction':HYPOTHESES[family]['prediction'],'risk':HYPOTHESES[family]['risk'],'operations':ops}
            record=run(spec,number);records.append(record);specs[record['id']]=spec
        best[family]=specs[min(records,key=rank)['id']]
    for families in [('H1','H2'),('H1','H3'),('H1','H2','H3')]:
        number+=1
        spec={'family':'+'.join(families),'hypothesis':'Compose disjoint, explicitly proposed corrections.',
              'prediction':'Combine current-column concentration, termination and readout corrections where applicable.',
              'risk':'Interactions through autoregressive prefixes may create additional errors.',
              'components':[{k:v for k,v in best[f].items() if k!='operations'} for f in families],
              'operations':[op for family in families for op in best[family]['operations']]}
        run(spec,number)
    assert number==12

if __name__=='__main__':main()
