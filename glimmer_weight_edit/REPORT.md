# General language model: Glimmer-1-Base weight editing

Model: 11,856 independent parameters, two Llama layers, hidden width 16, grouped-query attention, SiLU MLP width 24, tied 512-token embedding/output. The author pretrained on FineWeb-Edu; no arithmetic-specialist training is used here.

## Frozen BLiMP subset results

| Condition | N | Before % | After % | Delta pp | Wrong→right | Right→wrong | 95% paired CI pp |
|---|---:|---:|---:|---:|---:|---:|---|
| selected | 4928 | 53.186 | 53.105 | -0.081 | 0 | 4 | [-0.162, -0.020] |
| random_93101 | 4928 | 53.186 | 53.856 | +0.670 | 57 | 24 | [+0.325, +1.015] |
| random_93102 | 4928 | 53.186 | 53.003 | -0.183 | 11 | 20 | [-0.406, +0.041] |
| random_93103 | 4928 | 53.186 | 52.942 | -0.244 | 9 | 21 | [-0.467, -0.041] |

Observed change is negative. The predeclared percentile-bootstrap interval excludes zero, but only 4 pairs are discordant. A post-finalization exact paired sensitivity test gives p=0.125; a stable negative effect is not established. This sensitivity analysis does not change model selection or the recorded primary metric.

Selected candidate_04 after 12 validation evaluations. Changed 16 / 11856 independent scalars, L2 delta 1.09679198, relative L2 0.104372482.

## By paradigm: selected edit

| Paradigm | N | Before correct | After correct | W→R | R→W |
|---|---:|---:|---:|---:|---:|
| anaphor_gender_agreement | 616 | 207 | 207 | 0 | 0 |
| determiner_noun_agreement_1 | 616 | 309 | 309 | 0 | 0 |
| regular_plural_subject_verb_agreement_1 | 616 | 348 | 348 | 0 | 0 |
| irregular_plural_subject_verb_agreement_1 | 616 | 305 | 304 | 0 | 1 |
| principle_A_domain_1 | 616 | 616 | 616 | 0 | 0 |
| npi_present_1 | 616 | 32 | 29 | 0 | 3 |
| wh_vs_that_no_gap | 616 | 613 | 613 | 0 | 0 |
| irregular_past_participle_adjectives | 616 | 191 | 191 | 0 | 0 |

## Method and limits

Primary metric is the standard minimal-pair full-sentence method: summed causal log likelihood of the grammatical sentence must exceed the ungrammatical sentence. BOS is conditioned on; EOS is not scored; no length normalization. Ties count wrong. This scoring uses teacher-forced likelihood and is distinct from free-form task solving. Illustrative greedy generations cannot establish improved language quality.
Eight BLiMP paradigms were selected for linguistic coverage before running this model. This is not full 67-paradigm BLiMP, and its results must not be equated to the model-card 52.43% figure. Exact duplicate/shared sentences were removed before splitting. Templates and vocabulary remain shared. Original training samples are unavailable, so absence of benchmark examples from pretraining is not certified.
Fixed diagnostic and validation data were used for hypotheses and checkpoint selection, with a maximum of 12 validation evaluations. All attempted edits are retained. The final test baseline was evaluated in a separate process and not opened until the selected weights, edit specification, protocol and evaluator were frozen. After opening, no edits were selected from these test errors.
Zero save/reload, rollback, strict architecture/shapes/dtypes, buffers and tied weights, finite values and unchanged values outside declared masks were checked. Saved candidates were evaluated in fresh processes, without activation-modifying hooks. Three fixed-seed random controls match the actual nonzero support and L2 norm within float32 tolerance. They provide descriptive controls only.
Paired bootstrap is stratified by paradigm, with fixed observed stratum weights, 20,000 replicates and seed 94101. The interval describes test-example sampling for this selected edit, not independent pretraining runs, fresh models, novel grammar types, or adaptive model-search uncertainty. Per-paradigm changes are descriptive secondary results.
CPU float32, one thread, official Transformers eager attention, batch size 64. One warmed inference measurement per condition; loading/scoring overhead excluded. Candidate construction time measures code application/serialization, not agent reasoning time.
This study changes existing values only. No gradients, backward, optimizer steps, SFT, LoRA, regression, distillation, evolutionary search or retraining were used. The checkpoint remains the original pretrained architecture.

## Attribution

Model: https://huggingface.co/Glint-Research/Glimmer-1-Base (MIT declared in card). Pinned revision and SHA-256 in sources.json.
BLiMP: Warstadt et al. (2020), The Benchmark of Linguistic Minimal Pairs for English, TACL 8:377–392, https://doi.org/10.1162/tacl_a_00321 ; data https://github.com/alexwarstadt/blimp (CC-BY 4.0). The present subset and editing splits are modifications of the original evaluation setup.
