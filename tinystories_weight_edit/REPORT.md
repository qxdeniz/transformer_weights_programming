# TinyStories-1M: broad direct weight edits

Actual independent parameter count: 3,745,984, including embeddings; eight transformer blocks contain 398,336 parameters. Architecture: GPT-Neo, width 64, 16 heads, FFN 256, learned positions, alternating global/local attention, tied input/output. The source model was pretrained by its authors; our experiment performs no training.

## Final held-out comparisons

| Condition | N | Before % | After % | Delta pp | Wrong→right | Right→wrong | 95% stratified paired CI pp |
|---|---:|---:|---:|---:|---:|---:|---|
| selected | 4917 | 66.057 | 66.036 | -0.020 | 181 | 182 | [-0.773, +0.752] |
| random_103101 | 4917 | 66.057 | 58.044 | -8.013 | 437 | 831 | [-9.416, -6.630] |
| random_103102 | 4917 | 66.057 | 56.396 | -9.660 | 442 | 917 | [-11.104, -8.237] |
| random_103103 | 4917 | 66.057 | 54.891 | -11.165 | 406 | 955 | [-12.589, -9.762] |

Preregistered joint improvement criterion: not met.
Selected candidate_11 after 12 validation trials. Changed 66048 independent scalars (1.763% of the whole model); L2 delta 3.95519642, relative L2 0.0212976278.

## Preregistered same-token-length subset

N=4106; correct 2694 → 2694; delta +0.000 pp, 95% paired interval [-0.852, +0.877]. W→R 164, R→W 164.

## By paradigm

| Paradigm | N | Before correct | After correct | W→R | R→W |
|---|---:|---:|---:|---:|---:|
| anaphor_number_agreement | 616 | 439 | 433 | 23 | 29 |
| determiner_noun_agreement_2 | 616 | 415 | 400 | 15 | 30 |
| regular_plural_subject_verb_agreement_2 | 616 | 341 | 341 | 25 | 25 |
| irregular_plural_subject_verb_agreement_2 | 615 | 365 | 374 | 32 | 23 |
| irregular_past_participle_verbs | 616 | 592 | 588 | 7 | 11 |
| distractor_agreement_relational_noun | 616 | 244 | 241 | 34 | 37 |
| passive_1 | 614 | 425 | 449 | 30 | 6 |
| intransitive | 608 | 427 | 421 | 15 | 21 |

## Method and limits

Eight new BLiMP paradigms were selected before running this model; they differ from the earlier Glimmer study. Do not rank models by raw accuracy across these distinct subsets. Exact duplicate or shared sentences are removed before any model scores. Templates and vocabulary remain shared.
The primary metric is the standard minimal-pair preference by summed causal sentence log likelihood conditioned on BOS, without EOS or length normalization; ties count wrong. This is teacher-forced likelihood scoring, not answer generation. The predeclared positive result requires positive bootstrap lower bounds on BOTH the full held-out test and its same-token-length subset. This intersection rule limits a pure length-preference interpretation; it does not prove general grammar competence.
All candidates edit complete attention/FFN projections, with at least 10,000 actual changed independent scalars. No global-output-temperature-only candidate is allowed. Candidate ranking uses validation correct count, then fewer lost baseline successes, fewer changed values and smaller L2 norm. Budget 12, including coefficient variants. All attempts retained. No gradients, backward, optimizer, regression, SFT, LoRA, distillation or evolutionary search.
Original checkpoint unchanged. Model keys/shapes/dtypes, tied embeddings, fixed buffers and values outside the declared mask are checked; zero-save/reload and rollback must reproduce predictions. Candidate evaluations run in fresh processes with no diagnostic activation interventions. Legacy 4.28 nonpersistent mask buffers are checked against the current constructor and omitted only for compatible strict parameter loading; missing tied head is restored from the original embedding. These pre-baseline compatibility operations are not weight improvements.
The test baseline was saved by a separate process and kept sealed until checkpoint, spec, protocol and evaluator freeze. Selection does not read final outcomes. After opening, no further edits are selected using this test. This is organizational separation, not technical inaccessibility to the researcher.
Three fixed-seed random controls use the same actual changed mask and L2 norm within float32 rounding tolerance. No best-seed selection or broad superiority over random editing is claimed.
Bootstrap: 20,000 replicates, fixed seed, stratified by paradigm and observed stratum weights. Exact conditional two-sided paired tests are descriptive sensitivity checks, especially for few discordant pairs. Intervals are conditional on the selected edit and test examples, not fresh pretraining runs, independent templates, model searches or different editors. Per-paradigm results are descriptive secondary analyses.
Model training stream was not audited against benchmark examples. One checkpoint, one editor, a fixed finite search and eight paradigms cannot establish broad improvement in language or free-form generation. Short generation samples are illustrative and do not establish improved coherence.
CPU float32, one thread, official eager GPT-Neo, score batch 16. Single warmed inference timing excludes model loading and report/scoring overhead. Construction seconds measure edit application and serialization, not reasoning time.

## Sources

TinyStories model: https://huggingface.co/roneneldan/TinyStories-1M ; revision 77f1b168e219585646439073245fe87e56b3023e. No explicit model license found in the pinned card/tree; redistribution permission is not asserted. Public artifact bundle uses a pinned downloader and declarative edit, omitting source weights.
BLiMP: Warstadt et al. (2020), TACL 8:377–392, https://doi.org/10.1162/tacl_a_00321 ; https://github.com/alexwarstadt/blimp , CC-BY 4.0. Present subset/filtering/editing splits modify the original evaluation setup. Sources, revisions and SHA-256 are in sources.json.
