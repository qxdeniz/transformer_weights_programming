# Weight-space editing of a 1,644-parameter addition transformer

## Scope and provenance

The source model was trained by its author. This experiment performed inference and direct weight edits only; no gradients, optimizer, distillation, or new training. Architecture, tokenizer, positional encoding, forward, generation and decoding were unchanged.
Source: https://github.com/anadim/smallest-addition-transformer-codex ; commit 9de34061c8c268e0fc2198e473911a62b152254c.
No explicit license file was found in the pinned repository tree. Public availability is not treated as a redistribution license. Artifacts here are a private research workspace; sources can be fetched by the pinned hashes.

## Results

| Set | Model | N | Before | After | Delta pp | Wrong→right | Right→wrong | 95% paired CI pp |
|---|---|---:|---:|---:|---:|---:|---:|---|
| test_uniform | selected | 5000 | 98.860% | 98.860% | +0.000 | 9 | 9 | [-0.160, 0.160] |
| test_uniform | random_71001 | 5000 | 98.860% | 30.240% | -68.620 | 3 | 3434 | [-69.901, -67.320] |
| test_uniform | random_71002 | 5000 | 98.860% | 0.280% | -98.580 | 0 | 4929 | [-98.900, -98.240] |
| test_uniform | random_71003 | 5000 | 98.860% | 2.180% | -96.680 | 0 | 4834 | [-97.160, -96.180] |
| test_stress | selected | 2000 | 98.300% | 98.750% | +0.450 | 15 | 6 | [0.000, 0.900] |
| test_stress | random_71001 | 2000 | 98.300% | 21.650% | -76.650 | 2 | 1535 | [-78.500, -74.800] |
| test_stress | random_71002 | 2000 | 98.300% | 0.100% | -98.200 | 0 | 1964 | [-98.750, -97.600] |
| test_stress | random_71003 | 2000 | 98.300% | 0.400% | -97.900 | 0 | 1958 | [-98.500, -97.250] |
| test_exhaustive_00_99 | selected | 10000 | 99.970% | 99.930% | -0.040 | 0 | 4 | finite exhaustive domain |
| test_exhaustive_00_99 | random_71001 | 10000 | 99.970% | 44.080% | -55.890 | 0 | 5589 | finite exhaustive domain |
| test_exhaustive_00_99 | random_71002 | 10000 | 99.970% | 0.000% | -99.970 | 0 | 9997 | finite exhaustive domain |
| test_exhaustive_00_99 | random_71003 | 10000 | 99.970% | 0.000% | -99.970 | 0 | 9997 | finite exhaustive domain |

## Interpretation

uncertain change on test_uniform; interval includes zero.
Selected candidate_12 after 12 validation attempts. Changed 52 independent scalars; L2 delta 0.560993499.
All candidate outcomes are in results/attempts.json; selection used validation only. Random seeds were fixed in advance; no best-seed selection.

## Encoding

Each operand is zero-padded to 10 decimal digits. Ten Pab digit-pair tokens appear least-significant column first between BOS and equals. The unmodified model greedily generates 12 tokens: 11 reversed sum digits and EOS. Strict EM requires all 12 tokens; author-compatible digits-only EM and the upstream numeric decoder are secondary metrics.

## Limits

One model, checkpoint, arithmetic operation, and generation configuration. No conclusion about general mathematical reasoning, decimal multiplication, other models, or unseen implementations. The training stream is unavailable, so splits are held out from editing, not proven absent from original training. The 0…99 test intentionally includes both operand orders and four known manual examples (with 99+1 also represented in reverse). Bootstrap intervals summarize test-sampling uncertainty for a fixed selected edit, not variability across independent training runs. Stress strata are engineered distributions, not a natural usage distribution. Three random edits are descriptive controls and cannot establish superiority over all random editing.

## Timing

CPU float32, one thread, batch 256, greedy 12-token generation, fixed warmup. Individual timings are one run per condition and are not a speed benchmark; loading and scoring are excluded.

## Reproduction

See README.md. Protocol and code hashes, predictions, all metrics, unchanged buffers/ties, zero edit, rollback and fresh-process reload checks are retained. Final errors must not be used for further confirmatory edits on these same tests.
