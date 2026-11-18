# Baseline: open development sets only

Published author claim: 99.04% digit-only EM on their test. The measurements below are independent; strict EM includes EOS.

| Set | N | Strict EM | Digit EM | Format valid | Seconds |
|---|---:|---:|---:|---:|---:|
| diagnostic | 1024 | 98.5352% | 98.7305% | 99.8047% | 0.199 |
| validation | 2048 | 98.8281% | 98.9258% | 99.9023% | 0.391 |

## Diagnostic categories

| Category | N | Correct | EM |
|---|---:|---:|---:|
| label:carry_chain | 706 | 696 | 98.584% |
| label:leading_zeros_10_digit_input | 416 | 406 | 97.596% |
| label:no_carry | 127 | 126 | 99.213% |
| label:overflow11 | 312 | 308 | 98.718% |
| label:power10 | 64 | 64 | 100.000% |
| label:single_carry | 135 | 133 | 98.519% |
| label:zero_operand | 64 | 63 | 98.438% |
| max_chain:0 | 127 | 126 | 99.213% |
| max_chain:1 | 191 | 187 | 97.906% |
| max_chain:10 | 29 | 29 | 100.000% |
| max_chain:2 | 157 | 153 | 97.452% |
| max_chain:3 | 183 | 182 | 99.454% |
| max_chain:4 | 91 | 89 | 97.802% |
| max_chain:5 | 71 | 71 | 100.000% |
| max_chain:6 | 60 | 59 | 98.333% |
| max_chain:7 | 37 | 37 | 100.000% |
| max_chain:8 | 41 | 39 | 95.122% |
| max_chain:9 | 37 | 37 | 100.000% |
| source:chain_long | 128 | 126 | 98.438% |
| source:chain_medium | 128 | 128 | 100.000% |
| source:chain_short | 128 | 127 | 99.219% |
| source:leading_zeros | 64 | 57 | 89.062% |
| source:overflow11 | 64 | 63 | 98.438% |
| source:power10 | 64 | 64 | 100.000% |
| source:single_carry | 128 | 127 | 99.219% |
| source:uniform | 256 | 254 | 99.219% |
| source:zero | 64 | 63 | 98.438% |

## Diagnostic error examples

| A | B | Expected | Predicted | Max carry run |
|---:|---:|---:|---:|---:|
| 863092030 | 235324082 | 1098416112 | 98416112 | 1 |
| 7733657850 | 2266342402 | 10000000252 | 10000000252 | 8 |
| 2262591730 | 7737408743 | 10000000473 | 10000000473 | 8 |
| 104290422 | 6402290027 | 6506580449 | 6506480449 | 1 |
| 626210134 | 229197833 | 855407967 | 1855407967 | 1 |
| 8889133881 | 9919575843 | 18808709724 | 18809709724 | 4 |
| 7944006416 | 3269508705 | 11213515121 | 11213505121 | 4 |
| 880313858 | 13229326 | 893543184 | 1893543184 | 2 |
| 139199327 | 94665449 | 233864776 | 10233864776 | 2 |
| 947243353 | 878594471 | 1825837824 | 12825837824 | 3 |
| 877036085 | 10268500 | 887304585 | 1887304585 | 2 |
| 966385062 | 0 | 966385062 | 1966385062 | 0 |
| 397295248 | 266924807 | 664220055 | 10664220055 | 6 |
| 4108250365 | 4370620390 | 8478870755 | 8478880755 | 1 |
| 332404931 | 5947850 | 338352781 | 10338352781 | 2 |

Observed errors do not by themselves identify a causal mechanism. Activation analysis and controlled weight edits are required.
Final test outcomes are sealed and were not used for this report.
