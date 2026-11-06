# Evidence index

| File | Meaning |
|---|---|
| `submissions.json` / `.csv` | 30 completed accessible Abhi team attempts, freshly retrieved 6 November 2026; exact scored-version links and displayed public/private scores |
| `provenance.json` | Scope, dates and source of the final rank |
| `fresh_paired_bootstrap.json` | 28 October paired Gold58 analysis, including seed and rejected draws |
| `fresh_cache_audit.json` | Aggregate 4,407-study / 88,140-window metadata audit |
| `recomputed_scores.json` / `.csv` | 119 historical score variants, including target-level AUC and input-key flags |
| `experiments/` | Compact saved numerical summaries and adoption decisions; no patient-index files |
| `source_manifest.json` | Publication source origin and SHA-256 hashes |

The score-audit paths describe files in the original private competition workspace. Those underlying patient-level NPZs are intentionally not distributed. Aggregate results can be inspected and charted here; independently recomputing them requires the authorized source predictions and labels. A recorded metric does not establish absence of training leakage or the completion of every run in a family.

The final rank is transcribed from the participant's supplied competition summary screenshot. The screenshot itself is not redistributed. The ledger's score precision is the precision returned by Kaggle.
