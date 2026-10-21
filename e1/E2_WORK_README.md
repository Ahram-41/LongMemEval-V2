# ~/e1 E2 work files (LongMemEval-V2 Small, AgentRunbook-R raw-state retrieval comparison)

Copied to `~/e1/e2work/README.md` on the cluster. Code: fork Ahram-41/LongMemEval-V2, branch
`e1-checkpoint`, directory `e1/`. Every producing script writes a `manifest*.json` next to its
output (git commit, dirty flag, argv, inputs, params).

## Inputs frozen once (write-once)

| Path | Content | Produced by |
|---|---|---|
| `~/e1/cache/meta/<dom>.jsonl` | question id, type, abstention flag, eval_function, question text, gold answer (from B0 per_question) | `e1/pipeline/e2_prep.py meta` |
| `~/e1/cache/queries/<dom>.jsonl` | controller queries parsed from the B0 memory contexts: raw-state queries, event / note query, per-query block counts, first-query limit L, `context_truncated`, `missing_detected`, B0 raw blocks (query, rank, similarity, entry id) | `e1/pipeline/freeze_queries.py` |
| `~/e1/cache/queries_regen/<dom>.jsonl` | official query bundle regenerated once (same prompt / Qwen3.5-9B / thinking / T=0.6) for the 107 questions with truncated B0 context or missing queries; `<dom>.DONE` markers | `e1/pipeline/e2_gpu.py regen` (`slurm/e2_embed.sbatch STAGE=regen`, job 990970) |
| `~/e1/index_ext/queries/<dom>/{queries.jsonl,vectors.npy}` | official query embeddings (`_embed_texts(is_query=True)`, "Instruct: ...\nQuery:...") for every raw query (parsed + regenerated), question text, regenerated event / note query; rows append-only | `e2_gpu.py queries` (job 990957) |
| `~/e1/index_ext/dense_full/<dom>/pieces.jsonl` | every state's "State i\n<axtree>" cut into consecutive <=4,096-token pieces (Qwen3-Embedding-8B tokenizer, cut at token offsets) | `e2_prep.py pieces` |
| `~/e1/index_ext/dense_full/<dom>/{shards/,vectors.npy}` | piece embeddings (documents, no instruction), row = piece_id | `e2_gpu.py pieces` (job 990957) |
| `~/e1/index_ext/bm25/<dom>/<tok>.pkl` | BM25 inverted index over full slice_axtree_text; tok = basic / camel / noid | `e1/retrieval/bm25.py` (built on first use) |

## Work products (`~/e1/e2work/`)

| Path | Content | Produced by |
|---|---|---|
| `scores/<dom>/queries.json` | the query strings (rows) all score matrices refer to | `e1/pipeline/retrieve.py score` |
| `scores/<dom>/<method>.npy` | float32 [n_queries x n_entries] score of every pool entry per base method (V, Vfull, T_<tok>, G_<src>_<tok>) | same |
| `strata/<dom>.json` | §十一 pre-registered strata per question (qtype, premise, domain, anchors, qidf; pos, plen, rarity for phrase-answer questions) | `e1/analysis/strata.py` (definitions committed in 18c7f8f before any comparison) |
| `strata/<dom>_entrytext.pkl` | per entry: normalised full slice text, normalised embedded prefix (official 40k-char cut + 4,096 tokens), token count | same |
| `rounds/<round>/config.json` | exact config of an iteration round: variants, query source, set, reference, config hash, git commit, note (what changed, why) | `e1/analysis/retrieval_eval.py table --round` |
| `rounds/<round>/table.txt` | full printed tables: coverage per variant x question type, paired gain/loss vs V, 2x2 per stratum with net and CI | same |
| `rounds/<round>/grid.tsv` | one row per domain x variant x proxy (ans / anc) x stratum label: covered, eligible, both/gain/loss/neither vs reference, net, 95% CI, sign-test p (the fusion sweep grid is the round that holds the w / k variants) | same |
| `rounds/<round>/labels.json` | per question per variant: coverage labels and merged blocks | same |
| `rounds/<round>/dump_<variant>_<dom>.jsonl` | rich per-question record: type, premise, query source, every query with top-50 (entry, rank, score) under V / Vfull / T:basic / G:both:basic and the variant itself, merged raw blocks, answer phrases with per-phrase hits (entry, offset in normalised slice, inside/outside embedded prefix, slice tokens), anchors, coverage, strata | same, `--dump-variants` |
| `retrieved/<arm>/<dom>.jsonl` | frozen arm: raw blocks per question (query, method rank, official V similarity for the header, method score, entry id) | `retrieve.py select` |
| `contexts/<arm>/<dom>_contexts.jsonl` | rendered memory contexts (official renderers; notes / events copied from B0 for parsed questions, re-searched for regenerated ones) | `e1/pipeline/build_context.py` |
| `reports/s1_memory_form.txt` | §十 S1a retention / S1b section-of-B1-context tables | `e1/analysis/memory_form.py` |
| `reports/*.txt` | other printed reports (V gate, calibration, reader comparisons) | see each file's first line |

## Reader arms (`~/e1/runs/<arm>/<dom>_small/`)

`prompt_rows.jsonl` (rerun questions only), `rerun_ids.txt`, `reuse_ids.txt`, `rerun_reasons.jsonl`
(why each question was rerun or reused, B1 vs arm blocks), `reader_outputs_cache.jsonl`,
`per_question.jsonl` (judge scores), `manifest.json`; `<arm>_merged_small` = B1 rows with rerun
rows replaced (`e1/pipeline/merge_noted.py`). Compact per-question reader record:
`e1/analysis/reader_dump.py`.
