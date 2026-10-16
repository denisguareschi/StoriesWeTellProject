# Codebook of Stories We Tell

This is a file-by-file description of the code: what each file contains and what every function or class is for. For the overall picture of the project see [ROADMAP.md](ROADMAP.md).

## How the code is organised

The code lives in `src/`, split into four folders by type of work:

| Folder | What it does | Files |
|---|---|---|
| `src/common/` | Pieces used by everything else: vocabularies, schema checking, access to the models | `vocab.py`, `schema.py`, `llm_client.py` |
| `src/research/` | From text to annotation: reads the corpus and asks the model to annotate it | `annotator.py`, `corpus.py` |
| `src/analysis/` | From annotations to results: graphs, similarity metrics, statistics, validation | `narrative_graph.py`, `similarity.py`, `analysis.py`, `validation.py` |
| `src/retelling/` | The experiment of retelling a story in another culture | `cross_cultural.py` |

Dependencies go in one direction only: `common` imports nothing from the other folders, `research` uses `common`, `analysis` uses `common` and `research`, and `retelling` uses all three.

---

## `main.py`

The entry point from the terminal. Each subcommand is launched with `python main.py <command>`, and `--help` shows the options. The options `--data` and `--result` say where the corpus and the results are.

| Function | Command | What it does |
|---|---|---|
| `cmd_models` | `models` | Shows which free models can be used (Ollama installed, Gemini and Groq keys) |
| `cmd_annotate` | `annotate` | Annotates the whole corpus with `--model provider:model`. `--replica k` repeats the annotation, `--delay` respects request limits, `--max-new` limits the number of calls |
| `_pick_annotator(args)` | not a command | Uses `--annotator`, or the latest complete annotation folder (60 out of 60, no replicas) |
| `cmd_analyze` | `analyze` | Runs `analysis.run_corpus_analysis` |
| `cmd_validate` | `validate` | Computes grounding, the comparison with LitBank and spaCy, reliability (`--compare`) and agreement with the manual annotation (`--manual`); saves `validation.json` |
| `cmd_manual_sheet` | `manual-sheet` | Exports the excerpts and the CSV for blind human annotation |
| `cmd_cross` | `cross-cultural` | Runs the retelling experiment (`--generators`, `--annotator`, `--per-zone`, `--replicas` and others) |
| `cmd_graph` | `graph` | Draws the event graph of one story |
| `cmd_selftest` | `selftest` | Complete test with no network, using a mock model |
| `build_parser()` | not a command | Builds the argument parser |

---

## `src/common/vocab.py`

Gathers vocabularies and metadata in one place. It contains no functions, only constants.

| Constant | Content |
|---|---|
| `N_EVENTS` | 5 key events per story |
| `ARCS`, `FUNCTIONS`, `ROLES` | The controlled vocabularies: 5 arcs, 11 functions, 6 roles |
| `SETTING_DOMAINS` | The social domain of the story (it used to be called "cultural context"). It is not the culture |
| `RELATION_TYPES`, `TRANSFORMATION_TYPES` | Relations between characters and types of transformation |
| `TRADITIONS`, `ZONE_TO_TRADITION` | The cultural traditions (the "culture" axis) and the map from folder to tradition |
| `FUNCTION_SHORT`, `FUNCTION_GLOSS` | Short labels for the plots and operational definitions that end up in the prompt |
| `PROPP_MAPPING` | Documented correspondence between the 11 functions and those of Propp and Campbell |

---

## `src/common/schema.py`

Checks and repairs annotations, and brings all of them to the v2 format.

- `SchemaError` signals an unrecoverable annotation: no characters, fewer than 3 events or no valid function.
- `normalize(raw)` returns `(data, issues)`. It brings any annotation (v1 or v2) to the canonical format: it maps values to the vocabularies ignoring case and punctuation, lists every repair in `issues` (it never drops anything silently), adds missing participants as `SECONDARY`, guarantees a protagonist, truncates to 5 events and recomputes `NARRATIVE_FUNCTIONS` from the events. For old data it reconstructs `CAUSED_BY`, `RELATIONS` and `CHANGE_EVENT` and records this in `_meta.derived`.
- `_derive_causal_links(events)` is a word-based heuristic: event *i* is caused by the earlier event whose effect most resembles its cause (overlap coefficient of at least 0.25).
- `_derive_relations(characters, events)`: if the antagonist shares an event with the protagonist, an `OPPOSES` relation is created; the helper produces `HELPS`.
- `_clean_relations(...)` filters the relations given by the model: known names, valid type, no relation of a character with itself.
- `_canon`, `_key`, `_tokens`, `_as_int_list`, `_context_value` are internal utilities.
- `function_sequence(data)` returns the functions of the events in order.
- `event_text(data)` builds the text (event, cause, effect) used by the semantic metric.

---

## `src/common/llm_client.py`

Talks to the models. Only free modes are supported.

- `LLMError`, `RateLimited(wait)`, `QuotaExhausted` form the family of errors.
- `ModelSpec(provider, model)` has `.tag` (a folder-safe name) and `__str__` (`provider:model`).
- `parse_spec(spec)` turns the string `provider:model` into a `ModelSpec`, with a default model for each provider.
- `ollama_models()` lists the installed Ollama models (an empty list if the service does not answer).
- `default_spec()` picks a free model in the order `$LLM_MODEL`, Ollama, Gemini, Groq, and explains the error if none is available.
- `generate(prompt, spec, *, temperature, seed, json_mode, max_retries)` makes one request. It retries 429 errors with the suggested wait and raises `QuotaExhausted` when the daily quota runs out.
- `extract_json(text)` extracts the first JSON object from a response, even when it is inside a markdown block.
- Internal providers: `_ollama`, `_groq` (HTTP with `urllib`), `_gemini` (the `google-genai` library) and `_mock` (deterministic, used only by the tests). Support: `_post_json`, `_retry_delay`, `_raise_for_text`, `_load_env` (reads `.env` from the project root).

---

## `src/research/annotator.py`

Turns a text into a structured annotation. The protocol is called `v2-sampled-single-pass` and is the same for every text.

- `representative_excerpt(text, budget=12000)` returns the whole text if it is short; otherwise it takes three windows (beginning, middle, end) cut at paragraph boundaries. `_snap` aligns the cuts.
- `build_annotation_prompt(excerpt)` builds the prompt with vocabularies, operational definitions of the functions, JSON schema and rules. It forbids using outside knowledge about the work and includes neither title nor author. `_vocab_block()` generates the vocabulary block.
- `annotate_text(text, spec, *, temperature, seed, max_attempts)` queries the model, extracts the JSON and applies `schema.normalize`. It retries up to `max_attempts` and adds `_meta` (protocol, model, seed, attempts, lengths).

---

## `src/research/corpus.py`

Manages the corpus, the cache of annotations and the annotation of the whole set.

- `Story(name, zone, path)` has `.tradition`, `.text()` and `.n_chars`.
- `list_stories(data_dir)` and `find_story(query, stories)` find stories, by exact name or by unique substring.
- `annotation_dir` and `annotation_path` follow the layout `result/annotations/<tag>[@r<k>]/<zone>/<story>.json`.
- `available_annotators(result_dir)` lists the annotation folders with their number of files.
- `load_annotation` and `load_annotations` load the annotations and normalise them (that is, migrate them to the v2 format).
- `corpus_status(...)` shows progress by zone.
- `annotate_corpus(spec, ...)` annotates the missing stories. It skips those already in the cache, stops on `QuotaExhausted`, keeps going if a single story fails, and for replicas uses temperature 0.5 when none is given.

---

## `src/analysis/narrative_graph.py`

Builds the event graph.

- `build_graph(data)` returns a `MultiDiGraph` with nodes `CHARACTER`, `EVENT` and `LOCATION` (labels: role, function, "LOCATION") and edges `precedes`, `causes`, `participates`, `takes_place_at`, `transforms_in`, plus the relations between characters.
- `skeleton(G)` produces a simple `DiGraph`, without locations and with parallel edges merged. It is the input of the edit distance.
- `motif_sets(data)` produces the motif sets: *function* (unigrams, bigrams, causal pairs) and *role* (role-function, role-relation-role, transformation edges).
- `layout(G)` and `plot_graph(G, title, path, show)` draw the graph: events in a row, characters above, locations below, edges styled by type.

---

## `src/analysis/similarity.py`

The similarity metrics. All of them go from 0 (nothing in common) to 1 (identical).

- `Prepared(data)` and `prepare(datas)` compute graph, skeleton, motifs and WL histogram once.
- `jaccard(a, b)` measures the overlap between two sets.
- `wl_histogram(G, h=3)` and `wl_similarity(a, b)`: Weisfeiler-Lehman with node labels and edge types, then the cosine between histograms.
- `approx_ged(S1, S2)` approximates the graph edit distance (bipartite assignment with the Hungarian algorithm, then the exact cost of the mapping found). It is an upper bound of the exact distance and is deterministic.
- `ged_similarity(S1, S2)` equals `1 − GED / (max|V| + max|E|)`.
- `pair_metrics(p, q)` returns `ged`, `motif`, `role`, `wl` and `composite` (the mean).
- `similarity_matrices(datas, cache)` builds the N×N matrices for all metrics, with an `.npz` cache.
- `SemanticModel(reference_texts)` measures the similarity of meaning of the event descriptions, with sentence-transformers if installed and otherwise TF-IDF plus LSA; it has `.embed` and `.similarity`.
- `METRICS` and `WEIGHTS` list the structural metrics and the weights (all equal).

---

## `src/analysis/analysis.py`

The analysis of the whole corpus.

- `to_distance`, `_agglo`, `select_k`: distance, hierarchical clustering with average linkage and choice of k by silhouette.
- `cluster_stability(dist, k)` computes the mean ARI between the full clustering and those on 80% subsamples.
- `zone_separation(dist, zones, n_perm)` is the permutation test on the difference between mean distance across zones and within zones (overall and by pair), with the silhouette per zone.
- `cluster_zone_association(labels, zones)` computes the ARI between clusters and zones with a permutation p-value.
- `event_share_matrix` and `function_zone_stats` compute the share of events per function, the Kruskal-Wallis test across zones, the Benjamini-Hochberg correction, ε² and the label `universal`, `culture-bound` or `shared`.
- `position_profile` gives the frequency of each function by event position and by zone.
- `name_clusters` gives each cluster a name: over-represented functions and dominant zone.
- Plots, saved in `figures/`: `plot_story_map`, `plot_distance_heatmap`, `plot_function_heatmap`, `plot_radar`, `plot_cluster_breakdown`, `plot_arc_by_zone`, `plot_position_profiles`.
- `run_corpus_analysis(annotator_tag, ...)` coordinates everything and writes `metrics.json` and the plots; `print_summary(m)` prints the summary.

---

## `src/analysis/validation.py`

Checks whether the annotation tool is reliable.

- `grounding_report(pairs)` says how many extracted characters really appear in the text and how many repairs are needed on average, per zone.
- `gold_person_names(path, top_k)` and `litbank_evaluation(pairs, gold_dir)` measure recall@k of the LitBank names for the LLM and for a spaCy baseline on the same excerpt.
- `reliability(pairs_a, pairs_b)` measures agreement between two annotations: κ on arc and functions, position-wise agreement, composite similarity of the same story and *retrieval top-1* (compared with chance, 1/n).
- `export_manual_sheet(...)` and `score_manual(csv, pairs)` create the sheet for blind human annotation and compare it with the model's annotations.

---

## `src/retelling/cross_cultural.py`

The experiment in which a model retells a story in another tradition without knowing what it has to preserve.

- `build_retelling_prompt` and `generate_retelling` build the prompt with only the text and the target tradition.
- `select_stories(...)` uses an explicit list or a sample per zone with a fixed seed.
- `run_experiment(generators, annotator, ...)` for each story, generator, tradition and replica generates, annotates and saves to the cache. It estimates the *noise floor* (re-annotations of the original) and the *paraphrase control*, then moves on to `_analyse`.
- `compute_role_alignment(original, retelling)` computes for each role how far the functions coincide (`aligned`), which are lost (`lost`) and which are added (`added`); it uses `_role_to_functions`.
- `_analyse(...)` computes for each retelling the structural and semantic metrics, the arc, the functions preserved, lost and added, and the outcome (`preserved`, `assimilated`, `hybrid`, `distorted`). It also computes retention per function with bootstrap intervals over stories, role alignment, the drift matrix, the per-generator summary and the Wilcoxon test between retellings and control. It writes `summary.json` and `retellings.csv`.
- `_bootstrap_ci`, `_plots` (`drift_matrix.png`, `function_stability.png`, `structure_vs_semantics.png`) and `_print` complete the module.

---

## `tests/test_offline.py`

A complete test with the mock model, on a temporary synthetic corpus. It exercises sampling, schema repair, the graph, properties of the approximate edit distance, the cache, the analysis, reliability and the cross-cultural experiment. The function `main()` is called by `python main.py selftest`. It checks that the machinery runs, not that the results are scientifically correct.
