# Stories We Tell: project description and roadmap

The starting question is this: how do language models represent and reproduce narrative archetypes when the stories come from different cultures? The project takes stories from three traditions (European, Japanese and South African), turns them into **event graphs**, has an LLM retell the stories in another tradition, and compares the versions with metrics of structure and meaning (graph edit distance, recurring motifs, semantic roles, Weisfeiler-Lehman).

This document explains what the code does today and how each requirement of the assignment is covered. For a file-by-file description see [CODEBOOK.md](CODEBOOK.md).

---

## 1. The project on one page

```
data/<zone>/*.txt ──► research/annotator ──► schema.normalize ──► annotation v2 (JSON)
   (60 texts)         (free LLM)             (validation)          result/annotations/<annotator>/
                                                                       │
                       ┌───────────────────────────────────────────────┤
                       ▼                                               ▼
        analysis/narrative_graph.build_graph                retelling/cross_cultural.run_experiment
        (typed event graph)                                 (blind retelling in other traditions)
                       │                                               │
                       ▼                                               │
        analysis/similarity (GED, motifs, roles, WL, semantics) ◄──────┘
                       │
                       ▼
        analysis/analysis (clusters, permutation tests, universal or cultural functions, plots)
        analysis/validation (grounding, LitBank, spaCy, reliability, human annotation)
```

| Folder or file | What it is for |
|---|---|
| `main.py` | The command line: `models`, `annotate`, `analyze`, `validate`, `manual-sheet`, `cross-cultural`, `graph`, `selftest` |
| `src/common/` | `vocab.py` (vocabularies, traditions, mapping to Propp), `schema.py` (checking and repairing annotations), `llm_client.py` (access to the free models) |
| `src/research/` | `annotator.py` (text sampling and prompt), `corpus.py` (story discovery, cache, annotation of the whole corpus) |
| `src/analysis/` | `narrative_graph.py` (graph and motifs), `similarity.py` (metrics), `analysis.py` (clusters, statistics, plots), `validation.py` (validation of the annotator) |
| `src/retelling/` | `cross_cultural.py` (the retelling experiment) |
| `tests/test_offline.py` | A complete test with no network, using a mock model |
| `data/` | The corpus: `european/` (20 LitBank novels), `asian/` (20 Japanese folktales), `south_african/` (20 folktales) |
| `data_gold/` | Reference annotations: LitBank entities for the 20 novels (from `dbamman/litbank`) and the manual sheet in `manual/` |
| `result/annotations/` | Model annotations, one folder per annotator (`<tag>` and `<tag>@r<k>` for replicas) |
| `result/analysis/<tag>/` | `metrics.json`, `validation.json`, `figures/*.png` |
| `result/cross_cultural/` | Retellings and their annotations; the summaries are in `_summary_*` |

---

## 2. Which models are used

Only free models, available to anyone.

| Provider | Cost | What is needed | How to write it |
|---|---|---|---|
| Ollama (local, recommended) | free and unlimited | install [ollama.com](https://ollama.com) and run `ollama pull llama3.1:8b` | `ollama:llama3.1:8b`, `ollama:qwen2.5:7b` |
| Gemini, free tier | free, with a daily quota | a free Google AI Studio key in the `.env` file as `GEMINI_API_KEY` | `gemini:gemini-2.5-flash` |
| Groq, free tier | free, with a request limit | a free key from console.groq.com in the `.env` file as `GROQ_API_KEY` | `groq:llama-3.3-70b-versatile` |

`python main.py models` shows what is available. The comparison between models required by the assignment is obtained by annotating and generating with different specifications. 429 errors are retried with the suggested wait; if the daily quota runs out, the run stops cleanly and resumes from the cache the next day.

---

## 3. How the assignment is covered

### 3.1 Corpus with at least three cultural areas
- Three areas with 20 texts each: European (LitBank), Japanese, South African.
- The "culture" axis is the **tradition** (`vocab.TRADITIONS`, `ZONE_TO_TRADITION`). The old "cultural context" field is now called `SETTING_DOMAIN`, because it describes a social domain and not a culture.
- A fourth tradition, Indigenous North American, exists as a **target** for retellings but has no texts in the corpus.
- Declared limits: all texts are in English, and some may have been translated after acquisition to get one uniform language. "European" means literary fiction in English (including American authors). Some South African tales are variants of the same tale type.

### 3.2 Model of the narrative structure
- Each story has a structured annotation (schema v2): arc, characters with role and transformation type, five key events with function, participants, location, cause and effect, `CAUSED_BY`, and typed relations between characters (`OPPOSES`, `HELPS`, `DECEIVES`, `BONDS_WITH`).
- The **event graph** (`narrative_graph.build_graph`) has nodes `CHARACTER`, `EVENT` (ordered 1 to 5) and `LOCATION`, and edges `precedes`, `causes`, `participates`, `takes_place_at`, `transforms_in`, plus the relations between characters.
- There are 11 narrative functions, taken from Propp and Campbell, with operational definitions written into the prompt. The mapping to Propp is documented in `vocab.PROPP_MAPPING`.
- NLP tools: an LLM does the extraction, which is then validated (see 3.6). LitBank annotations and a spaCy NER baseline serve as a non-LLM reference.

### 3.3 Comparison across cultures and models
`cross_cultural.run_experiment` follows a **blind** protocol:
1. The generator receives only the text and the name of the target tradition, never the arc or the functions to preserve.
2. Each retelling is annotated by the same annotator as the corpus, which may be a different model from the generator.
3. Two controls: the retelling in the story's own tradition (*paraphrase control*) and the re-annotation of the original at a higher temperature (*noise floor*). Drift is read only beyond these thresholds.
4. Each retelling receives an outcome: `preserved`, `assimilated` (closer to the target tradition's corpus), `hybrid` (halfway) or `distorted`.
5. With several generators in the same experiment, models are compared against each other (`per_generator`).

### 3.4 Quantitative analysis
| Required | How it is done |
|---|---|
| Graph edit distance | `similarity.approx_ged`: bipartite approximation (Riesen and Bunke) on the labelled skeleton; deterministic and an upper bound of the exact distance (checked in the test) |
| Motif overlap | Jaccard over unigrams, temporal bigrams and causal pairs of functions (`motif`) |
| Semantic role alignment | Jaccard over (role, function) pairs, (role, relation, role) triples and transformation edges (`role`), plus `compute_role_alignment` per role |
| Graph similarity with content | Weisfeiler-Lehman histograms (h=3) on the full graph (`wl`) |
| Semantic metric | Cosine over TF-IDF and LSA (or sentence-transformers, if installed) of the event descriptions |
| Most stable or most distorted functions | `function_stability`: retention and intrusion per function, with 95% bootstrap intervals over stories |
| Clusters and recurring motifs | Hierarchical clustering, k chosen by silhouette, stability with ARI on bootstrap samples |
| Universal or culture-bound? | Permutation test by zone (between minus within), ARI of clusters against zone, Kruskal-Wallis with BH correction per function, labels `universal`, `culture-bound`, `shared` |

The **composite** is the mean of the four structural metrics. Semantic similarity is reported separately.

### 3.5 Extensions
- **Cross-cultural map** (`story_map.png`): MDS of the narrative universe, colour by tradition, marker by cluster, lines between the most similar pairs from different traditions.
- Visualisations: function by tradition heatmap, radar, profile by position, clusters by tradition, arc by tradition, *Narrative Drift Matrix* (source against target), retention by function.
- **Not done**: multimodal inputs (images or symbols).

### 3.6 Validation of the annotator (`validate`)
- **Grounding**: share of extracted characters that really appear in the text.
- **LitBank**: recall@5 of the gold names of the 20 novels, for the LLM and for spaCy on the same excerpt.
- **Reliability**: agreement between two runs (different seeds or different models): Cohen's κ on arc and functions, position-wise agreement, *retrieval top-1* (is the annotator's noise smaller than the variability between stories?).
- **Blind human annotation**: `manual-sheet` exports 12 excerpts and a CSV; `validate --manual` computes the agreement. The current CSV was filled in by hand entirely by the author, independently of the two LLM annotators, and covers 12 stories.

---

## 4. What was corrected compared with the first version

| Problem | Correction |
|---|---|
| The "star" graph did not enter the similarity calculation | Event graph with causal, temporal and relational edges, used by GED, motifs, roles and WL |
| Similarity as a cosine over 30 labels, with almost saturated features | Four structural metrics. Functions come from the **events**: in the old annotations the story-level list held on average 8.2 functions out of 11, against 4.3 distinct functions actually assigned to events |
| "Culture" equal to domain (Social, Professional, Health) | Culture equal to tradition; the domain became `SETTING_DOMAIN` |
| Retelling guided by the functions to preserve (circular), always the same targets, a single story | Blind protocol, all target traditions, noise controls, a sample of stories, replicas, several models |
| Asymmetric pipeline (novels summarised from the model's memory, tales read directly) | Same protocol for everything: beginning, middle and end windows, no title, no outside knowledge allowed |
| No validation of the LLM, and `TRAGEDY`, outside the vocabulary, silently dropped | Schema validator with reported repairs, retries, grounding, LitBank, κ and human annotation |
| No test of the research question | Permutation tests, ARI, Kruskal-Wallis with BH, bootstrap |
| Vocabularies duplicated in three files | A single `vocab.py` |
| Only Gemini, paid or by quota | Selectable free providers (Ollama, Gemini, Groq) |
| No reproducibility | Explicit temperature and seed, `_meta` in every annotation, cache, offline test |

---

## 5. Results obtained

Two annotators were run on the whole corpus (60 texts): Llama 3.1 8B locally and Gemini 2.5 Flash. Llama also has a complete replica (`@r1`) and noise replicas for the stories of the experiment (`@r2`).

**Corpus structure** (`result/analysis/<tag>/metrics.json`)
- Mean composite similarity of 0.205 for both models: the stories share little of the exact event structure.
- Stories are closer within the same tradition than between traditions, but only slightly: a difference of 0.016 for Llama and 0.034 for Gemini, both with p < 0.001. The effect comes mostly from the South African tales.
- **TRICKSTER** is the only culture-bound function (80-90% of South African tales, 15-35% of the others). **CALL_TO_ADVENTURE** and **ORDEAL** are universal. **HELPER_APPEARS** depends on the model.
- The clusters are unstable and only weakly aligned with the traditions: they should not be interpreted.
- The arc assigned depends on the model: European novels are Tragedy for Llama (13 out of 20) and Hero's Journey for Gemini (16 out of 20).

**Validation**
- Grounding between 0.98 and 1.00 for both models.
- Recall@5 on LitBank: 0.39 (Llama) and 0.44 (Gemini), against 0.48 for spaCy. It should be read as a plausibility check, not as a measure of quality: recall does not penalise precision, and the gold covers only roughly the first 2000 words of each novel.
- Llama replica: κ 0.97 on the arc, 0.45 on functions. Between the two models: κ 0.26 on the arc, 0.36 on functions.
- Manual sheet (12 stories): Llama agreement 0.42 on the arc and 0.55 on functions (κ 0.50); Gemini 0.67 and 0.45 (κ 0.37).

**Cross-cultural retellings** (`result/cross_cultural/_summary_ollama_9stories` and `_summary_gemini_5stories`)
- Llama retold 9 stories (36 retellings: 27 into other traditions and 9 controls): 18 preserved, 7 hybrid, 2 distorted, none assimilated. Into another tradition the similarity is 0.306, equal to the control (Wilcoxon p = 1.0).
- Gemini retold 5 stories (20 retellings): 5 preserved, 6 hybrid, 4 distorted; similarity 0.296 against 0.370 for the control.
- Annotator noise is high: a similarity of 0.42 between two annotations of the same original.
- Functions that are best preserved: CALL_TO_ADVENTURE and HELPER_APPEARS. SACRIFICE and RETURN are lost.

One caveat: the Wilcoxon test needs at least 6 pairs, so it was not run for Gemini.

---

## 6. How to rerun everything

```bash
python main.py selftest                                         # offline check
python main.py models                                           # available free LLMs
python main.py annotate --model ollama:llama3.1:8b              # about 60 calls (resumes from the cache)
python main.py annotate --model ollama:llama3.1:8b --replica 1  # replica for reliability
python main.py analyze  --annotator ollama-llama3.1-8b
python main.py manual-sheet                                     # CSV to fill in by hand (12 stories)
python main.py validate --annotator ollama-llama3.1-8b --compare ollama-llama3.1-8b@r1 \
                        --manual data_gold/manual/manual_annotation_template.csv
python main.py cross-cultural --generators ollama:llama3.1:8b,gemini:gemini-2.5-flash \
                              --annotator ollama:llama3.1:8b --per-zone 3 --replicas 2
python main.py graph --annotator ollama-llama3.1-8b --story "MIRROR OF MATSUYAMA"
```

With the Gemini free tier add `--delay 5` (about 12 requests per minute). If the daily quota runs out, the command stops and resumes from the cache the next day. If `cross-cultural` is launched with several generators and one runs out of quota, the run stops for the others too: it is better to launch them separately and then do a final run with all of them, which only reads the cache.

`validate` overwrites the annotator's `validation.json`: if a comparison is already saved (for example with the replica), copy the file before rerunning.

---

## 7. Known limits and future work

- The annotator is an LLM. How far it can be trusted depends on `validate` and on the human annotation, which covers 12 stories.
- Five events per story is a strong simplification, chosen so that texts can be compared. Novels are represented by three text windows.
- Sample size: 20 texts per tradition. The retelling experiment has few tales (9 for Llama, 5 for Gemini), so the intervals are wide and should be reported as such.
- The texts are English translations, and some may have been translated a second time after acquisition. The translator's effect cannot be separated from that of the tradition.
- Event functions are noisy even between replicas of the same model (κ 0.45).
- Multimodal extension not done.
- Possible developments: more tale types (ATU classification), a second non-English corpus, use of the LitBank *events* and coreference layers to validate events and coreference chains too, independent human annotation by several people.

---

## 8. Design choices and parameters

Every value is a constant or an argument with a default in the code (in the file listed in [CODEBOOK.md](CODEBOOK.md)) and can be changed. Where a choice is heuristic, this is stated; it should come with a sensitivity analysis when results are reported.
