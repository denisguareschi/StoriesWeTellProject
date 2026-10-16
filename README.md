# Stories We Tell

How do language models represent and reproduce narrative archetypes when the stories come from different cultures?

This project takes 60 stories from three traditions (European literary fiction, Japanese folktales, South African folktales) and turns each one into a typed **event graph**: five key events, each labelled with one of eleven narrative functions inspired by Propp and Campbell. A language model does the annotation, and its output is checked against LitBank, a spaCy baseline, a replica run and a small manual annotation. The graphs are compared with four structural metrics, and a blind retelling experiment shows what happens when a model retells a story in another tradition.

Two free models are used throughout: Llama 3.1 8B (local, through Ollama) and Gemini 2.5 Flash (free tier).

## What it found

- Stories share little of their exact event structure, whatever the tradition. Stories are slightly closer within the same tradition, and the effect comes mostly from the South African tales.
- The trickster is the only culture-bound function. Call to adventure and Ordeal appear in every tradition.
- The arc assigned to a tradition depends on the annotating model: the same European novels are mostly tragedies for one model and hero's journeys for the other.
- A retelling into another culture is as close to the original as a retelling in its own culture, and no retelling becomes closer to the target culture. The models keep the structural skeleton and change the surface.

Numbers, caveats and the full list of limits are in [ROADMAP.md](ROADMAP.md).

## Quick start

Python 3.13 is the version the dependencies were tested with.

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm    # only needed for `validate`
python main.py selftest                    # offline check with a mock model, no network
```

To use real models:

- **Ollama (local, free, recommended).** Install it from [ollama.com](https://ollama.com), then run `ollama pull llama3.1:8b`.
- **Gemini (free tier).** Create a free key in Google AI Studio and put it in a `.env` file in the project root: `GEMINI_API_KEY=your_key`. The `.env` file is ignored by Git and must never be committed.
- **Groq (free tier, optional).** Same idea with `GROQ_API_KEY`.

`python main.py models` shows which models are available.

## Running the experiments

```bash
python main.py annotate --model ollama:llama3.1:8b              # annotate the corpus (resumes from the cache)
python main.py annotate --model ollama:llama3.1:8b --replica 1  # replica, to measure reliability
python main.py analyze  --annotator ollama-llama3.1-8b
python main.py validate --annotator ollama-llama3.1-8b --compare ollama-llama3.1-8b@r1 \
                        --manual data_gold/manual/manual_annotation_template.csv
python main.py cross-cultural --generators ollama:llama3.1:8b,gemini:gemini-2.5-flash \
                              --annotator ollama:llama3.1:8b --per-zone 3 --replicas 2
python main.py graph --annotator ollama-llama3.1-8b --story "MIRROR OF MATSUYAMA"
```

With the Gemini free tier add `--delay 5` to stay within the request limit. If the daily quota runs out, the command stops cleanly and resumes from the cache the next day. All results are saved per story, so an interrupted run loses nothing.

## Repository layout

| Path | Content |
|---|---|
| `main.py` | Command line entry point |
| `src/common/` | Vocabularies, schema checking, access to the models |
| `src/research/` | Text sampling, prompts, annotation of the corpus |
| `src/analysis/` | Event graphs, similarity metrics, statistics, validation |
| `src/retelling/` | The cross-cultural retelling experiment |
| `tests/` | Offline end-to-end test with a mock model |
| `data/` | The corpus (20 texts for each of three traditions) |
| `data_gold/` | LitBank reference annotations and the manual annotation sheet |
| `result/` | Saved annotations, analyses and retellings |

A file-by-file description is in [CODEBOOK.md](CODEBOOK.md).

## Data

- **European**: 20 novels from [LitBank](https://github.com/dbamman/litbank).
- **Japanese**: folktales collected from [Il Bazar di Mari](https://www.ilbazardimari.net/approfondimenti/leggende-orientali/).
- **South African**: folktales from the [Sacred Texts](https://sacred-texts.com/book/south-african-folk-tales) collection.

All texts are in English. Some may have been translated a second time after acquisition to get one uniform language, so translator effects cannot be separated from tradition effects. Check the terms of each source before redistributing the texts.

## Limits

- The language model is both the instrument and the object of study.
- Five events per story is a coarse abstraction, and novels are represented by three text windows.
- The retelling sample is small (nine stories for Llama, five for Gemini because of the free-tier quota), and annotator noise is of the same order as the drift being measured.

## Use of generative AI

Llama 3.1 8B and Gemini 2.5 Flash are the objects of the research. Claude (Anthropic), through Claude Code, was also used as an assistant to write and restructure code, tests and documentation. The author reviewed and tested the result and is responsible for the final content.

## References

- Bamman, D., Popat, S., Shen, S. (2019). An Annotated Dataset of Literary Entities. NAACL-HLT.
- Propp, V. (1968). Morphology of the Folktale. University of Texas Press.
- Campbell, J. (1949). The Hero with a Thousand Faces. Pantheon Books.

## License

To be added.
