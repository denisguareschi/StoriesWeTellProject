"""
Stories We Tell — command line entry point.

Typical session (all LLM modes are free, see README in ROADMAP.md):

  python main.py models                                    # which free LLMs are usable
  python main.py annotate --model ollama:llama3.1:8b       # step 1: annotate the corpus
  python main.py annotate --model ollama:llama3.1:8b --replica 1   # re-run for reliability
  python main.py analyze  --annotator ollama-llama3.1-8b   # step 2: clustering + statistics
  python main.py validate --annotator ollama-llama3.1-8b --compare ollama-llama3.1-8b@r1
  python main.py cross-cultural --generators ollama:llama3.1:8b \\
                                --annotator ollama:llama3.1:8b  # step 3: retelling experiment
  python main.py graph --annotator <tag> --story "THE MIRROR OF MATSUYAMA"
  python main.py selftest                                  # offline end-to-end check (mock LLM)
"""

import argparse
import json
import sys
from pathlib import Path


def _stories(args):
    from src.research import corpus
    return corpus.list_stories(args.data)


def cmd_models(args):
    from src.research import corpus
    from src.common import llm_client
    inst = llm_client.ollama_models()
    print("Ollama (local, free, no key):", inst or "not running / no model installed")
    import os
    llm_client._load_env()
    print("Gemini free tier key set    :", bool(os.getenv("GEMINI_API_KEY")))
    print("Groq free tier key set      :", bool(os.getenv("GROQ_API_KEY")))
    try:
        print("Default model               :", llm_client.default_spec())
    except llm_client.LLMError as exc:
        print("Default model               : none —", exc)
    print("\nAnnotation sets in result/annotations:")
    for tag, n in corpus.available_annotators(args.result).items():
        print(f"  {tag:50s} {n:3d} files")


def cmd_annotate(args):
    from src.research import corpus
    from src.common import llm_client
    spec = llm_client.parse_spec(args.model) if args.model else llm_client.default_spec()
    print(f"Annotating corpus with {spec}  (tag: {spec.tag})")
    corpus.corpus_status(_stories(args), args.result, spec.tag, args.replica)
    summary = corpus.annotate_corpus(
        spec, args.data, args.result, replica=args.replica, temperature=args.temperature,
        seed=args.seed, delay_seconds=args.delay, max_new=args.max_new)
    print(summary)
    corpus.corpus_status(_stories(args), args.result, spec.tag, args.replica)


def _pick_annotator(args) -> str:
    from src.research import corpus
    if args.annotator:
        return args.annotator
    avail = corpus.available_annotators(args.result)
    full = [t for t, n in avail.items() if n >= len(_stories(args)) and "@r" not in t]
    if not full:
        sys.exit("No complete annotation set. Run `annotate` first (see `models`).")
    return full[-1]


def cmd_analyze(args):
    from src.analysis import analysis
    analysis.run_corpus_analysis(_pick_annotator(args), args.data, args.result,
                                 n_clusters=args.k, n_perm=args.perm)


def cmd_validate(args):
    from src.research import corpus
    from src.analysis import validation
    tag = _pick_annotator(args)
    pairs = corpus.load_annotations(_stories(args), args.result, tag)
    report = {"annotator": tag,
              "grounding": validation.grounding_report(pairs),
              "litbank": validation.litbank_evaluation(pairs, args.gold)}
    if args.compare:
        other = corpus.load_annotations(_stories(args), args.result, args.compare)
        report["reliability_vs_" + args.compare] = validation.reliability(pairs, other)
    if args.manual:
        report["manual"] = validation.score_manual(args.manual, pairs)
    out = Path(args.result) / "analysis" / tag
    out.mkdir(parents=True, exist_ok=True)
    report["litbank"].pop("per_novel", None) if not args.verbose else None
    (out / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def cmd_manual_sheet(args):
    from src.analysis import validation
    validation.export_manual_sheet(_stories(args), n=args.n, seed=args.seed)


def cmd_cross(args):
    from src.retelling import cross_cultural
    cross_cultural.run_experiment(
        [g.strip() for g in args.generators.split(",")],
        args.annotator,
        story_names=args.stories.split("|") if args.stories else None,
        per_zone=args.per_zone,
        targets=args.targets.split(",") if args.targets else None,
        replicas=args.replicas, noise_replicas=args.noise_replicas,
        data_dir=args.data, result_dir=args.result, seed=args.seed,
        delay_seconds=args.delay, corpus_tag=args.corpus_tag, force=args.force)


def cmd_graph(args):
    from src.research import corpus
    from src.analysis import narrative_graph
    tag = _pick_annotator(args)
    story = corpus.find_story(args.story, _stories(args))
    data = corpus.load_annotation(story, args.result, tag)
    if data is None:
        sys.exit(f"{story.name} has no annotation under '{tag}'")
    out = Path(args.result) / "analysis" / tag / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    G = narrative_graph.build_graph(data)
    p = out / f"{story.name}.png"
    narrative_graph.plot_graph(G, title=f"{story.name}  [{data['NARRATIVE_ARC']}]", path=p)
    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {p}")


def cmd_selftest(args):
    from tests import test_offline
    test_offline.main()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--result", default="./result")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="show usable free LLMs and annotation sets").set_defaults(fn=cmd_models)

    p = sub.add_parser("annotate", help="annotate the corpus with an LLM")
    p.add_argument("--model", help="provider:model (default: first free model found)")
    p.add_argument("--replica", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.0, help="seconds between calls (free tiers: ~5)")
    p.add_argument("--max-new", type=int, default=None)
    p.set_defaults(fn=cmd_annotate)

    p = sub.add_parser("analyze", help="clustering, statistics and figures")
    p.add_argument("--annotator")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--perm", type=int, default=5000)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("validate", help="grounding, LitBank gold, reliability, manual sheet")
    p.add_argument("--annotator")
    p.add_argument("--compare", help="annotation folder to compare with (e.g. '<tag>@r1')")
    p.add_argument("--manual", help="filled manual_annotation_template.csv")
    p.add_argument("--gold", default="./data_gold/litbank_entities")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("manual-sheet", help="export blind human-annotation template")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_manual_sheet)

    p = sub.add_parser("cross-cultural", help="blind retelling experiment")
    p.add_argument("--generators", required=True, help="comma list of provider:model")
    p.add_argument("--annotator", required=True, help="provider:model used to annotate")
    p.add_argument("--stories", help="'|'-separated story names (default: seeded sample)")
    p.add_argument("--per-zone", type=int, default=3)
    p.add_argument("--targets", help="comma list of traditions (default: all)")
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--noise-replicas", type=int, default=2)
    p.add_argument("--corpus-tag", help="annotation set used as reference corpus")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_cross)

    p = sub.add_parser("graph", help="draw the event graph of one story")
    p.add_argument("--annotator")
    p.add_argument("--story", required=True)
    p.set_defaults(fn=cmd_graph)

    sub.add_parser("selftest", help="offline end-to-end test with the mock LLM").set_defaults(fn=cmd_selftest)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
