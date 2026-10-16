"""
Offline end-to-end test (no network, no API key): runs the whole pipeline with
the deterministic `mock` LLM on a tiny synthetic corpus in a temp directory.

It checks plumbing and invariants, NOT scientific results.
Run:  python main.py selftest     (or: python -m tests.test_offline)
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.analysis import analysis, narrative_graph, similarity, validation
from src.research import corpus
from src.retelling import cross_cultural
from src.common import schema
from src.common import vocab as V
from src.research.annotator import representative_excerpt

_ZONES = {"european": 6, "asian": 6, "south_african": 6}


def _make_corpus(root: Path) -> None:
    for zone, n in _ZONES.items():
        (root / zone).mkdir(parents=True)
        for i in range(n):
            body = " ".join(f"Sentence {j} of story {zone}{i} about hero and rival." for j in range(60))
            (root / zone / f"{zone}_{i}.txt").write_text(body, encoding="utf-8")


def test_excerpt():
    long = "Para one. " * 5000
    ex = representative_excerpt(long)
    assert "[BEGINNING OF THE WORK]" in ex and "[END OF THE WORK]" in ex
    assert len(ex) < 13_500
    assert representative_excerpt("short text") == "short text"


def test_schema_repair():
    raw = {"NARRATIVE_ARC": "hero's journey", "CULTURAL_CONTEXT": "Social and Familiar",
           "CHARACTERS": [{"CHARACTER": "A", "ROLE": "protagonist"}, {"CHARACTER": "B", "ROLE": "villain"}],
           "LOCATIONS": [{"LOCATION": "X"}],
           "KEY EVENTS": [{"KEY EVENT": f"e{i}", "NARRATIVE_FUNCTION": "quest" if i < 3 else "TRAGEDY",
                           "CHARACTERS": ["A", "C"], "LOCATION": "X",
                           "CAUSE": "Story opening" if i == 1 else f"result of e{i-1}",
                           "EFFECT": f"result of e{i}"} for i in range(1, 6)]}
    d, issues = schema.normalize(raw)
    assert d["NARRATIVE_ARC"] == "Hero's Journey"
    assert d["CHARACTERS"][1]["ROLE"] == "SECONDARY"           # 'villain' repaired + reported
    assert any("ROLE invalid" in i for i in issues)
    assert any("FUNCTION invalid" in i for i in issues)        # TRAGEDY is not a function
    assert {c["CHARACTER"] for c in d["CHARACTERS"]} == {"A", "B", "C"}
    assert d["EVENTS"][1]["CAUSED_BY"] == [1]                  # lexical causal derivation
    try:
        schema.normalize({"CHARACTERS": [], "EVENTS": []})
        raise AssertionError("empty annotation must raise")
    except schema.SchemaError:
        pass


def test_graph_and_similarity(datas):
    G = narrative_graph.build_graph(datas[0])
    types = {t for _, _, t in G.edges(data="type")}
    assert {"precedes", "participates", "takes_place_at"} <= types
    P = similarity.prepare(datas[:4])
    m = similarity.pair_metrics(P[0], P[0])
    assert all(abs(m[k] - 1) < 1e-9 for k in similarity.METRICS + ["composite"])
    m01 = similarity.pair_metrics(P[0], P[1])
    assert 0 <= m01["composite"] <= 1
    # approximate GED is an upper bound of the exact GED on a small graph
    import networkx as nx
    a, b = P[0].skel, P[1].skel
    exact = nx.graph_edit_distance(
        a, b, node_match=lambda x, y: x["label"] == y["label"] and x["type"] == y["type"],
        edge_match=lambda x, y: x["type"] == y["type"], timeout=10)
    assert similarity.approx_ged(a, b) >= exact - 1e-9


def main():
    test_excerpt()
    test_schema_repair()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir, result_dir = tmp / "data", tmp / "result"
        _make_corpus(data_dir)
        spec = "mock:demo"

        s = corpus.annotate_corpus(spec, data_dir, result_dir)
        assert s["processed"] == sum(_ZONES.values()) and s["failed"] == 0, s
        s2 = corpus.annotate_corpus(spec, data_dir, result_dir)
        assert s2["processed"] == 0 and s2["skipped"] == sum(_ZONES.values())   # cache
        corpus.annotate_corpus(spec, data_dir, result_dir, replica=1)

        stories = corpus.list_stories(data_dir)
        pairs = corpus.load_annotations(stories, result_dir, "mock-demo")
        assert len(pairs) == 18
        test_graph_and_similarity([d for _, d in pairs])

        m = analysis.run_corpus_analysis("mock-demo", data_dir, result_dir, n_perm=200)
        assert m["n_stories"] == 18 and 0 <= m["zone_separation"]["overall"]["p_value"] <= 1
        assert (result_dir / "analysis" / "mock-demo" / "figures" / "story_map.png").exists()

        rel = validation.reliability(
            pairs, corpus.load_annotations(stories, result_dir, "mock-demo", replica=1))
        assert "retrieval_top1" in rel
        assert validation.grounding_report(pairs)

        res = cross_cultural.run_experiment(
            [spec], spec, per_zone=1, replicas=1, noise_replicas=1,
            data_dir=data_dir, result_dir=result_dir)
        assert res["n_retellings"] == 3 * len(V.TRADITIONS), res["n_retellings"]
        assert res["n_control"] == 3
        assert (result_dir / "cross_cultural" / "_summary" / "mock-demo" / "drift_matrix.png").exists()
        json.dumps(res)                                   # summary must be serialisable
    print("\nSELFTEST OK")


if __name__ == "__main__":
    main()
