"""
Structural and semantic similarity between narratives.

All similarities are in [0, 1].  Four structural metrics are computed on the
event graph of each story and averaged into the composite score:

  ged    1 - normalised approximate graph edit distance on the labelled
         skeleton (events + characters; unit costs; labels must match;
         bipartite assignment, deterministic upper bound of the exact GED)
  motif  Jaccard overlap of function motifs (unigrams, temporal bigrams,
         causal pairs of narrative functions)
  role   Jaccard overlap of semantic-role motifs ((role, function) pairs,
         (role, relation, role) triples, transformation arcs)
  wl     cosine similarity of Weisfeiler-Lehman subtree histograms (h=3) on
         the full labelled event graph

The semantic metric is separate: TF-IDF/LSA (or sentence-transformers when
installed) cosine on the event descriptions.  It measures surface/content
drift, and is deliberately kept out of the structural composite.
"""

import hashlib
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

from .narrative_graph import build_graph, motif_sets, skeleton
from ..common.schema import event_text

METRICS = ["ged", "motif", "role", "wl"]
WEIGHTS = {m: 1.0 / len(METRICS) for m in METRICS}


# ── per-story representation ──────────────────────────────────────────────────

class Prepared:
    """Everything needed to compare one story, computed once."""

    def __init__(self, data: dict):
        self.data = data
        self.graph = build_graph(data)
        self.skel = skeleton(self.graph)
        m = motif_sets(data)
        self.fmotif, self.rmotif = m["function"], m["role"]
        self.wl = wl_histogram(self.graph)


def prepare(datas: list[dict]) -> list[Prepared]:
    return [Prepared(d) for d in datas]


# ── metrics ───────────────────────────────────────────────────────────────────

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def wl_histogram(G: nx.MultiDiGraph, h: int = 3) -> Counter:
    labels = {n: f"{d['type']}:{d['label']}" for n, d in G.nodes(data=True)}
    hist: Counter = Counter(labels.values())
    for _ in range(h):
        new = {}
        for n in G:
            out = sorted(f"o:{t}>{labels[m]}" for _, m, t in G.out_edges(n, data="type"))
            inn = sorted(f"i:{t}<{labels[m]}" for m, _, t in G.in_edges(n, data="type"))
            sig = labels[n] + "|" + ",".join(out) + "|" + ",".join(inn)
            new[n] = hashlib.md5(sig.encode()).hexdigest()[:12]
        labels = new
        hist.update(labels.values())
    return hist


def wl_similarity(a: Counter, b: Counter) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _incident(G: nx.DiGraph, n) -> Counter:
    c: Counter = Counter()
    for _, _, t in G.out_edges(n, data="type"):
        c[("o", t)] += 1
    for _, _, t in G.in_edges(n, data="type"):
        c[("i", t)] += 1
    return c


def approx_ged(S1: nx.DiGraph, S2: nx.DiGraph) -> float:
    """
    Bipartite approximate graph edit distance (Riesen & Bunke, 2009).

    1. Build a node-assignment problem whose substitution cost is the label
       mismatch plus the mismatch of the incident-edge multisets; solve it with
       the Hungarian algorithm (scipy).
    2. Re-compute the *exact* edit cost of the node mapping found (node
       substitutions/insertions/deletions + edge substitutions/insertions/
       deletions, unit costs).  The result is a deterministic upper bound of
       the true GED.
    """
    from scipy.optimize import linear_sum_assignment

    A, B = list(S1.nodes), list(S2.nodes)
    n, m = len(A), len(B)
    if n == 0 and m == 0:
        return 0.0
    inc1 = {a: _incident(S1, a) for a in A}
    inc2 = {b: _incident(S2, b) for b in B}
    big = 1e6
    C = np.zeros((n + m, n + m))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            lab = 0.0 if (S1.nodes[a]["label"] == S2.nodes[b]["label"]
                          and S1.nodes[a]["type"] == S2.nodes[b]["type"]) else 1.0
            common = sum((inc1[a] & inc2[b]).values())
            loc = (sum(inc1[a].values()) + sum(inc2[b].values()) - 2 * common) / 2.0
            C[i, j] = lab + loc
    for i, a in enumerate(A):                       # deletions
        C[i, m:] = big
        C[i, m + i] = 1.0 + sum(inc1[a].values()) / 2.0
    for j, b in enumerate(B):                       # insertions
        C[n:, j] = big
        C[n + j, j] = 1.0 + sum(inc2[b].values()) / 2.0
    rows, cols = linear_sum_assignment(C)

    mapping = {}                                    # node of S1 -> node of S2 | None
    cost = 0.0
    used = set()
    for r, c in zip(rows, cols):
        if r < n and c < m:
            mapping[A[r]] = B[c]
            used.add(B[c])
            a, b = S1.nodes[A[r]], S2.nodes[B[c]]
            cost += 0.0 if (a["label"] == b["label"] and a["type"] == b["type"]) else 1.0
        elif r < n:
            mapping[A[r]] = None
            cost += 1.0
    cost += sum(1 for b in B if b not in used)      # inserted nodes

    matched2 = set()
    for u, v, t in S1.edges(data="type"):
        mu, mv = mapping.get(u), mapping.get(v)
        if mu is not None and mv is not None and S2.has_edge(mu, mv):
            matched2.add((mu, mv))
            cost += 0.0 if S2.edges[mu, mv]["type"] == t else 1.0
        else:
            cost += 1.0                             # edge deletion
    cost += sum(1 for e in S2.edges if e not in matched2)   # edge insertions
    return float(cost)


def ged_similarity(S1: nx.DiGraph, S2: nx.DiGraph) -> float:
    """1 - GED / bound with bound = max(|V1|,|V2|) + max(|E1|,|E2|), clipped to [0, 1]."""
    d = approx_ged(S1, S2)
    bound = max(S1.number_of_nodes(), S2.number_of_nodes())         + max(S1.number_of_edges(), S2.number_of_edges())
    if bound == 0:
        return 1.0
    return float(min(1.0, max(0.0, 1.0 - d / bound)))


def pair_metrics(p: Prepared, q: Prepared) -> dict[str, float]:
    out = {
        "ged": ged_similarity(p.skel, q.skel),
        "motif": jaccard(p.fmotif, q.fmotif),
        "role": jaccard(p.rmotif, q.rmotif),
        "wl": wl_similarity(p.wl, q.wl),
    }
    out["composite"] = sum(WEIGHTS[m] * out[m] for m in METRICS)
    return out


def similarity_matrices(
    datas: list[dict],
    cache: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    N x N matrices {ged, motif, role, wl, composite}.  `cache` (.npz) stores
    the result so that the (n(n-1)/2) pairwise comparisons run once.
    """
    n = len(datas)
    if cache and Path(cache).exists():
        z = np.load(cache)
        if z["composite"].shape == (n, n):
            return {k: z[k] for k in z.files}

    P = prepare(datas)
    mats = {k: np.eye(n) for k in METRICS + ["composite"]}
    total, done, t0 = n * (n - 1) // 2, 0, time.time()
    for i in range(n):
        for j in range(i + 1, n):
            m = pair_metrics(P[i], P[j])
            for k, v in m.items():
                mats[k][i, j] = mats[k][j, i] = v
            done += 1
        if verbose and (i % 10 == 0 or i == n - 1):
            print(f"  similarity matrix: {done}/{total} pairs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **mats)
    return mats


# ── semantic metric ───────────────────────────────────────────────────────────

class SemanticModel:
    """
    Content similarity of event descriptions.  Uses sentence-transformers
    (all-MiniLM-L6-v2, free & local) when installed, otherwise TF-IDF + LSA
    fitted on a reference corpus.
    """

    def __init__(self, reference_texts: list[str]):
        self.backend = "tfidf-lsa"
        try:                                         # optional dependency
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend = "sentence-transformers"
        except Exception:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                                          min_df=1, ngram_range=(1, 2))
            X = self._tfidf.fit_transform(reference_texts)
            k = max(2, min(50, X.shape[0] - 1, X.shape[1] - 1))
            self._svd = TruncatedSVD(n_components=k, random_state=0).fit(X)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            v = self._st.encode(texts, normalize_embeddings=True)
        else:
            v = self._svd.transform(self._tfidf.transform(texts))
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return np.asarray(v)

    def similarity(self, a: dict, b: dict) -> float:
        va, vb = self.embed([event_text(a), event_text(b)])
        return float(np.clip(va @ vb, 0.0, 1.0))
