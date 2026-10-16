"""
Corpus-level analysis: does narrative structure cluster by culture?

Pipeline (run_corpus_analysis):
  annotations -> event graphs -> 4 structural similarity matrices
  -> hierarchical clustering (k by silhouette) + bootstrap stability
  -> permutation tests: zone separation, cluster/zone association
  -> per-function universality vs cultural specificity (Kruskal-Wallis + BH)
  -> figures (PNG) and metrics.json under result/analysis/<annotator>/
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ..common import vocab as V
from ..research.corpus import Story, list_stories, load_annotations
from ..common.schema import function_sequence
from .similarity import METRICS, similarity_matrices

ZONE_COLORS = {"european": "#4C78A8", "asian": "#E45756", "south_african": "#54A24B"}
DEFAULT_COLOR = "#888888"


# ── distances & clustering ────────────────────────────────────────────────────

def to_distance(sim: np.ndarray) -> np.ndarray:
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    return np.clip((d + d.T) / 2, 0.0, None)


def _agglo(dist: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                   linkage="average").fit_predict(dist)


def select_k(dist: np.ndarray, k_max: int = 8) -> tuple[int, dict[int, float]]:
    scores = {}
    for k in range(2, min(k_max, len(dist) - 1) + 1):
        scores[k] = float(silhouette_score(dist, _agglo(dist, k), metric="precomputed"))
    return max(scores, key=scores.get), scores


def cluster_stability(dist: np.ndarray, k: int, n_boot: int = 200,
                      frac: float = 0.8, seed: int = 0) -> dict:
    """Mean ARI between the full clustering and clusterings of random 80% subsets."""
    rng = np.random.default_rng(seed)
    full = _agglo(dist, k)
    n, m = len(dist), int(round(len(dist) * frac))
    aris = []
    for _ in range(n_boot):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        sub = _agglo(dist[np.ix_(idx, idx)], k)
        aris.append(adjusted_rand_score(full[idx], sub))
    return {"mean_ari": float(np.mean(aris)), "sd_ari": float(np.std(aris)), "n_boot": n_boot}


# ── permutation tests ─────────────────────────────────────────────────────────

def _between_minus_within(dist: np.ndarray, zones: np.ndarray) -> float:
    same = zones[:, None] == zones[None, :]
    off = ~np.eye(len(zones), dtype=bool)
    return float(dist[off & ~same].mean() - dist[off & same].mean())


def zone_separation(dist: np.ndarray, zones: list[str], n_perm: int = 5000, seed: int = 0) -> dict:
    """
    Are stories of the same tradition structurally closer than stories of
    different traditions?  Statistic = mean between-zone distance minus mean
    within-zone distance; p-value by permuting zone labels.
    """
    z = np.array(zones)
    rng = np.random.default_rng(seed)

    def _test(d, labels):
        obs = _between_minus_within(d, labels)
        null = np.array([_between_minus_within(d, rng.permutation(labels))
                         for _ in range(n_perm)])
        return {"observed": obs, "null_mean": float(null.mean()),
                "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm))}

    out = {"overall": _test(dist, z), "pairwise": {}}
    out["silhouette_by_zone"] = float(silhouette_score(dist, z, metric="precomputed"))
    uz = sorted(set(zones))
    for i, a in enumerate(uz):
        for b in uz[i + 1:]:
            idx = np.where((z == a) | (z == b))[0]
            out["pairwise"][f"{a} vs {b}"] = _test(dist[np.ix_(idx, idx)], z[idx])
    return out


def cluster_zone_association(labels: np.ndarray, zones: list[str],
                             n_perm: int = 5000, seed: int = 0) -> dict:
    """ARI between clusters and zones with a permutation p-value."""
    rng = np.random.default_rng(seed)
    z = np.array(zones)
    obs = adjusted_rand_score(z, labels)
    null = np.array([adjusted_rand_score(rng.permutation(z), labels) for _ in range(n_perm)])
    return {"ari": float(obs), "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm))}


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


# ── narrative-function statistics ─────────────────────────────────────────────

def event_share_matrix(datas: list[dict]) -> np.ndarray:
    """n_stories x n_functions: fraction of the story's events carrying each function."""
    X = np.zeros((len(datas), len(V.FUNCTIONS)))
    for i, d in enumerate(datas):
        seq = function_sequence(d)
        for f in seq:
            X[i, V.FUNCTIONS.index(f)] += 1.0 / max(len(seq), 1)
    return X


def function_zone_stats(datas: list[dict], zones: list[str],
                        universal_min_presence: float = 0.5,
                        specific_effect: float = 0.2) -> list[dict]:
    """
    Per function: presence rate and mean event share per zone, Kruskal-Wallis
    test of the event share across zones (BH-corrected) and a label:
      universal      present (>=1 event) in >= `universal_min_presence` of the
                     stories of EVERY zone
      culture-bound  q < .05 and epsilon^2 >= `specific_effect`
      shared         otherwise
    """
    X = event_share_matrix(datas)
    z = np.array(zones)
    uz = sorted(set(zones))
    rows, pvals = [], []
    for j, f in enumerate(V.FUNCTIONS):
        groups = [X[z == g, j] for g in uz]
        try:
            h, p = stats.kruskal(*groups)
        except ValueError:                       # all values identical
            h, p = 0.0, 1.0
        n, k = len(X), len(uz)
        eps2 = max(0.0, (h - k + 1) / (n - k)) if n > k else 0.0
        rows.append({
            "function": f,
            "presence": {g: float((X[z == g, j] > 0).mean()) for g in uz},
            "mean_share": {g: float(X[z == g, j].mean()) for g in uz},
            "kruskal_H": float(h), "p_value": float(p), "epsilon2": float(eps2),
        })
        pvals.append(p)
    q = _bh(np.array(pvals))
    for r, qi in zip(rows, q):
        r["q_value"] = float(qi)
        if min(r["presence"].values()) >= universal_min_presence:
            r["label"] = "universal"
        elif qi < 0.05 and r["epsilon2"] >= specific_effect:
            r["label"] = "culture-bound"
        else:
            r["label"] = "shared"
    return rows


def position_profile(datas: list[dict], zones: list[str]) -> dict[str, np.ndarray]:
    """zone -> (n_functions x N_EVENTS) frequency of function at event position."""
    out = {g: np.zeros((len(V.FUNCTIONS), V.N_EVENTS)) for g in sorted(set(zones))}
    count = Counter(zones)
    for d, g in zip(datas, zones):
        for e in d["EVENTS"]:
            if e["FUNCTION"] and e["ID"] <= V.N_EVENTS:
                out[g][V.FUNCTIONS.index(e["FUNCTION"]), e["ID"] - 1] += 1
    return {g: m / count[g] for g, m in out.items()}


def name_clusters(datas: list[dict], labels: np.ndarray, zones: list[str]) -> dict[int, str]:
    """Label clusters by their over-represented functions and dominant zone."""
    X = event_share_matrix(datas)
    overall = X.mean(axis=0) + 1e-9
    names = {}
    for cid in sorted(set(labels)):
        m = labels == cid
        lift = X[m].mean(axis=0) / overall
        top = [j for j in np.argsort(-lift)[:2] if X[m, j].mean() >= 0.1]
        funcs = "+".join(V.FUNCTION_SHORT[V.FUNCTIONS[j]] for j in top) or "mixed"
        zc = Counter(np.array(zones)[m])
        zone, cnt = zc.most_common(1)[0]
        names[int(cid)] = f"C{cid}: {funcs} | {zone} {100 * cnt / m.sum():.0f}%"
    return names


# ── plots ─────────────────────────────────────────────────────────────────────

def _save(fig, out_dir: Path | None, name: str, show: bool):
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_story_map(sim, names, zones, labels, cluster_names, out_dir=None, show=False,
                   n_links: int = 12):
    """MDS map of the narrative universe: colour = tradition, marker = cluster;
    lines join the most similar story pairs across different traditions."""
    dist = to_distance(sim)
    kw = dict(n_components=2, random_state=0, n_init=4, init="random",
              normalized_stress="auto")
    try:                                   # scikit-learn >= 1.8
        xy = MDS(metric="precomputed", **kw).fit_transform(dist)
    except (TypeError, ValueError):        # older releases
        xy = MDS(dissimilarity="precomputed", **kw).fit_transform(dist)
    fig, ax = plt.subplots(figsize=(13, 9))
    z = np.array(zones)
    pairs = [(sim[i, j], i, j) for i in range(len(z)) for j in range(i + 1, len(z))
             if z[i] != z[j]]
    for s, i, j in sorted(pairs, reverse=True)[:n_links]:
        ax.plot(*zip(xy[i], xy[j]), color="#999", lw=0.8, alpha=0.7, zorder=1)
    markers = "osD^vP*X"
    for cid in sorted(set(labels)):
        for g in sorted(set(zones)):
            m = (labels == cid) & (z == g)
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=130, marker=markers[cid % len(markers)],
                           color=ZONE_COLORS.get(g, DEFAULT_COLOR), edgecolor="k", zorder=3)
    for (x, y), n in zip(xy, names):
        ax.annotate(n[:22], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=6)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in ZONE_COLORS.items()]
    h += [Line2D([], [], marker=markers[c % len(markers)], ls="", color="#666",
                 label=cluster_names[c]) for c in sorted(set(labels))]
    ax.legend(handles=h, fontsize=8, loc="best", title="tradition / cluster")
    ax.set_title("Narrative universe — composite structural distance (MDS)\n"
                 "grey lines: most similar story pairs from different traditions")
    ax.axis("off")
    _save(fig, out_dir, "story_map", show)


def plot_distance_heatmap(sim, names, zones, out_dir=None, show=False):
    order = np.argsort(zones, kind="stable")
    d = to_distance(sim)[np.ix_(order, order)]
    z = np.array(zones)[order]
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(d, cmap="YlOrRd", ax=ax, xticklabels=[names[i][:24] for i in order],
                yticklabels=[names[i][:24] for i in order],
                cbar_kws={"label": "structural distance (1 - composite similarity)"})
    for b in np.where(z[1:] != z[:-1])[0] + 1:
        ax.axhline(b, color="k", lw=2)
        ax.axvline(b, color="k", lw=2)
    ax.tick_params(labelsize=5)
    ax.set_title("Pairwise narrative distance, ordered by tradition (black lines)")
    _save(fig, out_dir, "distance_heatmap", show)


def plot_function_heatmap(rows, out_dir=None, show=False):
    zones = sorted(rows[0]["mean_share"])
    M = np.array([[r["mean_share"][g] for g in zones] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="Blues", ax=ax, xticklabels=zones,
                yticklabels=[f"{r['function']}  [{r['label']}]" + ("*" if r["q_value"] < .05 else "")
                             for r in rows], cbar_kws={"label": "mean share of a story's events"})
    ax.set_title("Narrative functions by tradition  (* = Kruskal-Wallis q < .05)")
    _save(fig, out_dir, "function_by_zone_heatmap", show)


def plot_radar(rows, out_dir=None, show=False):
    zones = sorted(rows[0]["mean_share"])
    ang = np.linspace(0, 2 * np.pi, len(rows), endpoint=False)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for g in zones:
        v = [r["mean_share"][g] for r in rows]
        ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], color=ZONE_COLORS.get(g), lw=2, label=g)
        ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], color=ZONE_COLORS.get(g), alpha=0.12)
    ax.set_xticks(ang)
    ax.set_xticklabels([V.FUNCTION_SHORT[r["function"]] for r in rows], fontsize=9)
    ax.set_title("Narrative-function profile per tradition", y=1.08)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05))
    _save(fig, out_dir, "function_radar", show)


def plot_cluster_breakdown(labels, zones, cluster_names, out_dir=None, show=False):
    uz = sorted(set(zones))
    ids = sorted(set(labels))
    M = np.array([[sum(1 for l, z in zip(labels, zones) if l == c and z == g) for c in ids]
                  for g in uz])
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(ids))
    for g, row in zip(uz, M):
        ax.bar([cluster_names[c] for c in ids], row, bottom=bottom,
               color=ZONE_COLORS.get(g), label=g, edgecolor="white")
        bottom += row
    ax.set_ylabel("stories")
    ax.set_title("Do clusters respect or cross tradition boundaries?")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    _save(fig, out_dir, "cluster_by_zone", show)


def plot_arc_by_zone(datas, zones, out_dir=None, show=False):
    uz = sorted(set(zones))
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.8 / len(uz)
    for i, g in enumerate(uz):
        sub = [d["NARRATIVE_ARC"] for d, z in zip(datas, zones) if z == g]
        c = Counter(sub)
        ax.bar(np.arange(len(V.ARCS)) + i * w, [c.get(a, 0) / len(sub) for a in V.ARCS],
               width=w, color=ZONE_COLORS.get(g), label=g)
    ax.set_xticks(np.arange(len(V.ARCS)) + w * (len(uz) - 1) / 2)
    ax.set_xticklabels(V.ARCS)
    ax.set_ylabel("share of stories")
    ax.set_title("Narrative arc by tradition")
    ax.legend()
    _save(fig, out_dir, "arc_by_zone", show)


def plot_position_profiles(prof, out_dir=None, show=False):
    zs = sorted(prof)
    fig, axes = plt.subplots(1, len(zs), figsize=(5.5 * len(zs), 5))
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, zs):
        sns.heatmap(prof[g], ax=ax, cmap="Purples", vmin=0, vmax=0.6, cbar=ax is axes[-1],
                    xticklabels=range(1, V.N_EVENTS + 1),
                    yticklabels=[V.FUNCTION_SHORT[f] for f in V.FUNCTIONS])
        ax.set_title(g)
        ax.set_xlabel("event position")
    fig.suptitle("Where in the story does each function occur?")
    _save(fig, out_dir, "function_position", show)


# ── orchestration ─────────────────────────────────────────────────────────────

def run_corpus_analysis(
    annotator_tag: str,
    data_dir="./data",
    result_dir="./result",
    n_clusters: int | None = None,
    n_perm: int = 5000,
    show: bool = False,
) -> dict:
    """Full corpus analysis for one annotator.  Returns the metrics dict."""
    if not show and "inline" not in matplotlib.get_backend().lower():
        plt.switch_backend("Agg")

    stories = list_stories(data_dir)
    pairs = load_annotations(stories, result_dir, annotator_tag)
    if len(pairs) < 6:
        raise RuntimeError(f"only {len(pairs)} annotated stories for '{annotator_tag}'")
    names = [s.name for s, _ in pairs]
    zones = [s.zone for s, _ in pairs]
    datas = [d for _, d in pairs]

    out_dir = Path(result_dir) / "analysis" / annotator_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysing {len(datas)} stories annotated by '{annotator_tag}' …")

    mats = similarity_matrices(datas, cache=out_dir / "similarity.npz")
    comp = mats["composite"]
    dist = to_distance(comp)

    k, scan = select_k(dist) if n_clusters is None else (n_clusters, {})
    labels = _agglo(dist, k)
    cnames = name_clusters(datas, labels, zones)
    stab = cluster_stability(dist, k)
    sep = zone_separation(dist, zones, n_perm)
    assoc = cluster_zone_association(labels, zones, n_perm)
    frows = function_zone_stats(datas, zones)

    off = ~np.eye(len(datas), dtype=bool)
    rho = {f"{a}~{b}": float(stats.spearmanr(mats[a][off], mats[b][off])[0])
           for i, a in enumerate(METRICS) for b in METRICS[i + 1:]}

    zarr = np.array(zones)
    cross = sorted(((float(comp[i, j]), names[i], names[j], zones[i], zones[j])
                    for i in range(len(names)) for j in range(i + 1, len(names))
                    if zones[i] != zones[j]), reverse=True)[:10]

    metrics = {
        "annotator": annotator_tag,
        "n_stories": len(datas),
        "stories_per_zone": dict(Counter(zones)),
        "protocol": dict(Counter(d["_meta"].get("protocol", "?") for d in datas)),
        "derived_fields": dict(Counter(f for d in datas for f in d["_meta"].get("derived", []))),
        "mean_similarity": {m: float(mats[m][off].mean()) for m in METRICS + ["composite"]},
        "metric_spearman": rho,
        "k": int(k), "silhouette_scan": scan,
        "cluster_sizes": {cnames[int(c)]: int((labels == c).sum()) for c in sorted(set(labels))},
        "cluster_stability": stab,
        "cluster_zone_association": assoc,
        "zone_separation": sep,
        "mean_within_zone_similarity": {g: float(comp[np.ix_(zarr == g, zarr == g)][
            ~np.eye((zarr == g).sum(), dtype=bool)].mean()) for g in sorted(set(zones))},
        "functions": frows,
        "top_cross_zone_pairs": [
            {"similarity": s, "a": a, "b": b, "zone_a": za, "zone_b": zb}
            for s, a, b, za, zb in cross],
        "arc_by_zone": {g: dict(Counter(d["NARRATIVE_ARC"] for d, z in zip(datas, zones) if z == g))
                        for g in sorted(set(zones))},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figs = out_dir / "figures"
    plot_story_map(comp, names, zones, labels, cnames, figs, show)
    plot_distance_heatmap(comp, names, zones, figs, show)
    plot_function_heatmap(frows, figs, show)
    plot_radar(frows, figs, show)
    plot_cluster_breakdown(labels, zones, cnames, figs, show)
    plot_arc_by_zone(datas, zones, figs, show)
    plot_position_profiles(position_profile(datas, zones), figs, show)

    print_summary(metrics)
    print(f"Saved metrics.json and figures to {out_dir}")
    return metrics


def print_summary(m: dict) -> None:
    sep = m["zone_separation"]
    print("\n=== CORPUS ANALYSIS ===")
    print(f"stories: {m['n_stories']}  per zone: {m['stories_per_zone']}  protocol: {m['protocol']}")
    print(f"mean similarity: " + ", ".join(f"{k}={v:.2f}" for k, v in m["mean_similarity"].items()))
    print(f"clusters: k={m['k']}  stability ARI={m['cluster_stability']['mean_ari']:.2f}"
          f"±{m['cluster_stability']['sd_ari']:.2f}")
    for n, c in m["cluster_sizes"].items():
        print(f"   {n}  ({c} stories)")
    print(f"cluster~zone ARI={m['cluster_zone_association']['ari']:.2f} "
          f"(p={m['cluster_zone_association']['p_value']:.4f})")
    print(f"zone separation (between-within distance): "
          f"{sep['overall']['observed']:.3f}  p={sep['overall']['p_value']:.4f}  "
          f"silhouette-by-zone={sep['silhouette_by_zone']:.2f}")
    for pair, r in sep["pairwise"].items():
        print(f"   {pair}: {r['observed']:.3f}  p={r['p_value']:.4f}")
    print("functions:")
    for r in m["functions"]:
        print(f"   {r['function']:18s} {r['label']:14s} q={r['q_value']:.3f} "
              f"eps2={r['epsilon2']:.2f}  presence={ {g: round(v, 2) for g, v in r['presence'].items()} }")
