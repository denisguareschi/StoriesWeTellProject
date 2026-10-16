"""
Cross-cultural retelling experiment.

Question: when an LLM acts as a "cultural translator", which structural
elements of a story survive, which are distorted, and does the result stay
close to the source tradition, drift toward the target tradition, or hybridise?

Design (every step is cached under result/cross_cultural/)
----------------------------------------------------------
1. Stories: explicit list or a seeded sample of `per_zone` stories per tradition.
2. Retelling: for each story x generator model x target tradition x replica the
   model receives ONLY the story text and the name of the target tradition.
   It is never told the narrative arc, the functions or what to preserve
   (the blind protocol removes the instruction-following circularity).
   The target equal to the story's own tradition is a *paraphrase control*.
3. Annotation: every retelling is annotated blind with the same annotator and
   protocol as the corpus (optionally a different model than the generator, to
   avoid self-annotation bias).
4. Noise floors: the original is re-annotated `noise_replicas` times
   (temperature > 0) -> annotator noise; the paraphrase control gives
   generator + annotator noise.  Drift is only interpreted beyond them.
5. Metrics per retelling vs. its original: structural similarity (ged, motif,
   role, wl, composite), semantic similarity of event descriptions, arc
   preserved, functions preserved / lost / added, role alignment.
6. Outcome per cross-tradition retelling:
     preserved    similarity >= noise/control floor - tolerance
     assimilated  below floor and closer to the target tradition's corpus
     hybrid       below floor and equally close to source and target corpus
     distorted    below floor and not closer to the target tradition
7. Aggregates: drift matrix (source x target), per-function stability with
   bootstrap CI over stories, per-role alignment, per-generator summary and a
   paired Wilcoxon test cross-tradition vs. control.
"""

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..common import vocab as V
from ..research.annotator import annotate_text, representative_excerpt
from ..research.corpus import (Story, annotation_path, list_stories, load_annotation,
                     load_annotations)
from ..common.llm_client import QuotaExhausted, ModelSpec, generate, parse_spec
from ..common.schema import SchemaError, event_text, function_sequence, normalize
from ..analysis.similarity import METRICS, Prepared, SemanticModel, pair_metrics

TRADITION_TO_ZONE = {t: z for z, t in V.ZONE_TO_TRADITION.items()}
PRESERVE_TOLERANCE = 0.05
HYBRID_MARGIN = 0.03
NOISE_TEMPERATURE = 0.5


# ── generation ────────────────────────────────────────────────────────────────

def build_retelling_prompt(excerpt: str, target_key: str) -> str:
    return f"""### TASK: RETELL
You are a storyteller and cultural translator. Retell the story below as it
would be told within this tradition:

TRADITION: {V.TRADITIONS[target_key]}

Adapt names, settings, customs, beliefs, imagery and narrative voice so that the
retelling feels native to that tradition, while remaining recognisable as an
adaptation of the same tale. Write 400-600 words of continuous prose. Output
only the retold story.

### STORY
{excerpt}
"""


def generate_retelling(text: str, target_key: str, spec, *, temperature=0.7, seed=0) -> str:
    prompt = build_retelling_prompt(representative_excerpt(text), target_key)
    return generate(prompt, spec, temperature=temperature, seed=seed)


# ── role alignment (semantic role alignment between two versions) ─────────────

def _role_to_functions(data: dict) -> dict[str, set[str]]:
    role = {c["CHARACTER"]: c["ROLE"] for c in data["CHARACTERS"]}
    out: dict[str, set[str]] = {}
    for e in data["EVENTS"]:
        if not e["FUNCTION"]:
            continue
        for ch in e["CHARACTERS"]:
            out.setdefault(role.get(ch, "SECONDARY"), set()).add(e["FUNCTION"])
    return out


def compute_role_alignment(original: dict, retelling: dict) -> dict[str, dict]:
    """role -> {status: aligned|lost|added, score: Jaccard of function sets}."""
    a, b = _role_to_functions(original), _role_to_functions(retelling)
    res = {}
    for r in set(a) | set(b):
        fa, fb = a.get(r, set()), b.get(r, set())
        if fa and fb:
            res[r] = {"status": "aligned", "score": len(fa & fb) / len(fa | fb)}
        elif fa:
            res[r] = {"status": "lost", "score": 0.0}
        else:
            res[r] = {"status": "added", "score": 0.0}
    return res


# ── experiment ────────────────────────────────────────────────────────────────

def select_stories(stories: list[Story], names: list[str] | None,
                   per_zone: int, seed: int) -> list[Story]:
    if names:
        out = []
        for n in names:
            m = [s for s in stories if s.name == n] or \
                [s for s in stories if n.lower() in s.name.lower()]
            if not m:
                raise KeyError(f"story {n!r} not found")
            out.append(m[0])
        return out
    rng = np.random.default_rng(seed)
    chosen = []
    for z in sorted({s.zone for s in stories}):
        pool = [s for s in stories if s.zone == z]
        idx = rng.choice(len(pool), size=min(per_zone, len(pool)), replace=False)
        chosen += [pool[i] for i in sorted(idx)]
    return chosen


def _ensure_annotation(story: Story, text: str, annotator: ModelSpec, result_dir,
                       replica: int = 0, force: bool = False) -> dict:
    """Corpus annotation of the ORIGINAL story (cached, shared with the corpus)."""
    p = annotation_path(story, result_dir, annotator.tag, replica)
    if p.exists() and not force:
        return load_annotation(story, result_dir, annotator.tag, replica)
    data = annotate_text(text, annotator,
                         temperature=NOISE_TEMPERATURE if replica else 0.0,
                         seed=replica)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def run_experiment(
    generators: list[str],
    annotator: str,
    *,
    story_names: list[str] | None = None,
    per_zone: int = 3,
    targets: list[str] | None = None,
    replicas: int = 1,
    noise_replicas: int = 2,
    data_dir="./data",
    result_dir="./result",
    seed: int = 0,
    delay_seconds: float = 0.0,
    corpus_tag: str | None = None,
    force: bool = False,
    show: bool = False,
) -> dict:
    """Run (or resume) the experiment and return the summary dict."""
    if not show and "inline" not in matplotlib.get_backend().lower():
        plt.switch_backend("Agg")
    annotator_spec = parse_spec(annotator)
    gens = [parse_spec(g) for g in generators]
    targets = targets or list(V.TRADITIONS)
    for t in targets:
        if t not in V.TRADITIONS:
            raise ValueError(f"unknown tradition {t!r}; use one of {list(V.TRADITIONS)}")
    result_dir = Path(result_dir)
    all_stories = list_stories(data_dir)
    chosen = select_stories(all_stories, story_names, per_zone, seed)
    cc_root = result_dir / "cross_cultural"

    # reference corpus (for hybrid classification and semantic fit)
    corpus_tag = corpus_tag or annotator_spec.tag
    corpus_pairs = load_annotations(all_stories, result_dir, corpus_tag)

    print(f"=== Cross-cultural experiment: {len(chosen)} stories x {len(gens)} generator(s) "
          f"x {len(targets)} targets x {replicas} replica(s) ===")
    rows: list[dict] = []
    stopped = None
    work: dict[str, dict] = {}

    for story in chosen:
        text = story.text()
        try:
            orig = _ensure_annotation(story, text, annotator_spec, result_dir)
            noise_annots = [_ensure_annotation(story, text, annotator_spec, result_dir, k)
                            for k in range(1, noise_replicas + 1)]
        except QuotaExhausted as exc:
            stopped = f"quota exhausted while annotating originals: {exc}"
            break
        work[story.name] = {"story": story, "orig": orig, "noise": noise_annots}
        story_dir = cc_root / story.name
        story_dir.mkdir(parents=True, exist_ok=True)

        for gen in gens:
            for target in targets:
                for k in range(replicas):
                    stem = f"{gen.tag}__{target}__r{k}"
                    txt_p = story_dir / f"{stem}.txt"
                    json_p = story_dir / f"{stem}.{annotator_spec.tag}.json"
                    try:
                        if not txt_p.exists() or force:
                            print(f"  [GEN] {story.name[:40]} -> {target} ({gen}, r{k})")
                            txt_p.write_text(
                                generate_retelling(text, target, gen, seed=seed + k),
                                encoding="utf-8")
                            if delay_seconds:
                                time.sleep(delay_seconds)
                        if not json_p.exists() or force:
                            data = annotate_text(txt_p.read_text(encoding="utf-8"),
                                                 annotator_spec, seed=seed)
                            json_p.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
                            if delay_seconds:
                                time.sleep(delay_seconds)
                        retold = normalize(json.loads(json_p.read_text(encoding="utf-8")))[0]
                    except QuotaExhausted as exc:
                        stopped = f"daily quota exhausted: {str(exc)[:100]}"
                        break
                    except (SchemaError, Exception) as exc:
                        print(f"  [FAIL] {story.name[:30]} -> {target}: {str(exc)[:120]}")
                        continue
                    rows.append({
                        "story": story.name, "zone": story.zone,
                        "source": story.tradition, "generator": str(gen),
                        "target": target, "control": target == story.tradition,
                        "replica": k, "_data": retold,
                    })
                if stopped:
                    break
            if stopped:
                break
        if stopped:
            break

    if not rows:
        print("No retellings available." + (f" Stopped: {stopped}" if stopped else ""))
        return {"n_retellings": 0, "stopped": stopped}

    summary = _analyse(rows, work, corpus_pairs, annotator_spec.tag, cc_root, show)
    summary["stopped"] = stopped
    return summary


# ── analysis ──────────────────────────────────────────────────────────────────

def _bootstrap_ci(values_by_story: dict[str, list[float]], n_boot: int = 2000, seed: int = 0):
    """Percentile CI of the mean, resampling STORIES (the independent unit)."""
    keys = list(values_by_story)
    if len(keys) < 2:
        m = np.mean([v for vs in values_by_story.values() for v in vs]) if keys else np.nan
        return float(m), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        vals = [v for i in pick for v in values_by_story[keys[i]]]
        means.append(np.mean(vals))
    allv = [v for vs in values_by_story.values() for v in vs]
    return float(np.mean(allv)), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _analyse(rows, work, corpus_pairs, tag, cc_root: Path, show: bool) -> dict:
    # semantic model fitted on the reference corpus + all event texts seen here
    texts = [event_text(d) for _, d in corpus_pairs] + [event_text(r["_data"]) for r in rows]
    texts += [event_text(w["orig"]) for w in work.values()]
    sem = SemanticModel(texts)

    corpus_prep = [(s, Prepared(d)) for s, d in corpus_pairs]

    # noise floors per story
    noise_floor = {}
    for name, w in work.items():
        po = Prepared(w["orig"])
        sims = [pair_metrics(po, Prepared(n))["composite"] for n in w["noise"]]
        if sims:
            noise_floor[name] = float(np.mean(sims))

    # paraphrase control per (story, generator)
    ctrl = defaultdict(list)
    for r in rows:
        if r["control"]:
            po = Prepared(work[r["story"]]["orig"])
            ctrl[(r["story"], r["generator"])].append(
                pair_metrics(po, Prepared(r["_data"]))["composite"])
    control_floor = {k: float(np.mean(v)) for k, v in ctrl.items()}

    for r in rows:
        orig = work[r["story"]]["orig"]
        po, pr = Prepared(orig), Prepared(r["_data"])
        m = pair_metrics(po, pr)
        r.update(m)
        r["semantic"] = sem.similarity(orig, r["_data"])
        fo, fr = set(function_sequence(orig)), set(function_sequence(r["_data"]))
        r["arc_preserved"] = orig["NARRATIVE_ARC"] == r["_data"]["NARRATIVE_ARC"]
        r["functions_preserved"] = sorted(fo & fr)
        r["functions_lost"] = sorted(fo - fr)
        r["functions_added"] = sorted(fr - fo)
        r["role_alignment"] = compute_role_alignment(orig, r["_data"])
        r["noise_floor"] = noise_floor.get(r["story"])
        r["control_floor"] = control_floor.get((r["story"], r["generator"]))
        ref = r["control_floor"] if r["control_floor"] is not None else r["noise_floor"]
        r["reference_floor"] = ref
        r["outcome"] = None
        if not r["control"] and ref is not None:
            if m["composite"] >= ref - PRESERVE_TOLERANCE:
                r["outcome"] = "preserved"
            else:
                src = [pair_metrics(pr, p)["composite"] for s, p in corpus_prep
                       if s.zone == r["zone"] and s.name != r["story"]]
                tz = TRADITION_TO_ZONE.get(r["target"])
                tgt = [pair_metrics(pr, p)["composite"] for s, p in corpus_prep
                       if s.zone == tz] if tz else []
                r["sim_source_zone"] = float(np.mean(src)) if src else None
                r["sim_target_zone"] = float(np.mean(tgt)) if tgt else None
                if tgt and src:
                    diff = r["sim_target_zone"] - r["sim_source_zone"]
                    r["outcome"] = ("assimilated" if diff > HYBRID_MARGIN else
                                    "hybrid" if abs(diff) <= HYBRID_MARGIN else "distorted")
                else:
                    r["outcome"] = "distorted"

    cross = [r for r in rows if not r["control"]]
    ctl = [r for r in rows if r["control"]]

    # per-function stability (cross-tradition retellings)
    fstab = {}
    for f in V.FUNCTIONS:
        kept, added = defaultdict(list), defaultdict(list)
        for r in cross:
            orig_f = set(function_sequence(work[r["story"]]["orig"]))
            new_f = set(function_sequence(r["_data"]))
            if f in orig_f:
                kept[r["story"]].append(float(f in new_f))
            else:
                added[r["story"]].append(float(f in new_f))
        mk, ck = _bootstrap_ci(kept) if kept else (float("nan"), (float("nan"),) * 2)
        ma, ca = _bootstrap_ci(added) if added else (float("nan"), (float("nan"),) * 2)
        fstab[f] = {"retention": mk, "retention_ci95": ck,
                    "n_original_stories": len(kept),
                    "intrusion": ma, "intrusion_ci95": ca}

    # role alignment aggregate
    roles = defaultdict(lambda: Counter())
    rscore = defaultdict(list)
    for r in cross:
        for role, a in r["role_alignment"].items():
            roles[role][a["status"]] += 1
            if a["status"] == "aligned":
                rscore[role].append(a["score"])
    role_summary = {
        role: {"preservation_rate": c["aligned"] / sum(c.values()),
               "mean_jaccard": float(np.mean(rscore[role])) if rscore[role] else 0.0,
               **{k: c[k] for k in ("aligned", "lost", "added")}}
        for role, c in roles.items()
    }

    # drift matrix source x target
    sources = sorted({r["source"] for r in rows})
    targets = [t for t in V.TRADITIONS if any(r["target"] == t for r in rows)]
    drift = {s: {t: [1 - r["composite"] for r in rows if r["source"] == s and r["target"] == t]
                 for t in targets} for s in sources}
    drift_mean = {s: {t: (float(np.mean(v)) if v else None) for t, v in d.items()}
                  for s, d in drift.items()}

    per_gen = {}
    for g in sorted({r["generator"] for r in rows}):
        sub = [r for r in cross if r["generator"] == g]
        c = [r for r in ctl if r["generator"] == g]
        per_gen[g] = {
            "n_cross": len(sub), "n_control": len(c),
            "mean_sim_cross": float(np.mean([r["composite"] for r in sub])) if sub else None,
            "mean_sim_control": float(np.mean([r["composite"] for r in c])) if c else None,
            "mean_semantic_cross": float(np.mean([r["semantic"] for r in sub])) if sub else None,
            "arc_preserved_rate": float(np.mean([r["arc_preserved"] for r in sub])) if sub else None,
            "outcomes": dict(Counter(r["outcome"] for r in sub)),
        }

    # paired test: cross-tradition vs paraphrase control (same story, generator)
    test = None
    pairs_x, pairs_c = [], []
    for (st, g), cv in control_floor.items():
        xs = [r["composite"] for r in cross if r["story"] == st and r["generator"] == g]
        if xs:
            pairs_x.append(float(np.mean(xs)))
            pairs_c.append(cv)
    if len(pairs_x) >= 6:
        from scipy.stats import wilcoxon
        diffs = np.array(pairs_x) - np.array(pairs_c)
        if np.any(diffs != 0):
            w = wilcoxon(pairs_x, pairs_c)
            test = {"n_pairs": len(pairs_x), "statistic": float(w.statistic),
                    "p_value": float(w.pvalue),
                    "mean_diff_cross_minus_control": float(diffs.mean())}
    elif pairs_x:
        test = {"n_pairs": len(pairs_x), "note": "fewer than 6 pairs: no test performed"}

    summary = {
        "annotator": tag,
        "n_stories": len({r["story"] for r in rows}),
        "n_retellings": len(rows), "n_cross": len(cross), "n_control": len(ctl),
        "noise_floor_mean": float(np.mean(list(noise_floor.values()))) if noise_floor else None,
        "control_floor_mean": float(np.mean(list(control_floor.values()))) if control_floor else None,
        "mean_composite_cross": float(np.mean([r["composite"] for r in cross])) if cross else None,
        "mean_semantic_cross": float(np.mean([r["semantic"] for r in cross])) if cross else None,
        "semantic_backend": sem.backend,
        "metrics_mean_cross": {m: float(np.mean([r[m] for r in cross])) for m in METRICS} if cross else {},
        "outcomes": dict(Counter(r["outcome"] for r in cross)),
        "arc_preserved_rate": float(np.mean([r["arc_preserved"] for r in cross])) if cross else None,
        "drift_matrix": drift_mean,
        "function_stability": fstab,
        "role_alignment": role_summary,
        "per_generator": per_gen,
        "cross_vs_control_test": test,
    }

    out = cc_root / "_summary" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "retellings.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["story", "zone", "source", "generator", "target", "control", "replica",
                "composite", *METRICS, "semantic", "arc_preserved", "noise_floor",
                "control_floor", "outcome", "functions_preserved", "functions_lost",
                "functions_added"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{k: ";".join(r[k]) for k in
                                ("functions_preserved", "functions_lost", "functions_added")}})
    _plots(rows, summary, out, show)
    _print(rows, summary)
    return summary


# ── output ────────────────────────────────────────────────────────────────────

def _plots(rows, S, out: Path, show: bool):
    cross = [r for r in rows if not r["control"]]
    # 1. drift matrix
    srcs = list(S["drift_matrix"])
    tgts = list(next(iter(S["drift_matrix"].values())))
    M = np.array([[S["drift_matrix"][s][t] if S["drift_matrix"][s][t] is not None else np.nan
                   for t in tgts] for s in srcs], dtype=float)
    fig, ax = plt.subplots(figsize=(1.8 * len(tgts) + 3, 1.0 * len(srcs) + 2.5))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax,
                xticklabels=[t.replace("_", "\n") for t in tgts],
                yticklabels=[s.replace("_", " ") for s in srcs],
                cbar_kws={"label": "narrative drift (1 - composite similarity)"})
    ax.set_xlabel("target tradition  (diagonal = paraphrase control)")
    ax.set_ylabel("source tradition")
    ax.set_title("Narrative Drift Matrix")
    fig.tight_layout()
    fig.savefig(out / "drift_matrix.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # 2. function stability with CI
    fs = S["function_stability"]
    funcs = [f for f in V.FUNCTIONS if fs[f]["n_original_stories"] > 0]
    if funcs:
        y = np.arange(len(funcs))
        ret = np.array([fs[f]["retention"] for f in funcs])
        lo = ret - np.array([fs[f]["retention_ci95"][0] for f in funcs])
        hi = np.array([fs[f]["retention_ci95"][1] for f in funcs]) - ret
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(y, ret, xerr=[np.nan_to_num(lo), np.nan_to_num(hi)], color="#4C78A8",
                capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{V.FUNCTION_SHORT[f]} (n={fs[f]['n_original_stories']})"
                            for f in funcs])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("retention rate in cross-tradition retellings (95% CI over stories)")
        ax.set_title("Which narrative functions survive cultural translation?")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out / "function_stability.png", dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    # 3. structural vs semantic similarity
    if cross:
        fig, ax = plt.subplots(figsize=(7, 6))
        cols = {"preserved": "#54A24B", "assimilated": "#4C78A8", "hybrid": "#F58518",
                "distorted": "#E45756", None: "#999999"}
        for oc in cols:
            sub = [r for r in cross if r["outcome"] == oc]
            if sub:
                ax.scatter([r["semantic"] for r in sub], [r["composite"] for r in sub],
                           color=cols[oc], label=str(oc), s=50, edgecolor="k")
        ax.set_xlabel("semantic similarity of event descriptions (surface content)")
        ax.set_ylabel("structural similarity (composite)")
        ax.set_title("Structure vs content: what changes in a retelling?")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "structure_vs_semantics.png", dpi=150)
        if show:
            plt.show()
        plt.close(fig)


def _print(rows, S):
    print("\n=== CROSS-CULTURAL SUMMARY ===")
    print(f"retellings: {S['n_retellings']} ({S['n_cross']} cross-tradition, "
          f"{S['n_control']} paraphrase controls) over {S['n_stories']} stories")
    print(f"noise floor (annotator re-runs) : {S['noise_floor_mean']}")
    print(f"control floor (paraphrase)      : {S['control_floor_mean']}")
    print(f"mean structural similarity      : {S['mean_composite_cross']}")
    print(f"mean semantic similarity        : {S['mean_semantic_cross']} ({S['semantic_backend']})")
    print(f"outcomes: {S['outcomes']}   arc preserved: {S['arc_preserved_rate']}")
    if S["cross_vs_control_test"]:
        print(f"cross vs control: {S['cross_vs_control_test']}")
    print("function retention:")
    for f, v in S["function_stability"].items():
        if v["n_original_stories"]:
            print(f"  {f:18s} retention={v['retention']:.2f} "
                  f"CI={tuple(round(x, 2) for x in v['retention_ci95'])} "
                  f"intrusion={v['intrusion']:.2f}")
