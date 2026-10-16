"""
Validation of the LLM annotator.  An LLM used as annotator is a measuring
instrument, so it must be checked:

  grounding_report      are the extracted characters really in the text?
  litbank_evaluation    character recall against LitBank gold entities, with a
                        non-LLM spaCy NER baseline on the SAME input
  reliability           agreement between two annotation runs (same model,
                        different seed — or two different models)
  export_manual_sheet / score_manual
                        blind human-annotation sheet and agreement with it
"""

import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ..common import vocab as V
from ..research.annotator import representative_excerpt
from ..research.corpus import Story
from ..common.schema import function_sequence
from .similarity import Prepared, pair_metrics

_TITLES = {"mr", "mrs", "miss", "ms", "dr", "sir", "lady", "lord", "madame", "mademoiselle",
           "the", "of", "de", "van", "von", "st", "captain", "colonel", "old", "young",
           "uncle", "aunt", "mister", "master"}


def _name_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z][A-Za-z'-]+", name.lower())
            if len(t) > 2 and t not in _TITLES}


# ── 1. grounding & repair statistics ──────────────────────────────────────────

def grounding_report(pairs: list[tuple[Story, dict]]) -> dict:
    """
    Per zone: share of CHARACTERS whose name occurs in the source text
    (hallucination check), mean number of schema repairs, derived fields.
    """
    by_zone: dict[str, dict] = {}
    for s, d in pairs:
        text = s.text().lower()
        names = [c["CHARACTER"] for c in d["CHARACTERS"]]
        # a character is grounded if one of its content tokens occurs in the text
        grounded = [any(re.search(rf"\b{re.escape(t)}", text) for t in _name_tokens(n)) or
                    n.lower() in text for n in names]
        z = by_zone.setdefault(s.zone, {"grounded": 0, "chars": 0, "repairs": 0, "n": 0})
        z["grounded"] += sum(grounded)
        z["chars"] += len(names)
        z["repairs"] += len(d["_meta"].get("issues", []))
        z["n"] += 1
    return {
        zone: {
            "grounding_rate": z["grounded"] / max(z["chars"], 1),
            "mean_schema_repairs": z["repairs"] / z["n"],
            "stories": z["n"],
        } for zone, z in by_zone.items()
    }


# ── 2. LitBank gold evaluation ────────────────────────────────────────────────

def _load_litbank_entities(path: Path) -> tuple[list[str], list[str]]:
    """(tokens, layer-0 BIO tags) from a LitBank *_brat.tsv file."""
    toks, tags = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0]:
            toks.append(parts[0])
            tags.append(parts[1])
    return toks, tags


def gold_person_names(path: Path, top_k: int = 5) -> tuple[list[str], str]:
    """
    Most frequent proper-name tokens inside gold PER mentions (+ the plain
    text of the annotated span, used as spaCy input for reference).
    """
    toks, tags = _load_litbank_entities(path)
    spans, cur = [], []
    for t, tag in zip(toks, tags):
        if tag.endswith("PER"):
            if tag.startswith("B-") and cur:
                spans.append(cur)
                cur = []
            cur.append(t)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    c: Counter = Counter()
    for sp in spans:
        for t in sp:
            if t[:1].isupper() and t.lower() not in _TITLES and len(t) > 2:
                c[t.lower()] += 1
    return [w for w, _ in c.most_common(top_k)], " ".join(toks)


def _spacy_person_tokens(text: str) -> set[str]:
    import spacy
    global _NLP
    try:
        _NLP
    except NameError:
        _NLP = spacy.load("en_core_web_sm")
    doc = _NLP(text[:100_000])
    out: set[str] = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            out |= _name_tokens(ent.text)
    return out


def litbank_evaluation(pairs: list[tuple[Story, dict]],
                       gold_dir: str | Path = "./data_gold/litbank_entities",
                       top_k: int = 5, with_spacy: bool = True) -> dict:
    """
    Recall@k of LitBank gold character names, for the LLM annotator and for a
    spaCy NER baseline, restricted to the gold names visible in the excerpt
    that the annotator actually received.
    """
    gold_dir = Path(gold_dir)
    rows = []
    for s, d in pairs:
        gp = gold_dir / f"{s.name}_brat.tsv"
        if not gp.exists():
            continue
        names, _ = gold_person_names(gp, top_k=top_k * 3)
        excerpt = representative_excerpt(s.text()).lower()
        visible = [n for n in names if re.search(rf"\b{re.escape(n)}", excerpt)][:top_k]
        if not visible:
            continue
        llm = set()
        for c in d["CHARACTERS"]:
            llm |= _name_tokens(c["CHARACTER"])
        for e in d["EVENTS"]:
            for c in e["CHARACTERS"]:
                llm |= _name_tokens(c)
        row = {"story": s.name, "k": len(visible),
               "llm_recall": sum(n in llm for n in visible) / len(visible)}
        if with_spacy:
            try:
                sp = _spacy_person_tokens(representative_excerpt(s.text()))
                row["spacy_recall"] = sum(n in sp for n in visible) / len(visible)
            except Exception:
                with_spacy = False
        rows.append(row)
    if not rows:
        return {"n_novels": 0, "note": f"no gold files found in {gold_dir}"}
    out = {"n_novels": len(rows),
           "llm_recall_at_k": float(np.mean([r["llm_recall"] for r in rows])),
           "per_novel": rows}
    if with_spacy and "spacy_recall" in rows[0]:
        out["spacy_recall_at_k"] = float(np.mean([r["spacy_recall"] for r in rows]))
    return out


# ── 3. agreement between two annotation runs ──────────────────────────────────

def reliability(pairs_a: list[tuple[Story, dict]], pairs_b: list[tuple[Story, dict]]) -> dict:
    """
    Agreement between two annotation sets of the same stories (two seeds of one
    model, or two models).  Reports:
      arc_agreement / arc_kappa      single-label agreement
      function_position_agreement    same function at the same event position
      function_kappa                 Cohen's kappa on position-wise functions
      mean_pair_composite            structural similarity of the two annotations
                                     of the SAME story
      retrieval_top1                 how often annotation A of a story is most
                                     similar to annotation B of that same story
                                     (chance = 1/n): is annotator noise smaller
                                     than between-story variation?
    """
    from sklearn.metrics import cohen_kappa_score

    b_map = {s.name: d for s, d in pairs_b}
    common = [(s, d, b_map[s.name]) for s, d in pairs_a if s.name in b_map]
    if len(common) < 3:
        return {"n": len(common), "note": "too few common stories"}
    arcs_a = [a["NARRATIVE_ARC"] or "NONE" for _, a, _ in common]
    arcs_b = [b["NARRATIVE_ARC"] or "NONE" for _, _, b in common]
    fa, fb = [], []
    for _, a, b in common:
        for ea, eb in zip(a["EVENTS"], b["EVENTS"]):
            fa.append(ea["FUNCTION"] or "NONE")
            fb.append(eb["FUNCTION"] or "NONE")
    Pa = [Prepared(a) for _, a, _ in common]
    Pb = [Prepared(b) for _, _, b in common]
    n = len(common)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = pair_metrics(Pa[i], Pb[j])["composite"]
    return {
        "n": n,
        "arc_agreement": float(np.mean([x == y for x, y in zip(arcs_a, arcs_b)])),
        "arc_kappa": float(cohen_kappa_score(arcs_a, arcs_b)),
        "function_position_agreement": float(np.mean([x == y for x, y in zip(fa, fb)])),
        "function_kappa": float(cohen_kappa_score(fa, fb)),
        "mean_pair_composite": float(np.mean(np.diag(S))),
        "mean_nonpair_composite": float((S.sum() - np.trace(S)) / (n * n - n)),
        "retrieval_top1": float(np.mean(S.argmax(axis=1) == np.arange(n))),
        "chance_top1": 1.0 / n,
    }


# ── 4. blind manual annotation ────────────────────────────────────────────────

def export_manual_sheet(stories: list[Story], out_dir: str | Path = "./data_gold/manual",
                        n: int = 12, seed: int = 0) -> Path:
    """
    Write `n` stories (balanced over zones, seeded) as excerpts plus a CSV
    template for a human annotator.  Fill the CSV without looking at any LLM
    output, then call score_manual().
    """
    out_dir = Path(out_dir)
    (out_dir / "excerpts").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    zones = sorted({s.zone for s in stories})
    chosen = []
    for z in zones:
        pool = [s for s in stories if s.zone == z]
        pick = rng.choice(len(pool), size=min(n // len(zones), len(pool)), replace=False)
        chosen += [pool[i] for i in sorted(pick)]
    path = out_dir / "manual_annotation_template.csv"
    header = ["story", "zone", "arc"] + [f"function_{i}" for i in range(1, V.N_EVENTS + 1)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in chosen:
            (out_dir / "excerpts" / f"{s.name}.txt").write_text(
                representative_excerpt(s.text()), encoding="utf-8")
            w.writerow([s.name, s.zone, ""] + [""] * V.N_EVENTS)
    print(f"Wrote {len(chosen)} excerpts to {out_dir/'excerpts'} and template {path}")
    print(f"Allowed arcs: {V.ARCS}\nAllowed functions: {V.FUNCTIONS}")
    return path


def score_manual(csv_path: str | Path, pairs: list[tuple[Story, dict]]) -> dict:
    """Agreement between the filled human sheet and the LLM annotations."""
    from sklearn.metrics import cohen_kappa_score

    ann = {s.name: d for s, d in pairs}
    ha, la, hf, lf = [], [], [], []
    with Path(csv_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = ann.get(row["story"])
            if d is None or not row["arc"].strip():
                continue
            ha.append(row["arc"].strip())
            la.append(d["NARRATIVE_ARC"] or "NONE")
            for i, e in enumerate(d["EVENTS"], start=1):
                h = row.get(f"function_{i}", "").strip().upper()
                if h:
                    hf.append(h)
                    lf.append(e["FUNCTION"] or "NONE")
    if not ha:
        return {"n": 0, "note": "no filled rows"}
    return {
        "n_stories": len(ha),
        "arc_agreement": float(np.mean([a == b for a, b in zip(ha, la)])),
        "function_agreement": float(np.mean([a == b for a, b in zip(hf, lf)])) if hf else None,
        "function_kappa": float(cohen_kappa_score(hf, lf)) if len(set(hf + lf)) > 1 else None,
    }
