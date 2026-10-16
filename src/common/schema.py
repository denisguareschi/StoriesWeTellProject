"""
Annotation schema (v2): validation, repair and migration of LLM output.

Every annotation that enters the pipeline goes through normalize().  It
  * accepts both the legacy v1 layout ("KEY EVENTS", "CULTURAL_CONTEXT", …)
    and the current v2 layout;
  * maps controlled-vocabulary values to their canonical spelling and *reports*
    (never silently drops) every value that had to be repaired;
  * derives the fields that v1 annotations lack (causal links, relations,
    transformation events) with transparent heuristics and flags them in
    `_meta.derived`, so downstream code can tell measured from inferred data.

Canonical v2 layout
-------------------
{
  "SCHEMA_VERSION": 2,
  "NARRATIVE_ARC": str,
  "SETTING_DOMAIN": str,
  "CHARACTERS": [{CHARACTER, ROLE, TRANSFORMATION, TRANSFORMATION_TYPE, CHANGE_EVENT}],
  "LOCATIONS": [str],
  "EVENTS": [{ID, EVENT, FUNCTION, CHARACTERS, LOCATION, CAUSE, EFFECT, CAUSED_BY}],
  "RELATIONS": [{SOURCE, TARGET, TYPE}],
  "NARRATIVE_FUNCTIONS": [str],        # derived from EVENTS, in order of appearance
  "_meta": {...}
}
"""

import copy
import re

from . import vocab as V

SCHEMA_VERSION = 2
MIN_EVENTS = 3


class SchemaError(ValueError):
    """The annotation cannot be repaired into a usable structure."""


_STOP = set(
    "the a an and or of to in on at by for with from that this it its is was were be been "
    "he she they them his her their him who whom which what when where then than as but "
    "not no so very into out up down over under after before again once only own same "
    "will would could should can had has have did does do done being such there here "
    "about against between through during".split()
)

_ARC_ALIASES = {"herojourney": "Hero's Journey", "trickster": "Trickster Tale",
                "origin": "Origin Myth", "cautionary": "Cautionary Tale"}


def _key(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _canon(value, allowed: list[str], aliases: dict | None = None):
    """Case/punctuation-insensitive match of `value` against `allowed`."""
    if not isinstance(value, str):
        return None
    k = _key(value)
    for a in allowed:
        if _key(a) == k:
            return a
    if aliases and k in aliases:
        return aliases[k]
    return None


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in _STOP}


def _as_int_list(value, lo: int, hi: int) -> list[int]:
    out = []
    if isinstance(value, (int, float)):
        value = [value]
    if isinstance(value, list):
        for x in value:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if lo <= i <= hi and i not in out:
                out.append(i)
    return out


def _context_value(d: dict):
    ctx = d.get("SETTING_DOMAIN") or d.get("CULTURAL_CONTEXT") or d.get("CULTURAL CONTEXT")
    if isinstance(ctx, dict):
        ctx = ctx.get("CULTURAL_CONTEXT") or ctx.get("CULTURAL CONTEXT") or ""
    return ctx


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize(raw: dict) -> tuple[dict, list[str]]:
    """
    Repair `raw` into canonical v2.  Returns (data, issues).
    Raises SchemaError if the annotation has too little structure to use.
    """
    if not isinstance(raw, dict):
        raise SchemaError("annotation is not a JSON object")

    issues: list[str] = []
    meta = copy.deepcopy(raw.get("_meta", {})) if isinstance(raw.get("_meta"), dict) else {}
    derived: list[str] = list(meta.get("derived", []))
    legacy = raw.get("SCHEMA_VERSION") != SCHEMA_VERSION
    if legacy:
        meta.setdefault("protocol", "v1-two-step-abstract")

    # ── story-level labels ───────────────────────────────────────────────────
    arc = _canon(raw.get("NARRATIVE_ARC"), V.ARCS, _ARC_ALIASES)
    if arc is None:
        issues.append(f"NARRATIVE_ARC invalid: {raw.get('NARRATIVE_ARC')!r}")
    domain = _canon(_context_value(raw), V.SETTING_DOMAINS)
    if domain is None:
        issues.append(f"SETTING_DOMAIN invalid: {_context_value(raw)!r}")

    # ── characters ───────────────────────────────────────────────────────────
    characters: list[dict] = []
    seen: set[str] = set()
    for c in raw.get("CHARACTERS", []) or []:
        if isinstance(c, str):
            c = {"CHARACTER": c}
        if not isinstance(c, dict):
            continue
        name = str(c.get("CHARACTER") or c.get("NAME") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        role = _canon(c.get("ROLE"), V.ROLES)
        if role is None:
            issues.append(f"ROLE invalid for {name!r}: {c.get('ROLE')!r} -> SECONDARY")
            role = "SECONDARY"
        ttype = _canon(c.get("TRANSFORMATION_TYPE"), V.TRANSFORMATION_TYPES)
        characters.append({
            "CHARACTER": name,
            "ROLE": role,
            "TRANSFORMATION": str(c.get("TRANSFORMATION") or "").strip(),
            "TRANSFORMATION_TYPE": ttype,      # None = unspecified (legacy)
            "CHANGE_EVENT": c.get("CHANGE_EVENT"),
        })
    if not characters:
        raise SchemaError("no characters")
    if not any(c["ROLE"] == "PROTAGONIST" for c in characters):
        issues.append("no PROTAGONIST: first character promoted")
        characters[0]["ROLE"] = "PROTAGONIST"

    # ── locations ────────────────────────────────────────────────────────────
    locations: list[str] = []
    for loc in raw.get("LOCATIONS", []) or []:
        name = loc.get("LOCATION") if isinstance(loc, dict) else loc
        name = str(name or "").strip()
        if name and name.lower() not in {l.lower() for l in locations}:
            locations.append(name)

    # ── events ───────────────────────────────────────────────────────────────
    raw_events = raw.get("EVENTS") or raw.get("KEY EVENTS") or []
    if not isinstance(raw_events, list) or len(raw_events) < MIN_EVENTS:
        raise SchemaError(f"fewer than {MIN_EVENTS} events")
    if len(raw_events) != V.N_EVENTS:
        issues.append(f"{len(raw_events)} events instead of {V.N_EVENTS}")
    raw_events = raw_events[: V.N_EVENTS]

    name_lookup = {c["CHARACTER"].lower(): c["CHARACTER"] for c in characters}

    def _resolve_char(n: str) -> str:
        n = str(n).strip()
        low = n.lower()
        if low in name_lookup:
            return name_lookup[low]
        for k, v in name_lookup.items():          # substring / first-name match
            if low and (low in k or k in low):
                return v
        issues.append(f"event participant {n!r} not in CHARACTERS: added as SECONDARY")
        name_lookup[low] = n
        characters.append({"CHARACTER": n, "ROLE": "SECONDARY", "TRANSFORMATION": "",
                           "TRANSFORMATION_TYPE": None, "CHANGE_EVENT": None})
        return n

    events: list[dict] = []
    for i, e in enumerate(raw_events, start=1):
        if not isinstance(e, dict):
            raise SchemaError(f"event {i} is not an object")
        func = _canon(e.get("FUNCTION") or e.get("NARRATIVE_FUNCTION"), V.FUNCTIONS)
        if func is None:
            issues.append(f"FUNCTION invalid in event {i}: "
                          f"{(e.get('FUNCTION') or e.get('NARRATIVE_FUNCTION'))!r}")
        chars = []
        for n in e.get("CHARACTERS", []) or []:
            r = _resolve_char(n)
            if r not in chars:
                chars.append(r)
        loc = str(e.get("LOCATION") or "").strip()
        if loc and loc.lower() not in {l.lower() for l in locations}:
            locations.append(loc)
        events.append({
            "ID": i,
            "EVENT": str(e.get("EVENT") or e.get("KEY EVENT") or "").strip(),
            "FUNCTION": func,
            "CHARACTERS": chars,
            "LOCATION": loc,
            "CAUSE": str(e.get("CAUSE") or "").strip(),
            "EFFECT": str(e.get("EFFECT") or "").strip(),
            "CAUSED_BY": _as_int_list(e.get("CAUSED_BY"), 1, i - 1),
        })
    if not any(e["FUNCTION"] for e in events):
        raise SchemaError("no valid narrative function in any event")
    n_ev = len(events)

    # ── derived fields (legacy / missing) ────────────────────────────────────
    if not any(e["CAUSED_BY"] for e in events) and "CAUSED_BY" not in derived:
        _derive_causal_links(events)
        if any(e["CAUSED_BY"] for e in events):
            derived.append("CAUSED_BY")

    if "RELATIONS" in raw and isinstance(raw["RELATIONS"], list):
        relations = _clean_relations(raw["RELATIONS"], name_lookup, issues)
    else:
        relations = _derive_relations(characters, events)
        if relations:
            derived.append("RELATIONS")

    for c in characters:
        ce = c.get("CHANGE_EVENT")
        c["CHANGE_EVENT"] = ce if isinstance(ce, int) and 1 <= ce <= n_ev else None
    if all(c["CHANGE_EVENT"] is None for c in characters) and legacy:
        # legacy: a character's transformation completes in the last event it joins
        for c in characters:
            last = [e["ID"] for e in events if c["CHARACTER"] in e["CHARACTERS"]]
            if last and c["TRANSFORMATION"]:
                c["CHANGE_EVENT"] = last[-1]
        derived.append("CHANGE_EVENT")

    funcs: list[str] = []
    for e in events:
        if e["FUNCTION"] and e["FUNCTION"] not in funcs:
            funcs.append(e["FUNCTION"])

    meta["derived"] = sorted(set(derived))
    if issues:
        meta["issues"] = issues
    data = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NARRATIVE_ARC": arc,
        "SETTING_DOMAIN": domain,
        "CHARACTERS": characters,
        "LOCATIONS": locations,
        "EVENTS": events,
        "RELATIONS": relations,
        "NARRATIVE_FUNCTIONS": funcs,
        "_meta": meta,
    }
    return data, issues


# ── helpers for derived fields ────────────────────────────────────────────────

def _derive_causal_links(events: list[dict], threshold: float = 0.25) -> None:
    """
    Lexical heuristic: event i is caused by the earlier event j whose EFFECT
    best overlaps event i's CAUSE (overlap coefficient on content words).
    """
    for i, ev in enumerate(events):
        cause = _tokens(ev["CAUSE"])
        if i == 0 or not cause:
            continue
        best, best_score = None, 0.0
        for j in range(i):
            eff = _tokens(events[j]["EFFECT"]) | _tokens(events[j]["EVENT"])
            if not eff:
                continue
            score = len(cause & eff) / min(len(cause), len(eff))
            if score > best_score:
                best, best_score = j, score
        if best is not None and best_score >= threshold:
            ev["CAUSED_BY"] = [events[best]["ID"]]


def _clean_relations(rels: list, name_lookup: dict, issues: list) -> list[dict]:
    out, seen = [], set()
    for r in rels:
        if not isinstance(r, dict):
            continue
        s = name_lookup.get(str(r.get("SOURCE", "")).strip().lower())
        t = name_lookup.get(str(r.get("TARGET", "")).strip().lower())
        ty = _canon(r.get("TYPE"), V.RELATION_TYPES)
        if not (s and t and ty and s != t):
            issues.append(f"relation dropped: {r!r}")
            continue
        if (s, t, ty) not in seen:
            seen.add((s, t, ty))
            out.append({"SOURCE": s, "TARGET": t, "TYPE": ty})
    return out


def _derive_relations(characters: list[dict], events: list[dict]) -> list[dict]:
    """
    Role-based heuristic for legacy data: an ANTAGONIST sharing an event with a
    PROTAGONIST OPPOSES it; a HELPER sharing an event HELPS it.
    """
    role = {c["CHARACTER"]: c["ROLE"] for c in characters}
    out, seen = [], set()
    for e in events:
        for a in e["CHARACTERS"]:
            for b in e["CHARACTERS"]:
                if role.get(b) != "PROTAGONIST" or a == b:
                    continue
                ty = {"ANTAGONIST": "OPPOSES", "HELPER": "HELPS"}.get(role.get(a))
                if ty and (a, b, ty) not in seen:
                    seen.add((a, b, ty))
                    out.append({"SOURCE": a, "TARGET": b, "TYPE": ty})
    return out


# ── Accessors used across the code base ───────────────────────────────────────

def function_sequence(data: dict) -> list[str]:
    """Ordered list of event functions (events without a valid function skipped)."""
    return [e["FUNCTION"] for e in data["EVENTS"] if e["FUNCTION"]]


def event_text(data: dict) -> str:
    """Language-level summary of the story used by the semantic metric."""
    return " ".join(
        f"{e['EVENT']} {e['CAUSE']} {e['EFFECT']}" for e in data["EVENTS"]
    ).strip()
