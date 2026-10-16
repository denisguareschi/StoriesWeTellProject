"""
Narrative annotator: text -> structured event graph description (schema v2).

Protocol "v2-sampled-single-pass" (identical for EVERY text, any length/culture):
  1. representative_excerpt(): short texts are used whole; long texts are
     reduced to three windows (beginning / middle / end) of equal size, so the
     annotator sees the real arc of the work and not just its opening.
  2. One annotation call with controlled vocabularies, operational definitions
     of the narrative functions and explicit instructions NOT to use outside
     knowledge of the work (limits memorisation of famous novels).
  3. normalize(): schema validation/repair; on failure the call is retried
     (up to `max_attempts`) before the story is reported as failed.

No title or author is ever sent to the model.
"""

import re

from ..common import vocab as V
from ..common.llm_client import ModelSpec, extract_json, generate, parse_spec
from ..common.schema import SchemaError, normalize

PROTOCOL = "v2-sampled-single-pass"
EXCERPT_BUDGET = 12_000          # characters sent to the annotator (~3k tokens)


# ── text sampling ─────────────────────────────────────────────────────────────

def _snap(text: str, pos: int, forward: bool) -> int:
    """Move `pos` to the nearest paragraph / sentence boundary (within 400 chars)."""
    window = 400
    if forward:
        seg = text[pos:pos + window]
        m = re.search(r"\n\s*\n|(?<=[.!?])\s", seg)
        return pos + m.end() if m else pos
    seg = text[max(0, pos - window):pos]
    ms = list(re.finditer(r"\n\s*\n|(?<=[.!?])\s", seg))
    return max(0, pos - window) + ms[-1].end() if ms else pos


def representative_excerpt(text: str, budget: int = EXCERPT_BUDGET) -> str:
    """Whole text if short; otherwise beginning / middle / end windows."""
    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if len(text) <= budget:
        return text
    w = budget // 3
    starts = [0, len(text) // 2 - w // 2, len(text) - w]
    parts = []
    for label, s in zip(("BEGINNING", "MIDDLE", "END"), starts):
        a = _snap(text, s, forward=True) if s > 0 else 0
        b = _snap(text, min(len(text), a + w), forward=False) if a + w < len(text) else len(text)
        parts.append(f"[{label} OF THE WORK]\n{text[a:b].strip()}")
    return "\n\n[...]\n\n".join(parts)


# ── prompt ────────────────────────────────────────────────────────────────────

def _vocab_block() -> str:
    gloss = "\n".join(f"  {f}: {V.FUNCTION_GLOSS[f]}" for f in V.FUNCTIONS)
    return f"""CONTROLLED VOCABULARIES (use ONLY these exact strings)
NARRATIVE_ARC       : {" | ".join(V.ARCS)}
FUNCTION            : {" | ".join(V.FUNCTIONS)}
{gloss}
ROLE                : {" | ".join(V.ROLES)}
SETTING_DOMAIN      : {" | ".join(V.SETTING_DOMAINS)}
RELATION TYPE       : {" | ".join(V.RELATION_TYPES)}
TRANSFORMATION_TYPE : {" | ".join(V.TRANSFORMATION_TYPES)}"""


def build_annotation_prompt(excerpt: str) -> str:
    return f"""### TASK: ANNOTATE
You are a computational narratologist. Convert the story below into a structured
event graph. Annotate ONLY what the text supports. The text may be a sample
(beginning / middle / end windows) of a longer work: reconstruct the arc from
the windows and do NOT use outside knowledge of the work, its title or author.

{_vocab_block()}

OUTPUT: one valid JSON object, nothing else (no markdown fences, no comments):
{{
  "NARRATIVE_ARC": "<one ARC value>",
  "SETTING_DOMAIN": "<one SETTING_DOMAIN value>",
  "CHARACTERS": [
    {{"CHARACTER": "<name as in the text>", "ROLE": "<ROLE>",
      "TRANSFORMATION": "<max 10 words: how the character changes>",
      "TRANSFORMATION_TYPE": "<TRANSFORMATION_TYPE>",
      "CHANGE_EVENT": <ID of the event where the change happens, or null>}}
  ],
  "LOCATIONS": ["<place name>"],
  "EVENTS": [
    {{"ID": 1, "EVENT": "<one sentence>", "FUNCTION": "<FUNCTION>",
      "CHARACTERS": ["<names>"], "LOCATION": "<place name>",
      "CAUSE": "<what triggered it; 'Story opening' for event 1>",
      "EFFECT": "<its direct consequence>",
      "CAUSED_BY": [<IDs of EARLIER events that causally lead to this one>]}}
  ],
  "RELATIONS": [
    {{"SOURCE": "<character>", "TARGET": "<character>", "TYPE": "<RELATION TYPE>"}}
  ]
}}

RULES
- 1 to 6 CHARACTERS, protagonist first; the protagonist always has ROLE PROTAGONIST.
- EXACTLY {V.N_EVENTS} EVENTS in chronological order, IDs 1..{V.N_EVENTS}; pick the
  {V.N_EVENTS} events that carry the causal skeleton of the story.
- Every event lists characters from CHARACTERS and a place from LOCATIONS.
- CAUSED_BY contains only IDs smaller than the event's own ID (empty for event 1
  and for events not caused by an earlier key event).
- RELATIONS lists the 1-5 most important character-to-character relations
  (e.g. antagonist OPPOSES protagonist, helper HELPS protagonist).
- Use each FUNCTION for the single event it best describes; the same function
  may appear more than once.

### STORY
{excerpt}
"""


# ── annotation ────────────────────────────────────────────────────────────────

def annotate_text(
    text: str,
    spec: "str | ModelSpec",
    *,
    temperature: float = 0.0,
    seed: int = 0,
    max_attempts: int = 3,
) -> dict:
    """
    Annotate one text.  Returns a canonical v2 dict (with `_meta`).
    Raises SchemaError if every attempt produced an unusable annotation.
    """
    spec = parse_spec(spec)
    excerpt = representative_excerpt(text)
    prompt = build_annotation_prompt(excerpt)

    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        reply = generate(
            prompt, spec,
            temperature=temperature if attempt == 1 else max(temperature, 0.3),
            seed=seed + attempt - 1,
            json_mode=True,
        )
        try:
            data, issues = normalize(extract_json(reply))
        except (ValueError, SchemaError) as exc:      # JSONDecodeError ⊂ ValueError
            last_err = exc
            continue
        data["_meta"].update({
            "protocol": PROTOCOL,
            "model": str(spec),
            "temperature": temperature,
            "seed": seed,
            "attempts": attempt,
            "source_chars": len(text),
            "excerpt_chars": len(excerpt),
        })
        return data
    raise SchemaError(f"{max_attempts} attempts failed: {last_err}")
