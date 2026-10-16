"""
Corpus handling: story discovery, annotation cache, batch annotation.

Layout
------
data/<zone>/<story>.txt                       raw texts (zone = european | asian | south_african)
result/annotations/<annotator-tag>/<zone>/<story>.json
                                              one folder per annotator; replicas of the
                                              same annotator live in "<tag>@r<k>"
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..common import vocab as V
from .annotator import annotate_text
from ..common.llm_client import ModelSpec, QuotaExhausted, parse_spec
from ..common.schema import SchemaError, normalize


@dataclass(frozen=True)
class Story:
    name: str          # file stem
    zone: str          # data sub-folder
    path: Path

    @property
    def tradition(self) -> str:
        return V.ZONE_TO_TRADITION.get(self.zone, self.zone)

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8", errors="replace")

    @property
    def n_chars(self) -> int:
        return len(self.text())


def list_stories(data_dir: str | Path = "./data") -> list[Story]:
    data_dir = Path(data_dir)
    return [
        Story(p.stem, p.parent.name, p)
        for p in sorted(data_dir.rglob("*.txt"))
        if p.parent != data_dir
    ]


def find_story(query: str, stories: list[Story]) -> Story:
    """Exact stem match, otherwise unique case-insensitive substring match."""
    exact = [s for s in stories if s.name == query]
    if exact:
        return exact[0]
    part = [s for s in stories if query.lower() in s.name.lower()]
    if len(part) == 1:
        return part[0]
    raise KeyError(
        f"story {query!r}: " + ("no match" if not part else
                                f"ambiguous ({', '.join(s.name for s in part[:5])} …)")
    )


# ── annotation cache ──────────────────────────────────────────────────────────

def annotation_dir(result_dir: str | Path, tag: str, replica: int = 0) -> Path:
    suffix = f"@r{replica}" if replica else ""
    return Path(result_dir) / "annotations" / f"{tag}{suffix}"


def annotation_path(story: Story, result_dir, tag: str, replica: int = 0) -> Path:
    return annotation_dir(result_dir, tag, replica) / story.zone / f"{story.name}.json"


def available_annotators(result_dir: str | Path = "./result") -> dict[str, int]:
    """{annotator folder name: number of annotation files}."""
    root = Path(result_dir) / "annotations"
    if not root.exists():
        return {}
    return {d.name: len(list(d.rglob("*.json"))) for d in sorted(root.iterdir()) if d.is_dir()}


def load_annotation(story: Story, result_dir, tag: str, replica: int = 0) -> dict | None:
    p = annotation_path(story, result_dir, tag, replica)
    if not p.exists():
        return None
    data, _ = normalize(json.loads(p.read_text(encoding="utf-8")))
    return data


def load_annotations(
    stories: list[Story], result_dir, tag: str, replica: int = 0
) -> list[tuple[Story, dict]]:
    """All stories that have an annotation under `tag` (schema-normalised)."""
    out = []
    for s in stories:
        d = load_annotation(s, result_dir, tag, replica)
        if d is not None:
            out.append((s, d))
    return out


def corpus_status(stories: list[Story], result_dir, tag: str, replica: int = 0) -> tuple[int, int]:
    by_zone: dict[str, list[int]] = {}
    for s in stories:
        done = annotation_path(s, result_dir, tag, replica).exists()
        z = by_zone.setdefault(s.zone, [0, 0])
        z[0] += done
        z[1] += 1
    print(f"Annotator '{tag}' (replica {replica}):")
    for z, (d, t) in sorted(by_zone.items()):
        print(f"  {z:15s} {d:3d}/{t:3d}")
    done = sum(v[0] for v in by_zone.values())
    total = sum(v[1] for v in by_zone.values())
    print(f"  {'TOTAL':15s} {done:3d}/{total:3d}")
    return done, total


# ── batch annotation ──────────────────────────────────────────────────────────

def annotate_corpus(
    spec: "str | ModelSpec",
    data_dir="./data",
    result_dir="./result",
    *,
    replica: int = 0,
    temperature: float = 0.0,
    seed: int = 0,
    delay_seconds: float = 0.0,
    max_new: int | None = None,
) -> dict:
    """
    Annotate every un-cached story with `spec`.  Safe to re-run: finished
    stories are skipped, a daily-quota error stops the run cleanly.
    Replicas (replica>=1) use seed = seed + replica and temperature 0.5 (unless
    given) and are stored separately; they measure annotator reliability and
    the noise floor of the cross-cultural experiment.
    """
    spec = parse_spec(spec)
    if replica and temperature == 0.0:
        temperature = 0.5          # replicas must differ from replica 0 to measure noise
    stories = list_stories(data_dir)
    summary = {"processed": 0, "skipped": 0, "failed": 0, "stopped": None}

    for s in stories:
        path = annotation_path(s, result_dir, spec.tag, replica)
        if path.exists():
            summary["skipped"] += 1
            continue
        if max_new is not None and summary["processed"] >= max_new:
            summary["stopped"] = "max_new reached"
            break
        print(f"  [LLM] {s.zone}/{s.name}")
        try:
            data = annotate_text(
                s.text(), spec,
                temperature=temperature, seed=seed + replica,
            )
        except QuotaExhausted as exc:
            summary["stopped"] = f"daily quota exhausted ({str(exc)[:80]})"
            break
        except (SchemaError, Exception) as exc:        # keep going: one bad story ≠ stop
            print(f"  [FAIL] {s.name}: {str(exc)[:160]}")
            summary["failed"] += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        issues = data["_meta"].get("issues", [])
        print(f"  [OK]  saved ({len(issues)} schema repair(s))")
        summary["processed"] += 1
        if delay_seconds:
            time.sleep(delay_seconds)
    return summary
