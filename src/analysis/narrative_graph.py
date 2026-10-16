"""
Narrative event graph.

Nodes
  CHARACTER  label = ROLE                 (PROTAGONIST, ANTAGONIST, …)
  EVENT      label = narrative FUNCTION   (QUEST, BETRAYAL, …), attr order = 1..5
  LOCATION   label = "LOCATION"

Edges (MultiDiGraph, attribute `type`)
  precedes        EVENT_i  -> EVENT_i+1     temporal order
  causes          EVENT_j  -> EVENT_i       causal link (CAUSED_BY)
  participates    CHARACTER -> EVENT
  takes_place_at  EVENT -> LOCATION
  transforms_in   CHARACTER -> EVENT        the event where its transformation
                                            arc is completed (attr ttype)
  OPPOSES | HELPS | DECEIVES | BONDS_WITH   CHARACTER -> CHARACTER relations
"""

import networkx as nx

from ..common.schema import function_sequence

EDGE_TYPES = ["precedes", "causes", "participates", "takes_place_at", "transforms_in"]


def build_graph(data: dict) -> nx.MultiDiGraph:
    """Canonical v2 annotation -> labelled event graph."""
    G = nx.MultiDiGraph()
    role_of = {}

    for c in data["CHARACTERS"]:
        n = f"C:{c['CHARACTER']}"
        role_of[c["CHARACTER"]] = n
        G.add_node(n, type="CHARACTER", label=c["ROLE"], name=c["CHARACTER"],
                   ttype=c.get("TRANSFORMATION_TYPE"))
    for loc in data["LOCATIONS"]:
        G.add_node(f"L:{loc}", type="LOCATION", label="LOCATION", name=loc)

    events = data["EVENTS"]
    for e in events:
        G.add_node(f"E{e['ID']}", type="EVENT", label=e["FUNCTION"] or "NONE",
                   name=e["EVENT"], order=e["ID"])

    for a, b in zip(events, events[1:]):
        G.add_edge(f"E{a['ID']}", f"E{b['ID']}", type="precedes")
    for e in events:
        for j in e["CAUSED_BY"]:
            G.add_edge(f"E{j}", f"E{e['ID']}", type="causes")
        for ch in e["CHARACTERS"]:
            if ch in role_of:
                G.add_edge(role_of[ch], f"E{e['ID']}", type="participates")
        loc = f"L:{e['LOCATION']}"
        if e["LOCATION"] and loc in G:
            G.add_edge(f"E{e['ID']}", loc, type="takes_place_at")

    for c in data["CHARACTERS"]:
        ce = c.get("CHANGE_EVENT")
        if ce:
            G.add_edge(role_of[c["CHARACTER"]], f"E{ce}", type="transforms_in",
                       ttype=c.get("TRANSFORMATION_TYPE"))
    for r in data["RELATIONS"]:
        if r["SOURCE"] in role_of and r["TARGET"] in role_of:
            G.add_edge(role_of[r["SOURCE"]], role_of[r["TARGET"]], type=r["TYPE"])
    return G


def skeleton(G: nx.MultiDiGraph) -> nx.DiGraph:
    """
    Simple DiGraph without locations (names carry no cross-story signal) and
    with parallel edges merged ("participates+transforms_in").  Used for the
    graph edit distance, which needs a simple labelled graph.
    """
    S = nx.DiGraph()
    for n, d in G.nodes(data=True):
        if d["type"] != "LOCATION":
            S.add_node(n, label=d["label"], type=d["type"])
    merged: dict[tuple, set] = {}
    for u, v, t in G.edges(data="type"):
        if u in S and v in S:
            merged.setdefault((u, v), set()).add(t)
    for (u, v), ts in merged.items():
        S.add_edge(u, v, type="+".join(sorted(ts)))
    return S


def motif_sets(data: dict) -> dict[str, set]:
    """
    Motif inventories used by the overlap metrics.

    function motifs : unigrams, temporal bigrams and causal pairs of narrative
                      functions along the event chain
    role motifs     : (role, function) pairs and (role, relation, role) triples
                      = semantic-role alignment features
    """
    events = data["EVENTS"]
    fseq = function_sequence(data)
    by_id = {e["ID"]: e for e in events}
    role = {c["CHARACTER"]: c["ROLE"] for c in data["CHARACTERS"]}

    fm: set = {("F", f) for f in fseq}
    fm |= {("FF", a, b) for a, b in zip(fseq, fseq[1:])}
    for e in events:
        for j in e["CAUSED_BY"]:
            if e["FUNCTION"] and by_id[j]["FUNCTION"]:
                fm.add(("CAUSE", by_id[j]["FUNCTION"], e["FUNCTION"]))

    rm: set = set()
    for e in events:
        if not e["FUNCTION"]:
            continue
        for ch in e["CHARACTERS"]:
            rm.add(("RF", role.get(ch, "SECONDARY"), e["FUNCTION"]))
    for r in data["RELATIONS"]:
        rm.add(("REL", role.get(r["SOURCE"], "SECONDARY"), r["TYPE"],
                role.get(r["TARGET"], "SECONDARY")))
    for c in data["CHARACTERS"]:
        if c.get("TRANSFORMATION_TYPE") and c["TRANSFORMATION_TYPE"] != "NONE":
            rm.add(("ARC", c["ROLE"], c["TRANSFORMATION_TYPE"]))
    return {"function": fm, "role": rm}


# ── plotting ──────────────────────────────────────────────────────────────────

_COLORS = {"CHARACTER": "#A2D2FF", "EVENT": "#FFD166", "LOCATION": "#B7E4C7"}
_EDGE_STYLE = {
    "precedes":       dict(color="#555555", style="solid", rad=0.0),
    "causes":         dict(color="#C0392B", style="solid", rad=0.35),
    "participates":   dict(color="#9AA5B1", style="dotted", rad=0.0),
    "takes_place_at": dict(color="#52B788", style="dashed", rad=0.0),
    "transforms_in":  dict(color="#8E44AD", style="dashdot", rad=0.0),
    "OPPOSES":        dict(color="#E67E22", style="solid", rad=0.12),
    "HELPS":          dict(color="#2E86C1", style="solid", rad=0.12),
    "DECEIVES":       dict(color="#B03A2E", style="dashed", rad=0.12),
    "BONDS_WITH":     dict(color="#117A65", style="solid", rad=0.12),
}


def layout(G: nx.MultiDiGraph) -> dict:
    """Events on a left-to-right line, characters above, locations below."""
    pos = {}
    ev = sorted([n for n, d in G.nodes(data=True) if d["type"] == "EVENT"],
                key=lambda n: G.nodes[n]["order"])
    for i, n in enumerate(ev):
        pos[n] = (i * 2.0, 0.0)
    for kind, y in (("CHARACTER", 2.0), ("LOCATION", -2.0)):
        nodes = [n for n, d in G.nodes(data=True) if d["type"] == kind]
        xs = [i * 2.0 * max(len(ev) - 1, 1) / max(len(nodes) - 1, 1) for i in range(len(nodes))]
        for n, x in zip(nodes, xs if len(nodes) > 1 else [(len(ev) - 1)]):
            pos[n] = (x, y)
    return pos


def plot_graph(G: nx.MultiDiGraph, title: str = "", path=None, show: bool = False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(14, 7))
    pos = layout(G)
    for kind, shape, size in (("CHARACTER", "o", 2600), ("EVENT", "s", 2600), ("LOCATION", "h", 2200)):
        nodes = [n for n, d in G.nodes(data=True) if d["type"] == kind]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=_COLORS[kind],
                               node_shape=shape, node_size=size, ax=ax, edgecolors="#333")
    for t, st in _EDGE_STYLE.items():
        edges = [(u, v) for u, v, tt in G.edges(data="type") if tt == t]
        if edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, edge_color=st["color"], style=st["style"],
                connectionstyle=f"arc3,rad={st['rad']}", arrows=True, arrowsize=14,
                node_size=2600, ax=ax, width=1.4)
    labels = {}
    for n, d in G.nodes(data=True):
        if d["type"] == "EVENT":
            labels[n] = f"{d['order']}. {d['label'].replace('_', chr(10))}"
        else:
            nm = d["name"] if len(d["name"]) <= 16 else d["name"][:15] + "…"
            labels[n] = nm + (f"\n[{d['label']}]" if d["type"] == "CHARACTER" else "")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
    handles = [Line2D([0], [0], color=s["color"], linestyle=s["style"], label=t)
               for t, s in _EDGE_STYLE.items() if any(tt == t for _, _, tt in G.edges(data="type"))]
    ax.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.08))
    ax.set_title(title or "Narrative event graph", fontsize=13)
    ax.margins(0.08, 0.2)
    ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
