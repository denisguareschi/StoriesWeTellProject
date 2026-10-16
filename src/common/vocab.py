"""
Single source of truth for controlled vocabularies, cultural traditions and
the mapping of the project's narrative functions to Propp's functions.

Every other module imports from here: nothing is duplicated.
"""

N_EVENTS = 5   # every story is reduced to exactly 5 key events (comparability)

ARCS = [
    "Hero's Journey", "Tragedy", "Trickster Tale", "Origin Myth", "Cautionary Tale",
]

FUNCTIONS = [
    "QUEST", "SACRIFICE", "TRICKSTER", "TRANSFORMATION", "MORAL_REWARD",
    "BETRAYAL", "HELPER_APPEARS", "RETURN", "CALL_TO_ADVENTURE", "ORDEAL",
    "RESURRECTION",
]

ROLES = [
    "PROTAGONIST", "ANTAGONIST", "DEUTERAGONIST", "HELPER", "TRICKSTER", "SECONDARY",
]

# Narrative setting / social domain of the story.  This is NOT the culture of
# the story: culture is the corpus tradition (see TRADITIONS below).
SETTING_DOMAINS = [
    "Social and Familiar", "Historical and Geographical", "Spiritual and Religious",
    "Professional and Organizational", "Artistic and Creative", "Socio-Health",
]

# Typed relations between characters (edges character -> character).
RELATION_TYPES = ["OPPOSES", "HELPS", "DECEIVES", "BONDS_WITH"]

# How a character changes between the beginning and the end of the story.
TRANSFORMATION_TYPES = ["GROWTH", "FALL", "REVERSAL", "NONE"]

# ── Cultural traditions (the real "culture" axis of the project) ──────────────

TRADITIONS = {
    "european":           "European literary tradition (English-language fiction)",
    "east_asian":         "East Asian (Japanese) folk tradition",
    "southern_african":   "Southern African oral tradition",
    "indigenous_american": "Indigenous North American oral tradition",
}

# data/<zone>/ folder  ->  tradition key
ZONE_TO_TRADITION = {
    "european":     "european",
    "asian":        "east_asian",
    "south_african": "southern_african",
}

# ── Short labels for charts ───────────────────────────────────────────────────

FUNCTION_SHORT = {
    "QUEST": "Quest", "SACRIFICE": "Sacrifice", "TRICKSTER": "Trickster",
    "TRANSFORMATION": "Transform", "MORAL_REWARD": "Reward", "BETRAYAL": "Betrayal",
    "HELPER_APPEARS": "Helper", "RETURN": "Return", "CALL_TO_ADVENTURE": "Call",
    "ORDEAL": "Ordeal", "RESURRECTION": "Rebirth",
}

# ── Operational definitions given to the annotator (improves reliability) ────

FUNCTION_GLOSS = {
    "CALL_TO_ADVENTURE": "a lack, danger, villainy or invitation sets the story in motion",
    "QUEST":             "the hero departs or actively pursues a goal",
    "HELPER_APPEARS":    "a helper, donor or magical aid intervenes for the hero",
    "ORDEAL":            "a struggle, trial, difficult task or confrontation with the antagonist",
    "TRICKSTER":         "success or harm obtained through deception, cunning or a trick",
    "BETRAYAL":          "a character breaks trust, harms an ally or acts treacherously",
    "SACRIFICE":         "a character gives up something valuable (life, goods, freedom) for others",
    "TRANSFORMATION":    "a character or object changes nature, form or status",
    "RESURRECTION":      "a character is restored, reborn or saved from apparent death or ruin",
    "RETURN":            "the hero returns home or to the starting situation",
    "MORAL_REWARD":      "virtue is rewarded or vice punished; the moral order is restored",
}

# Documented alignment with Propp's 31 functions (Valls-Vargas et al., 2016)
# and Campbell's monomyth; used in the report, not in the computation.
PROPP_MAPPING = {
    "CALL_TO_ADVENTURE": "Propp 6-10 (villainy / lack, mediation, beginning counteraction)",
    "QUEST":             "Propp 11 (departure)",
    "HELPER_APPEARS":    "Propp 12-14 (donor test, hero's reaction, receipt of magical agent)",
    "ORDEAL":            "Propp 16, 25 (struggle, difficult task)",
    "TRICKSTER":         "Propp 6-7 (trickery, complicity)",
    "BETRAYAL":          "Propp 8 (villainy)",
    "SACRIFICE":         "no direct Propp equivalent (Campbell: atonement / apotheosis)",
    "TRANSFORMATION":    "Propp 29 (transfiguration)",
    "RESURRECTION":      "Campbell: resurrection (Propp 19 liquidation of lack)",
    "RETURN":            "Propp 20 (return)",
    "MORAL_REWARD":      "Propp 30-31 (punishment, wedding / reward)",
}
