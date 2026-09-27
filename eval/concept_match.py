"""Deterministic, negation-aware presence tests shared by the scorer and the label scripts.

Three layers, all auditable:
1. lexical: the abbreviation-expanded finding name is in the text, or >= 70% of its content tokens
   are (eval.semantic_match.phrase_in_text), and the mention is not negated
   (eval.value_match.negated_mention);
2. value polarity: a lab reported as a value with a direction matching an interpreted finding
   (eval.value_match.value_polarity_match);
3. concept: the finding's ontology concept ids (eval.imaging_concepts.ConceptExtractor: graph names,
   SNOMED descriptions, curated synonyms) all occur in the text with negated clauses removed.

`mentioned_polarity` reports whether a finding is stated as present, stated as absent, or not
mentioned, which the content-based retrieval grading needs. `grounded_precision`,
`hallucination_rate_one` and `length_factor` are the precision-side terms of the summarization reward.

Stage 3 of audit/ROADMAP.md; replaces the recall-only, negation-blind whole-patient matcher (F§13).
"""

from __future__ import annotations

import re
from functools import lru_cache

from eval import semantic_match, value_match

SUMMARY_WORD_BUDGET = 350
"""A 5-10 sentence clinical summary. Longer output is discounted linearly; a chart dump (thousands of
words) cannot score by pasting."""

_CLAUSE_SPLIT = re.compile(r"(?<=[.;!?])\s+|\s+(?:but|however|although|whereas)\s+", re.I)
_NEG = tuple(c.strip() for c in value_match.NEG_CUES)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class Phrase:
    """A finding name with its normalization precomputed (for scoring many texts)."""

    __slots__ = ("raw", "norm", "tokens")

    def __init__(self, raw: str):
        self.raw = raw or ""
        self.norm = semantic_match._normalized_phrase(self.raw)
        self.tokens = semantic_match._content_tokens(self.raw)


class TextIndex:
    """A text with its normalization precomputed (for scoring many phrases against it)."""

    __slots__ = ("raw", "low", "norm", "tokens")

    def __init__(self, raw: str):
        self.raw = raw or ""
        self.low = self.raw.lower()
        self.norm = semantic_match._normalized_phrase(self.raw)
        self.tokens = semantic_match._content_tokens(self.raw)


def lexical_hit(p: Phrase, t: TextIndex, coverage: float = semantic_match._PHRASE_COVERAGE) -> bool:
    """Same rule as semantic_match.phrase_in_text, on precomputed objects."""
    if not p.norm or not t.norm:
        return False
    if p.norm in t.norm:
        return True
    if not p.tokens:
        return False
    return len(p.tokens & t.tokens) / len(p.tokens) >= coverage


def mentioned_polarity(p: Phrase, t: TextIndex) -> str | None:
    """'present' if the finding is stated and not only negated, 'absent' if stated only in negated
    contexts, None if not mentioned. A value-polarity match counts as 'present'."""
    if lexical_hit(p, t):
        return "absent" if value_match.negated_mention(p.raw, t.raw) else "present"
    if value_match.value_polarity_match(p.raw, t.raw):
        return "present"
    return None


def unnegated_text(text: str) -> str:
    """The text with clauses that carry a negation cue removed (for concept extraction)."""
    keep = []
    for clause in _CLAUSE_SPLIT.split(text or ""):
        low = clause.lower()
        if not any(c in low for c in _NEG):
            keep.append(clause)
    return " ".join(keep)


@lru_cache(maxsize=4096)
def _concepts_cached(extractor_id: int, text: str) -> frozenset[str]:
    return frozenset(_EXTRACTORS[extractor_id].concepts(text))


_EXTRACTORS: dict[int, object] = {}


def concepts(text: str, extractor) -> frozenset[str]:
    """Concept ids in `text` (cached per extractor and text)."""
    _EXTRACTORS.setdefault(id(extractor), extractor)
    return _concepts_cached(id(extractor), text or "")


def present(name: str, text: str, extractor=None, text_index: TextIndex | None = None,
            text_concepts: frozenset[str] | None = None) -> bool:
    """Is the finding `name` stated as present in `text`? Lexical (not negated) or value-polarity,
    else all of its concepts occur in the un-negated text."""
    if not name or not text:
        return False
    t = text_index or TextIndex(text)
    pol = mentioned_polarity(Phrase(name), t)
    if pol == "present":
        return True
    if pol == "absent" or extractor is None:
        return False
    cn = concepts(name, extractor)
    if not cn:
        return False
    tc = text_concepts if text_concepts is not None else concepts(unnegated_text(text), extractor)
    return cn <= tc


def count_present(names: list[str], text: str, extractor=None) -> int:
    t = TextIndex(text)
    tc = concepts(unnegated_text(text), extractor) if extractor is not None else None
    return sum(1 for n in names if present(n, text, extractor, t, tc))


class Support:
    """What a summary may state without fabricating: the record's content.

    A concept is supported when one of its surface forms (the extractor's token sets for that
    concept id) is covered by the tokens of the chart text plus the patient's annotated terms
    (finding and diagnosis names of the source questions, the record's normalized content even
    when the note phrases them as values: "glucose 680" for "Severe hyperglycemia"). Curated
    concepts, which have pattern forms, are supported when extracted from those texts.
    """

    def __init__(self, chart_text: str, patient_terms: str | None, extractor):
        from eval.imaging_concepts import ctoks
        self.extractor = extractor
        base = (chart_text or "") + " . " + (patient_terms or "")
        self.tokens = frozenset(extractor._snap(t) for t in ctoks(base))
        self.extracted = concepts(chart_text, extractor) | (concepts(patient_terms, extractor) if patient_terms else frozenset())

    def __contains__(self, cid: str) -> bool:
        if cid in self.extracted:
            return True
        return any(form <= self.tokens for form in _forms_of(self.extractor).get(cid, ()))


_FORMS: dict[int, dict[str, list[frozenset[str]]]] = {}


def _forms_of(extractor) -> dict[str, list[frozenset[str]]]:
    """concept id -> its surface forms (token sets), built once per extractor."""
    key = id(extractor)
    if key not in _FORMS:
        idx: dict[str, list[frozenset[str]]] = {}
        for tk, cid in extractor.forms.items():
            idx.setdefault(extractor.absorb.get(cid, cid), []).append(tk)
        _FORMS[key] = idx
    return _FORMS[key]


def supported_concepts(chart_text: str, patient_terms: str | None, extractor) -> Support:
    return Support(chart_text, patient_terms, extractor)


def grounded_precision(summary: str, chart_text: str, extractor, patient_terms: str | None = None,
                       supported: Support | None = None) -> float | None:
    """Share of the summary's clinical concepts that the record supports (see Support). None when
    the summary mentions no concept."""
    sc = concepts(unnegated_text(summary), extractor)
    if not sc:
        return None
    sup = supported if supported is not None else Support(chart_text, patient_terms, extractor)
    return sum(1 for c in sc if c in sup) / len(sc)


def hallucination_rate_one(summary: str, chart_text: str, extractor, patient_terms: str | None = None,
                           supported: Support | None = None) -> float | None:
    """Share of the summary's sentences that carry clinical concepts none of which the record
    supports. None when no sentence carries a concept."""
    sup = supported if supported is not None else Support(chart_text, patient_terms, extractor)
    flagged = counted = 0
    for sent in _SENT_SPLIT.split((summary or "").strip()):
        sc = concepts(unnegated_text(sent), extractor)
        if not sc:
            continue
        counted += 1
        if not any(c in sup for c in sc):
            flagged += 1
    return flagged / counted if counted else None


def length_factor(text: str, budget_words: int = SUMMARY_WORD_BUDGET) -> float:
    n = len((text or "").split())
    return 1.0 if n <= budget_words else budget_words / n


def harmonic(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0
