"""Deterministic parsing of a finding's `value_text` into FHIR value elements.

The ETL stored each measurement as free text ("24 mEq/L (Normal: 22-29)", "120/78 mmHg", "<0.04 ng/mL")
plus a lossy `value_numeric` (thousands separators dropped: "250,000/mm³" → 250.0). This module reads the
text: number, comparator, unit (mapped to UCUM), blood pressure as systolic/diastolic components, and a
"(normal: low-high unit)" parenthetical as a reference range. Anything it does not understand is kept
verbatim in `note`; it never guesses a unit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# unit token (normalized: lower-case, no spaces) -> (display unit, UCUM code)
_U = {
    "mg/dl": ("mg/dL", "mg/dL"), "g/dl": ("g/dL", "g/dL"), "g/l": ("g/L", "g/L"), "mg/l": ("mg/L", "mg/L"),
    "meq/l": ("mEq/L", "meq/L"), "mmol/l": ("mmol/L", "mmol/L"), "µmol/l": ("µmol/L", "umol/L"), "umol/l": ("µmol/L", "umol/L"),
    "nmol/l": ("nmol/L", "nmol/L"), "pmol/l": ("pmol/L", "pmol/L"), "mosm/kg": ("mOsm/kg", "mosm/kg"), "mosm/l": ("mOsm/L", "mosm/L"),
    "ng/ml": ("ng/mL", "ng/mL"), "ng/dl": ("ng/dL", "ng/dL"), "pg/ml": ("pg/mL", "pg/mL"), "pg/dl": ("pg/dL", "pg/dL"),
    "µg/dl": ("µg/dL", "ug/dL"), "ug/dl": ("µg/dL", "ug/dL"), "mcg/dl": ("µg/dL", "ug/dL"), "µg/l": ("µg/L", "ug/L"), "ug/l": ("µg/L", "ug/L"),
    "µg/ml": ("µg/mL", "ug/mL"), "ug/ml": ("µg/mL", "ug/mL"), "mcg/ml": ("µg/mL", "ug/mL"), "µg/g": ("µg/g", "ug/g"), "ug/g": ("µg/g", "ug/g"),
    "u/l": ("U/L", "U/L"), "iu/l": ("IU/L", "[IU]/L"), "miu/l": ("mIU/L", "m[IU]/L"), "miu/ml": ("mIU/mL", "m[IU]/mL"),
    "µiu/ml": ("µIU/mL", "u[IU]/mL"), "uiu/ml": ("µIU/mL", "u[IU]/mL"), "u/ml": ("U/mL", "U/mL"), "units/l": ("U/L", "U/L"),
    "%": ("%", "%"), "percent": ("%", "%"),
    "/mm³": ("/mm³", "/mm3"), "/mm^3": ("/mm³", "/mm3"), "/mm3": ("/mm³", "/mm3"), "cells/mm³": ("/mm³", "/mm3"), "cells/mm3": ("/mm³", "/mm3"),
    "/µl": ("/µL", "/uL"), "/ul": ("/µL", "/uL"), "/mcl": ("/µL", "/uL"), "cells/µl": ("/µL", "/uL"), "cells/ul": ("/µL", "/uL"),
    "k/µl": ("10³/µL", "10*3/uL"), "k/ul": ("10³/µL", "10*3/uL"), "×10³/µl": ("10³/µL", "10*3/uL"), "x10³/µl": ("10³/µL", "10*3/uL"),
    "×10⁹/l": ("10⁹/L", "10*9/L"), "x10^9/l": ("10⁹/L", "10*9/L"), "m/µl": ("10⁶/µL", "10*6/uL"), "×10⁶/µl": ("10⁶/µL", "10*6/uL"),
    "mmhg": ("mmHg", "mm[Hg]"), "cmh2o": ("cmH2O", "cm[H2O]"), "cmh₂o": ("cmH2O", "cm[H2O]"),
    "°c": ("°C", "Cel"), "ºc": ("°C", "Cel"), "c": ("°C", "Cel"), "celsius": ("°C", "Cel"),
    "°f": ("°F", "[degF]"), "ºf": ("°F", "[degF]"), "f": ("°F", "[degF]"), "fahrenheit": ("°F", "[degF]"),
    "/min": ("/min", "/min"), "bpm": ("/min", "/min"), "beats/min": ("/min", "/min"), "beats/minute": ("/min", "/min"),
    "beatsperminute": ("/min", "/min"), "beats": ("/min", "/min"), "breaths/min": ("/min", "/min"), "breaths": ("/min", "/min"),
    "breathsperminute": ("/min", "/min"), "respirations/min": ("/min", "/min"),
    "kg/m^2": ("kg/m²", "kg/m2"), "kg/m2": ("kg/m²", "kg/m2"), "kg/m²": ("kg/m²", "kg/m2"),
    "pounds": ("lb", "[lb_av]"), "pound": ("lb", "[lb_av]"), "copies/ml": ("copies/mL", "{copies}/mL"), "iu/ml": ("IU/mL", "[IU]/mL"),
    "kg": ("kg", "kg"), "g": ("g", "g"), "mg": ("mg", "mg"), "mcg": ("µg", "ug"), "µg": ("µg", "ug"), "lb": ("lb", "[lb_av]"), "lbs": ("lb", "[lb_av]"),
    "cm": ("cm", "cm"), "mm": ("mm", "mm"), "m": ("m", "m"), "in": ("in", "[in_i]"), "inches": ("in", "[in_i]"),
    "ml": ("mL", "mL"), "l": ("L", "L"), "dl": ("dL", "dL"), "ml/min": ("mL/min", "mL/min"), "ml/min/1.73m²": ("mL/min/1.73m²", "mL/min/{1.73_m2}"),
    "ml/min/1.73m2": ("mL/min/1.73m²", "mL/min/{1.73_m2}"), "ml/kg": ("mL/kg", "mL/kg"), "ml/hr": ("mL/h", "mL/h"), "ml/h": ("mL/h", "mL/h"), "l/min": ("L/min", "L/min"),
    "mm/hr": ("mm/h", "mm/h"), "mm/h": ("mm/h", "mm/h"), "fl": ("fL", "fL"), "pg": ("pg", "pg"),
    "seconds": ("s", "s"), "second": ("s", "s"), "sec": ("s", "s"), "s": ("s", "s"),
    "minutes": ("min", "min"), "minute": ("min", "min"), "min": ("min", "min"), "mins": ("min", "min"),
    "hours": ("h", "h"), "hour": ("h", "h"), "hr": ("h", "h"), "hrs": ("h", "h"), "h": ("h", "h"),
    "days": ("d", "d"), "day": ("d", "d"), "d": ("d", "d"),
    "weeks": ("wk", "wk"), "week": ("wk", "wk"), "wk": ("wk", "wk"), "wks": ("wk", "wk"),
    "months": ("mo", "mo"), "month": ("mo", "mo"), "mo": ("mo", "mo"), "mos": ("mo", "mo"),
    "years": ("a", "a"), "year": ("a", "a"), "yr": ("a", "a"), "yrs": ("a", "a"), "y": ("a", "a"), "yo": ("a", "a"),
    "mg/kg": ("mg/kg", "mg/kg"), "mg/kg/day": ("mg/kg/d", "mg/kg/d"), "mg/day": ("mg/d", "mg/d"), "g/day": ("g/d", "g/d"), "mg/24h": ("mg/(24.h)", "mg/(24.h)"),
    "meq/24h": ("mEq/(24.h)", "meq/(24.h)"), "meq/day": ("mEq/d", "meq/d"), "mmol/day": ("mmol/d", "mmol/d"), "g/24h": ("g/(24.h)", "g/(24.h)"),
    "mg/g": ("mg/g", "mg/g"), "µg/min": ("µg/min", "ug/min"), "ug/min": ("µg/min", "ug/min"), "mg/mmol": ("mg/mmol", "mg/mmol"),
    "mm²": ("mm²", "mm2"), "cm²": ("cm²", "cm2"), "cm³": ("cm³", "cm3"), "ml/m²": ("mL/m²", "mL/m2"),
    "dioptres": ("[diop]", "[diop]"), "db": ("dB", "dB"), "hz": ("Hz", "Hz"), "kcal": ("kcal", "kcal"), "kcal/day": ("kcal/d", "kcal/d"),
    "mmol/mol": ("mmol/mol", "mmol/mol"), "ppm": ("ppm", "[ppm]"), "mm/hg": ("mmHg", "mm[Hg]"), "/hpf": ("/HPF", "/[HPF]"), "/lpf": ("/LPF", "/[LPF]"),
}

_SNOMED = "http://snomed.info/sct"
_LOINC = "http://loinc.org"
_UCUM = "http://unitsofmeasure.org"

_NUMBER = re.compile(r"(?P<cmp>[<>≤≥]=?)?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
_BP = re.compile(r"(?P<sys>\d{2,3})\s*/\s*(?P<dia>\d{2,3})\s*(?:mm\s*hg|mmhg)", re.I)
_RANGE = re.compile(
    r"\((?:normal|nl|ref(?:erence)?(?:\s+range)?|range)\s*[:=]?\s*(?P<lo>\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(?P<hi>\d+(?:\.\d+)?)\s*(?P<unit>[^)]*)\)",
    re.I,
)
_PAREN = re.compile(r"\([^)]*\)")
_CMP = {"<": "<", ">": ">", "≤": "<=", "≥": ">=", "<=": "<=", ">=": ">="}


@dataclass
class Quantity:
    value: float
    comparator: str | None = None
    unit: str | None = None
    code: str | None = None

    def fhir(self) -> dict:
        d: dict = {"value": self.value}
        if self.comparator:
            d["comparator"] = self.comparator
        if self.unit:
            d.update(unit=self.unit, system=_UCUM, code=self.code)
        return d


@dataclass
class Parsed:
    quantity: Quantity | None = None
    string: str | None = None                  # value_string when there is no number
    components: list[dict] = field(default_factory=list)
    reference_range: list[dict] = field(default_factory=list)
    note: str | None = None                    # text the parser did not consume


def _norm_unit(tok: str) -> str:
    t = tok.strip().strip(".,;").lower().replace("\u03bc", "\u00b5")   # Greek mu -> micro sign
    t = t.lstrip("-")                          # "-year-old", "-day"
    t = re.sub(r"-old$", "", t)
    t = t.replace(" per ", "/").replace("per ", "/")
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"^cells/", "/", t)
    t = t.replace("mm/hg", "mmhg")
    return t


def unit_for(token: str) -> tuple[str, str] | None:
    """(display, UCUM) for a raw unit token, or None when the token is not a known unit."""
    t = _norm_unit(token)
    if t in _U:
        return _U[t]
    # "beats/min" style compounds normalized above; try dropping a trailing plural/time word
    for suffix in ("/minute", "perminute"):
        if t.endswith(suffix) and t[: -len(suffix)] in ("beats", "breaths", "respirations"):
            return _U["/min"]
    return None


def _number(s: str) -> float:
    return float(s.replace(",", ""))


def parse_value(text: str | None, numeric: float | None = None) -> Parsed:
    """Parse one measurement. `numeric` is the ETL's value_numeric, used only when the text has no number."""
    out = Parsed()
    raw = (text or "").strip()
    if not raw:
        if numeric is not None:
            out.quantity = Quantity(value=float(numeric))
        return out

    rest = raw
    # reference range parenthetical -> referenceRange
    m = _RANGE.search(rest)
    if m:
        u = unit_for(m.group("unit")) if m.group("unit").strip() else None
        lo = Quantity(_number(m.group("lo")), unit=u[0] if u else None, code=u[1] if u else None)
        hi = Quantity(_number(m.group("hi")), unit=u[0] if u else None, code=u[1] if u else None)
        out.reference_range.append({"low": lo.fhir(), "high": hi.fhir()})
        rest = (rest[: m.start()] + rest[m.end():]).strip()

    # blood pressure -> components (and no top-level value)
    bp = _BP.search(rest)
    if bp:
        mmhg = _U["mmhg"]
        for code, display, key in (("8480-6", "Systolic blood pressure", "sys"), ("8462-4", "Diastolic blood pressure", "dia")):
            out.components.append({
                "code": {"coding": [{"system": _LOINC, "code": code, "display": display}], "text": display},
                "valueQuantity": Quantity(float(bp.group(key)), unit=mmhg[0], code=mmhg[1]).fhir(),
            })
        leftover = (rest[: bp.start()] + rest[bp.end():]).strip(" ,;")
        out.note = leftover or None
        if out.reference_range and not out.reference_range[0]["low"].get("unit"):
            for q in out.reference_range[0].values():
                q.update(unit=mmhg[0], system=_UCUM, code=mmhg[1])
        return out

    m = _NUMBER.search(rest)
    if not m:
        out.string = raw
        if numeric is not None:
            out.quantity = None
        return out

    value = _number(m.group("num"))
    comparator = _CMP.get(m.group("cmp") or "", None)
    after = rest[m.end():]
    before = rest[: m.start()].strip()
    # the unit is the first token after the number, up to a parenthesis / comma / conjunction
    tail = re.split(r"[(,;]| and | on | at | with | in | of | ago| since| for | after| prior", after, 1)[0].strip()
    unit = None
    consumed_end = m.end()
    if tail:
        words = tail.split()
        # longest candidate first: "beats per minute" before "beats"; "-year-old" is glued to the number
        for n in (3, 2, 1):
            cand = " ".join(words[:n])
            u = unit_for(cand)
            if u is not None:
                unit = u
                consumed_end = m.end() + after.index(words[0]) + len(cand)
                break
    out.quantity = Quantity(value, comparator=comparator, unit=unit[0] if unit else None, code=unit[1] if unit else None)
    leftover = (before + " " + rest[consumed_end:]).strip(" ,;")
    leftover = re.sub(r"\s+", " ", leftover)
    if leftover:
        out.note = leftover
    if out.reference_range and unit and not out.reference_range[0]["low"].get("unit"):
        for q in out.reference_range[0].values():
            q.update(unit=unit[0], system=_UCUM, code=unit[1])
    return out


def presence_concept(present: bool) -> dict:
    """valueCodeableConcept for a finding that has no measured value."""
    if present:
        return {"coding": [{"system": _SNOMED, "code": "52101004", "display": "Present"}], "text": "Present"}
    return {"coding": [{"system": _SNOMED, "code": "2667000", "display": "Absent"}], "text": "Absent"}


def interpretation(present: bool) -> dict:
    code, display = ("POS", "Positive") if present else ("NEG", "Negative")
    return {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": code, "display": display}], "text": display}
