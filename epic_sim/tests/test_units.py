"""Pure tests for the measurement text parser behind FHIR Observation (Stage 4b). No database."""

from epic_sim.app.fhir.units import interpretation, parse_value, presence_concept

UCUM = "http://unitsofmeasure.org"


def test_quantity_with_ucum_unit_and_reference_range():
    p = parse_value("24 mEq/L (Normal: 22-29)", 24.0)
    assert p.quantity.value == 24.0 and p.quantity.unit == "mEq/L" and p.quantity.code == "meq/L"
    assert p.quantity.fhir()["system"] == UCUM
    assert p.reference_range == [{"low": {"value": 22.0, "unit": "mEq/L", "system": UCUM, "code": "meq/L"},
                                  "high": {"value": 29.0, "unit": "mEq/L", "system": UCUM, "code": "meq/L"}}]
    assert p.note is None and p.string is None and not p.components


def test_thousands_separator_is_read_from_the_text_not_the_lossy_numeric():
    p = parse_value("250,000/mm³", 250.0)
    assert p.quantity.value == 250000.0 and p.quantity.code == "/mm3"


def test_blood_pressure_becomes_two_components():
    p = parse_value("120/78 mmHg", 120.0)
    assert p.quantity is None
    codes = [c["code"]["coding"][0]["code"] for c in p.components]
    assert codes == ["8480-6", "8462-4"]
    assert [c["valueQuantity"]["value"] for c in p.components] == [120.0, 78.0]
    assert all(c["valueQuantity"]["code"] == "mm[Hg]" for c in p.components)


def test_comparator_and_parenthetical_note():
    p = parse_value("<0.04 ng/mL (normal)", 0.04)
    assert p.quantity.comparator == "<" and p.quantity.value == 0.04 and p.quantity.code == "ng/mL"
    assert p.note == "(normal)"


def test_age_and_duration_units():
    assert parse_value("45-year-old", 45.0).quantity.code == "a"
    p = parse_value("5 days ago", 5.0)
    assert p.quantity.code == "d" and p.note == "ago"


def test_unknown_unit_is_not_guessed():
    p = parse_value("89% on room air", 89.0)
    assert p.quantity.code == "%" and p.note == "on room air"
    p = parse_value("7.32", 7.32)                        # pH: unitless
    assert p.quantity.value == 7.32 and p.quantity.unit is None and p.note is None
    p = parse_value("1.0 (Normal: 0.8-1.2)", 1.0)
    assert p.quantity.value == 1.0 and p.reference_range[0]["low"]["value"] == 0.8 and "unit" not in p.reference_range[0]["low"]


def test_text_without_a_number_is_a_string_value():
    p = parse_value("denies", None)
    assert p.string == "denies" and p.quantity is None


def test_no_text_no_number_gives_nothing_so_the_resource_carries_a_presence_concept():
    p = parse_value(None, None)
    assert p.quantity is None and p.string is None and not p.components
    assert presence_concept(False)["coding"][0]["code"] == "2667000"
    assert presence_concept(True)["coding"][0]["code"] == "52101004"
    assert interpretation(False)["coding"][0]["code"] == "NEG"
    assert interpretation(True)["coding"][0]["code"] == "POS"


def test_words_per_minute_rates():
    assert parse_value("78 beats per minute", 78.0).quantity.code == "/min"
    assert parse_value("110 beats/min", 110.0).quantity.code == "/min"
    assert parse_value("22 breaths/min", 22.0).quantity.code == "/min"
    assert parse_value("37.2°C (99°F)", 37.2).quantity.code == "Cel"
