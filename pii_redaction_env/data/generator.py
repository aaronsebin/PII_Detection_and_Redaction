from __future__ import annotations

import random
from collections.abc import Iterable

from faker import Faker

try:
    from ..models import PIIType, RedactionSpan
except ImportError:
    from models import PIIType, RedactionSpan


def _append_segment(
    parts: list[str],
    spans: list[RedactionSpan],
    text: str,
    pii_type: PIIType | None = None,
) -> None:
    start = len(_join_with_space(parts))
    if parts:
        start += 1
    parts.append(text)
    if pii_type is not None:
        spans.append(
            RedactionSpan(
                start=start,
                end=start + len(text),
                pii_type=pii_type,
                text=text,
            )
        )


def _join_with_space(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part)


def generate_document(task_id: str, seed: int = 42) -> tuple[str, list[RedactionSpan]]:
    rng = random.Random(seed)
    faker = Faker()
    faker.seed_instance(seed)

    person = faker.name()
    email = faker.email()
    phone = faker.phone_number()
    ssn = faker.ssn()
    address = faker.address().replace("\n", ", ")
    dob = faker.date_of_birth(minimum_age=21, maximum_age=73).isoformat()
    credit_card = faker.credit_card_number(card_type="visa")
    ip_address = faker.ipv4_public()
    passport = f"{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{rng.randint(10000000, 99999999)}"
    company = faker.company()
    job_title = faker.job()
    city = faker.city()
    state_name = faker.state()
    postal_code = faker.postcode()
    age = str(rng.randint(24, 68))
    hospital = faker.company() + " Medical Center"
    vehicle = f"{faker.color_name()} {faker.word().title()} sedan"

    parts: list[str] = []
    spans: list[RedactionSpan] = []

    if task_id == "basic_pii_detection":
        _append_segment(parts, spans, "Patient intake note:")
        _append_segment(parts, spans, person, PIIType.PERSON)
        _append_segment(parts, spans, "can be reached at")
        _append_segment(parts, spans, email, PIIType.EMAIL)
        _append_segment(parts, spans, "or")
        _append_segment(parts, spans, phone, PIIType.PHONE)
        _append_segment(parts, spans, ".")
        return _join_with_space(parts), spans

    if task_id == "mixed_pii_redaction":
        _append_segment(parts, spans, "Fraud review file for")
        _append_segment(parts, spans, person, PIIType.PERSON)
        _append_segment(parts, spans, "lists mailing address")
        _append_segment(parts, spans, address, PIIType.ADDRESS)
        _append_segment(parts, spans, ", SSN")
        _append_segment(parts, spans, ssn, PIIType.SSN)
        _append_segment(parts, spans, ", date of birth")
        _append_segment(parts, spans, dob, PIIType.DATE_OF_BIRTH)
        _append_segment(parts, spans, ", corporate email")
        _append_segment(parts, spans, email, PIIType.EMAIL)
        _append_segment(parts, spans, ", and card ending with")
        _append_segment(parts, spans, credit_card, PIIType.CREDIT_CARD)
        _append_segment(parts, spans, ". Login from")
        _append_segment(parts, spans, ip_address, PIIType.IP_ADDRESS)
        _append_segment(parts, spans, "was flagged during verification.")
        return _join_with_space(parts), spans

    if task_id == "adversarial_quasi_identification":
        _append_segment(parts, spans, "Incident summary:")
        _append_segment(parts, spans, person, PIIType.PERSON)
        _append_segment(parts, spans, "works as")
        _append_segment(parts, spans, job_title, PIIType.QUASI_IDENTIFIER)
        _append_segment(parts, spans, "at")
        _append_segment(parts, spans, company, PIIType.QUASI_IDENTIFIER)
        _append_segment(parts, spans, "and lives near")
        _append_segment(
            parts,
            spans,
            f"{city}, {state_name} {postal_code}",
            PIIType.QUASI_IDENTIFIER,
        )
        _append_segment(parts, spans, ". The case references passport")
        _append_segment(parts, spans, passport, PIIType.PASSPORT)
        _append_segment(parts, spans, ", vehicle")
        _append_segment(parts, spans, vehicle, PIIType.QUASI_IDENTIFIER)
        _append_segment(parts, spans, ", age")
        _append_segment(parts, spans, age, PIIType.QUASI_IDENTIFIER)
        _append_segment(parts, spans, ", treatment site")
        _append_segment(parts, spans, hospital, PIIType.QUASI_IDENTIFIER)
        _append_segment(parts, spans, ", and contact phone")
        _append_segment(parts, spans, phone, PIIType.PHONE)
        _append_segment(parts, spans, ".")
        return _join_with_space(parts), spans

    raise ValueError(f"Unknown task_id: {task_id}")
