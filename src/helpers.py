import logging
import pandas as pd
import re

from src.constants import REGEX_PATTERN

logger = logging.getLogger(__name__)

def extract_field_from_row(text: str) -> dict:
    clean_text = text.replace("\r\n", "\n").replace("*", "").strip()

    pattern = re.compile(
        REGEX_PATTERN,
        re.IGNORECASE | re.DOTALL
    )

    match = pattern.search(clean_text)
    if not match:
        logger.debug("No match found.")
        return None

    important = match.group("important").strip()
    reason = match.group("reason").strip()
    confidence = match.group("confidence").strip()

    return {"important": important, "reason": reason, "confidence": confidence}

def extract_fields_from_raw_result(results: list[str]) -> list[dict]:
    extracted_fields: list = []

    for i in results:
        logger.info(f"Extracting fields from {len(results)} results...")
        extracted_fields.append(extract_field_from_row(i))

    return extracted_fields
