import logging
import pandas as pd
import re

from src.constants import REGEX_PATTERN

logger = logging.getLogger(__name__)


def extract_field_from_row(text: str) -> dict | None:
    """
    Extract structured fields ('important', 'reason', 'confidence')
    from a single raw LLM output string.

    The function cleans the input text of Markdown formatting and
    newline inconsistencies, then applies a compiled regex pattern
    (defined in `REGEX_PATTERN`) to extract the relevant classification fields.

    Args:
        text (str): The raw output text from the language model.
            Expected format:
            ```
            IMPORTANT: Yes
            REASON: This email is a work-related invoice.
            CONFIDENCE: High
            ```

    Returns:
        dict | None: A dictionary containing the extracted fields:
            - "important": str — classification label ("Yes" or "No")
            - "reason": str — brief rationale provided by the model
            - "confidence": str — confidence level ("High", "Medium", "Low")
        Returns `None` if no match is found.

    Example:
        >>> text = "IMPORTANT: No\\nREASON: Promotional content\\nCONFIDENCE: High"
        >>> extract_field_from_row(text)
        {'important': 'No', 'reason': 'Promotional content', 'confidence': 'High'}
    """
    clean_text = text.replace("\r\n", "\n").replace("*", "").strip()

    pattern = re.compile(REGEX_PATTERN, re.IGNORECASE | re.DOTALL)
    match = pattern.search(clean_text)

    if not match:
        logger.debug("No match found in text.")
        return None

    important = match.group("important").strip()
    reason = match.group("reason").strip()
    confidence = match.group("confidence").strip()

    return {"important": important, "reason": reason, "confidence": confidence}


def extract_fields_from_raw_result(results: list[str]) -> list[dict | None]:
    """
    Process a list of raw model outputs and extract structured classification fields.

    This function loops through each string response produced by the LLM,
    applies `extract_field_from_row()` to parse it, and aggregates
    the results into a list of dictionaries. It logs progress as it iterates.

    Args:
        results (list[str]): A list of raw string responses from the model.
            Each string should follow the expected classification format.

    Returns:
        list[dict | None]: A list of parsed results, where each item corresponds
            to one model output. If a result could not be parsed,
            its entry will be `None`.

    Example:
        >>> results = [
        ...     "IMPORTANT: Yes\\nREASON: Urgent invoice\\nCONFIDENCE: High",
        ...     "IMPORTANT: No\\nREASON: Promotional content\\nCONFIDENCE: Medium"
        ... ]
        >>> extract_fields_from_raw_result(results)
        [
            {'important': 'Yes', 'reason': 'Urgent invoice', 'confidence': 'High'},
            {'important': 'No', 'reason': 'Promotional content', 'confidence': 'Medium'}
        ]
    """
    extracted_fields: list[dict | None] = []

    logger.info(f"Extracting fields from {len(results)} results...")

    for result in results:
        parsed = extract_field_from_row(result)
        extracted_fields.append(parsed)

    return extracted_fields
