DEFAULT_LLM = "Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"

REGEX_PATTERN = (
        r"IMPORTANT\s*:\s*(?P<important>[^\n]+)\s*"
        r"(?:\n)+REASON\s*:\s*(?P<reason>.+?)\s*"
        r"(?:\n)+CONFIDENCE\s*:\s*(?P<confidence>[A-Za-z]+)"
        )

REVIEW_CSV_COLUMNS = [
    "message_id",
    "subject",
    "sender",
    "date_sent",
    "important",
    "reason",
    "confidence",
    "needs_review",
    "suggested_decision",
    "decision",
    "notes",
]

CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}
