DEFAULT_LLM = "Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"

REGEX_PATTERN = (
        r"IMPORTANT\s*:\s*(?P<important>[^\n]+)\s*"
        r"(?:\n)+REASON\s*:\s*(?P<reason>.+?)\s*"
        r"(?:\n)+CONFIDENCE\s*:\s*(?P<confidence>[A-Za-z]+)"
        )

REVIEW_CSV_COLUMNS = [
    "Message_ID",
    "Subject",
    "Sender",
    "important",
    "reason",
    "confidence",
    "Decision",
    "Notes",
]

CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}
