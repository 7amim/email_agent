DEFAULT_LLM = "Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"

REGEX_PATTERN = (
        r"IMPORTANT\s*:\s*(?P<important>[^\n]+)\s*"
        r"(?:\n)+REASON\s*:\s*(?P<reason>.+?)\s*"
        r"(?:\n)+CONFIDENCE\s*:\s*(?P<confidence>[A-Za-z]+)"
        )
