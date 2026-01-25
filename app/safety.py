import re
from typing import Tuple

# -------------------------------
# BLOCKLIST (cheap & effective)
# -------------------------------
BLOCKED_PHRASES = [
    "ignore previous instructions",
    "ignore system prompt",
    "reveal system prompt",
    "show system prompt",
    "show your prompt",
    "show your instructions",
    "bypass safety",
    "jailbreak",
    "act as system",
    "act as developer"
]

BLOCKED_REGEX = [
    re.compile(r"ignore\s+all\s+previous", re.IGNORECASE),
    re.compile(r"reveal\s+.*prompt", re.IGNORECASE),
    re.compile(r"system\s+instructions", re.IGNORECASE),
]

# -------------------------------
# INPUT SAFETY
# -------------------------------
def check_input_safety(text: str) -> Tuple[bool, str]:
    lowered = text.lower()

    for phrase in BLOCKED_PHRASES:
        if phrase in lowered:
            return False, "Unsafe request detected"

    for pattern in BLOCKED_REGEX:
        if pattern.search(text):
            return False, "Unsafe request detected"

    return True, ""

# -------------------------------
# OUTPUT SAFETY (LIGHTWEIGHT)
# -------------------------------
def check_output_safety(text: str) -> Tuple[bool, str]:
    # Placeholder for moderation APIs later
    # (OpenAI / Perspective / internal classifiers)

    # Example: prevent leaking system internals
    if "system prompt" in text.lower():
        return False, "Response blocked due to policy"

    return True, ""
