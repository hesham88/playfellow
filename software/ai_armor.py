"""AI Armor & Responsible AI Guardrails for PlayFellow. / درع الذكاء الاصطناعي وضوابط الأمان المسؤولة لبلاي فيلو.

Protects players from prompt injections, personal-data leaks, rogue keystroke commands,
and hostile web content when the AI Buddy plans or performs actions.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("playfellow.ai_armor")

# 1. Hostile Prompt Injection & Jailbreak Patterns
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(DAN|unfiltered|jailbroken|evil|unrestricted)", re.IGNORECASE),
    re.compile(r"bypass\s+(safety|rules|restrictions|guardrails)", re.IGNORECASE),
    re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", re.IGNORECASE),
    re.compile(r"\[SYSTEM_PROMPT\]|\[INSTRUCTION\]|\[DEVELOPER\]", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
]

# 2. Privacy & PII Redaction Patterns
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
_IPV4_RE = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

# 3. Hazardous OS Hotkeys and Actions
_DANGEROUS_KEYS = {
    "alt+f4",
    "ctrl+alt+del",
    "ctrl+shift+esc",
    "win+l",
    "win+r",
    "ctrl+w",
    "format c:",
    "del /f",
    "rmdir /s",
}

# 4. Basic profanity and explicit-content filter for what the buddy says
_PROFANITY_PATTERNS = [
    re.compile(r"\b(fuck|shit|bitch|asshole|bastard|dick|pussy|cunt)\b", re.IGNORECASE),
    re.compile(r"\b(porn|nsfw|sex|nude|naked)\b", re.IGNORECASE),
]


class AIArmor:
    """Responsible AI engine providing prompt sanitization, privacy redaction, and action clamping."""

    def __init__(self, redact_privacy: bool = True, enforce_family_safe: bool = True):
        self.redact_privacy = redact_privacy
        self.enforce_family_safe = enforce_family_safe

    def sanitize_prompt(self, user_text: str) -> tuple[str, bool]:
        """Strip hostile prompt injections and jailbreaks.

        Returns:
            (sanitized_text, injection_detected)
        """
        if not user_text:
            return "", False

        detected = False
        clean = user_text
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(clean):
                detected = True
                clean = pattern.sub("[REDACTED_SECURITY_PROMPT]", clean)

        return clean.strip(), detected

    def redact_pii(self, text: str) -> str:
        """Mask personal identifiable information (emails, IPs, phones, credit cards)."""
        if not self.redact_privacy or not text:
            return text

        result = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        result = _CREDIT_CARD_RE.sub("[REDACTED_CARD]", result)
        result = _PHONE_RE.sub("[REDACTED_PHONE]", result)
        result = _IPV4_RE.sub("[REDACTED_IP]", result)
        return result

    def is_child_safe(self, text: str) -> bool:
        """Check that output text has no profanity or explicit content (the name is older than the 18+ decision)."""
        if not self.enforce_family_safe or not text:
            return True

        for pattern in _PROFANITY_PATTERNS:
            if pattern.search(text):
                return False
        return True

    def validate_action(self, action_key: str) -> tuple[bool, str]:
        """Verify that an action key or combination is safe to execute on the player's OS.

        Returns:
            (is_safe, sanitized_key_or_reason)
        """
        if not action_key:
            return True, ""

        normalized = action_key.strip().lower()
        if normalized in _DANGEROUS_KEYS:
            logger.warning("Blocked hazardous action key: %s", action_key)
            return False, f"Blocked hazardous key combination: {action_key}"

        # Guard against recursive / runaway command injection strings
        for dangerous in ("cmd.exe", "powershell.exe", "taskkill"):
            if dangerous in normalized:
                return False, f"Blocked command injection attempt: {action_key}"
        for dangerous in ("bash", "sh"):
            if re.search(rf"\b{dangerous}\b", normalized):
                return False, f"Blocked command injection attempt: {action_key}"

        return True, action_key


# Default singleton instance
armor = AIArmor()
