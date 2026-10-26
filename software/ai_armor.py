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
# Phone numbers (PRIV-2, PRIV-3). What counts as a phone depends on who wrote the text: the player's own words are
# masked generously (a bare score is rare in chat), the model's and the game's text only where a number is clearly a phone.
_SEP = r"[\s.-]?"  # the separators people type inside a number


def _digits(first: str, more: int) -> str:
    """`first` (a pattern for the leading digits) then `more` digits, each optionally preceded by a separator."""
    return first + (_SEP + r"\d") * more


_LEFT, _RIGHT = r"(?<![\w+])", r"(?!\w)"
_INTERNATIONAL_RE = re.compile(_LEFT + r"(?:\+|00)\d(?:[\s().-]?\d){7,14}" + _RIGHT)  # +20 100 123 4567, 0020...
_PLAYER_RE = re.compile(_LEFT + r"\(?\d(?:[\s().-]{0,2}\d){8,14}" + _RIGHT)  # any 9-15 digit run, as the player typed it
_CLEAR_LOCAL_RE = re.compile(_LEFT + "(?:" + "|".join([
    r"(?:\d{1,3}[-.\s])?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}",  # US/Canada only with separators: 555-123-4567
    _digits(r"01[0125]", 8),  # Egypt: 01001234567
    _digits(r"05\d", 7),  # Saudi Arabia, UAE: 0501234567
    _digits(r"07", 9),  # UK: 07911 123456
    _digits(r"07[789]", 7),  # Jordan
    _digits(r"0[67]", 8),  # Morocco, Algeria
    _digits(r"07[3-9]", 8),  # Iraq
]) + ")" + _RIGHT)
# 8-digit mobiles (Kuwait, Qatar, Bahrain, Oman, Tunisia, Lebanon, ...) look like scores: a separator or a phone word is needed.
_SHORT_LOCAL_RE = re.compile(_LEFT + r"(?:[2-9]\d{7}|03\d{6})" + _RIGHT + r"|" + _LEFT + r"[2-9]\d(?:[\s.-]\d{2}){3}" + _RIGHT
                             + r"|" + _LEFT + r"\d{4}[\s.-]\d{4}" + _RIGHT)
_PHONE_WORD_RE = re.compile(r"phone|call|number|mobile|whatsapp|tel\b|هاتف|اتصل|رقم|موبايل|جوال|واتساب", re.IGNORECASE)
_DATE_RE = re.compile(r"^(?:19|20)\d{2}[-. ](?:0[1-9]|1[0-2])[-. ](?:0[1-9]|[12]\d|3[01])$")
PLAYER, MODEL = "player", "model"
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

    def redact_pii(self, text: str, source: str = PLAYER) -> str:
        """Mask personal identifiable information (emails, IPs, phones, credit cards).

        `source` says who wrote the text. "player" (the default, so a new caller fails safe) is the player's own words:
        any 9-15 digit run is a phone. "model" is the AI's or the game's text: only clearly formatted numbers are, so a
        score survives. Anything else is treated as "player"."""
        if not self.redact_privacy or not text:
            return text

        result = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        result = _CREDIT_CARD_RE.sub("[REDACTED_CARD]", result)
        result = _IPV4_RE.sub("[REDACTED_IP]", result)
        result = _INTERNATIONAL_RE.sub("[REDACTED_PHONE]", result)
        if source == MODEL:
            result = _CLEAR_LOCAL_RE.sub("[REDACTED_PHONE]", result)
            result = _SHORT_LOCAL_RE.sub(lambda m: self._short_phone(result, m), result)
        else:
            result = _PLAYER_RE.sub(lambda m: m.group(0) if _DATE_RE.match(m.group(0)) else "[REDACTED_PHONE]", result)
        return result

    @staticmethod
    def _short_phone(text: str, match: "re.Match[str]") -> str:
        """An 8-digit number is a phone only with a separator inside it or a phone word close by (never a date)."""
        number = match.group(0)
        if _DATE_RE.match(number):
            return number
        near = text[max(0, match.start() - 30):match.end() + 30]
        if re.search(r"[\s.-]", number) or _PHONE_WORD_RE.search(near):
            return "[REDACTED_PHONE]"
        return number

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
