"""Reads settings.yaml and the AI keys. / يقرأ ملف الإعدادات ومفاتيح الذكاء الاصطناعي."""
from __future__ import annotations

import dataclasses
import difflib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from messages import LANGUAGES, msg, set_language, tr

BRAIN_NAMES = ("gemini", "claude", "openai", "local")
MAIN_BRAIN_NAMES = BRAIN_NAMES + ("builtin",)  # the built-in brain plays by itself: no account, no key, no network
POWER_LEVELS = ("smart", "thrifty")

# The only place model names live. smart = plays best (default); thrifty = costs least.
MODEL_TIERS = {
    "smart": {
        "gemini": "gemini-3.8-flash",
        "claude": "claude-sonnet-5-5",
        "openai": "gpt-5-mini",
        "local": "gemma4:e4b",
        "builtin": "navigator",
    },
    "thrifty": {
        "gemini": "gemini-3.1-flash-lite",
        "claude": "claude-haiku-5-5",
        "openai": "gpt-5-nano",
        "local": "gemma4:e2b",
        "builtin": "navigator",
    },
}
# Claude models that accept server-side fallbacks when a request is declined.
CLAUDE_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5", "claude-sonnet-5-5")

KEY_NAMES = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
KEYRING_SERVICE = "PlayFellow research"


@dataclass
class Settings:
    buddy_name: str = "PlayFellow"
    language: str = "english"
    game_window: str = "PlayFellow Practice Arena"
    controls_card: str = "practice_arena"
    goal: str = "Collect all the coins, then grab the key and open the door."
    personality: str = "Friendly, careful and clear. You explain your plan in one short sentence."
    look_every_seconds: float = 1.0
    memory_steps: int = 3
    max_actions_per_step: int = 4
    image_width: int = 640
    brain: str = "builtin"  # no account, no key, no network: it learns the game from the pictures
    brain_power: str = "smart"
    model: str = ""
    max_minutes: int = 20
    max_ai_calls: int = 600

    def brain_model(self) -> str:
        return self.model or MODEL_TIERS[self.brain_power][self.brain]


RANGES = {
    "look_every_seconds": (0.2, 30.0),
    "memory_steps": (0, 10),
    "max_actions_per_step": (1, 8),
    "image_width": (256, 1600),
    "max_minutes": (1, 120),
    "max_ai_calls": (1, 5000),
}
LOCAL_FILE = "settings.local.yaml"  # optional, git-ignored: this PC's own values on top of the shipped settings
CHOICES = {"brain": MAIN_BRAIN_NAMES, "language": LANGUAGES, "brain_power": POWER_LEVELS}


class SettingsError(Exception):
    """A mistake in settings.yaml, explained in both languages."""

    def __init__(self, detail: str):
        super().__init__(msg("settings_error", detail=detail))


def load_settings(path) -> Settings:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SettingsError(tr(f"I can't find {path.name}", f"لا أجد الملف {path.name}")) from None
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark else "?"
        raise SettingsError(
            tr(f"check line {line} (spaces and colons)", f"راجع السطر {line} (المسافات والنقطتين)")
        ) from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SettingsError(tr("each line should look like  name: value", "كل سطر يجب أن يكون بالشكل  الاسم: القيمة"))
    local = path.with_name(LOCAL_FILE)
    if local.is_file() and path.name != LOCAL_FILE:
        try:
            extra = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            raise SettingsError(tr(f"check {LOCAL_FILE} (spaces and colons)", f"راجع {LOCAL_FILE} (المسافات والنقطتين)")) from None
        if not isinstance(extra, dict):
            raise SettingsError(tr(f"{LOCAL_FILE}: each line should look like  name: value",
                                   f"{LOCAL_FILE}: كل سطر يجب أن يكون بالشكل  الاسم: القيمة"))
        data = {**data, **extra}  # this PC's own values win over the shipped ones

    # Set language before validation so error messages come out in the chosen language
    language_value = data.get("language")
    if language_value:
        lang = str(language_value).strip().lower()
        if lang in LANGUAGES:
            set_language(lang)

    defaults = Settings()
    names = [field.name for field in dataclasses.fields(Settings)]
    values = {}
    for raw_key, value in data.items():
        key = str(raw_key)
        if key not in names:
            guess = difflib.get_close_matches(key, names, n=1)
            hint = tr(f" Did you mean '{guess[0]}'?", f" هل تقصد '{guess[0]}'؟") if guess else ""
            raise SettingsError(tr(f"unknown setting '{key}'", f"إعداد غير معروف '{key}'") + f".{hint}")
        values[key] = _check(key, value, getattr(defaults, key))
    return Settings(**values)


def _check(key: str, value, default):
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise SettingsError(tr(f"'{key}' must be true or false", f"'{key}' يجب أن تكون true أو false"))
        return value
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(tr(f"'{key}' must be a number like {default}", f"'{key}' يجب أن يكون رقمًا مثل {default}"))
        if isinstance(default, int) and not float(value).is_integer():
            raise SettingsError(
                tr(f"'{key}' must be a whole number like {default}", f"'{key}' يجب أن يكون رقمًا صحيحًا مثل {default}")
            )
        low, high = RANGES[key]
        if not low <= value <= high:
            raise SettingsError(tr(f"'{key}' must be between {low} and {high}", f"'{key}' يجب أن يكون بين {low} و {high}"))
        return int(value) if isinstance(default, int) else float(value)
    text = "" if value is None else str(value).strip()
    if key in CHOICES:
        text = text.lower()
        if text not in CHOICES[key]:
            options = ", ".join(CHOICES[key])
            raise SettingsError(tr(f"'{key}' must be one of: {options}", f"'{key}' يجب أن يكون واحدًا من: {options}"))
    if not text and default:
        return default
    return text


def load_api_key(brain: str) -> str | None:
    """Find the key for a brain: first the environment, then Windows Credential Manager."""
    if brain not in KEY_NAMES:
        return None
    name = KEY_NAMES[brain]
    if os.environ.get(name, "").strip():
        return os.environ[name].strip()
    import keyring

    try:
        value = keyring.get_password(KEYRING_SERVICE, name)
    except Exception:
        return None
    return value.strip() if value and value.strip() else None


def save_api_key(brain: str, key: str) -> None:
    """Keep the key safe in Windows Credential Manager, never in a plain file."""
    if brain not in KEY_NAMES:
        raise ValueError(tr(f"'{brain}' doesn't use a key", f"'{brain}' لا يحتاج إلى مفتاح"))
    stripped_key = key.strip()
    if not stripped_key:
        raise ValueError(tr("The key is empty", "المفتاح فارغ"))
    import keyring

    keyring.set_password(KEYRING_SERVICE, KEY_NAMES[brain], stripped_key)
