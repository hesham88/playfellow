"""Friendly messages PlayFellow shows the player. / رسائل ودودة يعرضها بلاي فيلو للاعب."""
from __future__ import annotations

LANGUAGES = ("english", "arabic")
_language = "english"


def set_language(language: str) -> None:
    global _language
    if language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}")
    _language = language


def get_language() -> str:
    return _language


def tr(english: str, arabic: str) -> str:
    """Pick the text for the current language."""
    return arabic if _language == "arabic" else english


MESSAGES: dict[str, tuple[str, str]] = {
    "no_window": (
        "I can't find the game window '{title}'. Open the game, then press Start.",
        "لا أجد نافذة اللعبة '{title}'. افتح اللعبة ثم اضغط ابدأ.",
    ),
    "click_game": (
        "Click on your game window so I can play.",
        "اضغط على نافذة اللعبة حتى أستطيع اللعب.",
    ),
    "blocked_window": (
        "That window isn't a game, so I won't touch it. Click your game to continue.",
        "هذه النافذة ليست لعبة، لذلك لن ألمسها. اضغط على لعبتك للمتابعة.",
    ),
    "refocusing": (
        "Bringing your game back to the front…",
        "أعيد لعبتك إلى الأمام…",
    ),
    "no_key": (
        "My brain isn't connected yet. Press 'Connect brain' and add your AI key.",
        "عقلي غير متصل بعد. اضغط على 'توصيل العقل' وأضف مفتاح الذكاء الاصطناعي الخاص بك.",
    ),
    "builtin_fallback": (
        "No AI account yet, so I'm playing with my built-in brain.",
        "لا يوجد حساب ذكاء اصطناعي بعد، لذلك ألعب بعقلي المدمج.",
    ),
    "bad_key": (
        "The AI key isn't working. Please check it in 'Connect brain'.",
        "مفتاح الذكاء الاصطناعي لا يعمل. يُرجى التحقق منه في 'توصيل العقل'.",
    ),
    "model_missing": ("This AI model isn't on this PC yet: {detail}", "نموذج الذكاء الاصطناعي هذا غير موجود على هذا الجهاز بعد: {detail}"),
    "quota": (
        "Your AI account has used up its quota for now. Check your plan, or switch the brain to 'local'.",
        "انتهت حصة حساب الذكاء الاصطناعي الآن. تحقّق من خطتك، أو غيّر العقل إلى 'local'.",
    ),
    "busy": (
        "The AI is busy right now. I'll wait {seconds} seconds and try again.",
        "الذكاء الاصطناعي مشغول الآن. سأنتظر {seconds} ثانية ثم أحاول مرة أخرى.",
    ),
    "token_cap_minute": (
        "That's a lot of AI questions in one minute ({detail} tokens). I'll wait {seconds} seconds before asking again.",
        "هذه أسئلة كثيرة للذكاء الاصطناعي في دقيقة واحدة ({detail} رمز). سأنتظر {seconds} ثانية قبل أن أسأل مرة أخرى.",
    ),
    "token_cap_day": (
        "Your daily AI budget is used up ({detail} tokens in the last 24 hours). Raise it in Settings, or play with the "
        "built-in brain for now.",
        "انتهت ميزانيتك اليومية للذكاء الاصطناعي ({detail} رمز خلال آخر 24 ساعة). ارفعها من الإعدادات، أو العب "
        "بالعقل المدمج الآن.",
    ),
    "offline": (
        "I can't reach the internet. Check the Wi-Fi.",
        "لا أستطيع الوصول إلى الإنترنت. تحقق من الواي فاي.",
    ),
    "local_offline": (
        "I can't reach the brain on this computer. Is Ollama running?",
        "لا أستطيع الوصول إلى العقل على هذا الكمبيوتر. هل برنامج Ollama يعمل؟",
    ),
    "brain_trouble": (
        "My brain had a problem: {detail}",
        "حدثت مشكلة في عقلي: {detail}",
    ),
    "confused": (
        "I got confused that time. Trying again!",
        "ارتبكت هذه المرة. سأحاول مرة أخرى!",
    ),
    "stopped_button": ("Stopped. You pressed Stop.", "توقفت. لقد ضغطت إيقاف."),
    "stopped_f12": ("Emergency stop! (F12)", "إيقاف طوارئ! (F12)"),
    "stopped_corner": (
        "Emergency stop! (mouse in the top-left corner)",
        "إيقاف طوارئ! (الفأرة في الزاوية العلوية اليسرى)",
    ),
    "stopped_time": (
        "Time's up: I played for {minutes} minutes. Take a break!",
        "انتهى الوقت: لعبت لمدة {minutes} دقيقة. خذ استراحة!",
    ),
    "stopped_calls": (
        "I used all {calls} brain questions for this session.",
        "استخدمت كل أسئلة العقل ({calls}) لهذه الجلسة.",
    ),
    "too_many_errors": (
        "Too many problems in a row, so I stopped.",
        "حدثت مشاكل كثيرة متتالية، لذلك توقفت.",
    ),
    "round_done": (
        "Challenge attempt finished!",
        "انتهت محاولة التحدي!",
    ),
    "coach_resting": (
        "The coach is resting (AI limit), so I'll check the screen myself.",
        "المدرب يستريح (حد الذكاء الاصطناعي)، لذلك سأتحقق من الشاشة بنفسي.",
    ),
    "online_game_warning": (
        "Heads up: \"{title}\" is played online against real people and uses anti-cheat. Games like this can ban accounts that use automated play, and other players may not want to play against a bot. Continue only if the game's rules allow it and you accept the risk.",
        "تنبيه: «{title}» تُلعب عبر الإنترنت ضد أشخاص حقيقيين وتستخدم نظامًا ضد الغش. قد تحظر هذه الألعاب الحسابات التي تستخدم اللعب الآلي، وقد لا يرغب اللاعبون الآخرون في اللعب ضد روبوت. تابع فقط إذا كانت قواعد اللعبة تسمح بذلك وكنت تقبل هذا الخطر.",
    ),
    "waiting_for_game": (
        "Click on your game window so I can play — I'm waiting.",
        "اضغط على نافذة اللعبة حتى أستطيع اللعب — أنا أنتظر.",
    ),
    "stopped_hint": (
        "I'm stopped — press Start (or type 'continue').",
        "أنا متوقف — اضغط ابدأ (أو اكتب 'أكمل').",
    ),
    "settings_error": (
        "There's a mistake in settings.yaml: {detail}",
        "يوجد خطأ في ملف الإعدادات settings.yaml: {detail}",
    ),
    "speed_hint": (
        "Tip: for faster steps, lower image_width to 640 or less in settings.yaml.",
        "نصيحة: لخطوات أسرع، خفض image_width إلى 640 أو أقل في settings.yaml.",
    ),
    "card_mismatch": (
        "The game '{game}' works best with the {card} card. Want to switch?",
        "اللعبة '{game}' تعمل بشكل أفضل مع بطاقة {card}. هل تريد التبديل؟",
    ),
    "no_knowledge_yet": (
        "No game knowledge yet. Play a few rounds to build it.",
        "لا توجد معرفة باللعبة بعد. العب بضع جولات لبنائها.",
    ),
    "mode_you": (
        "Your turn! I let go of every key and I'm just watching. Press F9 to give me the controls back.",
        "دورك! تركت كل الأزرار وأنا أتفرج فقط. اضغط F9 لتعيد لي التحكم.",
    ),
    "mode_brain": (
        "My turn again! (Press F9 any time to take over.)",
        "دوري من جديد! (اضغط F9 في أي وقت لتتولى القيادة.)",
    ),
    "recording_on": (
        "● Recording your game keys",
        "● أسجّل أزرار لعبتك",
    ),
    "recording_frames": (
        "● Recording your game keys: {frames} pictures",
        "● أسجّل أزرار لعبتك: {frames} صورة",
    ),
    "recording_paused": (
        "● Recording paused: click your game to go on (nothing is kept from other windows)",
        "● التسجيل متوقف مؤقتًا: اضغط على لعبتك للمتابعة (لا يُحفظ شيء من النوافذ الأخرى)",
    ),
    "recorded": (
        "Saved your recording on this PC: {folder}",
        "حُفظ تسجيلك على هذا الجهاز: {folder}",
    ),
    "nothing_recorded": (
        "Nothing was recorded (your game wasn't in front).",
        "لم يُسجَّل شيء (لعبتك لم تكن في الأمام).",
    ),
    "start_first": (
        "Press Start first, then you can take over.",
        "اضغط ابدأ أولًا، ثم يمكنك تولي القيادة.",
    ),
    "warming_local": (
        "Waking up the local brain… (first time can take a few minutes)",
        "أوقظ العقل المحلي… (قد يستغرق ذلك بضع دقائق في المرة الأولى)",
    ),
}


class _Blank(dict):
    def __missing__(self, key: str) -> str:
        return ""


def msg(key: str, **values) -> str:
    """Return the message in the current language only."""
    english, arabic = MESSAGES[key]
    filled = _Blank(values)
    return tr(english.format_map(filled), arabic.format_map(filled))
