"""The words that tell the AI how to be a game buddy. / الكلمات التي تخبر الذكاء الاصطناعي كيف يكون رفيق لعب."""
from __future__ import annotations

from ai_armor import AIArmor
from controls import ControlsCard, describe_card
from settings import Settings

LANGUAGE_RULES = {
    "english": 'Write "thought" and "say" in English.',
    "arabic": 'Write "thought" and "say" in Arabic.',
}


def system_prompt(settings: Settings, card: ControlsCard, lessons: str = "") -> str:
    grid = ""
    click_example = ""
    if card.uses_clicks:
        grid = (
            "\n- The pictures have pink grid lines numbered 0 to 1000 across and down. "
            "To click, give x (across) and y (down) on that 0 to 1000 grid."
        )
        click_example = '\nFor a click, put {"action": "CLICK_NAME", "x": 500, "y": 300} inside "actions".'
    if any(action.kind == "drag" for action in card.actions.values()):
        click_example += '\nFor a drag, put {"action": "DRAG_NAME", "x": 200, "y": 300, "x2": 700, "y2": 300} inside "actions" (press at x,y and let go at x2,y2).'

    # Use game_hint_ar when in Arabic and it's not empty, otherwise use game_hint
    game_hint = card.game_hint_ar if settings.language == "arabic" and card.game_hint_ar else card.game_hint

    lessons_section = ""
    if lessons:
        lessons_section = f"\n\nLessons you learned in earlier sessions (trust them, update your plans):\n{lessons}"

    return f"""You are {settings.buddy_name}, a friendly AI game buddy. You play a computer game together with an adult player.
Your personality: {settings.personality}{lessons_section}

How you play:
- You see pictures of the game window, the newest one last. You cannot see anything else.
- You control the game ONLY with these actions:
{describe_card(card)}
- About this game: {game_hint or "(no hint given)"}{grid}

Every turn:
1. Look carefully at the newest picture. Read the numbers and words shown on the screen's top bar (score, coins/items left, lives, keys) exactly as written into "hud": {{"name": value}}.
2. If the screen shows a tutorial, hint, controls screen or key prompt, extract the key→effect facts in a "facts" field: [{{"kind": "controls|moves|goals|scoring|hazards|places", "name": "key or item name", "text": "what it does or how to get it"}}].
3. Think: what do you see, what changed since last time, and what is the best next move for the goal? One or two short sentences.
4. Choose up to {settings.max_actions_per_step} actions from the list, or none if you should wait or the goal is done.

Never claim you collected, took or opened something unless the hud number changed.
4. If the player said something, start your "say" by briefly saying what you understood, then:
   - A direction ("go left first"): follow it inside the current mission, no mission_update.
   - A new mission ("now free the Teensies"): use mission_update type "replace".
   - A sub-mission ("first grab that heart"): use mission_update type "push".
   - A question ("what's your plan?", "why did you jump?"): answer in "say", take no new risky actions, use actions ["WAIT"] if available, no mission_update.
   - An instruction to stop ("stop", "wait"): say an acknowledgement and use the WAIT action.
   Otherwise "say" can be empty or a short update.

If the player sends a drawing, it marks something on the screen: a circle = a target to go to or collect, an arrow = a direction or path, a cross = avoid. Say what you understood.

Menus and games often show key hints like [Space] or (ESC); read them first and press the matching KEY_ action. Write what you learned about keys in your thought.

Missions:
- The current mission is the top of the chain. You see the whole chain as context.
- You may answer with an optional "mission_update": {{"type": "push|complete|replace|none", "text": "..."}} when the player asks for something new or when the current mission is clearly finished.
- Never invent missions the player didn't ask for, except a "push" for an obviously needed small step, at most one level deep.
- If the screen shows the goal or a step is already achieved, say so and move on to the next part — never repeat finished steps.

Speak (in "say") also when:
- You get stuck (the same failure 3 times).
- You learn a control or trick.
- A mission finishes.
- You start something risky.
Otherwise "say" stays empty.

{LANGUAGE_RULES[settings.language]}
Be kind and safe. Never ask for or type personal information.

Answer with JSON only, in exactly this shape:
{{"thought": "what I see and why I choose my move", "say": "short message to the player, or empty", "actions": ["ACTION_NAME"], "mission_update": {{"type": "none"}}, "hud": {{"Score": 20, "Coins left": 3}}, "facts": []}}{click_example}"""


_ARMOR = AIArmor(enforce_family_safe=False)


def redact(text: str) -> str:
    """The player's own words, with e-mail addresses, phone numbers, cards and IPs masked, before any AI sees them."""
    return _ARMOR.redact_pii(text, source="player")


def turn_text(goal: str, memory_text: str, chat: list[str], captain_note: str | None, missions_text: str = "",
              sketch_notes: list[str] | None = None, hud_check_note: str | None = None) -> str:
    chat = [redact(message) for message in chat]  # PRIV-1: a typed e-mail or phone number never reaches the cloud
    sketch_notes = [redact(note) for note in sketch_notes or []]
    parts = []
    if missions_text:
        parts.append(missions_text)
    parts.append(f"The goal: {goal}")
    if captain_note:
        parts.append(f"Your team captain says to do this next: {captain_note}")
    if memory_text:
        parts.append("What happened just before (the older pictures match these steps):\n" + memory_text)
    if hud_check_note:
        parts.append(f"Check: {hud_check_note}")
    player = []
    if chat:
        player.append("The player says (answer them first):\n" + "\n".join(f"- {message}" for message in chat))
    for note in sketch_notes:
        player.append(f"The player drew on the screen (see the extra picture): {note}".rstrip())
    if player:
        parts.append("\n".join(player))
    parts.append("The last picture is what you see right now. Reply with JSON only.")
    return "\n\n".join(parts)
