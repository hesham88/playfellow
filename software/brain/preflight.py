"""Look before you load: is there room for this model? / تحقق قبل التحميل: هل يتسع الجهاز لهذا النموذج؟

A local model that does not fit in memory can freeze this laptop (it did, twice: D90). Like checking a lorry's height before
the bridge, the check comes BEFORE the engine is asked to load anything, including LM Studio's load-on-first-request and
Ollama's load on the first chat. Standard library only; it asks no server and starts no other program, so it can run on the window's thread.
Memory is read at the moment of every call, never cached.
"""
from __future__ import annotations

import ctypes
import re
from dataclasses import dataclass
from typing import Callable

from brain import BrainError

GIB = 1024 ** 3
FILE_FACTOR = 1.25                          # the model file in memory is a quarter bigger than on disk (buffers, compute)
KV_BYTES_PER_TOKEN = 128 * 1024             # the attention cache a small vision model keeps per token of context
VISION_ENCODER_BYTES = 768 * 1024 * 1024    # the picture reader that rides along with a vision model
OS_MARGIN = GIB                             # what Windows and the game itself keep
DEFAULT_CONTEXT_TOKENS = 8192
UNKNOWN_MODEL_BYTES = 8 * GIB               # a model whose size we cannot tell is assumed big, never small
VRAM_USABLE = 0.9                           # keep a tenth of the graphics memory for the display
VRAM_CREDIT = 0.5                           # the engine, not us, decides what goes to the GPU: only half counts until real offload is verified here
BLOCKED = re.compile(r"qwen[\W_]*3[\W_]*vl[\W_]*4b", re.IGNORECASE)  # D90: crashed the owner's laptop twice


@dataclass(frozen=True)
class Memory:
    free_ram: int
    free_vram: int | None  # None = no graphics card we can read


@dataclass(frozen=True)
class Plan:
    need: int        # bytes the model needs altogether
    offload: int     # bytes planned for the graphics memory (0 when it cannot help)
    ram_needed: int  # bytes that must be free in ordinary memory after the offload


def check_blocked(model: str) -> None:
    """D90: refused by name, before ANY request of any kind (not even a model-list request), whatever the memory says."""
    if BLOCKED.search(model or ""):
        raise BrainError("model_blocked", retry=False, detail=model)


def _size(value) -> int | None:
    """A usable size in bytes, else None: a missing, non-numeric or zero size is unknown, never trusted."""
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def pick_ollama(tags, running, model: str) -> tuple[int | None, bool]:
    """(size, already loaded) for THIS exact model from Ollama's /api/tags and /api/ps answers; a malformed answer gives (None, False).
    Fields: name, model, size (https://github.com/ollama/ollama/blob/main/docs/api.md)."""
    wanted = (model, f"{model}:latest") if ":" not in model else (model,)  # a bare name means its :latest tag, never another tag

    def find(answer):
        for item in (answer.get("models") if isinstance(answer, dict) else None) or []:
            if isinstance(item, dict) and (item.get("name") in wanted or item.get("model") in wanted):
                return item
        return None
    listed, active = find(tags), find(running)
    return _size((active or listed or {}).get("size")), active is not None


def pick_lmstudio(answer, model: str) -> tuple[int | None, bool]:
    """(size, already loaded) for THIS exact model from LM Studio's model list. Documented fields: size_bytes and loaded_instances
    (https://lmstudio.ai/docs/developer/rest/list); the older list's state == "loaded" counts too. Malformed gives (None, False).
    Not yet checked against a real response (LM Studio did not answer when asked): an absent field means the conservative guess."""
    items = (answer.get("data") or answer.get("models")) if isinstance(answer, dict) else None
    for item in items or []:
        if isinstance(item, dict) and model in (item.get("id"), item.get("key")):
            loaded = bool(item.get("loaded_instances")) or item.get("state") == "loaded"
            return _size(item.get("size_bytes")), loaded
    return None, False


def estimate_need(model_file_bytes: int, context_tokens: int, has_vision: bool) -> int:
    """The one place the formula lives: file x 1.25 + the context's cache + the picture reader (if any) + 1 GB for the OS."""
    return int(model_file_bytes * FILE_FACTOR + context_tokens * KV_BYTES_PER_TOKEN
               + (VISION_ENCODER_BYTES if has_vision else 0) + OS_MARGIN)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                ("avail", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _free_vram() -> int | None:
    """The first graphics card's free memory through the driver's own library (no other program is started); None when it cannot be read, in which
    case the preflight counts ordinary memory only (fail safe)."""
    for path in (r"C:\Windows\System32\nvml.dll",):  # where current drivers install it; if it is missing, VRAM is unknown
        try:
            nvml = ctypes.WinDLL(path)
            if nvml.nvmlInit_v2() != 0:
                continue
            try:
                handle, info = ctypes.c_void_p(), _NvmlMemory()
                if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                    continue
                if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)) != 0:
                    continue
                return int(info.free)
            finally:
                nvml.nvmlShutdown()
        except (OSError, AttributeError):
            continue
    return None


def read_ram() -> int:
    """Free ordinary memory in bytes (Windows' own counter), right now: microseconds. Raises OSError when unreadable."""
    status = _MemoryStatus()
    status.length = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("memory status unavailable")
    return int(status.avail)


def read_memory() -> Memory:
    """Free ordinary memory and free graphics memory, right now (the graphics library's first load can take about a second)."""
    return Memory(read_ram(), _free_vram())


def ensure_fits(model: str, model_bytes: int | None, context_tokens: int = DEFAULT_CONTEXT_TOKENS, has_vision: bool = True,
                reader: Callable[[], Memory] | None = None, already_loaded: bool = False) -> Plan:
    """Raise BrainError (model_blocked, memory_unknown, model_too_big) before a load that must not happen; else the plan.
    `reader` is looked up on every call so a test (or a later setting) can swap it; nothing is remembered between calls."""
    check_blocked(model)
    if already_loaded:  # the engine says THIS model is in memory already: its memory is spent, so free memory proves nothing
        return Plan(need=0, offload=0, ram_needed=0)
    size = model_bytes if model_bytes else UNKNOWN_MODEL_BYTES
    need = estimate_need(size, context_tokens, has_vision)
    try:
        if reader is None:  # the real thing: ordinary memory first; the graphics card is only asked when RAM alone is not enough
            memory = Memory(read_ram(), None)
            if memory.free_ram < need:
                memory = read_memory()
        else:
            memory = reader()
        if memory.free_ram <= 0:
            raise OSError("no usable reading")
    except Exception:
        raise BrainError("memory_unknown", retry=False, detail=model) from None
    offload = 0
    if memory.free_vram:
        offload = min(int(memory.free_vram * VRAM_USABLE * VRAM_CREDIT), int(size * FILE_FACTOR))  # only what the graphics card can hold
    ram_needed = need - offload
    if memory.free_ram < ram_needed:
        raise BrainError("model_too_big", retry=False,
                         detail=f"{model}: needs {ram_needed / GIB:.1f} GB, only {memory.free_ram / GIB:.1f} GB free"
                                + ("" if model_bytes else f" (size unknown, assumed {UNKNOWN_MODEL_BYTES // GIB} GB)"))
    return Plan(need=need, offload=offload, ram_needed=ram_needed)
