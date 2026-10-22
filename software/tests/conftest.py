"""Test setup that runs for every test: no real keys, mouse, credential store or personal settings ever."""
from pathlib import Path

import pytest

# This PC's own settings.local.yaml must never leak into tests: the shipped defaults are what tests check. Only that
# one real file is hidden; a settings.local.yaml a test writes in its own tmp_path still works.
_REAL_LOCAL_SETTINGS = (Path(__file__).resolve().parents[1] / "settings.local.yaml")


@pytest.fixture(autouse=True)
def hide_this_pcs_local_settings(monkeypatch):
    real_is_file = Path.is_file

    def is_file(self):
        try:
            if self.resolve() == _REAL_LOCAL_SETTINGS:
                return False
        except OSError:
            pass
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", is_file)


@pytest.fixture(autouse=True)
def reset_language():
    """Reset to English before and after each test."""
    from messages import set_language
    set_language("english")
    yield
    set_language("english")


@pytest.fixture
def simple_card():
    """A simple card for testing with basic actions."""
    from controls import ControlsCard, ControlAction

    return ControlsCard(
        name="test",
        game_hint="test game",
        actions={
            "UP": ControlAction(name="UP", kind="keys", keys=["up"]),
            "DOWN": ControlAction(name="DOWN", kind="keys", keys=["down"]),
            "JUMP": ControlAction(name="JUMP", kind="keys", keys=["space"]),
            "MOVE": ControlAction(name="MOVE", kind="keys", keys=["w"]),
            "KEY": ControlAction(name="KEY", kind="keys", keys=["k"]),
            "WAIT": ControlAction(name="WAIT", kind="keys", keys=["w"])
        }
    )


@pytest.fixture(autouse=True)
def no_real_input(monkeypatch):
    """A test must never press a real key or move the real mouse: if one ever reaches the real pydirectinput, it fails
    loudly instead of acting on the PC."""
    import sys
    import types

    stub = types.ModuleType("pydirectinput")
    stub.FAILSAFE = True
    stub.PAUSE = 0.0

    def forbidden(name):
        def call(*args, **kwargs):
            raise AssertionError(f"a test tried to use the real pydirectinput.{name}()")
        return call

    for name in ("keyDown", "keyUp", "press", "click", "moveTo", "moveRel", "mouseDown", "mouseUp", "typewrite",
                 "write", "scroll", "dragTo", "dragRel"):
        setattr(stub, name, forbidden(name))
    monkeypatch.setitem(sys.modules, "pydirectinput", stub)
    yield


@pytest.fixture(autouse=True)
def no_real_keyring():
    """No test ever reads or writes the real credential store (AI keys): every test gets an empty in-memory keyring
    and the real one is put back afterwards."""
    import keyring
    import keyring.backend

    class Memory(keyring.backend.KeyringBackend):
        priority = 1

        def __init__(self):
            super().__init__()
            self.store = {}

        def get_password(self, service, username):
            return self.store.get((service, username))

        def set_password(self, service, username, password):
            self.store[(service, username)] = password

        def delete_password(self, service, username):
            self.store.pop((service, username), None)

    previous = keyring.get_keyring()
    keyring.set_keyring(Memory())
    yield
    keyring.set_keyring(previous)


@pytest.fixture(autouse=True)
def no_real_data_folder(monkeypatch, tmp_path_factory):
    """No test writes into the real per-user data folder: the default is a temp one."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path_factory.mktemp("localappdata")))
