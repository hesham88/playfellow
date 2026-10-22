from PIL import Image

from memory import Memory, Moment


def moment(n):
    return Moment(Image.new("RGB", (4, 4)), f"thought {n}", (f"ACT{n}",), "")


def test_keeps_only_the_last_n():
    memory = Memory(2)
    for n in range(4):
        memory.add(moment(n))
    assert [m.thought for m in memory.recent()] == ["thought 2", "thought 3"]


def test_size_zero_remembers_nothing():
    memory = Memory(0)
    memory.add(moment(1))
    assert memory.recent() == [] and memory.as_text() == ""


def test_as_text_reads_oldest_first():
    memory = Memory(3)
    memory.add(Moment(Image.new("RGB", (4, 4)), "saw a coin", ("UP",), "Going up!"))
    memory.add(Moment(Image.new("RGB", (4, 4)), "", (), ""))
    lines = memory.as_text().splitlines()
    assert lines[0].startswith("2 steps ago") and "saw a coin" in lines[0] and "UP" in lines[0] and "Going up!" in lines[0]
    assert lines[1].startswith("1 step ago") and "nothing" in lines[1]


def test_clear():
    memory = Memory(3)
    memory.add(moment(1))
    memory.clear()
    assert memory.recent() == []
