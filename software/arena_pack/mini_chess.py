"""Arena level 3 (board): Mini Chess. You play White against a simple computer (it looks 2 moves ahead).
Click one of your pieces, then the square to move it to. Win by checkmate, or by lasting 30 moves without being mated.
/ المستوى 3: شطرنج ضد الحاسوب. انقر قطعتك ثم المربع. افز بكش مات أو بالصمود 30 نقلة."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack import chess_rules as rules
from arena_pack.base import ArenaPackGame, font, run_window

SQUARE = 60
SURVIVE = 30
LIGHT, DARK = (238, 226, 200), (150, 110, 80)
PICKED, LAST = (124, 92, 255), (190, 220, 120)
MATE_POINTS = 500


class MiniChess(ArenaPackGame):
    name = "Mini Chess"
    name_ar = "شطرنج صغير"
    keys = ()  # mouse only
    max_ticks = 400  # clicks
    state_fields = ("pos", "picked", "moves_made", "result", "last")

    def new_game(self) -> None:
        self.pos = rules.start()
        self.picked = -1
        self.moves_made = 0
        self.result = None  # "mate" (you won), "mated", "draw"
        self.last = ()

    def on_click(self, x: float, y: float) -> None:
        col, row = int(x // SQUARE), int((y - self.pixel_size[1] + 8 * SQUARE) // SQUARE)
        if not (0 <= col < 8 and 0 <= row < 8):
            return
        sq = row * 8 + col
        piece = self.pos[0][sq]
        if piece != "." and piece.isupper():
            self.picked = sq  # pick (or re-pick) one of your pieces
            return
        if self.picked < 0:
            return
        move = (self.picked, sq)
        self.picked = -1
        if move not in rules.legal(self.pos):
            return  # not a legal move: nothing happens
        self.pos = rules.make(self.pos, move)
        self.moves_made += 1
        self.last = move
        if self._settle():
            return
        replies = rules.best_moves(self.pos, 2)
        reply = self.rng.choice(sorted(replies))
        self.pos = rules.make(self.pos, reply)
        self.last = reply
        self._settle()

    def _settle(self) -> bool:
        """Check for the end of the game after a move. True when it's over."""
        state = rules.status(self.pos)
        if state == "checkmate":
            self.result = "mated" if self.pos[1] else "mate"
            if self.result == "mate":
                self.score += MATE_POINTS
        elif state == "stalemate":
            self.result = "draw"
        elif self.moves_made >= SURVIVE and self.pos[1]:
            self.result = "survived"
            self.score += 100
        return self.result is not None

    def is_won(self) -> bool:
        return self.result in ("mate", "survived")

    def is_lost(self) -> bool:
        return self.result in ("mated", "draw")

    def progress(self) -> dict:
        return {"moves": self.moves_made, "survive": SURVIVE, "in_check": rules.in_check(self.pos, True),
                "material": rules.evaluate(self.pos[0]), "picked": self.picked}

    def progress_text(self) -> str:
        check = rules.in_check(self.pos, True)
        if self.language == "arabic":
            return f"نقلاتك: {self.moves_made}/{SURVIVE}" + ("    كش!" if check else "")
        return f"Your moves: {self.moves_made}/{SURVIVE}" + ("    Check!" if check else "")

    def board_pixels(self) -> tuple[int, int]:
        return 8 * SQUARE, 8 * SQUARE

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        letters = font(26)
        for sq, piece in enumerate(self.pos[0]):
            row, col = divmod(sq, 8)
            left, upper = col * SQUARE, top + row * SQUARE
            colour = LIGHT if (row + col) % 2 == 0 else DARK
            if sq == self.picked:
                colour = PICKED
            elif sq in self.last:
                colour = LAST
            draw.rectangle((left, upper, left + SQUARE - 1, upper + SQUARE - 1), fill=colour)
            if piece != ".":
                mine = piece.isupper()
                draw.ellipse((left + 8, upper + 8, left + SQUARE - 8, upper + SQUARE - 8),
                             fill=(250, 250, 250) if mine else (30, 30, 40), outline=(20, 20, 20), width=2)
                text = piece.upper()
                box = draw.textbbox((0, 0), text, font=letters)
                draw.text((left + (SQUARE - box[2]) / 2, upper + (SQUARE - box[3]) / 2 - 3), text,
                          fill=(30, 30, 40) if mine else (250, 250, 250), font=letters)


if __name__ == "__main__":
    run_window(MiniChess, "PlayFellow Arena: Mini Chess")
