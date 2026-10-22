"""Chess rules for Mini Chess: moves, check, checkmate, castling, en passant, promotion (always to a queen).

A position is (board, white_to_move, castling, en_passant): board is 64 characters, square 0 is a8 (top-left as
drawn) and 63 is h1; upper case is White (the player), lower case is Black (the computer), '.' is empty.
"""
from __future__ import annotations

START = tuple("rnbqkbnr" "pppppppp" "........" "........" "........" "........" "PPPPPPPP" "RNBQKBNR")
VALUES = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0}
KNIGHT = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
STRAIGHT = ((-1, 0), (1, 0), (0, -1), (0, 1))
CENTRE = [[0, 1, 2, 3, 3, 2, 1, 0][c] + [0, 1, 2, 3, 3, 2, 1, 0][r] for r in range(8) for c in range(8)]

Position = tuple  # (board tuple, white_to_move, castling str, en_passant square or -1)


def start() -> Position:
    return START, True, "KQkq", -1


def white(piece: str) -> bool:
    return piece.isupper()


def _rays(board, sq, steps, slide):
    r, c = divmod(sq, 8)
    for dr, dc in steps:
        nr, nc = r + dr, c + dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            yield nr * 8 + nc
            if not slide or board[nr * 8 + nc] != ".":
                break
            nr, nc = nr + dr, nc + dc


def attacked(board, sq: int, by_white: bool) -> bool:
    """Is square `sq` attacked by the given side?"""
    r, c = divmod(sq, 8)
    pawn_row = r + 1 if by_white else r - 1  # white pawns attack upwards (towards row 0)
    for dc in (-1, 1):
        if 0 <= pawn_row < 8 and 0 <= c + dc < 8:
            p = board[pawn_row * 8 + c + dc]
            if p == ("P" if by_white else "p"):
                return True
    for steps, slide, kinds in ((KNIGHT, False, "n"), (KING, False, "k"), (DIAGONAL, True, "bq"), (STRAIGHT, True, "rq")):
        for t in _rays(board, sq, steps, slide):
            p = board[t]
            if p != "." and white(p) == by_white and p.lower() in kinds:
                return True
    return False


def king_square(board, of_white: bool) -> int:
    return board.index("K" if of_white else "k")


def in_check(pos: Position, of_white: bool) -> bool:
    return attacked(pos[0], king_square(pos[0], of_white), not of_white)


def moves(pos: Position) -> list[tuple[int, int]]:
    """Pseudo-legal moves (from, to) for the side to move; castling only through unattacked squares."""
    board, w, castling, ep = pos
    out = []
    for sq, p in enumerate(board):
        if p == "." or white(p) != w:
            continue
        kind = p.lower()
        r, c = divmod(sq, 8)
        if kind == "p":
            step = -8 if w else 8
            one = sq + step
            if 0 <= one < 64 and board[one] == ".":
                out.append((sq, one))
                if (r == 6 if w else r == 1) and board[one + step] == ".":
                    out.append((sq, one + step))
            for dc in (-1, 1):
                if 0 <= c + dc < 8:
                    t = one + dc
                    if 0 <= t < 64 and ((board[t] != "." and white(board[t]) != w) or t == ep):
                        out.append((sq, t))
            continue
        steps, slide = {"n": (KNIGHT, False), "k": (KING, False), "b": (DIAGONAL, True), "r": (STRAIGHT, True),
                        "q": (DIAGONAL + STRAIGHT, True)}[kind]
        for t in _rays(board, sq, steps, slide):
            if board[t] == "." or white(board[t]) != w:
                out.append((sq, t))
        if kind == "k":
            home = 60 if w else 4
            rights = "KQ" if w else "kq"
            if sq == home and not attacked(board, home, not w):
                if rights[0] in castling and board[home + 1] == board[home + 2] == "." \
                        and not attacked(board, home + 1, not w) and board[home + 3] == ("R" if w else "r"):
                    out.append((home, home + 2))
                if rights[1] in castling and board[home - 1] == board[home - 2] == board[home - 3] == "." \
                        and not attacked(board, home - 1, not w) and board[home - 4] == ("R" if w else "r"):
                    out.append((home, home - 2))
    return out


def make(pos: Position, move: tuple[int, int]) -> Position:
    board, w, castling, ep = pos
    frm, to = move
    b = list(board)
    piece = b[frm]
    kind = piece.lower()
    b[frm] = "."
    if kind == "p" and to == ep:
        b[to + (8 if w else -8)] = "."  # en passant takes the pawn beside us
    if kind == "k" and abs(to - frm) == 2:  # castling moves the rook too
        rook_from, rook_to = (frm + 3, frm + 1) if to > frm else (frm - 4, frm - 1)
        b[rook_to], b[rook_from] = b[rook_from], "."
    if kind == "p" and (to < 8 or to >= 56):
        piece = "Q" if w else "q"
    b[to] = piece
    new_ep = (frm + to) // 2 if kind == "p" and abs(to - frm) == 16 else -1
    lost = {60: "KQ", 4: "kq", 63: "K", 56: "Q", 7: "k", 0: "q"}
    for sq in (frm, to):
        for right in lost.get(sq, ""):
            castling = castling.replace(right, "")
    return tuple(b), not w, castling, new_ep


def legal(pos: Position) -> list[tuple[int, int]]:
    w = pos[1]
    return [m for m in moves(pos) if not in_check(make(pos, m), w)]


def status(pos: Position) -> str:
    """'play', 'checkmate' (the side to move has lost) or 'stalemate'."""
    if legal(pos):
        return "play"
    return "checkmate" if in_check(pos, pos[1]) else "stalemate"


def evaluate(board) -> int:
    """Material and a little centre control, from White's point of view."""
    score = 0
    for sq, p in enumerate(board):
        if p != ".":
            v = VALUES[p.lower()] + (4 * CENTRE[sq] if p.lower() in "nbp" else 0)
            score += v if white(p) else -v
    return score


def search(pos: Position, depth: int, alpha: float = -1e9, beta: float = 1e9) -> float:
    """Negamax with alpha-beta over pseudo-legal moves; taking the king ends the line (fast legality)."""
    board, w = pos[0], pos[1]
    if "K" not in board or "k" not in board:
        return -100000 if ("K" if w else "k") not in board else 100000
    if depth == 0:
        return evaluate(board) if w else -evaluate(board)
    best = -1e9
    ordered = sorted(moves(pos), key=lambda m: VALUES.get(board[m[1]].lower(), 0) if board[m[1]] != "." else -1, reverse=True)
    for m in ordered:
        score = -search(make(pos, m), depth - 1, -beta, -alpha)
        if score > best:
            best = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    if best <= -99000 and not in_check(pos, w):
        return 0  # every move loses the king but we're not in check: stalemate
    return best


def best_moves(pos: Position, depth: int) -> list[tuple[int, int]]:
    """All legal moves that share the best search score (callers break ties)."""
    best, out = -1e9, []
    for m in legal(pos):
        # Only "is it at least as good as the best so far?" matters, so search with the window [best - 1, ∞):
        # a worse move fails low cheaply, and equal moves still come back with their exact score.
        score = -search(make(pos, m), depth - 1, -1e9, -(best - 1))
        if score > best:
            best, out = score, [m]
        elif score == best:
            out.append(m)
    return out
