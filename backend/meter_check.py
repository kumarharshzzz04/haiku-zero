"""Module 10 - syllable counting and the 5-7-5 check.

A heuristic counter with no dependency: vowel groups, silent-e and a table of words the
rules get wrong. It is accurate to about one syllable on unusual words, so anything that
compares against it uses a tolerance.
"""
from __future__ import annotations

import re
import sys

VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
WORD_RE = re.compile(r"[a-z]+")

# Words the generic rules get wrong (poetry-common ones first).
IRREGULAR = {
    "poem": 2, "poems": 2, "poet": 2, "poetry": 3,
    "being": 2, "doing": 2, "going": 2, "seeing": 2, "dying": 2, "lying": 2,
    "create": 2, "creates": 2, "created": 3, "creating": 3,
    "people": 2, "quiet": 2, "often": 2, "every": 2, "evening": 2,
    "science": 2, "silence": 2, "violet": 3, "diary": 3, "camera": 3,
    "fire": 1, "fired": 1, "fires": 1, "hour": 1, "hours": 1, "our": 1,
    "flower": 2, "flowers": 2, "power": 2, "shower": 2, "tower": 2,
    "heaven": 2, "seven": 2, "never": 2, "river": 2, "over": 2,
    "the": 1, "a": 1, "i": 1, "you": 1, "are": 1, "was": 1, "were": 1,
}


def count_syllables(word: str) -> int:
    w = word.lower().strip()
    plain = "".join(c for c in w if c.isalpha())
    if not plain:
        return 0                       # digits/punctuation only
    if plain in IRREGULAR:
        return IRREGULAR[plain]
    n = len(VOWEL_GROUP_RE.findall(plain))
    if plain.endswith("e") and not plain.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1                         # silent e: "time" = 1, "gone" = 1
    if plain.endswith("ed") and n > 1 and plain[-3] not in "td":
        n -= 1                         # "walked" = 1, but "wanted" stays 2
    return max(1, n)


def line_syllables(line: str) -> int:
    return sum(count_syllables(w) for w in WORD_RE.findall(line.lower()))


def pattern(lines) -> tuple:
    return tuple(line_syllables(l) for l in lines[:3])


def distance(pat: tuple) -> int:
    """Total syllabic distance from strict 5-7-5. 0 = exact haiku."""
    pat = tuple(pat) + (0,) * (3 - len(pat))
    return abs(pat[0] - 5) + abs(pat[1] - 7) + abs(pat[2] - 5)


def is_haiku(lines) -> bool:
    return distance(pattern(lines)) == 0


def closest(candidates) -> tuple[list[str], tuple, bool]:
    """Best candidate by meter. Returns (lines, pattern, strict_haiku?)."""
    best = min(candidates, key=lambda c: distance(pattern(c)))
    pat = pattern(best)
    return best, pat, distance(pat) == 0


def main(argv=None) -> None:
    lines = list(argv if argv is not None else sys.argv[1:])
    if not lines:
        print("usage: python -m backend.meter_check \"line1\" \"line2\" \"line3\"")
        return
    pat = pattern(lines)
    print(f"pattern {pat[0]}-{pat[1]}-{pat[2]}  (target 5-7-5, distance {distance(pat)})")
    for l in lines[:3]:
        print(f"  {line_syllables(l):>2}  {l}")


if __name__ == "__main__":
    main()
