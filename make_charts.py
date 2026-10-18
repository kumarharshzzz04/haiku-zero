"""Build the README charts as plain SVG from the training logs (no plotting dependency).

    python make_charts.py            # writes docs/images/*.svg
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "images"

INK = "#241a20"
MUTED = "#8a7480"
GRID = "#eadfe4"
SERIES = ["#b4415f", "#d98a6a", "#5c7d8a"]


def read_log(name: str, field: str = "val_loss") -> list[tuple[int, float]]:
    path = ROOT / name
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return [(int(r["step"]), float(r[field])) for r in rows if r[field]]


def svg(width: int, height: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Georgia, serif">\n'
            f'<rect width="{width}" height="{height}" fill="#fffaf8"/>\n{body}</svg>\n')


class Frame:
    def __init__(self, width, height, title, subtitle, x_range, y_range, x_label, y_label):
        self.w, self.h = width, height
        self.left, self.right, self.top, self.bottom = 78, 28, 84, 96
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.parts = [
            f'<text x="{self.left}" y="38" font-size="21" fill="{INK}">{title}</text>',
            f'<text x="{self.left}" y="62" font-size="13" fill="{MUTED}">{subtitle}</text>',
        ]
        self._axes(x_label, y_label)

    def px(self, x):
        return self.left + (x - self.x0) / (self.x1 - self.x0) * (self.w - self.left - self.right)

    def py(self, y):
        return self.h - self.bottom - (y - self.y0) / (self.y1 - self.y0) * (self.h - self.top - self.bottom)

    def _axes(self, x_label, y_label):
        for i in range(5):
            y = self.y0 + (self.y1 - self.y0) * i / 4
            py = self.py(y)
            self.parts.append(f'<line x1="{self.left}" y1="{py:.1f}" x2="{self.w - self.right}" '
                              f'y2="{py:.1f}" stroke="{GRID}" stroke-width="1"/>')
            self.parts.append(f'<text x="{self.left - 10}" y="{py + 4:.1f}" font-size="12" '
                              f'fill="{MUTED}" text-anchor="end">{y:g}</text>')
        for i in range(6):
            x = self.x0 + (self.x1 - self.x0) * i / 5
            px = self.px(x)
            self.parts.append(f'<text x="{px:.1f}" y="{self.h - self.bottom + 18}" font-size="12" '
                              f'fill="{MUTED}" text-anchor="middle">{x:,.0f}</text>')
        self.parts.append(f'<line x1="{self.left}" y1="{self.h - self.bottom}" x2="{self.w - self.right}" '
                          f'y2="{self.h - self.bottom}" stroke="{MUTED}" stroke-width="1"/>')
        self.parts.append(f'<text x="{(self.left + self.w) / 2:.0f}" y="{self.h - 44}" font-size="12" '
                          f'fill="{MUTED}" text-anchor="middle">{x_label}</text>')
        self.parts.append(f'<text transform="translate(22 {self.h / 2:.0f}) rotate(-90)" font-size="12" '
                          f'fill="{MUTED}" text-anchor="middle">{y_label}</text>')

    def line(self, points, color, label):
        path = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in points)
        self.parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.4" '
                          f'stroke-linejoin="round"/>')
        self.parts.append(f'<circle cx="{self.px(points[-1][0]):.1f}" cy="{self.py(points[-1][1]):.1f}" '
                          f'r="3.6" fill="{color}"/>')

    def legend(self, entries):
        x = self.left
        for label, color in entries:
            self.parts.append(f'<line x1="{x}" y1="{self.h - 22}" x2="{x + 22}" y2="{self.h - 22}" '
                              f'stroke="{color}" stroke-width="2.6"/>')
            self.parts.append(f'<text x="{x + 30}" y="{self.h - 18}" font-size="12.5" '
                              f'fill="{INK}">{label}</text>')
            x += 44 + 7.2 * len(label)

    def note(self, text):
        self.parts.append(f'<text x="{self.w - self.right}" y="{self.h - 18}" font-size="12" '
                          f'fill="{MUTED}" text-anchor="end">{text}</text>')

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(svg(self.w, self.h, "\n".join(self.parts)), encoding="utf-8")
        print("wrote", (OUT / name).relative_to(ROOT))


def pretrain_curve():
    runs = [("training_log_pretrain.csv", "v1 · 15.4M tokens", SERIES[1]),
            ("training_log_pretrain_v2.csv", "v2 · 54.4M tokens", SERIES[0]),
            ("training_log_pretrain_cap.csv", "v2 · 33.6M params", SERIES[2])]
    frame = Frame(760, 340, "Stage 1 — learning English on public-domain poetry",
                  "validation loss after each epoch, same tokenizer and format for all three runs",
                  (0, 34000), (3.2, 4.4), "training steps", "val loss")
    entries = []
    for log, label, color in runs:
        frame.line(read_log(log), color, label)
        entries.append((label, color))
    frame.legend(entries)
    frame.note("lower is better")
    frame.save("stage1.svg")


def finetune_curve():
    frame = Frame(760, 340, "Stage 2 — learning haiku form",
                  "validation loss per epoch on the authentic haiku corpus",
                  (1, 25), (3.7, 4.05), "epoch", "val loss")
    runs = [("training_log_poetry_v2.csv", "15.8M · 25 epochs", SERIES[0]),
            ("training_log_cap.csv", "33.6M · overfits", SERIES[2])]
    entries = []
    for log, label, color in runs:
        pts = [(int(r["step"]), float(r["val_loss"]))
               for r in csv.DictReader((ROOT / log).open(encoding="utf-8")) if r["val_loss"]]
        # collapse to one point per epoch
        best: dict[int, float] = {}
        for row in csv.DictReader((ROOT / log).open(encoding="utf-8")):
            if row["val_loss"]:
                best[int(row["epoch"])] = float(row["val_loss"])
        frame.line(sorted(best.items()), color, label)
        entries.append((label, color))
    frame.legend(entries)
    frame.note("33.6M memorises the 1M-token corpus")
    frame.save("stage2.svg")


def corpus_bars():
    frame = Frame(660, 360, "What changed between the two pretraining corpora",
                  "same pipeline, a wider budget: 3.5× the text and 2.1× the vocabulary",
                  (0, 3), (0, 3), "", "")
    frame.parts.clear()  # this one is a bar chart, not a line plot
    items = [("BPE tokens", 15.41, 54.35, "M"), ("unique words", 120.5, 272.0, "k"),
             ("books", 611, 1954, "")]
    bar_w, gap = 46, 74
    x = 110
    base = 300
    frame.parts.append(f'<line x1="70" y1="{base}" x2="600" y2="{base}" stroke="{MUTED}" stroke-width="1"/>')
    for label, before, after, unit in items:
        peak = max(before, after)
        for i, (value, color, tag) in enumerate(((before, "#e7c3ce", "v1"), (after, "#b4415f", "v2"))):
            h = 170 * value / peak
            bx = x + i * (bar_w + 12)
            by = base - h
            frame.parts.append(f'<rect x="{bx}" y="{by:.1f}" width="{bar_w}" height="{h:.1f}" '
                               f'rx="3" fill="{color}"/>')
            frame.parts.append(f'<text x="{bx + bar_w / 2}" y="{by - 8:.1f}" font-size="13" '
                               f'fill="{INK}" text-anchor="middle">{value:g}{unit}</text>')
            frame.parts.append(f'<text x="{bx + bar_w / 2}" y="{base + 18}" font-size="11.5" '
                               f'fill="{MUTED}" text-anchor="middle">{tag}</text>')
        frame.parts.append(f'<text x="{x + bar_w + 6}" y="{base + 38}" font-size="12.5" fill="{INK}" '
                           f'text-anchor="middle">{label}</text>')
        x += 2 * bar_w + 12 + gap
    frame.save("corpus.svg")


if __name__ == "__main__":
    pretrain_curve()
    finetune_curve()
    corpus_bars()
