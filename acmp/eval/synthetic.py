"""Synthetic comic-page generator with ground-truth panel boxes.

Real labeled comic datasets (e.g. Manga109) are license-restricted, so we
generate deterministic synthetic pages with *known* panel layouts. This gives
us a reproducible labeled set to:

  * unit-test the panel detector,
  * compute detection metrics (IoU / precision / recall / mAP),
  * and weakly-supervise / sanity-check a learned detector.

Each page is a white "paper" with a grid of solid-colored panels separated by
white gutters and bordered in black — the high-contrast layout typical of
manga/comic pages, which contour detection is designed to find.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

# (x, y, w, h) integer pixel box — the format used throughout acmp.
Box = tuple[int, int, int, int]

# A palette of distinct mid-tone fills so panels are clearly non-white.
_PALETTE = [
    (110, 140, 200), (200, 120, 110), (120, 190, 140), (200, 190, 110),
    (170, 130, 200), (120, 200, 200), (200, 150, 180), (150, 170, 120),
]


@dataclass
class SyntheticPage:
    """A generated page paired with its ground-truth panel boxes."""

    image: Image.Image
    boxes: list[Box]

    def __iter__(self):
        # Allow tuple-unpacking: image, boxes = page
        yield self.image
        yield self.boxes


def generate_comic_page(
    rows: int = 2,
    cols: int = 2,
    width: int = 1000,
    height: int = 1500,
    margin: int = 40,
    gutter: int = 40,
    border: int = 4,
    add_content: bool = True,
    seed: int = 0,
) -> SyntheticPage:
    """Generate one synthetic comic page with a regular grid of panels.

    Args:
        rows, cols: panel grid dimensions.
        width, height: page size in pixels.
        margin: white border around the whole page.
        gutter: white gap between adjacent panels (must exceed the detector's
            dilation so neighbours don't merge — keep >= ~20px).
        border: black panel border thickness in pixels.
        add_content: draw a few shapes inside each panel (more realistic).
        seed: RNG seed for deterministic content/colours.

    Returns:
        SyntheticPage with the rendered image and ground-truth (x, y, w, h) boxes.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")

    rng = random.Random(seed)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    avail_w = width - 2 * margin - (cols - 1) * gutter
    avail_h = height - 2 * margin - (rows - 1) * gutter
    if avail_w <= 0 or avail_h <= 0:
        raise ValueError("margin/gutter too large for the given page size")

    panel_w = avail_w // cols
    panel_h = avail_h // rows

    boxes: list[Box] = []
    for r in range(rows):
        for c in range(cols):
            x = margin + c * (panel_w + gutter)
            y = margin + r * (panel_h + gutter)
            fill = _PALETTE[(r * cols + c) % len(_PALETTE)]

            # Panel body + black border (rectangle outline is drawn inside the box).
            draw.rectangle([x, y, x + panel_w, y + panel_h], fill=fill,
                           outline=(20, 20, 20), width=border)

            if add_content:
                _draw_content(draw, x, y, panel_w, panel_h, rng)

            boxes.append((x, y, panel_w, panel_h))

    return SyntheticPage(image=img, boxes=boxes)


def _draw_content(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
                  rng: random.Random) -> None:
    """Draw a couple of dark shapes inside a panel so it isn't a flat block."""
    for _ in range(rng.randint(1, 3)):
        cx = rng.randint(x + w // 5, x + 4 * w // 5)
        cy = rng.randint(y + h // 5, y + 4 * h // 5)
        rad = rng.randint(min(w, h) // 10, min(w, h) // 4)
        shade = rng.randint(40, 90)
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                     fill=(shade, shade, shade))


def generate_dataset(
    n: int = 12,
    layouts: tuple[tuple[int, int], ...] = ((2, 2), (3, 2), (2, 1), (3, 3), (1, 2)),
    width: int = 1000,
    height: int = 1500,
    seed: int = 0,
) -> list[SyntheticPage]:
    """Generate a small labeled dataset spanning several panel layouts.

    Layouts are cycled and the gutter/margin jittered per page so the detector
    (and any learned model) sees variety rather than one fixed grid.
    """
    rng = random.Random(seed)
    pages: list[SyntheticPage] = []
    for i in range(n):
        rows, cols = layouts[i % len(layouts)]
        pages.append(
            generate_comic_page(
                rows=rows,
                cols=cols,
                width=width,
                height=height,
                margin=rng.choice([30, 40, 50]),
                gutter=rng.choice([28, 36, 48]),
                seed=seed + i,
            )
        )
    return pages


def boxes_to_yolo(boxes: list[Box], img_w: int, img_h: int, cls: int = 0) -> list[str]:
    """Convert (x, y, w, h) pixel boxes to YOLO txt lines.

    YOLO format per line: ``cls cx cy w h`` with all coords normalised to [0, 1]
    and (cx, cy) the box centre.
    """
    lines = []
    for (x, y, w, h) in boxes:
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def yolo_to_boxes(lines: list[str], img_w: int, img_h: int) -> list[Box]:
    """Inverse of :func:`boxes_to_yolo` — parse YOLO txt lines to pixel boxes."""
    boxes: list[Box] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, nw, nh = parts[:5]
        cx, cy, nw, nh = float(cx), float(cy), float(nw), float(nh)
        w = nw * img_w
        h = nh * img_h
        x = cx * img_w - w / 2
        y = cy * img_h - h / 2
        boxes.append((int(round(x)), int(round(y)), int(round(w)), int(round(h))))
    return boxes
