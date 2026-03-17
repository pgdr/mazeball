#!/usr/bin/env python3
"""
Convert simple SVG path geometry into game wall segments.

Supported SVG path commands:
  M/m  move to
  L/l  line to
  H/h  horizontal line to
  V/v  vertical line to
  Z/z  close path

Output format:
[
  [x1, y1, x2, y2],
  ...
]

Coordinates are scaled into the rectangle:
  (0, 0) .......... (target_width, target_height)

Example:
  python svg_to_walls.py drawing-1.svg
  python svg_to_walls.py drawing-1.svg --width 1200 --height 1000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


COMMANDS = set("MmLlHhVvZz")
TOKEN_RE = re.compile(r"[MmLlHhVvZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def parse_length(s: str) -> float:
    """Parse an SVG length like '297', '297mm', '1200px'."""
    m = re.match(r"^\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", s)
    if not m:
        raise ValueError(f"Cannot parse length: {s!r}")
    return float(m.group(1))


def get_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    viewbox = root.get("viewBox")
    if viewbox:
        parts = [float(x) for x in re.split(r"[,\s]+", viewbox.strip()) if x]
        if len(parts) != 4:
            raise ValueError(f"Invalid viewBox: {viewbox!r}")
        return tuple(parts)  # min_x, min_y, width, height

    width = root.get("width")
    height = root.get("height")
    if width is None or height is None:
        raise ValueError("SVG must have either a viewBox or both width and height")
    return (0.0, 0.0, parse_length(width), parse_length(height))


def is_command(token: str) -> bool:
    return token in COMMANDS


def parse_path_d(d: str) -> list[list[tuple[float, float]]]:
    """
    Parse a simple SVG path into a list of subpaths.
    Each subpath is a list of points.
    """
    tokens = TOKEN_RE.findall(d)
    i = 0
    cmd = None

    x = y = 0.0
    start_x = start_y = 0.0

    subpaths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    def require_numbers(n: int) -> None:
        if i + n > len(tokens):
            raise ValueError(f"Not enough parameters for command {cmd!r}")

    while i < len(tokens):
        if is_command(tokens[i]):
            cmd = tokens[i]
            i += 1
        elif cmd is None:
            raise ValueError("Path data starts with numbers but no command")

        if cmd in ("M", "m"):
            require_numbers(2)

            # First pair is move-to
            nx = float(tokens[i])
            ny = float(tokens[i + 1])
            i += 2

            if cmd == "m":
                x += nx
                y += ny
            else:
                x, y = nx, ny

            if current:
                subpaths.append(current)
            current = [(x, y)]
            start_x, start_y = x, y

            # Remaining pairs after M/m are treated as L/l
            line_cmd = "l" if cmd == "m" else "L"
            while i < len(tokens) and not is_command(tokens[i]):
                require_numbers(2)
                nx = float(tokens[i])
                ny = float(tokens[i + 1])
                i += 2

                if line_cmd == "l":
                    x += nx
                    y += ny
                else:
                    x, y = nx, ny
                current.append((x, y))

        elif cmd in ("L", "l"):
            while i < len(tokens) and not is_command(tokens[i]):
                require_numbers(2)
                nx = float(tokens[i])
                ny = float(tokens[i + 1])
                i += 2

                if cmd == "l":
                    x += nx
                    y += ny
                else:
                    x, y = nx, ny
                current.append((x, y))

        elif cmd in ("H", "h"):
            while i < len(tokens) and not is_command(tokens[i]):
                require_numbers(1)
                nx = float(tokens[i])
                i += 1

                if cmd == "h":
                    x += nx
                else:
                    x = nx
                current.append((x, y))

        elif cmd in ("V", "v"):
            while i < len(tokens) and not is_command(tokens[i]):
                require_numbers(1)
                ny = float(tokens[i])
                i += 1

                if cmd == "v":
                    y += ny
                else:
                    y = ny
                current.append((x, y))

        elif cmd in ("Z", "z"):
            if current and current[-1] != (start_x, start_y):
                current.append((start_x, start_y))
            if current:
                subpaths.append(current)
                current = []
            cmd = None

        else:
            raise ValueError(f"Unsupported SVG path command: {cmd!r}")

    if current:
        subpaths.append(current)

    return subpaths


def scale_point(
    x: float,
    y: float,
    min_x: float,
    min_y: float,
    vb_w: float,
    vb_h: float,
    target_w: int,
    target_h: int,
) -> tuple[int, int]:
    sx = (x - min_x) * target_w / vb_w
    sy = (y - min_y) * target_h / vb_h

    # Clamp to bounds and round to integer indices
    ix = max(0, min(target_w, int(round(sx))))
    iy = max(0, min(target_h, int(round(sy))))
    return ix, iy


def svg_to_wall_segments(
    svg_path: str | Path,
    target_w: int = 1200,
    target_h: int = 1000,
) -> list[list[int]]:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    min_x, min_y, vb_w, vb_h = get_viewbox(root)

    walls: list[list[int]] = []

    for path_el in root.findall(".//svg:path", SVG_NS):
        d = path_el.get("d")
        if not d:
            continue

        subpaths = parse_path_d(d)
        for subpath in subpaths:
            for (x1, y1), (x2, y2) in zip(subpath, subpath[1:]):
                p1 = scale_point(x1, y1, min_x, min_y, vb_w, vb_h, target_w, target_h)
                p2 = scale_point(x2, y2, min_x, min_y, vb_w, vb_h, target_w, target_h)

                if p1 != p2:
                    walls.append([p1[0], p1[1], p2[0], p2[1]])

    return walls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg_file", help="Input SVG file")
    parser.add_argument("--width", type=int, default=1600, help="Target width")
    parser.add_argument("--height", type=int, default=900, help="Target height")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation",
    )
    args = parser.parse_args()

    try:
        walls = svg_to_wall_segments(args.svg_file, args.width, args.height)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print("""{
  "level_id": "level06",
  "name": "SVG",
  "start_pos": [1550, 150],
  "goal_hole": [300, 140],
    "red_holes": [[466,85],[466,470]],
  "wall_thickness": 2,
  "walls":
""")
    print(json.dumps(walls, indent=args.indent))
    print("}")

if __name__ == "__main__":
    main()
