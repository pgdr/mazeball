from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True, eq=True)
class Pos:
    x: float
    y: float

    def __add__(v1, v2):
        return Pos(v1.x + v2.x, v1.y + v2.y)

    def __sub__(v1, v2):
        return Pos(v1.x - v2.x, v1.y - v2.y)

    def __mul__(v, scalar):
        return Pos(v.x * scalar, v.y * scalar)

    @property
    def as_tuple(v):
        return (v.x, v.y)


@dataclass(frozen=True)
class Level:
    level_id: str
    start_pos: Pos
    win_hole: Pos
    red_holes: list[Pos]
    walls: list[tuple[Pos, Pos]]  # segments
    wall_thickness: float
    name: str | None = None


def _require_keys(obj: dict, keys: list[str], path: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(missing)}")


def _as_pos(value, key_name: str, path: str) -> Pos:
    if not (isinstance(value, list) and len(value) == 2):
        raise ValueError(f"{path}: '{key_name}' must be a length-2 array like [x, y]")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: '{key_name}' must contain numbers") from e
    return Pos(x, y)


def load_level(path: str, default_wall_thickness: float) -> Level:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"{path}: failed to read/parse JSON") from e

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")

    _require_keys(data, ["level_id", "start_pos", "win_hole", "red_holes", "walls"], path)

    level_id = data["level_id"]
    if not isinstance(level_id, str) or not level_id.strip():
        raise ValueError(f"{path}: 'level_id' must be a non-empty string")

    name = data.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"{path}: 'name' must be a string if provided")

    start_pos = _as_pos(data["start_pos"], "start_pos", path)
    win_hole = _as_pos(data["win_hole"], "win_hole", path)

    red_holes_raw = data["red_holes"]
    if not isinstance(red_holes_raw, list):
        raise ValueError(f"{path}: 'red_holes' must be a list of [x, y] entries")
    red_holes: list[Pos] = []
    for i, rh in enumerate(red_holes_raw):
        red_holes.append(_as_pos(rh, f"red_holes[{i}]", path))

    wall_thickness = data.get("wall_thickness", default_wall_thickness)
    try:
        wall_thickness = float(wall_thickness)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: 'wall_thickness' must be a number if provided") from e

    walls_raw = data["walls"]
    if not isinstance(walls_raw, list):
        raise ValueError(f"{path}: 'walls' must be a list of [x1,y1,x2,y2] entries")

    walls: list[tuple[Pos, Pos]] = []
    for i, w in enumerate(walls_raw):
        if not (isinstance(w, list) and len(w) == 4):
            raise ValueError(f"{path}: 'walls[{i}]' must be a length-4 array like [x1,y1,x2,y2]")
        try:
            x1, y1, x2, y2 = (float(w[0]), float(w[1]), float(w[2]), float(w[3]))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path}: 'walls[{i}]' must contain numbers") from e
        walls.append((Pos(x1, y1), Pos(x2, y2)))

    return Level(
        level_id=level_id,
        name=name,
        start_pos=start_pos,
        win_hole=win_hole,
        red_holes=red_holes,
        walls=walls,
        wall_thickness=wall_thickness,
    )


def load_levels(dir_path: str, default_wall_thickness: float) -> list[Level]:
    d = Path(dir_path)
    if not d.exists() or not d.is_dir():
        raise ValueError(f"{dir_path}: not a directory")

    files = sorted(d.glob("*.json"), key=lambda p: p.name)
    if not files:
        raise ValueError(f"{dir_path}: no .json level files found")

    return [load_level(str(p), default_wall_thickness) for p in files]
