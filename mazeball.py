import sys
import math
from dataclasses import dataclass
import pygame

from levels import Pos, Level, load_level


@dataclass(frozen=True)
class Settings:

    WIDTH: int = 800
    HEIGHT: int = 600
    FPS: int = 60
    FONT_SIZE: int = 48
    CAPTION: str = "Move the Ball to the Black Hole"

    SPRITE_PATH: str = "blue.png"

    BALL_RADIUS: int = 20
    BLACK_HOLE_RADIUS: int = 30
    RED_HOLE_RADIUS: int = 26
    WALL_THICKNESS: int = 10

    DEADZONE: float = 0.20
    BASE_ACCEL: float = 900.0
    BOOST_MULT: float = 2.0
    MAX_SPEED: float = 650.0
    DRAG: float = 2.0
    RESTITUTION: float = 0.60
    UNSTUCK_KICK: float = 80.0
    COLLISION_PASSES: int = 4

    SPIN_MULT: float = 1.0

    JOY_ACCEL_BUTTON: int = 0
    KEY_ACCEL: tuple[int, ...] = (
        pygame.K_SPACE,
        pygame.K_LSHIFT,
        pygame.K_RSHIFT,
    )
    KEY_RESTART: int = pygame.K_r
    KEY_QUIT: int = pygame.K_q

    WHITE: tuple[int, int, int] = (255, 255, 255)
    GRAY: tuple[int, int, int] = (192, 192, 192)
    BLUE: tuple[int, int, int] = (0, 0, 255)
    BLACK: tuple[int, int, int] = (0, 0, 0)
    RED: tuple[int, int, int] = (220, 0, 0)
    GREEN_TEXT: tuple[int, int, int] = (0, 140, 0)
    RED_TEXT: tuple[int, int, int] = (180, 0, 0)


S = Settings()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dot(a: Pos, b: Pos) -> float:
    return a.x * b.x + a.y * b.y


def length(v: Pos) -> float:
    return (v.x * v.x + v.y * v.y) ** 0.5


def normalize(v: Pos) -> Pos:
    L = length(v)
    return Pos(0.0, 0.0) if L < 1e-9 else Pos(v.x / L, v.y / L)


def dist(a: Pos, b: Pos) -> float:
    return length(a - b)


def closest_point_on_segment(p: Pos, a: Pos, b: Pos) -> Pos:
    ab = b - a
    ab_len2 = dot(ab, ab)
    if ab_len2 < 1e-12:
        return a
    t = dot(p - a, ab) / ab_len2
    t = clamp(t, 0.0, 1.0)
    return Pos(a.x + ab.x * t, a.y + ab.y * t)


class Maze:
    def __init__(
        self, segments: list[tuple[Pos, Pos]], thickness: float
    ):
        self.segments = segments
        self.thickness = thickness

    def draw(self, surface, color):
        for a, b in self.segments:
            pygame.draw.line(
                surface,
                color,
                a.as_tuple,
                b.as_tuple,
                int(self.thickness),
            )


def resolve_circle_against_segment(
    pos: Pos,
    vel: Pos,
    radius: float,
    a: Pos,
    b: Pos,
    thickness: float,
    restitution: float,
    unstuck_kick: float,
) -> tuple[Pos, Pos, bool]:
    cp = closest_point_on_segment(pos, a, b)
    delta = pos - cp
    d = length(delta)
    min_d = radius + thickness / 2.0

    if d >= min_d:
        return pos, vel, False

    if d > 1e-9:
        n = Pos(delta.x / d, delta.y / d)
    else:
        ab = b - a
        perp = Pos(-ab.y, ab.x)
        n = normalize(perp) if length(perp) > 1e-9 else Pos(1.0, 0.0)

    penetration = min_d - d
    pos = pos + n * (penetration + 0.5)

    vn = dot(vel, n)
    if vn < 0.0:
        vel = vel - n * ((1.0 + restitution) * vn)
    else:
        if length(vel) < 1e-6:
            vel = vel + n * unstuck_kick

    return pos, vel, True


def resolve_bounds(
    pos: Pos,
    vel: Pos,
    radius: float,
    w: int,
    h: int,
    restitution: float,
    unstuck_kick: float,
):
    if pos.x < radius:
        pos = Pos(radius, pos.y)
        vel = Pos(abs(vel.x) * restitution, vel.y)
        if length(vel) < 1e-6:
            vel = Pos(unstuck_kick, vel.y)
    elif pos.x > w - radius:
        pos = Pos(w - radius, pos.y)
        vel = Pos(-abs(vel.x) * restitution, vel.y)
        if length(vel) < 1e-6:
            vel = Pos(-unstuck_kick, vel.y)

    if pos.y < radius:
        pos = Pos(pos.x, radius)
        vel = Pos(vel.x, abs(vel.y) * restitution)
        if length(vel) < 1e-6:
            vel = Pos(vel.x, unstuck_kick)
    elif pos.y > h - radius:
        pos = Pos(pos.x, h - radius)
        vel = Pos(vel.x, -abs(vel.y) * restitution)
        if length(vel) < 1e-6:
            vel = Pos(vel.x, -unstuck_kick)

    return pos, vel


@dataclass
class GameState:
    ball_pos: Pos
    ball_vel: Pos
    ball_angle_deg: float
    status: str


def new_game_state(level: Level) -> GameState:
    return GameState(
        ball_pos=level.start_pos,
        ball_vel=Pos(0.0, 0.0),
        ball_angle_deg=0.0,
        status="playing",
    )


def update(
    state: GameState,
    maze: Maze,
    settings: Settings,
    level: Level,
    raw_dir: Pos,
    accelerate: bool,
    dt: float,
) -> GameState:
    pos, vel, angle = (
        state.ball_pos,
        state.ball_vel,
        state.ball_angle_deg,
    )

    if state.status == "playing":
        mag = length(raw_dir)
        throttle = clamp(mag, 0.0, 1.0)
        dir_n = (
            normalize(raw_dir) if throttle > 1e-6 else Pos(0.0, 0.0)
        )

        accel_strength = settings.BASE_ACCEL * (
            settings.BOOST_MULT if accelerate else 1.0
        )
        a = dir_n * (accel_strength * throttle)

        vel = vel + a * dt
        vel = vel * max(0.0, 1.0 - settings.DRAG * dt)

        sp = length(vel)
        if sp > settings.MAX_SPEED:
            vel = vel * (settings.MAX_SPEED / sp)

        pos = pos + vel * dt

        sp = length(vel)
        if sp > 1e-3:
            sign = (
                1.0
                if (abs(vel.x) >= abs(vel.y) and vel.x >= 0)
                or (abs(vel.y) > abs(vel.x) and vel.y >= 0)
                else -1.0
            )
            deg_per_sec = (
                (sp / max(1.0, settings.BALL_RADIUS))
                * (180.0 / math.pi)
                * settings.SPIN_MULT
            )
            angle = (angle + sign * deg_per_sec * dt) % 360.0

    pos, vel = resolve_bounds(
        pos,
        vel,
        settings.BALL_RADIUS,
        settings.WIDTH,
        settings.HEIGHT,
        settings.RESTITUTION,
        settings.UNSTUCK_KICK,
    )

    for _ in range(settings.COLLISION_PASSES):
        any_hit = False
        for a0, b0 in maze.segments:
            pos, vel, hit = resolve_circle_against_segment(
                pos,
                vel,
                settings.BALL_RADIUS,
                a0,
                b0,
                maze.thickness,
                settings.RESTITUTION,
                settings.UNSTUCK_KICK,
            )
            any_hit |= hit
        if not any_hit:
            break

    status = state.status
    if status == "playing":
        if (
            dist(pos, level.win_hole)
            < settings.BALL_RADIUS + settings.BLACK_HOLE_RADIUS
        ):
            status = "won"
        else:
            for hp in level.red_holes:
                if (
                    dist(pos, hp)
                    < settings.BALL_RADIUS + settings.RED_HOLE_RADIUS
                ):
                    status = "dead"
                    break

    return GameState(
        ball_pos=pos,
        ball_vel=vel,
        ball_angle_deg=angle,
        status=status,
    )


pygame.init()
screen = pygame.display.set_mode((S.WIDTH, S.HEIGHT))
pygame.display.set_caption(S.CAPTION)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, S.FONT_SIZE)


level = load_level(
    "levels/level01.json", default_wall_thickness=S.WALL_THICKNESS
)
maze = Maze(level.walls, thickness=level.wall_thickness)


ball_img = None
try:
    src = pygame.image.load(S.SPRITE_PATH).convert_alpha()
    ball_img = pygame.transform.smoothscale(
        src, (S.BALL_RADIUS * 2, S.BALL_RADIUS * 2)
    )
except Exception as e:
    print(
        f"Could not load '{S.SPRITE_PATH}': {e} (falling back to circle)"
    )


def draw_ball(surface, center: Pos, angle_deg: float):
    if ball_img is None:
        pygame.draw.circle(
            surface,
            S.BLUE,
            (int(center.x), int(center.y)),
            S.BALL_RADIUS,
        )
        return
    rotated = pygame.transform.rotozoom(ball_img, -angle_deg, 1.0)
    rect = rotated.get_rect(center=(int(center.x), int(center.y)))
    surface.blit(rotated, rect)


joystick_available = False
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    joystick_available = True

state = new_game_state(level)
state = update(
    state,
    maze,
    S,
    level,
    raw_dir=Pos(0.0, 0.0),
    accelerate=False,
    dt=0.0,
)


while True:
    dt = clock.tick(S.FPS) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == S.KEY_QUIT
        ):
            pygame.quit()
            sys.exit()
        if (
            event.type == pygame.KEYDOWN
            and event.key == S.KEY_RESTART
        ):
            state = new_game_state(level)
            state = update(
                state,
                maze,
                S,
                level,
                raw_dir=Pos(0.0, 0.0),
                accelerate=False,
                dt=0.0,
            )

    if joystick_available:
        x = joystick.get_axis(0)
        y = joystick.get_axis(1)
        if abs(x) < S.DEADZONE:
            x = 0.0
        if abs(y) < S.DEADZONE:
            y = 0.0
        direction = Pos(x, y)
        accelerate = bool(joystick.get_button(S.JOY_ACCEL_BUTTON))
    else:
        keys = pygame.key.get_pressed()
        dx = (1 if keys[pygame.K_RIGHT] else 0) - (
            1 if keys[pygame.K_LEFT] else 0
        )
        dy = (1 if keys[pygame.K_DOWN] else 0) - (
            1 if keys[pygame.K_UP] else 0
        )
        direction = Pos(float(dx), float(dy))
        accelerate = any(keys[k] for k in S.KEY_ACCEL)

    state = update(state, maze, S, level, direction, accelerate, dt)

    screen.fill(S.WHITE)
    maze.draw(screen, S.GRAY)

    for hp in level.red_holes:
        pygame.draw.circle(
            screen, S.RED, (int(hp.x), int(hp.y)), S.RED_HOLE_RADIUS
        )
    pygame.draw.circle(
        screen,
        S.BLACK,
        (int(level.win_hole.x), int(level.win_hole.y)),
        S.BLACK_HOLE_RADIUS,
    )

    draw_ball(screen, state.ball_pos, state.ball_angle_deg)

    if state.status == "won":
        msg = font.render(
            "YOU WIN! (R to restart)", True, S.GREEN_TEXT
        )
        screen.blit(msg, (20, 20))
    elif state.status == "dead":
        msg = font.render(
            "YOU DIED! (R to restart)", True, S.RED_TEXT
        )
        screen.blit(msg, (20, 20))

    pygame.display.flip()
