import math
import sys
from dataclasses import dataclass
import pygame


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

    @staticmethod
    def from_flat_list(
        flat_xy: list[float], thickness: float
    ) -> "Maze":
        if len(flat_xy) % 2 != 0:
            raise ValueError(
                "walls must contain an even number of values: x1,y1,x2,y2,..."
            )
        pts = [
            Pos(float(flat_xy[i]), float(flat_xy[i + 1]))
            for i in range(0, len(flat_xy), 2)
        ]
        segs = list(zip(pts, pts[1:]))
        return Maze(segs, thickness)

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
    win_hole_pos: Pos
    lose_holes: list[Pos]
    status: str = "playing"


DEADZONE = 0.20

BASE_ACCEL = 900.0
BOOST_MULT = 2
MAX_SPEED = 650.0

DRAG = 2.0
RESTITUTION = 0.60
UNSTUCK_KICK = 80.0
COLLISION_PASSES = 4


def update(
    state: GameState,
    maze: Maze,
    raw_dir: Pos,
    accelerate: bool,
    dt: float,
    screen_w: int,
    screen_h: int,
    ball_radius: float,
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

        accel_strength = BASE_ACCEL * (
            BOOST_MULT if accelerate else 1.0
        )
        a = dir_n * (accel_strength * throttle)

        vel = vel + a * dt

        drag_factor = max(0.0, 1.0 - DRAG * dt)
        vel = vel * drag_factor

        sp = length(vel)
        if sp > MAX_SPEED:
            vel = vel * (MAX_SPEED / sp)

        pos = pos + vel * dt

        sp = length(vel)
        if sp > 1e-3:
            sign = 1.0 if vel.x >= 0 else -1.0
            angle = (
                angle
                + sign
                * (sp / max(1.0, ball_radius))
                * dt
                * (180.0 / math.pi)
            ) % 360.0

    pos, vel = resolve_bounds(
        pos,
        vel,
        ball_radius,
        screen_w,
        screen_h,
        RESTITUTION,
        UNSTUCK_KICK,
    )

    for _ in range(COLLISION_PASSES):
        any_hit = False
        for a0, b0 in maze.segments:
            pos, vel, hit = resolve_circle_against_segment(
                pos,
                vel,
                ball_radius,
                a0,
                b0,
                maze.thickness,
                RESTITUTION,
                UNSTUCK_KICK,
            )
            any_hit |= hit
        if not any_hit:
            break

    status = state.status
    if status == "playing":
        if (
            dist(pos, state.win_hole_pos)
            < ball_radius + BLACK_HOLE_RADIUS
        ):
            status = "won"
        else:
            for hp in state.lose_holes:
                if dist(pos, hp) < ball_radius + RED_HOLE_RADIUS:
                    status = "dead"
                    break

    return GameState(
        ball_pos=pos,
        ball_vel=vel,
        ball_angle_deg=angle,
        win_hole_pos=state.win_hole_pos,
        lose_holes=state.lose_holes,
        status=status,
    )


pygame.init()

WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 20
BLACK_HOLE_RADIUS = 30
RED_HOLE_RADIUS = 26
WALL_THICKNESS = 10

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (220, 0, 0)
GRAY = (192, 192, 192)
WHITE = (255, 255, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Move the Ball to the Black Hole")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)


SPRITE_PATH = "blue.png"
try:
    ball_img_src = pygame.image.load(SPRITE_PATH).convert_alpha()
    ball_img = pygame.transform.smoothscale(
        ball_img_src, (BALL_RADIUS * 2, BALL_RADIUS * 2)
    )
    sprite_ok = True
except Exception as e:
    print(f"Could not load '{SPRITE_PATH}': {e}")
    ball_img = None
    sprite_ok = False


walls = [
    100,
    100,
    700,
    100,
    700,
    250,
    200,
    250,
    200,
    500,
    700,
    500,
]
maze = Maze.from_flat_list(walls, thickness=WALL_THICKNESS)


win_hole = Pos(WIDTH / 2, HEIGHT / 2 + 200)
red_holes = [
    Pos(300, 180),
    Pos(520, 360),
    Pos(420, 520),
]


def new_game_state() -> GameState:
    return GameState(
        ball_pos=Pos(WIDTH / 2, HEIGHT / 2),
        ball_vel=Pos(0.0, 0.0),
        ball_angle_deg=0.0,
        win_hole_pos=win_hole,
        lose_holes=red_holes,
        status="playing",
    )


state = new_game_state()
state = update(
    state, maze, Pos(0.0, 0.0), False, 0.0, WIDTH, HEIGHT, BALL_RADIUS
)


joystick_available = False
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    joystick_available = True


def draw_ball(surface, center: Pos, angle_deg: float):
    if ball_img is None:
        pygame.draw.circle(
            surface, BLUE, (int(center.x), int(center.y)), BALL_RADIUS
        )
        return

    rotated = pygame.transform.rotozoom(ball_img, -angle_deg, 1.0)
    rect = rotated.get_rect(center=(int(center.x), int(center.y)))
    surface.blit(rotated, rect)


while True:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_q
        ):
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            state = new_game_state()
            state = update(
                state,
                maze,
                Pos(0.0, 0.0),
                False,
                0.0,
                WIDTH,
                HEIGHT,
                BALL_RADIUS,
            )

    if joystick_available:
        x = joystick.get_axis(0)
        y = joystick.get_axis(1)

        if abs(x) < DEADZONE:
            x = 0.0
        if abs(y) < DEADZONE:
            y = 0.0

        direction = Pos(x, y)
        accelerate = bool(joystick.get_button(0))
    else:
        keys = pygame.key.get_pressed()
        dx = (1 if keys[pygame.K_RIGHT] else 0) - (
            1 if keys[pygame.K_LEFT] else 0
        )
        dy = (1 if keys[pygame.K_DOWN] else 0) - (
            1 if keys[pygame.K_UP] else 0
        )
        direction = Pos(float(dx), float(dy))
        accelerate = bool(
            keys[pygame.K_SPACE]
            or keys[pygame.K_LSHIFT]
            or keys[pygame.K_RSHIFT]
        )

    state = update(
        state,
        maze,
        direction,
        accelerate,
        dt,
        WIDTH,
        HEIGHT,
        BALL_RADIUS,
    )

    screen.fill(WHITE)
    maze.draw(screen, GRAY)

    for hp in state.lose_holes:
        pygame.draw.circle(
            screen, RED, (int(hp.x), int(hp.y)), RED_HOLE_RADIUS
        )
    pygame.draw.circle(
        screen,
        BLACK,
        (int(state.win_hole_pos.x), int(state.win_hole_pos.y)),
        BLACK_HOLE_RADIUS,
    )

    draw_ball(screen, state.ball_pos, state.ball_angle_deg)

    if state.status == "won":
        msg = font.render(
            "YOU WIN! (R to restart)", True, (0, 140, 0)
        )
        screen.blit(msg, (20, 20))
    elif state.status == "dead":
        msg = font.render(
            "YOU DIED! (R to restart)", True, (180, 0, 0)
        )
        screen.blit(msg, (20, 20))

    pygame.display.flip()
