import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# all shot types
class ShotType(str, Enum):
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    SERVE    = "serve"
    SMASH    = "smash"
    LOB      = "lob"
    UNKNOWN  = "unknown"


# one shot result
@dataclass
class ShotEvent:
    shot_id:   int
    frame_idx: int
    timestamp: float
    shot_type: ShotType
    player_id: Optional[int]
    ball_x:    Optional[int]
    ball_y:    Optional[int]
    speed_kmh: Optional[float]
    zone:      str   # "net" / "mid" / "back"
    side:      str   # "right" / "left" / "overhead"
    direction: str   # "cross_court" / "down_line"

    def to_dict(self):
        return {
            "shot_id":   self.shot_id,
            "frame":     self.frame_idx,
            "time_sec":  round(self.timestamp, 2),
            "shot_type": self.shot_type.value,
            "player_id": self.player_id,
            "ball_x":    self.ball_x,
            "ball_y":    self.ball_y,
            "speed_kmh": round(self.speed_kmh, 1) if self.speed_kmh else None,
            "zone":      self.zone,
            "side":      self.side,
            "direction": self.direction,
        }


# remembers where the ball has been
class BallTracker:

    def __init__(self):
        self.positions = []  # list of (frame_idx, (x, y) or None)

    def update(self, frame_idx, ball):
        pos = ball.center if ball else None
        self.positions.append((frame_idx, pos))
        if len(self.positions) > 30:
            self.positions.pop(0)

    def get_velocity(self):
        # returns (vx, vy) = how fast ball moving per frame
        valid = [(f, p) for f, p in self.positions[-6:] if p]
        if len(valid) < 2:
            return None
        (f0, p0), (f1, p1) = valid[0], valid[-1]
        df = f1 - f0
        if df == 0:
            return None
        return ((p1[0] - p0[0]) / df, (p1[1] - p0[1]) / df)

    def is_going_up(self):
        # y=0 is top of screen so going up = y decreasing
        valid = [p[1] for _, p in self.positions[-8:] if p]
        if len(valid) < 4:
            return False
        return valid[-1] < valid[0]

    def is_bounce(self):
        # V-shape in Y values = ball bounced
        valid = [p[1] for _, p in self.positions[-10:] if p]
        if len(valid) < 4:
            return False
        mid = len(valid) // 2
        return valid[mid] > valid[0] and valid[-1] < valid[mid]


# returns which side the player hit from
def get_contact_side(player, racket):
    if not player:
        return "right"
    x1, y1, x2, y2 = player.bbox
    player_cx = (x1 + x2) / 2
    player_cy = (y1 + y2) / 2
    player_h  = y2 - y1

    if racket:
        rx, ry = racket.center
        if ry < player_cy - player_h * 0.2:
            return "overhead"
        return "right" if rx > player_cx else "left"
    return "right"


# tells us where on the court the ball is
def get_zone(ball, players):
    if not ball or not players:
        return "mid"
    ball_y = ball.center[1]
    max_y  = max(p.center[1] for p in players)
    if ball_y < max_y * 0.4:
        return "net"
    if ball_y > max_y * 0.75:
        return "back"
    return "mid"


# main shot classifier
class ShotClassifier:

    def __init__(self, fps=30.0):
        self.fps          = fps
        self.tracker      = BallTracker()
        self.shot_count   = 0
        self.last_shot_at = -20  # frame of last shot

    def update(self, frame_det, frame=None) -> Optional[ShotEvent]:
        self.tracker.update(frame_det.frame_idx, frame_det.ball)

        if not frame_det.ball:
            return None

        # don't detect another shot too soon
        if frame_det.frame_idx - self.last_shot_at < 15:
            return None

        vel = self.tracker.get_velocity()
        if vel is None:
            return None

        speed_px = np.hypot(vel[0], vel[1])
        if speed_px < 3.0:
            return None  # ball barely moving

        # find the player closest to the ball
        player = self._nearest_player(frame_det.ball, frame_det.players)
        racket = self._player_racket(player, frame_det.rackets)

        side      = get_contact_side(player, racket)
        zone      = get_zone(frame_det.ball, frame_det.players)
        going_up  = self.tracker.is_going_up()
        vx, vy    = vel
        direction = "cross_court" if abs(vx) > 2.0 else "down_line"
        speed_kmh = (speed_px * self.fps / 30.0) * 3.6

        shot_type = self._classify(side, zone, going_up, speed_px, vy)

        self.shot_count  += 1
        self.last_shot_at = frame_det.frame_idx

        return ShotEvent(
            shot_id   = self.shot_count,
            frame_idx = frame_det.frame_idx,
            timestamp = frame_det.timestamp,
            shot_type = shot_type,
            player_id = player.track_id if player else None,
            ball_x    = frame_det.ball.center[0],
            ball_y    = frame_det.ball.center[1],
            speed_kmh = speed_kmh,
            zone      = zone,
            side      = side,
            direction = direction,
        )

    def _classify(self, side, zone, going_up, speed_px, vy):
        if zone == "back" and side == "overhead" and speed_px > 15:
            return ShotType.SERVE
        if side == "overhead" and speed_px > 12:
            return ShotType.SMASH
        if going_up and vy < -2:
            return ShotType.LOB
        if side == "right":
            return ShotType.FOREHAND
        if side == "left":
            return ShotType.BACKHAND
        return ShotType.UNKNOWN

    def _nearest_player(self, ball, players):
        if not players:
            return None
        bx, by = ball.center
        return min(players, key=lambda p: np.hypot(p.center[0] - bx, p.center[1] - by))

    def _player_racket(self, player, rackets):
        if not player or not rackets:
            return None
        for r in rackets:
            if r.track_id == player.track_id:
                return r
        return min(rackets, key=lambda r: np.hypot(r.center[0] - player.center[0],
                                                    r.center[1] - player.center[1]))
