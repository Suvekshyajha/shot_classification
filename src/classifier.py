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
    VOLLEY   = "volley"
    LOB      = "lob"
    UNKNOWN  = "unknown"


# stores result of one shot
@dataclass
class ShotEvent:
    shot_id:        int
    frame_idx:      int
    timestamp:      float
    shot_type:      ShotType
    player_id:      Optional[int]
    ball_x:         Optional[int]
    ball_y:         Optional[int]
    speed_kmh:      Optional[float]
    zone:           str    # "net" "mid" "back"
    side:           str    # "right" "left" "overhead"
    direction:      str    # "cross_court" "down_line"
    confidence:     float

    def to_dict(self):
        return {
            "shot_id":    self.shot_id,
            "frame":      self.frame_idx,
            "time_sec":   round(self.timestamp, 2),
            "shot_type":  self.shot_type.value,
            "player_id":  self.player_id,
            "ball_x":     self.ball_x,
            "ball_y":     self.ball_y,
            "speed_kmh":  round(self.speed_kmh, 1) if self.speed_kmh else None,
            "zone":       self.zone,
            "side":       self.side,
            "direction":  self.direction,
            "confidence": round(self.confidence, 2),
        }


# tracks last N ball positions
class BallTracker:

    def __init__(self):
        self.positions = []   # list of (frame_idx, (x,y) or None)

    def update(self, frame_idx, ball):
        pos = ball.center if ball else None
        self.positions.append((frame_idx, pos))
        # keep only last 30
        if len(self.positions) > 30:
            self.positions.pop(0)

    def get_speed(self):
        # get valid positions only
        valid = [(f, p) for f, p in self.positions[-6:] if p]
        if len(valid) < 2:
            return None
        (f0, p0), (f1, p1) = valid[0], valid[-1]
        df = f1 - f0
        if df == 0:
            return None
        vx = (p1[0] - p0[0]) / df
        vy = (p1[1] - p0[1]) / df
        return (vx, vy)

    def is_going_up(self):
        # check if ball arc is upward = lob
        valid = [p[1] for _, p in self.positions[-8:] if p]
        if len(valid) < 4:
            return False
        # if y values decreasing = going up (y=0 is top of screen)
        return valid[-1] < valid[0]

    def is_bounce(self):
        # V shape in Y = bounce
        valid = [p[1] for _, p in self.positions[-10:] if p]
        if len(valid) < 4:
            return False
        mid = len(valid) // 2
        going_down = valid[mid] > valid[0]
        going_up   = valid[-1] < valid[mid]
        return going_down and going_up


# figures out which side player is hitting from
def get_contact_side(player, racket):
    if not player:
        return "right"   # default

    x1, y1, x2, y2 = player.bbox
    player_cx = (x1 + x2) / 2
    player_cy = (y1 + y2) / 2
    player_h  = y2 - y1

    if racket:
        rx, ry = racket.center
        # overhead if racket is high up
        if ry < player_cy - player_h * 0.2:
            return "overhead"
        # right or left based on racket x vs player center
        return "right" if rx > player_cx else "left"

    return "right"


# get court zone from ball position
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


# main classifier
class ShotClassifier:

    def __init__(self, fps=30.0):
        self.fps          = fps
        self.tracker      = BallTracker()
        self.shot_count   = 0
        self.last_shot_at = -20   # frame index of last shot

    def update(self, frame_det, frame) -> Optional[ShotEvent]:
        self.tracker.update(frame_det.frame_idx, frame_det.ball)

        # no ball = no shot
        if not frame_det.ball:
            return None

        # too soon after last shot
        if frame_det.frame_idx - self.last_shot_at < 15:
            return None

        # get ball speed
        vel = self.tracker.get_speed()
        if vel is None:
            return None

        speed_px = np.hypot(vel[0], vel[1])

        # ball not moving enough
        if speed_px < 3.0:
            return None

        # find nearest player to ball
        player = self._nearest_player(frame_det.ball, frame_det.players)

        # find that player's racket
        racket = self._player_racket(player, frame_det.rackets)

        # get info
        side      = get_contact_side(player, racket)
        zone      = get_zone(frame_det.ball, frame_det.players)
        bounce    = self.tracker.is_bounce()
        going_up  = self.tracker.is_going_up()
        vx, vy    = vel
        direction = "cross_court" if abs(vx) > 2.0 else "down_line"
        speed_kmh = (speed_px * self.fps / 30.0) * 3.6

        # classify
        shot, conf = self._classify(side, zone, bounce,
                                     going_up, speed_px, vy)

        self.shot_count  += 1
        self.last_shot_at = frame_det.frame_idx

        return ShotEvent(
            shot_id    = self.shot_count,
            frame_idx  = frame_det.frame_idx,
            timestamp  = frame_det.timestamp,
            shot_type  = shot,
            player_id  = player.track_id if player else None,
            ball_x     = frame_det.ball.center[0],
            ball_y     = frame_det.ball.center[1],
            speed_kmh  = speed_kmh,
            zone       = zone,
            side       = side,
            direction  = direction,
            confidence = conf,
        )

    def _classify(self, side, zone, bounce, going_up, speed_px, vy):

        # serve: back court + overhead + fast
        if zone == "back" and side == "overhead" and speed_px > 15:
            return ShotType.SERVE, 0.80

        # smash: overhead + fast
        if side == "overhead" and speed_px > 12:
            return ShotType.SMASH, 0.78

        # lob: ball going up
        if going_up and vy < -2:
            return ShotType.LOB, 0.74

        # volley: at net, no bounce
        if zone == "net" and not bounce:
            if side == "right":
                return ShotType.FOREHAND, 0.70
            return ShotType.BACKHAND, 0.70

        # ground strokes
        if side == "right":
            return ShotType.FOREHAND, 0.76
        if side == "left":
            return ShotType.BACKHAND, 0.75

        return ShotType.UNKNOWN, 0.30

    def _nearest_player(self, ball, players):
        if not players:
            return None
        bx, by = ball.center
        return min(players,
                   key=lambda p: np.hypot(p.center[0]-bx,
                                          p.center[1]-by))

    def _player_racket(self, player, rackets):
        if not player or not rackets:
            return None
        # match by track_id first
        for r in rackets:
            if r.track_id == player.track_id:
                return r
        # else nearest racket to player
        return min(rackets,
                   key=lambda r: np.hypot(r.center[0]-player.center[0],
                                          r.center[1]-player.center[1]))