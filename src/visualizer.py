import cv2
import numpy as np


# colors for each thing we draw (BGR format)
COLORS = {
    "player":   (50,  220,  50),   # green
    "ball":     (0,   230, 255),   # yellow
    "racket":   (255, 165,   0),   # orange
    "forehand": (0,   255, 127),   # mint
    "backhand": (0,   165, 255),   # light blue
    "serve":    (255,  50,  50),   # red
    "smash":    (255,   0, 200),   # pink
    "volley":   (255, 200,   0),   # gold
    "lob":      (180, 120, 255),   # purple
    "unknown":  (150, 150, 150),   # grey
}


class FrameVisualizer:

    def __init__(self):
        self.ball_trail   = []          # stores last ball positions
        self.last_shot    = None        # last shot event
        self.shot_timer   = 0           # how many frames to show shot label

    def draw(self, frame, det, shot_event):
        out = frame.copy()

        # 1. draw ball trail
        self._draw_trail(out, det.ball)

        # 2. draw player boxes
        for player in det.players:
            self._draw_box(out, player.bbox,
                           COLORS["player"],
                           f"P{player.track_id or '?'}")

        # 3. draw racket boxes
        for racket in det.rackets:
            self._draw_box(out, racket.bbox,
                           COLORS["racket"], "racket",
                           thickness=1)

        # 4. draw ball circle
        if det.ball:
            cx, cy = det.ball.center
            cv2.circle(out, (cx, cy), 10, COLORS["ball"], 2)
            cv2.circle(out, (cx, cy),  3, COLORS["ball"], -1)

        # 5. show shot label when shot happens
        if shot_event:
            self.last_shot  = shot_event
            self.shot_timer = 40        # show for 40 frames

        if self.last_shot and self.shot_timer > 0:
            self._draw_shot_label(out, self.last_shot, self.shot_timer)
            self.shot_timer -= 1

        # 6. draw info panel top left
        self._draw_panel(out, det)

        return out

    # ── draw fading ball trail ───────────────────────────────────────
    def _draw_trail(self, frame, ball):
        pos = ball.center if ball else None
        self.ball_trail.append(pos)

        # keep only last 20 positions
        if len(self.ball_trail) > 20:
            self.ball_trail.pop(0)

        # draw lines between positions
        for i in range(1, len(self.ball_trail)):
            p1 = self.ball_trail[i - 1]
            p2 = self.ball_trail[i]
            if p1 is None or p2 is None:
                continue
            # fade: older = more transparent
            alpha = i / len(self.ball_trail)
            color = (0, int(alpha * 230), int(alpha * 255))
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    # ── draw a bounding box with label ──────────────────────────────
    def _draw_box(self, frame, bbox, color, label, thickness=2):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # small filled rectangle behind text
        (tw, th), _ = cv2.getTextSize(label,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame,
                      (x1, y1 - th - 6),
                      (x1 + tw + 4, y1),
                      color, -1)
        cv2.putText(frame, label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── draw shot type label near ball ───────────────────────────────
    def _draw_shot_label(self, frame, shot, timer):
        h, w   = frame.shape[:2]
        label  = shot.shot_type.value.upper()
        conf   = f"{shot.confidence:.0%}"
        color  = COLORS.get(shot.shot_type.value, COLORS["unknown"])

        # fade out as timer decreases
        alpha  = timer / 40.0

        # position: center top area
        cx, cy = w // 2, 80

        # big shot label
        scale  = 1.2
        (tw, th), _ = cv2.getTextSize(label,
                                       cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        px = cx - tw // 2

        # dark background box
        pad = 10
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (px - pad, cy - th - pad),
                      (px + tw + pad, cy + pad),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha * 0.6,
                        frame,   1 - alpha * 0.6, 0, frame)

        # shot name
        cv2.putText(frame, label, (px, cy),
                    cv2.FONT_HERSHEY_DUPLEX,
                    scale, color, 2, cv2.LINE_AA)

        # confidence below
        cv2.putText(frame, conf,
                    (cx - 15, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # ── small info panel top left ────────────────────────────────────
    def _draw_panel(self, frame, det):
        # dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (210, 80), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        lines = [
            f"Frame:   {det.frame_idx}",
            f"Time:    {det.timestamp:.2f}s",
            f"Players: {len(det.players)}",
            f"Ball:    {'yes' if det.ball else 'no'}",
        ]

        for i, line in enumerate(lines):
            cv2.putText(frame, line,
                        (8, 18 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)