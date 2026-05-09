import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ── Single detected object ──────────────────────────────────────────
@dataclass
class Detection:
    bbox: tuple        # (x1, y1, x2, y2)
    confidence: float
    class_name: str    # "player", "ball", "racket"
    track_id: Optional[int] = None

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.width   = x2 - x1
        self.height  = y2 - y1


# ── All detections for one frame ────────────────────────────────────
@dataclass
class FrameDetections:
    frame_idx: int
    timestamp: float
    players:   list
    rackets:   list
    ball:      Optional[Detection]


# ── Main Detector ───────────────────────────────────────────────────
class PadelDetector:

    # YOLO COCO class IDs
    CLASS_MAP = {0: "player", 32: "ball", 38: "racket"}

    def __init__(self, model_path="yolov8n.pt", conf=0.35):
        self.conf  = conf
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info("YOLO loaded OK")
        except Exception as e:
            logger.warning(f"YOLO not available ({e}) — using color fallback")

    # ── Main entry point ────────────────────────────────────────────
    def detect(self, frame, frame_idx, timestamp) -> FrameDetections:
        if self.model:
            return self._yolo(frame, frame_idx, timestamp)
        return self._fallback(frame, frame_idx, timestamp)

    # ── YOLO path ───────────────────────────────────────────────────
    def _yolo(self, frame, frame_idx, timestamp) -> FrameDetections:
        results = self.model.track(frame, persist=True,
                                   conf=self.conf, verbose=False)
        players, rackets, ball = [], [], None

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.CLASS_MAP:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det = Detection(
                    bbox       = (x1, y1, x2, y2),
                    confidence = float(box.conf[0]),
                    class_name = self.CLASS_MAP[cls_id],
                    track_id   = int(box.id[0]) if box.id is not None else None,
                )

                if cls_id == 0:
                    players.append(det)
                elif cls_id == 32:
                    # keep only the most confident ball
                    if ball is None or det.confidence > ball.confidence:
                        ball = det
                elif cls_id == 38:
                    rackets.append(det)

        return FrameDetections(frame_idx, timestamp, players, rackets, ball)

    # ── Fallback: pure OpenCV color/shape detection ─────────────────
    def _fallback(self, frame, frame_idx, timestamp) -> FrameDetections:
        ball    = self._find_ball(frame)
        players = self._find_players(frame)
        rackets = self._guess_rackets(players)
        return FrameDetections(frame_idx, timestamp, players, rackets, ball)

    def _find_ball(self, frame):
        """Detect ball by yellow-green color + circularity check."""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([25, 80, 80]),
                                np.array([65, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (20 < area < 2000):
                continue
            _, radius = cv2.minEnclosingCircle(cnt)
            circularity = area / (np.pi * radius ** 2 + 1e-5)
            if circularity > 0.5:
                x, y, w, h = cv2.boundingRect(cnt)
                det = Detection((x, y, x+w, y+h), circularity, "ball")
                if best is None or det.confidence > best.confidence:
                    best = det
        return best

    def _find_players(self, frame):
        """Detect players by finding large upright blobs."""
        h, w  = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (21, 21), 0)
        _, th = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        players = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if not (5000 < area < 80000):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh / (bw + 1e-5) > 1.2 and y > h * 0.1:
                players.append(Detection((x, y, x+bw, y+bh),
                                         0.6, "player", track_id=i))
        return players[:4]  # max 4 in padel

    def _guess_rackets(self, players):
        """Estimate racket location from upper body of each player."""
        rackets = []
        for p in players:
            x1, y1, x2, y2 = p.bbox
            # racket lives in top-third of player box
            rackets.append(Detection((x1, y1, x2, y1 + (y2-y1)//3),
                                      0.4, "racket", track_id=p.track_id))
        return rackets