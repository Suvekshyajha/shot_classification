import cv2
import json
import csv
import time
import logging
from pathlib import Path
from tqdm import tqdm

from .detector   import PadelDetector
from .classifier import ShotClassifier
from .visualizer import FrameVisualizer
from .analytics  import ShotAnalytics

logger = logging.getLogger(__name__)


class PadelPipeline:

    def __init__(self,
                 model_path    = "yolov8n.pt",
                 conf          = 0.35,
                 annotate      = True,
                 output_dir    = "outputs"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.annotate   = annotate

        # create all modules
        self.detector   = PadelDetector(model_path, conf)
        self.visualizer = FrameVisualizer()
        self.analytics  = ShotAnalytics()

    def run(self, video_path, max_frames=None):
        """
        Main function.
        Opens video → loops frames → detects → classifies → saves.
        Returns a summary dict.
        """

        # ── open video ───────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        logger.info(f"Video : {video_path}")
        logger.info(f"Size  : {width}x{height}  FPS: {fps}  Frames: {total_frames}")

        # ── create classifier with correct fps ───────────────────────
        classifier = ShotClassifier(fps=fps)

        # ── setup output video writer ────────────────────────────────
        writer       = None
        out_vid_path = None

        if self.annotate:
            name         = Path(video_path).stem
            out_vid_path = str(self.output_dir / f"{name}_annotated.mp4")
            fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
            writer       = cv2.VideoWriter(out_vid_path, fourcc,
                                           fps, (width, height))

        # ── main loop ────────────────────────────────────────────────
        shot_events = []
        frame_idx   = 0
        start_time  = time.time()

        with tqdm(total=total_frames, desc="Processing", unit="frame") as bar:
            while True:
                ret, frame = cap.read()

                # end of video or max frames reached
                if not ret or (max_frames and frame_idx >= max_frames):
                    break

                timestamp = frame_idx / fps

                # detect objects in this frame
                det = self.detector.detect(frame, frame_idx, timestamp)

                # attach raw frame for pose analysis
                det.raw_frame = frame

                # classify shot
                event = classifier.update(det, frame)
                if event:
                    shot_events.append(event)
                    logger.debug(f"Shot {event.shot_id} | "
                                 f"{event.shot_type.value} | "
                                 f"{timestamp:.2f}s")

                # draw annotations and write frame
                if writer:
                    annotated = self.visualizer.draw(frame, det, event)
                    writer.write(annotated)

                frame_idx += 1
                bar.update(1)

        # cleanup
        cap.release()
        if writer:
            writer.release()

        elapsed = time.time() - start_time
        logger.info(f"Done in {elapsed:.1f}s  "
                    f"({frame_idx / elapsed:.1f} fps processing speed)")

        # ── save outputs ─────────────────────────────────────────────
        name     = Path(video_path).stem
        json_out = self.output_dir / f"{name}_shots.json"
        csv_out  = self.output_dir / f"{name}_shots.csv"

        self._save_json(shot_events, json_out, video_path, fps, frame_idx)
        self._save_csv(shot_events, csv_out)

        # ── compute analytics ────────────────────────────────────────
        analytics = self.analytics.compute(shot_events, frame_idx, fps)

        # ── return summary ───────────────────────────────────────────
        return {
            "shots":      [e.to_dict() for e in shot_events],
            "analytics":  analytics,
            "outputs": {
                "json":  str(json_out),
                "csv":   str(csv_out),
                "video": out_vid_path,
            },
            "meta": {
                "video":          video_path,
                "total_frames":   frame_idx,
                "fps":            fps,
                "duration_sec":   round(frame_idx / fps, 1),
                "process_time_s": round(elapsed, 2),
            }
        }

    # ── save all shots to JSON ───────────────────────────────────────
    def _save_json(self, events, path, video_path, fps, total_frames):
        data = {
            "video":       video_path,
            "fps":         fps,
            "total_frames":total_frames,
            "total_shots": len(events),
            "shots":       [e.to_dict() for e in events],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON saved → {path}")

    # ── save all shots to CSV ────────────────────────────────────────
    def _save_csv(self, events, path):
        if not events:
            logger.warning("No shots to save in CSV")
            return
        rows = [e.to_dict() for e in events]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CSV saved → {path}")