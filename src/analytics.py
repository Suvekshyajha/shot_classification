from collections import defaultdict
import numpy as np


class ShotAnalytics:

    def compute(self, events, total_frames, fps):

        if not events:
            return {"total_shots": 0, "message": "no shots detected"}

        duration = total_frames / fps
        total    = len(events)

        # count each shot type
        shot_counts = defaultdict(int)
        for e in events:
            shot_counts[e.shot_type.value] += 1

        # percentage of each shot type
        shot_pct = {k: round(v / total * 100, 1) for k, v in shot_counts.items()}

        # per player shot counts
        player_counts = defaultdict(lambda: defaultdict(int))
        for e in events:
            pid = e.player_id if e.player_id is not None else "unknown"
            player_counts[pid][e.shot_type.value] += 1

        # average ball speed per shot type
        speeds = defaultdict(list)
        for e in events:
            if e.speed_kmh:
                speeds[e.shot_type.value].append(e.speed_kmh)
        avg_speed = {k: round(float(np.mean(v)), 1) for k, v in speeds.items()}

        # where on court shots happened
        zone_counts = defaultdict(int)
        for e in events:
            zone_counts[e.zone] += 1

        # cross court vs down the line
        direction_counts = defaultdict(int)
        for e in events:
            direction_counts[e.direction] += 1

        # shots per minute
        shots_per_min = round(total / (duration / 60), 1)

        # how many shots in each 10 second window
        timeline = defaultdict(int)
        for e in events:
            bucket = int(e.timestamp / 10) * 10
            timeline[bucket] += 1

        return {
            "total_shots":      total,
            "duration_sec":     round(duration, 1),
            "shots_per_minute": shots_per_min,
            "shot_counts":      dict(shot_counts),
            "shot_percentages": shot_pct,
            "avg_speed_kmh":    avg_speed,
            "zone_counts":      dict(zone_counts),
            "direction_counts": dict(direction_counts),
            "player_counts":    {str(k): dict(v) for k, v in player_counts.items()},
            "timeline_per_10s": dict(timeline),
        }
