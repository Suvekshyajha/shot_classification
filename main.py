import argparse
import logging
from src import PadelPipeline

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt = "%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Padel Shot Classifier")
    parser.add_argument("--video",       required=True,        help="path to input video")
    parser.add_argument("--model",       default="models/yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf",        default=0.35, type=float,    help="detection confidence")
    parser.add_argument("--output",      default="outputs",    help="folder to save results")
    parser.add_argument("--no-annotate", action="store_true",  help="skip annotated video")
    parser.add_argument("--max-frames",  default=None, type=int, help="limit frames (for testing)")
    args = parser.parse_args()

    pipeline = PadelPipeline(
        model_path = args.model,
        conf       = args.conf,
        annotate   = not args.no_annotate,
        output_dir = args.output,
    )

    result    = pipeline.run(video_path=args.video, max_frames=args.max_frames)
    meta      = result["meta"]
    analytics = result["analytics"]

    print("\n" + "=" * 50)
    print("       PADEL ANALYTICS — RESULTS")
    print("=" * 50)
    print(f"  Video      : {meta['video']}")
    print(f"  Duration   : {meta['duration_sec']}s")
    print(f"  Processed  : {meta['process_time_s']}s")
    print("-" * 50)
    print(f"  Total Shots: {analytics['total_shots']}")
    print(f"  Shots/min  : {analytics.get('shots_per_minute', 'N/A')}")
    print("-" * 50)
    print("  Shot Breakdown:")
    for shot, count in analytics.get("shot_counts", {}).items():
        pct = analytics.get("shot_percentages", {}).get(shot, 0)
        bar = "█" * int(pct / 5)
        print(f"    {shot:<12} {count:>3}  {bar} {pct}%")
    print("-" * 50)
    print("  Outputs saved:")
    for k, v in result["outputs"].items():
        if v:
            print(f"    {k:<8} → {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
