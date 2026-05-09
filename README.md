# Padel Game Analytics — Shot Classification System
### Approach Explanation

---

## Table of Contents
1. [Methodology](#methodology)
2. [System Architecture](#system-architecture)
3. [Challenges Faced](#challenges-faced)
4. [Improvements I Would Make](#improvements-i-would-make)

---

## Methodology

### Overview

The basic idea is simple — take a padel match video, go through it frame by frame, figure out where the players and ball are, and then try to guess what kind of shot was just played (forehand, backhand, or smash).

I split the work into separate files so each file handles just one job. That way it's easier to fix or change one part without breaking everything else.

### How It Works Step by Step

**1. Detecting players and the ball (`detector.py`)**

I used a pre-trained model called YOLOv8 to find the players, ball, and rackets in each frame. I didn't train this model myself — it already knows how to find people and objects. I just pointed it at each video frame and it tells me where everything is.

I picked the "nano" (smallest) version of YOLO because it's fast enough to run on a normal laptop without a GPU.

**2. Figuring out what shot was played (`classifier.py`)**

Once I know where the ball is in each frame, I look at how the ball is moving — left, right, up, or down — and which side of the court the player is on. From that I make a simple guess:

| What I see | Shot I call it |
|---|---|
| Ball going left to right, player on left | Forehand |
| Ball going right to left, player on left | Backhand |
| Ball coming sharply downward from high up | Smash / Serve |
| Can't tell | Unclassified |

I also made sure it doesn't count the same shot 10 times — once a shot is detected, it waits a bit before looking for the next one.

**3. Putting it all together (`pipeline.py`)**

This file just connects everything. It reads the video, sends each frame to the detector, sends the result to the classifier, draws labels on the frame, and saves everything to a file. Think of it as the manager that tells everyone else what to do.

```
Video → detect players/ball → classify shot → draw on frame → save output
```

**4. Counting and stats (`analytics.py`)**

After the video is done, this file adds everything up — how many forehands, how many backhands, how fast the ball was moving on average, etc. It then saves a JSON and CSV file with all the results.

**5. Drawing on the video (`visualizer.py`)**

This draws coloured boxes around the players and ball, shows the ball's path as a trail, and puts a label on screen when a shot is detected. It makes the output video look nice and easy to understand.

**Tools I Used**

| Tool | What I used it for |
|---|---|
| Python | Main programming language |
| OpenCV | Reading/writing video and drawing on frames |
| YOLOv8 | Detecting players, ball, rackets |
| NumPy | Basic math calculations |
| Pandas | Saving data to CSV |
| tqdm | Showing a progress bar in the terminal |

---

## System Architecture

```
shot/
├── src/
│   ├── detector.py       # finds players and ball
│   ├── classifier.py     # guesses the shot type
│   ├── pipeline.py       # runs everything in order
│   ├── analytics.py      # counts and stats
│   ├── visualizer.py     # draws on frames
│   └── __init__.py
├── models/               # YOLO weights go here (yolov8n.pt)
├── data/                 # your input video goes here
├── outputs/              # results saved here (JSON, CSV, video)
├── main.py               # run this to start
└── requirements.txt
```

Run command:
```bash
python main.py --video data/match.mp4 --model models/yolov8n.pt --conf 0.35
```

---

## Challenges Faced

**1. The ball was really hard to detect**

The ball in padel is tiny — sometimes just a few pixels on screen. When it moves fast it gets blurry and YOLO just misses it completely. I worked around this by remembering the last known position of the ball and using that as a guess when detection fails. It's not perfect but it helped a lot.

**2. Knowing exactly when a shot happened**

A player's swing takes about 10–20 frames. If I logged a shot on every single frame of the swing, I'd end up with 15 "forehands" for one actual shot. I fixed this by adding a short waiting period — once a shot is saved, the system ignores everything for the next half second or so.

**3. Telling left from right wasn't always easy**

My shot detection logic depends on knowing which side of the court the player is on. But depending on the camera angle, both players can look like they're in the middle. This made it hard to tell forehand from backhand reliably for every camera setup.

**4. No pre-made padel dataset exists**

I wanted to train a proper AI model to recognise shots, but there's no ready-made collection of labelled padel shots out there. Because of that I had to go with the simpler rule-based approach (just looking at ball direction). It's not as accurate but it works without needing any training data.

**5. Ball speed numbers aren't exact**

I estimated ball speed by looking at how many pixels the ball moved per frame, then converting that to km/h. The problem is I don't know the real size of the court in pixels, so the conversion is just a rough guess. The numbers give a general idea but aren't accurate enough to quote in a match report.

---

## Improvements I Would Make

**1. Track body movements, not just the ball**

Right now I only look at where the ball goes to guess the shot. A much better way would be to also look at how the player is moving — specifically their arm and wrist. Tools like MediaPipe can give you the exact position of joints on the body, so you could tell a forehand from a backhand just by watching the swing direction. I didn't have time to add this but it would make classification way more accurate.

**2. Use a better ball tracker**

YOLO is a general-purpose tool — it was built to find people, cars, dogs, etc. It's not great at tracking a tiny fast-moving ball. There are other tools built specifically for tracking small sports balls that would do a much better job here. This would fix most of the missed detections.

**3. Train a proper shot classifier with real examples**

The current system just uses simple if/else rules. If I could collect a few hundred clips of labelled shots (here's a forehand, here's a smash, etc.) I could train a small model to actually learn what different shots look like — rather than me hard-coding the logic myself.

**4. Fix the speed calculation**

To get real km/h values, I'd need to figure out how many pixels correspond to one metre on the court. This is doable by detecting the court lines and using the known court size (10m × 20m) to set the scale. It's not super complicated but I ran out of time to add it properly.

**5. Make it work live, not just on saved videos**

Right now it only works on a video file you already have. It would be cool to plug it into a live camera feed so coaches could see shot stats in real time during a match. This would just need some threading work to read frames faster.

**6. A simple dashboard to show the results**

Instead of reading a CSV file, it would be much nicer to have a simple webpage or app that shows charts — like a pie chart of shot types, a timeline of the match, and a count per player. Something a coach could open without being a developer.

---

*Built for the Layman.ai Padel Game Analytics Assignment.*
