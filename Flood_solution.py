"""DroneRescue: illustrative four-drone, YOLOv8-assisted rescue simulation.

The visualized drones and lifejacket deliveries are simulated, not physical actions.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# -------- Settings --------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "yolov8n.pt"
VIDEO_PATH = ROOT / "flood_video.mp4"
OUTPUT_PATH = ROOT / "flood_annotated_output.mp4"

NUM_DRONES = 4
CONFIDENCE_THRESHOLD = 0.40
MAX_OUTPUT_WIDTH = 1280
MAX_OUTPUT_HEIGHT = 720
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540
SHOW_PREVIEW = True

# Animation lengths in seconds of *video* time, not real-world time.
THROW_DURATION_SECONDS = 1.2
COOLDOWN_SECONDS = 1.0

# OpenCV uses BGR, not RGB.
BLUE = (245, 160, 35)
GREEN = (75, 230, 80)
WHITE = (245, 245, 245)
DARK = (20, 35, 45)
YELLOW = (80, 225, 245)


def label(img, text, pos, scale=0.53, color=WHITE, thickness=2):
    """Outlined labels stay visible on light or dark footage."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                DARK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_drone(img, center, number, flying=True, frame_index=0):
    """Prominent four-rotor drone icon with simple propeller animation."""
    x, y = center
    arms = [(-27, -18), (27, -18), (-27, 18), (27, 18)]
    # Drop-shadow / contrast pad.
    cv2.circle(img, center, 34, (25, 28, 32), -1, cv2.LINE_AA)
    cv2.circle(img, center, 34, (180, 205, 220), 2, cv2.LINE_AA)
    for dx, dy in arms:
        cx, cy = x + dx, y + dy
        cv2.line(img, center, (cx, cy), (235, 240, 245), 5, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 9, (50, 55, 65), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 9, GREEN if flying else BLUE, 2, cv2.LINE_AA)
        if frame_index % 2 == 0:
            cv2.line(img, (cx - 10, cy), (cx + 10, cy), WHITE, 2, cv2.LINE_AA)
        else:
            cv2.line(img, (cx, cy - 10), (cx, cy + 10), WHITE, 2, cv2.LINE_AA)
    cv2.circle(img, center, 11, BLUE, -1, cv2.LINE_AA)
    cv2.circle(img, center, 11, WHITE, 2, cv2.LINE_AA)
    label(img, f"DRONE {number}", (x - 36, y - 44), 0.47)


def draw_lifejacket(img, pos, color=BLUE, size=16):
    """Small vest-like pictogram, instead of the previous orange circle."""
    x, y = pos
    s = size
    contour = [
        (x - s, y - s), (x - 5, y - s), (x - 3, y - 4),
        (x + 3, y - 4), (x + 5, y - s), (x + s, y - s),
        (x + s - 2, y + s), (x - s + 2, y + s)
    ]
    cv2.fillPoly(img, [__import__('numpy').array(contour, dtype='int32')], color)
    cv2.polylines(img, [__import__('numpy').array(contour, dtype='int32')], True,
                  WHITE, 2, cv2.LINE_AA)
    cv2.line(img, (x, y - 3), (x, y + s - 3), WHITE, 2, cv2.LINE_AA)
    cv2.line(img, (x - s + 4, y + 2), (x + s - 4, y + 2), WHITE, 2, cv2.LINE_AA)


def ease_in_out(t):
    return t * t * (3 - 2 * t)


def render_throw(img, start, end, progress):
    """An arcing 2D illustrative path (not a physics simulation)."""
    p = ease_in_out(max(0.0, min(1.0, progress)))
    sx, sy = start
    ex, ey = end
    # An arc upward relative to a straight interpolation.
    arc = -45 * math.sin(math.pi * p)
    x = round(sx + (ex - sx) * p)
    y = round(sy + (ey - sy) * p + arc)
    prev = max(0.0, p - 0.07)
    tx = round(sx + (ex - sx) * prev)
    ty = round(sy + (ey - sy) * prev - 45 * math.sin(math.pi * prev))
    cv2.line(img, (tx, ty), (x, y), GREEN, 5, cv2.LINE_AA)
    draw_lifejacket(img, (x, y), BLUE)


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Missing video: {VIDEO_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input: {VIDEO_PATH}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if orig_w <= 0 or orig_h <= 0:
        cap.release()
        raise RuntimeError("Invalid input video dimensions")
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    scale = min(MAX_OUTPUT_WIDTH / orig_w, MAX_OUTPUT_HEIGHT / orig_h, 1.0)
    width = max(2, int(orig_w * scale) // 2 * 2)
    height = max(2, int(orig_h * scale) // 2 * 2)

    out = cv2.VideoWriter(str(OUTPUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to create output: {OUTPUT_PATH}")

    model = YOLO(str(MODEL_PATH))
    # Stable source video coordinates for this *visual* drone swarm.
    margin = max(70, min(110, width // 10))
    drone_y = max(85, height // 8)
    drone_points = [
        (round(margin + i * (width - 2 * margin) / max(1, NUM_DRONES - 1)), drone_y)
        for i in range(NUM_DRONES)
    ]
    throw_frames = max(2, round(THROW_DURATION_SECONDS * fps))
    cycle_frames = throw_frames + max(1, round(COOLDOWN_SECONDS * fps))

    if SHOW_PREVIEW:
        cv2.namedWindow("DroneRescue - Swarm Simulation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DroneRescue - Swarm Simulation", PREVIEW_WIDTH, PREVIEW_HEIGHT)

    print(f"Input: {orig_w}x{orig_h} | Output: {width}x{height} | FPS: {fps:.2f}")
    print(f"Total frames reported: {total_frames if total_frames > 0 else 'unknown'}")
    print("Processing... press Esc to stop early (preview window).")

    count = 0
    user_stopped = False
    started = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            detected = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            people = []
            for box in detected.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                people.append((x1, y1, x2, y2, confidence))

            # Stable left-to-right order within each frame. Does not perform identity tracking.
            people.sort(key=lambda p: (p[0] + p[2]) / 2)
            targets = people[:NUM_DRONES]

            for x1, y1, x2, y2, confidence in people:
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                label(frame, f"PERSON {confidence:.0%}", (x1, max(22, y1 - 9)), 0.43, GREEN, 1)

            phase = count % cycle_frames
            progress = min(1.0, phase / (throw_frames - 1))
            for i, center in enumerate(drone_points):
                draw_drone(frame, center, i + 1, i < len(targets), count)
            for i, (x1, y1, x2, y2, _) in enumerate(targets):
                target_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                drone_point = drone_points[i]
                cv2.line(frame, drone_point, target_point, BLUE, 1, cv2.LINE_AA)
                if phase < throw_frames:
                    render_throw(frame, drone_point, target_point, progress)
                    label(frame, f"D{i+1}: DEPLOYING", (x1, min(height - 12, y2 + 22)), 0.42, BLUE, 1)
                else:
                    draw_lifejacket(frame, target_point, GREEN)
                    cv2.circle(frame, target_point, 26 + count % 5, GREEN, 2, cv2.LINE_AA)
                    label(frame, f"D{i+1}: SIMULATED DELIVERY", (x1, min(height - 12, y2 + 22)), 0.4, GREEN, 1)

            # Translucent overlays help legibility with busy footage.
            panel = frame.copy()
            cv2.rectangle(panel, (0, 0), (width, 32), DARK, -1)
            cv2.rectangle(panel, (0, height - 67), (width, height), DARK, -1)
            frame = cv2.addWeighted(panel, 0.54, frame, 0.46, 0)
            label(frame, "DRONERESCUE | AI FLOOD RESPONSE SIMULATION", (14, 23), 0.60)
            label(frame, f"DETECTED: {len(people)}   DRONES: {NUM_DRONES}   TARGETS: {len(targets)}",
                  (16, height - 39), 0.57)
            percent = min(100.0, ((count + 1) / total_frames * 100)) if total_frames > 0 else None
            progress_text = f"ANNOTATION: {percent:.1f}%" if percent is not None else f"FRAMES: {count + 1}"
            label(frame, progress_text, (16, height - 13), 0.50, GREEN, 1)
            if total_frames > 0:
                cv2.rectangle(frame, (width // 2, height - 23), (width - 18, height - 13), WHITE, 1)
                cv2.rectangle(frame, (width // 2 + 1, height - 22),
                              (width // 2 + int((width // 2 - 20) * min(1., (count + 1) / total_frames)), height - 14),
                              GREEN, -1)

            out.write(frame)
            count += 1
            if count == 1 or count % max(1, int(fps * 2)) == 0:
                elapsed = max(0.01, time.monotonic() - started)
                rate = count / elapsed
                if percent is not None:
                    eta = max(0, (total_frames - count) / max(rate, 0.01))
                    print(f"\rAnnotation {percent:5.1f}% | {count}/{total_frames} frames | ~{eta:.0f}s remaining", end="", flush=True)
                else:
                    print(f"\rAnnotated {count} frames", end="", flush=True)

            if SHOW_PREVIEW:
                preview_scale = min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height, 1.0)
                preview = cv2.resize(frame, (max(1, round(width * preview_scale)),
                                            max(1, round(height * preview_scale))))
                cv2.imshow("DroneRescue - Swarm Simulation", preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    user_stopped = True
                    break
    except KeyboardInterrupt:
        user_stopped = True
    finally:
        cap.release()
        out.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    print("\n" + "=" * 55)
    if user_stopped:
        print("STOPPED EARLY: Only the processed portion was saved.")
    else:
        print("VIDEO ANNOTATION COMPLETE")
    print(f"Frames: {count} / {total_frames if total_frames > 0 else '?'}")
    print(f"Elapsed: {elapsed:.1f} seconds")
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()
