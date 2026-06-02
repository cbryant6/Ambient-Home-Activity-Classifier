"""
record_clips.py — Captures audio clips from INMP441 via serial

Usage:
    python record_clips.py --port COM20 --output_dir ./device_recordings

In Arduino Serial Monitor type:
    quiet / human / appliance / impact / ambient  → start recording that class
    stop → pause

This script saves each 1-second clip as a WAV file automatically.
"""

import os
import argparse
import serial
import numpy as np
import soundfile as sf
from datetime import datetime

CLASS_NAMES = ["quiet", "human", "appliance", "impact", "ambient"]

FULL_CLASS_NAMES = {
    "quiet":     "quiet",
    "human":     "human_activity",
    "appliance": "appliance_mechanical",
    "impact":    "transient_impact",
    "ambient":   "environmental_ambient"
}

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True,
                        help="Serial port e.g. COM20")
    parser.add_argument("--output_dir", type=str, default="./device_recordings",
                        help="Where to save WAV files")
    parser.add_argument("--target_per_class", type=int, default=30,
                        help="Target clips per class before warning")
    args = parser.parse_args()

    # Create output folders
    for cls in FULL_CLASS_NAMES.values():
        os.makedirs(os.path.join(args.output_dir, cls), exist_ok=True)

    # Count existing clips
    clip_counts = {}
    for short, full in FULL_CLASS_NAMES.items():
        cls_dir = os.path.join(args.output_dir, full)
        existing = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]
        clip_counts[short] = len(existing)

    print(f"\n=== Ambient Classifier — Device Data Recorder ===")
    print(f"Saving to: {args.output_dir}")
    print(f"\nCurrent clip counts:")
    for short, full in FULL_CLASS_NAMES.items():
        count = clip_counts[short]
        status = "✓" if count >= args.target_per_class else f"need {args.target_per_class - count} more"
        print(f"  {full:30s}: {count:3d} clips  {status}")

    print(f"\nConnecting to {args.port} at 921600 baud...")

    try:
        ser = serial.Serial(args.port, 921600, timeout=5)
    except Exception as e:
        print(f"ERROR: Could not open {args.port}: {e}")
        print("Make sure Arduino Serial Monitor is CLOSED before running this script")
        return

    print("Connected! Type commands in this terminal or use Arduino IDE serial monitor")
    print("\nCommands: quiet / human / appliance / impact / ambient / stop / quit\n")

    current_class = None
    samples = []
    collecting = False
    expected_samples = CLIP_SAMPLES

    try:
        while True:
            # Check for local keyboard input
            import select
            import sys
            if sys.platform == 'win32':
                import msvcrt
                if msvcrt.kbhit():
                    cmd = input().strip().lower()
                    if cmd == 'quit':
                        break
                    elif cmd in CLASS_NAMES or cmd == 'stop':
                        ser.write((cmd + '\n').encode())
                        print(f">> Sent: {cmd}")

            # Read from serial
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except:
                    continue

                if not line:
                    continue

                # Clip header
                if line.startswith("CLIP:"):
                    parts = line.split(":")
                    short_cls = parts[1]
                    clip_num = int(parts[2])
                    current_class = short_cls
                    samples = []
                    collecting = True

                # Sample data
                elif collecting and line != "END":
                    try:
                        samples.append(float(line))
                    except:
                        pass

                # End of clip
                elif line == "END" and collecting:
                    collecting = False
                    if len(samples) >= CLIP_SAMPLES * 0.95:  # Allow 5% tolerance
                        audio = np.array(samples[:CLIP_SAMPLES], dtype=np.float32)
                        full_cls = FULL_CLASS_NAMES.get(current_class, current_class)
                        cls_dir = os.path.join(args.output_dir, full_cls)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{full_cls}_{timestamp}.wav"
                        filepath = os.path.join(cls_dir, filename)
                        sf.write(filepath, audio, SAMPLE_RATE)
                        clip_counts[current_class] = clip_counts.get(current_class, 0) + 1
                        count = clip_counts[current_class]
                        target = args.target_per_class
                        bar = "█" * min(count, target) + "░" * max(0, target - count)
                        print(f"  [{bar}] {full_cls}: {count}/{target}")
                        if count == target:
                            print(f"  ✓ {full_cls} target reached! Switch to next class.")
                    else:
                        print(f"  WARNING: Short clip ({len(samples)} samples) — skipped")

                # Regular messages
                elif not line.startswith("CLIP:"):
                    print(f"  ESP32: {line}")

    except KeyboardInterrupt:
        print("\n\nRecording stopped.")
    finally:
        ser.close()

    print("\nFinal clip counts:")
    for short, full in FULL_CLASS_NAMES.items():
        cls_dir = os.path.join(args.output_dir, full)
        count = len([f for f in os.listdir(cls_dir) if f.endswith('.wav')])
        status = "✓ DONE" if count >= args.target_per_class else f"need {args.target_per_class - count} more"
        print(f"  {full:30s}: {count:3d}  {status}")

    print(f"\nNext: python 01_prepare_dataset.py --device_dir {args.output_dir}")

if __name__ == "__main__":
    main()
