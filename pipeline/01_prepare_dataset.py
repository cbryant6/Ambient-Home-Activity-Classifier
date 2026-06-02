"""
Stage 1: Dataset Preparation (ESC-50 + LibriSpeech Mini)

- ESC-50: appliance, transient impact, environmental, human activity (non-speech)
- LibriSpeech dev-clean: speech/talking → added to human_activity class
- Synthesized: quiet samples

Usage:
    python 01_prepare_dataset.py --esc50_dir ./ESC-50-master --librispeech_dir ./LibriSpeech/dev-clean --output_dir ./processed

LibriSpeech download (no account needed):
    wget https://www.openslr.org/resources/12/dev-clean.tar.gz
    tar xzf dev-clean.tar.gz
"""

import os
import argparse
import random
import glob
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split

try:
    from audiomentations import Compose, PitchShift, AddGaussianNoise, Gain, TimeStretch
    HAS_AUGMENTATION = True
except ImportError:
    print("WARNING: audiomentations not installed. Skipping augmentation.")
    print("Install with: pip install audiomentations")
    HAS_AUGMENTATION = False

# =============================================================================
# Configuration
# =============================================================================

TARGET_SR = 16000
CLIP_DURATION = 1.0
TARGET_SAMPLES_PER_CLASS = 300
RANDOM_SEED = 42
LIBRISPEECH_CLIPS = 60  # How many speech clips to add

CLASS_NAMES = [
    "quiet",
    "human_activity",
    "appliance_mechanical",
    "transient_impact",
    "environmental_ambient"
]

ESC50_CLASS_MAP = {
    # Human Activity — non-speech (class 1)
    "footsteps":         "human_activity",
    "clapping":          "human_activity",
    "breathing":         "human_activity",
    "coughing":          "human_activity",
    "laughing":          "human_activity",
    "sneezing":          "human_activity",

    # Appliance / Mechanical (class 2)
    "vacuum_cleaner":    "appliance_mechanical",
    "washing_machine":   "appliance_mechanical",
    "clock_tick":        "appliance_mechanical",
    "engine":            "appliance_mechanical",

    # Transient Impact (class 3)
    "door_wood_knock":   "transient_impact",
    "glass_breaking":    "transient_impact",
    "can_opening":       "transient_impact",
    "mouse_click":       "transient_impact",
    "keyboard_typing":   "transient_impact",

    # Environmental / Ambient (class 4)
    "rain":              "environmental_ambient",
    "wind":              "environmental_ambient",
    "crickets":          "environmental_ambient",
    "chirping_birds":    "environmental_ambient",
    "sea_waves":         "environmental_ambient",
}


# =============================================================================
# Audio Utilities
# =============================================================================

def load_and_normalize(filepath, target_sr=TARGET_SR, duration=CLIP_DURATION):
    """Load, resample, normalize, pad/trim to fixed duration."""
    try:
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None

    target_len = int(target_sr * duration)

    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    elif len(audio) > target_len:
        start = random.randint(0, len(audio) - target_len)
        audio = audio[start:start + target_len]

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio


def create_augmentation_pipeline():
    if not HAS_AUGMENTATION:
        return None
    return Compose([
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
    ])


def augment_clip(audio, augmenter, sr=TARGET_SR):
    if augmenter is None:
        return audio
    return augmenter(samples=audio, sample_rate=sr)


def generate_quiet_samples(n_samples, sr=TARGET_SR, duration=CLIP_DURATION):
    samples = []
    target_len = int(sr * duration)
    for _ in range(n_samples):
        noise_level = random.uniform(0.001, 0.02)
        audio = np.random.randn(target_len) * noise_level
        samples.append(audio.astype(np.float32))
    return samples


# =============================================================================
# ESC-50 Processing
# =============================================================================

def process_esc50(esc50_dir):
    """Process ESC-50 and map to target classes."""
    meta_file = os.path.join(esc50_dir, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_dir, "audio")

    if not os.path.exists(meta_file):
        print(f"ERROR: ESC-50 not found at {meta_file}")
        print("Download: https://github.com/karolpiczak/ESC-50")
        return None

    import csv
    processed = {c: [] for c in CLASS_NAMES}
    source_counts = {}

    with open(meta_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            esc_class = row['category']
            if esc_class in ESC50_CLASS_MAP:
                target_class = ESC50_CLASS_MAP[esc_class]
                filepath = os.path.join(audio_dir, row['filename'])
                audio = load_and_normalize(filepath)
                if audio is not None:
                    processed[target_class].append(audio)
                    source_counts[esc_class] = source_counts.get(esc_class, 0) + 1

    print("\nESC-50 source breakdown:")
    for esc_cls, count in sorted(source_counts.items()):
        target = ESC50_CLASS_MAP[esc_cls]
        print(f"  {esc_cls:20s} -> {target:25s} ({count} clips)")

    print("\nMapped totals:")
    for cls in CLASS_NAMES:
        print(f"  {cls:25s}: {len(processed[cls])}")

    return processed


# =============================================================================
# LibriSpeech Processing
# =============================================================================

def process_librispeech(librispeech_dir, max_clips=LIBRISPEECH_CLIPS):
    """
    Load speech clips from LibriSpeech dev-clean.
    Directory structure: dev-clean/SPEAKER/CHAPTER/SPEAKER-CHAPTER-UTTERANCE.flac

    These get added to human_activity to include talking/speech.
    """
    if not os.path.exists(librispeech_dir):
        print(f"  LibriSpeech not found at {librispeech_dir} — skipping")
        print(f"  Download: wget https://www.openslr.org/resources/12/dev-clean.tar.gz")
        print(f"  Extract:  tar xzf dev-clean.tar.gz")
        return []

    # Find all .flac files recursively
    flac_files = glob.glob(os.path.join(librispeech_dir, "**", "*.flac"), recursive=True)

    if len(flac_files) == 0:
        print(f"  No .flac files found in {librispeech_dir}")
        return []

    print(f"  Found {len(flac_files)} LibriSpeech clips")

    # Randomly sample max_clips
    random.shuffle(flac_files)
    selected = flac_files[:max_clips]

    speech_clips = []
    for filepath in selected:
        audio = load_and_normalize(filepath)
        if audio is not None:
            speech_clips.append(audio)

    print(f"  Loaded {len(speech_clips)} speech clips → human_activity")
    return speech_clips



# =============================================================================
# Device Recordings
# =============================================================================

def process_device_recordings(device_dir):
    """Load clips recorded from INMP441 via data_recorder.ino."""
    processed = {c: [] for c in CLASS_NAMES}
    if not os.path.exists(device_dir):
        print(f"  Device recordings not found at {device_dir} — skipping")
        return processed
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(device_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.endswith(".wav")]
        for fname in files:
            audio = load_and_normalize(os.path.join(cls_dir, fname))
            if audio is not None:
                processed[cls].append(audio)
                total += 1
        print(f"  {cls:25s}: {len(processed[cls])} device clips")
    print(f"  Total device clips: {total}")
    return processed

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50_dir", type=str, default="./ESC-50-master")
    parser.add_argument("--librispeech_dir", type=str, default="./LibriSpeech/dev-clean")
    parser.add_argument("--output_dir", type=str, default="./processed")
    parser.add_argument("--device_dir", type=str, default="./device_recordings",
                        help="Device recordings from data_recorder.ino")
    parser.add_argument("--augment_to", type=int, default=TARGET_SAMPLES_PER_CLASS)
    parser.add_argument("--speech_clips", type=int, default=LIBRISPEECH_CLIPS,
                        help="Number of speech clips to add from LibriSpeech")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- ESC-50 ---
    print("=" * 60)
    print("Processing ESC-50...")
    print("=" * 60)
    all_data = process_esc50(args.esc50_dir)
    if all_data is None:
        return

    # --- LibriSpeech speech clips → human_activity ---
    print("\n" + "=" * 60)
    print("Processing LibriSpeech (speech → human_activity)...")
    print("=" * 60)
    speech_clips = process_librispeech(args.librispeech_dir, args.speech_clips)
    all_data["human_activity"].extend(speech_clips)

    # --- Device recordings ---
    print("\n" + "=" * 60)
    print("Processing device recordings (INMP441)...")
    print("=" * 60)
    device_data = process_device_recordings(args.device_dir)
    for cls in CLASS_NAMES:
        all_data[cls].extend(device_data[cls])
    device_counts = {cls: len(device_data[cls]) for cls in CLASS_NAMES}

    # --- Synthetic quiet ---
    print("\n" + "=" * 60)
    print("Generating quiet samples...")
    print("=" * 60)
    quiet = generate_quiet_samples(80)
    all_data["quiet"].extend(quiet)
    print(f"  Generated {len(quiet)} quiet samples")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Raw counts (before augmentation):")
    print("=" * 60)
    for cls in CLASS_NAMES:
        device_note = f"  ({device_counts.get(cls, 0)} device clips)"
        print(f"  {cls:25s}: {len(all_data[cls])}{device_note}")

    # --- Augment ---
    print("\n" + "=" * 60)
    print(f"Augmenting to {args.augment_to} per class...")
    print("=" * 60)
    augmenter = create_augmentation_pipeline()

    for cls in CLASS_NAMES:
        originals = list(all_data[cls])
        if len(originals) == 0:
            print(f"  WARNING: {cls} has 0 samples!")
            continue
        while len(all_data[cls]) < args.augment_to:
            source = random.choice(originals)
            all_data[cls].append(augment_clip(source, augmenter))
        print(f"  {cls:25s}: {len(all_data[cls])}")

    # --- Split ---
    print("\n" + "=" * 60)
    print("Splitting 70/15/15...")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(args.output_dir, split, cls), exist_ok=True)

    for cls in CLASS_NAMES:
        samples = all_data[cls]
        if len(samples) == 0:
            continue
        train, temp = train_test_split(samples, test_size=0.30, random_state=RANDOM_SEED)
        val, test = train_test_split(temp, test_size=0.50, random_state=RANDOM_SEED)

        for name, data in [("train", train), ("val", val), ("test", test)]:
            for i, audio in enumerate(data):
                path = os.path.join(args.output_dir, name, cls, f"{cls}_{name}_{i:04d}.wav")
                sf.write(path, audio, TARGET_SR)

        print(f"  {cls:25s}: train={len(train)}, val={len(val)}, test={len(test)}")

    with open(os.path.join(args.output_dir, "class_names.txt"), 'w') as f:
        for i, cls in enumerate(CLASS_NAMES):
            f.write(f"{i},{cls}\n")

    print(f"\nDone! Next: python 02_fine_tune_yamnet.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
