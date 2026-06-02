"""
Stage 2: Train Models
Model A (optional): YAMNet transfer learning
Model B (deploy):   Small MFCC CNN for ESP32S3

Usage:
    python 02_fine_tune_yamnet.py --data_dir ./processed --epochs 50
    python 02_fine_tune_yamnet.py --data_dir ./processed --epochs 50 --skip_yamnet
"""

import os
import argparse
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json

TARGET_SR = 16000
CLIP_DURATION = 1.0
N_MFCC = 14
HOP_LENGTH = 512
N_FFT = 1024

CLASS_NAMES = [
    "quiet",
    "human_activity",
    "appliance_mechanical",
    "transient_impact",
    "environmental_ambient"
]
N_CLASSES = len(CLASS_NAMES)


def load_dataset(data_dir, split):
    split_dir = os.path.join(data_dir, split)
    audio_clips, labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"  WARNING: {cls_dir} not found")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith('.wav'):
                continue
            audio, _ = librosa.load(os.path.join(cls_dir, fname), sr=TARGET_SR, mono=True)
            target_len = int(TARGET_SR * CLIP_DURATION)
            audio = np.pad(audio, (0, max(0, target_len - len(audio))))[:target_len]
            audio_clips.append(audio)
            labels.append(cls_idx)
    return np.array(audio_clips, dtype=np.float32), np.array(labels, dtype=np.int32)


def compute_mfccs(audio_clips, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT):
    """Compute MFCCs matching the Arduino on-device implementation exactly."""
    fixed_frames = 1 + int(TARGET_SR * CLIP_DURATION) // hop_length
    features = []
    n_mels = 40
    
    # Build mel filterbank matching Arduino implementation
    mel_min = 2595.0 * np.log10(1.0 + 0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + 8000.0 / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / TARGET_SR).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        for k in range(bin_points[m], bin_points[m+1]+1):
            if bin_points[m+1] != bin_points[m]:
                filterbank[m][k] = (k - bin_points[m]) / (bin_points[m+1] - bin_points[m])
        for k in range(bin_points[m+1], bin_points[m+2]+1):
            if bin_points[m+2] != bin_points[m+1]:
                filterbank[m][k] = (bin_points[m+2] - k) / (bin_points[m+2] - bin_points[m+1])

    for audio in audio_clips:
        mfcc_frames = []
        for f in range(fixed_frames):
            start = f * hop_length
            frame = np.zeros(n_fft)
            end = min(start + n_fft, len(audio))
            frame[:end-start] = audio[start:end]
            
            # Hann window
            window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1)))
            frame *= window
            
            # FFT power spectrum — matches Arduino: (Re^2 + Im^2) / N_FFT
            fft_out = np.fft.rfft(frame)
            power = (fft_out.real**2 + fft_out.imag**2) / n_fft
            
            # Mel filterbank + log
            mel_energy = filterbank @ power
            log_mel = np.log(mel_energy + 1e-10)
            
            # DCT without normalization — matches Arduino cosine sum
            mfcc = np.zeros(n_mfcc)
            for c in range(n_mfcc):
                for m in range(n_mels):
                    mfcc[c] += log_mel[m] * np.cos(np.pi * c * (m + 0.5) / n_mels)
            
            mfcc_frames.append(mfcc)
        
        features.append(np.array(mfcc_frames))
    
    return np.array(features, dtype=np.float32)


def extract_yamnet_embeddings(audio_clips):
    print("  Loading YAMNet...")
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    embeddings = []
    for i, audio in enumerate(audio_clips):
        if i % 50 == 0:
            print(f"  {i}/{len(audio_clips)}...")
        _, emb, _ = yamnet(audio)
        embeddings.append(np.mean(emb.numpy(), axis=0))
    return np.array(embeddings, dtype=np.float32)


def build_yamnet_classifier(n_classes=N_CLASSES):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_small_cnn(input_shape, n_classes=N_CLASSES):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(8, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_and_plot(model, X_test, y_test, name, save_dir):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3)
    print(f"\n{name}\n{report}")
    with open(os.path.join(save_dir, f"{name}_report.txt"), 'w') as f:
        f.write(report)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    fig.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_confusion.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./processed")
    parser.add_argument("--output_dir", default="./models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_yamnet", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    train_audio, train_labels = load_dataset(args.data_dir, "train")
    val_audio,   val_labels   = load_dataset(args.data_dir, "val")
    test_audio,  test_labels  = load_dataset(args.data_dir, "test")
    print(f"  Train:{len(train_audio)} Val:{len(val_audio)} Test:{len(test_audio)}")

    if not args.skip_yamnet:
        print("\n=== MODEL A: YAMNet Transfer Learning ===")
        tr_emb = extract_yamnet_embeddings(train_audio)
        va_emb = extract_yamnet_embeddings(val_audio)
        te_emb = extract_yamnet_embeddings(test_audio)
        clf = build_yamnet_classifier()
        clf.fit(tr_emb, train_labels, validation_data=(va_emb, val_labels),
                epochs=args.epochs, batch_size=args.batch_size,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        evaluate_and_plot(clf, te_emb, test_labels, "yamnet_transfer", args.output_dir)
        clf.save(os.path.join(args.output_dir, "yamnet_classifier.keras"))

    print("\n=== MODEL B: Small MFCC CNN ===")
    tr_mfcc = compute_mfccs(train_audio)
    va_mfcc = compute_mfccs(val_audio)
    te_mfcc = compute_mfccs(test_audio)
    print(f"  MFCC shape: {tr_mfcc.shape}")

    cnn = build_small_cnn((tr_mfcc.shape[1], tr_mfcc.shape[2]))
    cnn.summary()
    p = cnn.count_params()
    print(f"  Params:{p:,}  Est INT8:{p/1024:.1f}KB")
    cnn.fit(tr_mfcc, train_labels, validation_data=(va_mfcc, val_labels),
            epochs=args.epochs, batch_size=args.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)])
    evaluate_and_plot(cnn, te_mfcc, test_labels, "small_cnn", args.output_dir)
    cnn.save(os.path.join(args.output_dir, "small_cnn.keras"))

    cfg = {"n_mfcc": N_MFCC, "hop_length": HOP_LENGTH, "n_fft": N_FFT,
           "sample_rate": TARGET_SR, "clip_duration": CLIP_DURATION,
           "input_shape": [tr_mfcc.shape[1], tr_mfcc.shape[2]],
           "class_names": CLASS_NAMES, "n_classes": N_CLASSES}
    with open(os.path.join(args.output_dir, "mfcc_config.json"), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"\nNext: python 03_quantize_and_profile.py --model_dir {args.output_dir}")

if __name__ == "__main__":
    main()
