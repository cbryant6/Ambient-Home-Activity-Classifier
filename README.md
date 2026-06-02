# 🔊 Ambient Home Activity Classifier

> A battery-powered embedded ML device that classifies household audio environments in real time — entirely on-device, no cloud, no internet.

Built on a **XIAO ESP32S3** microcontroller with a custom on-device DSP pipeline, a quantized neural network via **TensorFlow Lite Micro**, and a **GC9A01 circular display** showing live predictions with a red-to-green confidence ring.

---

## 📸 

<img width="1092" height="1115" alt="image" src="https://github.com/user-attachments/assets/c6d2ef29-758e-405d-859b-14678a19a340" />
<img width="1064" height="1129" alt="image" src="https://github.com/user-attachments/assets/73b86fb4-2d08-4529-b275-73cbb1161647" />
<img width="1204" height="986" alt="image" src="https://github.com/user-attachments/assets/b7950a2c-14e5-4fcd-8c9b-2cf49f0070af" />


---

## 🎯 What It Does

The device listens to its environment continuously and classifies the current audio scene into one of five classes:

| Class | Label | Description |
|---|---|---|
| 0 | **Quiet** | Low-activity background — silence or near-silence |
| 1 | **Human Activity** | Footsteps, clapping, coughing, speech, general human presence |
| 2 | **Appliance / Mechanical** | Fans, vacuums, washing machines, sustained mechanical noise |
| 3 | **Transient Impact** | Knocks, door slams, drops, sudden sharp sounds |
| 4 | **Environmental / Ambient** | Rain, wind, birds, crickets, outdoor nature sounds |

The circular display shows the predicted class, confidence percentage, a class-specific symbol, and a red-to-green ring indicating model confidence — all updating every 3 seconds from a fully on-device inference pipeline.

---

## 🔧 Hardware

| Component | Role | Notes |
|---|---|---|
| XIAO ESP32S3 | Microcontroller | 240 MHz, 8 MB flash, 8 MB PSRAM, hardware FPU |
| INMP441 | I2S digital microphone | 16 kHz, left channel, breakout board |
| GC9A01 | 240×240 circular SPI display | Red-to-green confidence ring UI |
| 1200 mAh LiPo | Power supply | ~7.5–9 hour runtime estimate |
| Slide switch | Hard power cutoff | In series on battery positive line |
| 3D printed enclosure | Housing | Custom designed, black PLA |

---

## 🧠 ML Pipeline

The full inference pipeline runs entirely on the ESP32S3 — no external compute, no API calls.

```
I2S Microphone (16 kHz)
        ↓
  Audio Capture (1 second)
        ↓
  Peak Normalization
        ↓
  On-Device MFCC (n_mfcc=14, 32 frames)
  [Hann window → FFT → Mel filterbank → DCT]
        ↓
  TFLite Micro Inference (Float32, 25.2 KB)
  [Small 1D CNN: Conv(8)→Conv(16)→Conv(32)→Dense(32)→Dense(5)]
        ↓
  Majority-Vote Smoothing (5-window)
  + Transient Impact Immediate Override
        ↓
  GC9A01 Display Update
```

### Model Details

| Property | Value |
|---|---|
| Architecture | 1D CNN (3 conv layers + global avg pool + dense) |
| Parameters | ~7,500 |
| Model size | 25.2 KB (float32) |
| Inference time | 18–21 ms |
| MFCC computation | ~263 ms |
| Total cycle | ~3 seconds |
| Test accuracy | 74.9% (5-class) |
| Flash usage | 16% of 8 MB |
| RAM usage | 74% of 327 KB |

### Training Dataset

| Source | Purpose | Clips |
|---|---|---|
| ESC-50 | Appliance, impact, environmental classes | ~800 clips |
| LibriSpeech dev-clean | Speech component of human activity | 60 clips |
| Device recordings (INMP441) | Domain adaptation to real hardware | 180 clips |
| Synthesized quiet | Quiet class baseline | 80 clips |
| Augmentation | Pitch shift, noise, gain, time stretch | → ~300/class |

> **Key finding:** Training on ESC-50 alone caused the model to fail in deployment due to domain shift (studio mics ≠ INMP441 in a real room). Adding 34–38 device recordings per class improved the two weakest classes from ~50% to ~75% F1.

---

## 📁 Repository Structure

```
├── pipeline/
│   ├── 01_prepare_dataset.py       # Download ESC-50 + LibriSpeech, augment, split
│   ├── 02_fine_tune_yamnet.py       # Train small CNN with custom MFCC features
│   ├── 03_quantize_and_profile.py   # Convert to TFLite (float32/float16/INT8)
│   ├── 04_export_to_c_header.py     # Convert .tflite → C array for Arduino
│   └── README.md
│
├── arduino/
│   ├── ambient_classifier/
│   │   ├── ambient_classifier.ino   # Main inference + display sketch
│   │   ├── model_data.h             # Trained model as C byte array
│   │   └── class_config.h           # Class names, colors, MFCC config
│   └── data_recorder/
│       └── data_recorder.ino        # Device recording sketch for data collection
│
├── tools/
│   └── record_clips.py              # Python serial capture for device recordings
│
├── models/
│   ├── small_cnn.keras              # Trained Keras model
│   ├── model_float32.tflite         # Float32 TFLite
│   ├── model_float16.tflite         # Float16 TFLite
│   ├── model_int8.tflite            # INT8 TFLite (note: calibration issues)
│   ├── mfcc_config.json             # Feature config for deployment
│   └── quantization_summary.json    # Size/accuracy tradeoff data
│
└── docs/
    └── Development_Journal.docx     # Full design decision log
```

---

## ⚡ Key Technical Decisions

**Why float32 instead of INT8?**
INT8 post-training quantization reduced model size from 25.2 KB to 18.3 KB but caused a 54-percentage-point accuracy drop. The cause: the custom on-device MFCC produces features in a different numerical distribution than the calibration dataset expected. Since 25.2 KB is only 0.3% of available flash and the ESP32S3 has a hardware FPU, float32 is the correct operating point — the size saving was not worth the accuracy loss.

**Why a custom MFCC instead of a library?**
Librosa (Python) uses center padding and orthonormal DCT normalization that cannot be replicated identically in the microcontroller C++ environment. A custom MFCC was implemented in both Python (training) and Arduino (deployment) with identical computation: same filterbank, same unnormalized DCT, same power spectrum scaling. This eliminated the feature mismatch that was causing all inference to fail.

**The single most important fix:**
Peak amplitude normalization. The training pipeline normalized every clip to peak 1.0. The device was feeding raw audio (RMS ~0.002) to the MFCC. One missing line — `audio = audio / np.max(np.abs(audio))` in training — caused the model to see feature values orders of magnitude outside its training distribution, making it predict Impact for everything. Adding the equivalent normalization on-device immediately resolved all classification failures.

---

## 🚀 Running the Code — Grader Instructions

### Prerequisites

**Python environment:**
```bash
pip install tensorflow tensorflow-hub librosa soundfile scikit-learn matplotlib audiomentations
```

**Arduino libraries** (install via Arduino Library Manager):
- Adafruit GC9A01A
- Adafruit GFX Library
- TensorFlowLite_ESP32
- arduinoFFT

**Arduino board package:** ESP32 by Espressif **version 2.0.17** (not 3.x — TFLite library incompatibility)

**Board:** XIAO_ESP32S3

---

### Step 1 — Prepare Dataset

Download ESC-50:
```bash
git clone https://github.com/karolpiczak/ESC-50.git
```

Download LibriSpeech dev-clean (~350 MB, no account needed):
```bash
curl -O https://www.openslr.org/resources/12/dev-clean.tar.gz
tar xzf dev-clean.tar.gz
```

Run dataset preparation (uses ESC-50 + LibriSpeech + optional device recordings):
```bash
python pipeline/01_prepare_dataset.py \
  --esc50_dir ./ESC-50 \
  --librispeech_dir ./LibriSpeech/dev-clean \
  --output_dir ./processed
```

---

### Step 2 — Train Model

```bash
python pipeline/02_fine_tune_yamnet.py \
  --data_dir ./processed \
  --output_dir ./models \
  --epochs 150 \
  --skip_yamnet
```

This trains the small 1D CNN on custom MFCC features matching the on-device implementation. Add `--skip_yamnet` to skip the optional YAMNet upper-bound comparison. Expected accuracy: ~72–75%.

---

### Step 3 — Quantize and Profile

```bash
python pipeline/03_quantize_and_profile.py \
  --model_dir ./models \
  --data_dir ./processed
```

Outputs float32, float16, and INT8 TFLite variants with size and accuracy comparison. The tradeoff table is printed to console.

---

### Step 4 — Export to C Header

```bash
python pipeline/04_export_to_c_header.py \
  --tflite_path ./models/model_float32.tflite \
  --config_path ./models/mfcc_config.json \
  --output_dir ./arduino_deploy
```

Generates `model_data.h` and `class_config.h` for the Arduino sketch.

---

### Step 5 — Flash to ESP32S3

1. Copy `model_data.h` and `class_config.h` from `arduino_deploy/` into the `arduino/ambient_classifier/` folder
2. Open `arduino/ambient_classifier/ambient_classifier.ino` in Arduino IDE
3. Select **Tools → Board → XIAO_ESP32S3**
4. Select the correct COM port
5. Click Upload

Open Serial Monitor at **115200 baud** to see live predictions:
```
Raw:Quiet       ( 97%) Show:Quiet        Audio:878ms MFCC:263ms Infer:20ms
Raw:Human       ( 71%) Show:Human        Audio:901ms MFCC:264ms Infer:19ms
Raw:Impact      ( 93%) Show:Impact       Audio:878ms MFCC:263ms Infer:21ms
```

---

### Optional — Collect Device Recordings

To record your own clips directly from the INMP441 for domain adaptation:

1. Flash `arduino/data_recorder/data_recorder.ino` to the ESP32S3
2. Close Arduino Serial Monitor
3. Run the capture script:
```bash
python tools/record_clips.py --port COM20 --output_dir ./device_recordings
```
4. Type class names in the terminal (`quiet`, `human`, `appliance`, `impact`, `ambient`) to start/stop recording each class
5. Re-run Step 1 with `--device_dir ./device_recordings` to include them in training

---

## 📊 Results

### Quantization Tradeoff

| Variant | Size | Accuracy | Decision |
|---|---|---|---|
| Float32 | 25.2 KB | **74.9%** | ✅ Deployed |
| Float16 | 20.7 KB | 74.5% | Negligible saving |
| INT8 | 18.3 KB | 20.8% | ❌ Calibration fails |

### Per-Class Performance (Final Model)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Quiet | 0.900 | 1.000 | **0.947** |
| Human Activity | 0.771 | 0.529 | 0.628 |
| Appliance/Mechanical | 0.739 | 0.756 | **0.747** |
| Transient Impact | 0.627 | 0.711 | 0.667 |
| Environmental/Ambient | 0.714 | 0.778 | **0.745** |
| **Overall** | | | **74.9%** |

---

## 🔬 Built With

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Arduino](https://img.shields.io/badge/Arduino-00979D?style=flat&logo=arduino&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=cplusplus&logoColor=white)

**ML:** TensorFlow / Keras, TensorFlow Lite Micro, librosa, audiomentations  
**Embedded:** Arduino (ESP32S3), I2S, SPI, TFLite Micro, ArduinoFFT  
**Hardware:** XIAO ESP32S3, INMP441, GC9A01, LiPo battery  
**Data:** ESC-50, LibriSpeech dev-clean, custom device recordings  

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
