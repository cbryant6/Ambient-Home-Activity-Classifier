# 🔊 Ambient Home Activity Classifier

> A battery-powered embedded ML device that classifies household audio environments in real time — entirely on-device, no cloud, no internet.

Built on a **XIAO ESP32S3** microcontroller with a custom on-device DSP pipeline, a quantized neural network via **TensorFlow Lite Micro**, and a **GC9A01 circular display** showing live predictions with a red-to-green confidence ring.

---

<img width="7200" height="10800" alt="Ambient_Classifier_Poster (1)" src="https://github.com/user-attachments/assets/31c53b43-7302-4e62-a289-3bbe24f74582" />

<img width="1920" height="1031" alt="Ambient Home Activity Classifier Schematic" src="https://github.com/user-attachments/assets/ab5bb583-a6e4-4005-b95f-343d94fcc520" />




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

## ⚡ Key Technical Decisions

**Why float32 instead of INT8?**
INT8 post-training quantization reduced model size from 25.2 KB to 18.3 KB but caused a 54-percentage-point accuracy drop. The cause: the custom on-device MFCC produces features in a different numerical distribution than the calibration dataset expected. Since 25.2 KB is only 0.3% of available flash and the ESP32S3 has a hardware FPU, float32 is the correct operating point — the size saving was not worth the accuracy loss.

**Why a custom MFCC instead of a library?**
Librosa (Python) uses center padding and orthonormal DCT normalization that cannot be replicated identically in the microcontroller C++ environment. A custom MFCC was implemented in both Python (training) and Arduino (deployment) with identical computation: same filterbank, same unnormalized DCT, same power spectrum scaling. This eliminated the feature mismatch that was causing all inference to fail.

**The single most important fix:**
Peak amplitude normalization. The training pipeline normalized every clip to peak 1.0. The device was feeding raw audio (RMS ~0.002) to the MFCC. One missing line — `audio = audio / np.max(np.abs(audio))` in training — caused the model to see feature values orders of magnitude outside its training distribution, making it predict Impact for everything. Adding the equivalent normalization on-device immediately resolved all classification failures.

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
