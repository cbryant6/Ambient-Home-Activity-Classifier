#ifndef CLASS_CONFIG_H
#define CLASS_CONFIG_H

#define NUM_CLASSES 5
#define SAMPLE_RATE 16000
#define N_MFCC 14
#define HOP_LENGTH 512
#define N_FFT 1024
#define CLIP_DURATION_MS 1000

const char* CLASS_NAMES[NUM_CLASSES] = {
  "Quiet",
  "Human",
  "Appliance",
  "Impact",
  "Ambient"
};

const uint16_t CLASS_COLORS[NUM_CLASSES] = {
  0xFFFF,  // white - quiet
  0x07FF,  // cyan - human
  0xFFE0,  // yellow - appliance
  0xF800,  // red - impact
  0x07E0  // green - ambient
};

#endif
