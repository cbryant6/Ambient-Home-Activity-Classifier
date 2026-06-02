/*
 * Ambient Home Activity Classifier — Deployment Sketch
 * =====================================================
 * I2S Mic (INMP441) → MFCC → TFLite → GC9A01 Display
 * With confidence ring and class symbols
 */

#include <driver/i2s.h>
#include <math.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_GC9A01A.h>
#include <arduinoFFT.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "class_config.h"

// =============================================================================
// Pins
// =============================================================================
#define I2S_SCK     2
#define I2S_WS      3
#define I2S_SD      4
#define TFT_CS      5
#define TFT_DC      6
#define TFT_RST     1
#define TFT_MOSI    9
#define TFT_SCK_PIN 7

// =============================================================================
// Audio
// =============================================================================
#define AUDIO_SAMPLE_RATE   16000
#define AUDIO_BUFFER_SIZE   512
#define CLIP_SAMPLES        16000

// =============================================================================
// MFCC
// =============================================================================
#define MFCC_N_FFT      1024
#define MFCC_HOP        512
#define MFCC_N_MFCC     14
#define MFCC_N_MELS     40
#define MFCC_FMAX       8000.0f
#define MFCC_N_FRAMES   (1 + CLIP_SAMPLES / MFCC_HOP)

// =============================================================================
// Inference & Display
// =============================================================================
#define SMOOTHING_WINDOW        5
#define IMPACT_OVERRIDE_MS      500
#define IMPACT_CLASS_ID         3
#define INFERENCE_INTERVAL      3000
#define TENSOR_ARENA_SIZE       (60 * 1024)
#define DISPLAY_REINIT_INTERVAL 30000
#define CONF_REDRAW_THRESHOLD   0.05f   // only redraw ring if conf changes by >5%

// =============================================================================
// Globals
// =============================================================================
Adafruit_GC9A01A tft(TFT_CS, TFT_DC, TFT_MOSI, TFT_SCK_PIN, TFT_RST);

int32_t i2s_raw[AUDIO_BUFFER_SIZE];
float   audio_buf[CLIP_SAMPLES];
int     audio_pos = 0;

float mel_fb[MFCC_N_MELS][MFCC_N_FFT / 2 + 1];
float mfcc_out[MFCC_N_FRAMES][MFCC_N_MFCC];

ArduinoFFT<float> FFT = ArduinoFFT<float>();
float vR[MFCC_N_FFT], vI[MFCC_N_FFT];

static tflite::MicroErrorReporter micro_error_reporter;
alignas(16) uint8_t arena[TENSOR_ARENA_SIZE];
tflite::AllOpsResolver resolver;
const tflite::Model*      tfl_model = nullptr;
tflite::MicroInterpreter* interp    = nullptr;
TfLiteTensor* t_in  = nullptr;
TfLiteTensor* t_out = nullptr;

int   pred_hist[SMOOTHING_WINDOW];
int   hist_idx  = 0;
int   cur_class = 0;
float cur_conf  = 0.0f;
int   last_disp = -1;
float last_conf_drawn = -1.0f;
unsigned long impact_t0           = 0;
bool  impact_on                   = false;
unsigned long last_infer          = 0;
unsigned long last_display_reinit = 0;

// =============================================================================
// Mel filterbank
// =============================================================================
float hz2mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
float mel2hz(float m)  { return 700.0f  * (powf(10.0f, m / 2595.0f) - 1.0f); }

void init_mel() {
    float mmin = hz2mel(0), mmax = hz2mel(MFCC_FMAX);
    float mp[MFCC_N_MELS + 2];
    int   bp[MFCC_N_MELS + 2];
    int   bins = MFCC_N_FFT / 2 + 1;
    for (int i = 0; i < MFCC_N_MELS + 2; i++) {
        mp[i] = mmin + (mmax - mmin) * i / (MFCC_N_MELS + 1);
        int b = (int)floorf((MFCC_N_FFT + 1) * mel2hz(mp[i]) / AUDIO_SAMPLE_RATE);
        bp[i] = b < bins ? b : bins - 1;
    }
    memset(mel_fb, 0, sizeof(mel_fb));
    for (int m = 0; m < MFCC_N_MELS; m++) {
        for (int k = bp[m];   k <= bp[m+1] && k < bins; k++)
            if (bp[m+1] != bp[m])   mel_fb[m][k] = (float)(k - bp[m])   / (bp[m+1] - bp[m]);
        for (int k = bp[m+1]; k <= bp[m+2] && k < bins; k++)
            if (bp[m+2] != bp[m+1]) mel_fb[m][k] = (float)(bp[m+2] - k) / (bp[m+2] - bp[m+1]);
    }
    Serial.println("Mel filterbank ready");
}

// =============================================================================
// MFCC
// =============================================================================
void compute_mfccs() {
    int bins = MFCC_N_FFT / 2 + 1;
    for (int f = 0; f < MFCC_N_FRAMES; f++) {
        int start = f * MFCC_HOP;
        for (int i = 0; i < MFCC_N_FFT; i++) {
            float w = 0.5f * (1.0f - cosf(2.0f * PI * i / (MFCC_N_FFT - 1)));
            vR[i] = (start + i < CLIP_SAMPLES) ? audio_buf[start + i] * w : 0.0f;
            vI[i] = 0.0f;
        }
        FFT.compute(vR, vI, MFCC_N_FFT, FFT_FORWARD);
        float ps[MFCC_N_FFT / 2 + 1];
        for (int k = 0; k < bins; k++)
            ps[k] = (vR[k]*vR[k] + vI[k]*vI[k]) / MFCC_N_FFT;
        float me[MFCC_N_MELS];
        for (int m = 0; m < MFCC_N_MELS; m++) {
            me[m] = 0.0f;
            for (int k = 0; k < bins; k++) me[m] += mel_fb[m][k] * ps[k];
            me[m] = logf(me[m] + 1e-10f);
        }
        for (int c = 0; c < MFCC_N_MFCC; c++) {
            mfcc_out[f][c] = 0.0f;
            for (int m = 0; m < MFCC_N_MELS; m++)
                mfcc_out[f][c] += me[m] * cosf(PI * c * (m + 0.5f) / MFCC_N_MELS);
        }
    }
}

// =============================================================================
// I2S
// =============================================================================
void init_i2s() {
    i2s_driver_uninstall(I2S_NUM_0);
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = AUDIO_SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 4,
        .dma_buf_len          = AUDIO_BUFFER_SIZE,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = 0
    };
    i2s_pin_config_t pins = {
        .bck_io_num   = I2S_SCK,
        .ws_io_num    = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_SD
    };
    i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pins);
    size_t br = 0;
    i2s_read(I2S_NUM_0, i2s_raw, sizeof(i2s_raw), &br, 100);
    Serial.println("I2S ready");
}

void capture_audio() {
    audio_pos = 0;
    while (audio_pos < CLIP_SAMPLES) {
        size_t br = 0;
        int n = (CLIP_SAMPLES - audio_pos < AUDIO_BUFFER_SIZE)
                ? (CLIP_SAMPLES - audio_pos) : AUDIO_BUFFER_SIZE;
        i2s_read(I2S_NUM_0, i2s_raw, n * sizeof(int32_t), &br, portMAX_DELAY);
        int got = (int)(br / sizeof(int32_t));
        for (int i = 0; i < got && audio_pos < CLIP_SAMPLES; i++)
            audio_buf[audio_pos++] = (float)(i2s_raw[i] >> 8) / 8388608.0f;
    }
    // Normalize to peak 1.0
    float peak = 0.0f;
    for (int i = 0; i < CLIP_SAMPLES; i++)
        if (fabsf(audio_buf[i]) > peak) peak = fabsf(audio_buf[i]);
    if (peak > 0.0f)
        for (int i = 0; i < CLIP_SAMPLES; i++) audio_buf[i] /= peak;
}

// =============================================================================
// TFLite
// =============================================================================
void init_tflite() {
    tfl_model = tflite::GetModel(model_data);
    static tflite::MicroInterpreter si(
        tfl_model, resolver, arena, TENSOR_ARENA_SIZE, &micro_error_reporter);
    interp = &si;
    if (interp->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors failed"); while (1);
    }
    t_in  = interp->input(0);
    t_out = interp->output(0);
    Serial.printf("TFLite ready — arena: %d bytes\n", (int)interp->arena_used_bytes());
}

int run_inference(float* conf) {
    float* d = t_in->data.f;
    for (int f = 0; f < MFCC_N_FRAMES; f++)
        for (int c = 0; c < MFCC_N_MFCC; c++)
            d[f * MFCC_N_MFCC + c] = mfcc_out[f][c];
    if (interp->Invoke() != kTfLiteOk) return -1;
    float scores[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) scores[i] = t_out->data.f[i];
    int best = 0;
    for (int i = 1; i < NUM_CLASSES; i++)
        if (scores[i] > scores[best]) best = i;
    *conf = scores[best];
    return best;
}

// =============================================================================
// Smoothing
// =============================================================================
int smoothed_class() {
    int v[NUM_CLASSES] = {0};
    for (int i = 0; i < SMOOTHING_WINDOW; i++)
        if (pred_hist[i] >= 0 && pred_hist[i] < NUM_CLASSES)
            v[pred_hist[i]]++;
    int b = 0;
    for (int i = 1; i < NUM_CLASSES; i++) if (v[i] > v[b]) b = i;
    return b;
}

// =============================================================================
// Display — Symbols
// =============================================================================

// Quiet: three small horizontal lines (silence/pause)
void draw_symbol_quiet(int cx, int cy, uint16_t color) {
    tft.drawFastHLine(cx - 12, cy - 6, 24, color);
    tft.drawFastHLine(cx - 8,  cy,     16, color);
    tft.drawFastHLine(cx - 12, cy + 6, 24, color);
}

// Human: stick figure (circle head + body + arms + legs)
void draw_symbol_human(int cx, int cy, uint16_t color) {
    tft.drawCircle(cx, cy - 8, 5, color);          // head
    tft.drawFastVLine(cx, cy - 3, 12, color);       // body
    tft.drawLine(cx, cy, cx - 7, cy + 5, color);    // left arm
    tft.drawLine(cx, cy, cx + 7, cy + 5, color);    // right arm
    tft.drawLine(cx, cy + 9, cx - 6, cy + 16, color); // left leg
    tft.drawLine(cx, cy + 9, cx + 6, cy + 16, color); // right leg
}

// Appliance: gear/cog (circle with small rectangles around it)
void draw_symbol_appliance(int cx, int cy, uint16_t color) {
    tft.drawCircle(cx, cy, 6, color);               // inner circle
    for (int i = 0; i < 6; i++) {
        float angle = i * 60.0f * PI / 180.0f;
        int x1 = cx + (int)(9 * cosf(angle));
        int y1 = cy + (int)(9 * sinf(angle));
        int x2 = cx + (int)(12 * cosf(angle));
        int y2 = cy + (int)(12 * sinf(angle));
        tft.drawLine(x1, y1, x2, y2, color);        // spokes
    }
    tft.drawCircle(cx, cy, 12, color);               // outer ring
}

// Impact: exclamation mark (tall rectangle + dot)
void draw_symbol_impact(int cx, int cy, uint16_t color) {
    tft.fillRect(cx - 2, cy - 10, 5, 14, color);    // bar
    tft.fillRect(cx - 2, cy + 7, 5, 5, color);      // dot
}

// Ambient: three wavy lines (like water/wind)
void draw_symbol_ambient(int cx, int cy, uint16_t color) {
    for (int row = -1; row <= 1; row++) {
        int y = cy + row * 8;
        for (int x = -14; x < 14; x++) {
            float wave = sinf(x * 0.4f) * 3.0f;
            tft.drawPixel(cx + x, y + (int)wave, color);
            tft.drawPixel(cx + x, y + (int)wave + 1, color);
        }
    }
}

void draw_symbol(int cls, int cx, int cy, uint16_t color) {
    switch (cls) {
        case 0: draw_symbol_quiet(cx, cy, color); break;
        case 1: draw_symbol_human(cx, cy, color); break;
        case 2: draw_symbol_appliance(cx, cy, color); break;
        case 3: draw_symbol_impact(cx, cy, color); break;
        case 4: draw_symbol_ambient(cx, cy, color); break;
    }
}

// =============================================================================
// Display — Confidence Ring
// =============================================================================
void draw_ring(float confidence) {
    int cx = 120, cy = 120, r = 108, ht = 4;

    // Background ring (dark gray)
    for (float a = 0.0f; a < 360.0f; a += 1.2f) {
        float rd = a * PI / 180.0f;
        tft.fillCircle(cx + (int)(r*cosf(rd)), cy + (int)(r*sinf(rd)), ht, 0x2104);
    }

    // Confidence arc red → green
    float end = -90.0f + confidence * 360.0f;
    for (float a = -90.0f; a <= end; a += 1.2f) {
        float rd = a * PI / 180.0f;
        float t = (a + 90.0f) / 360.0f;
        uint16_t col = tft.color565((uint8_t)(255*(1.0f-t)), (uint8_t)(255*t), 0);
        tft.fillCircle(cx + (int)(r*cosf(rd)), cy + (int)(r*sinf(rd)), ht, col);
    }
}

// =============================================================================
// Display — Full Update
// =============================================================================
void reinit_display() {
    pinMode(TFT_RST, OUTPUT);
    digitalWrite(TFT_RST, LOW);
    delay(10);
    digitalWrite(TFT_RST, HIGH);
    delay(120);
    tft.begin();
    tft.setSPISpeed(20000000);
    tft.setRotation(2);
    tft.fillScreen(GC9A01A_BLACK);
    last_disp = -1;
    last_conf_drawn = -1.0f;
    last_display_reinit = millis();
    Serial.println("Display reinitialized");
}

void full_redraw(int cls, float conf) {
    // Clear center area only (preserve ring if possible)
    tft.fillCircle(120, 120, 90, GC9A01A_BLACK);

    // Class label at y=95
    tft.setTextColor(CLASS_COLORS[cls]);
    tft.setTextSize(3);
    int16_t x1, y1;
    uint16_t w, h;
    tft.getTextBounds(CLASS_NAMES[cls], 0, 0, &x1, &y1, &w, &h);
    tft.setCursor(120 - w/2, 90 - h/2);
    tft.print(CLASS_NAMES[cls]);

    // Confidence percentage at y=120
    char conf_str[8];
    snprintf(conf_str, sizeof(conf_str), "%d%%", (int)(conf * 100));
    tft.setTextColor(0xBDF7);  // light gray
    tft.setTextSize(2);
    tft.getTextBounds(conf_str, 0, 0, &x1, &y1, &w, &h);
    tft.setCursor(120 - w/2, 118);
    tft.print(conf_str);

    // Symbol at y=152
    draw_symbol(cls, 120, 152, CLASS_COLORS[cls]);

    // Ring
    draw_ring(conf);

    last_disp = cls;
    last_conf_drawn = conf;
}

void update_display(int cls, float conf) {
    bool class_changed = (cls != last_disp);
    bool conf_changed = (fabsf(conf - last_conf_drawn) > CONF_REDRAW_THRESHOLD);

    if (class_changed) {
        // Full redraw on class change
        full_redraw(cls, conf);
    } else if (conf_changed) {
        // Only update confidence text and ring
        // Clear confidence text area
        tft.fillRect(80, 115, 80, 20, GC9A01A_BLACK);
        char conf_str[8];
        snprintf(conf_str, sizeof(conf_str), "%d%%", (int)(conf * 100));
        tft.setTextColor(0xBDF7);
        tft.setTextSize(2);
        int16_t x1, y1;
        uint16_t w, h;
        tft.getTextBounds(conf_str, 0, 0, &x1, &y1, &w, &h);
        tft.setCursor(120 - w/2, 118);
        tft.print(conf_str);

        // Redraw ring
        draw_ring(conf);
        last_conf_drawn = conf;
    }
}

// =============================================================================
// Setup
// =============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== Ambient Home Activity Classifier ===\n");

    for (int i = 0; i < SMOOTHING_WINDOW; i++) pred_hist[i] = 0;

    pinMode(TFT_RST, OUTPUT);
    digitalWrite(TFT_RST, LOW);
    delay(100);
    digitalWrite(TFT_RST, HIGH);
    delay(200);

    tft.begin();
    tft.setSPISpeed(20000000);
    tft.setRotation(2);
    tft.fillScreen(GC9A01A_BLACK);
    last_display_reinit = millis();

    // Startup text
    tft.setTextColor(GC9A01A_WHITE);
    tft.setTextSize(2);
    int16_t x1, y1;
    uint16_t w, h;
    tft.getTextBounds("Starting...", 0, 0, &x1, &y1, &w, &h);
    tft.setCursor(120 - w/2, 110);
    tft.print("Starting...");

    init_mel();
    init_i2s();
    init_tflite();

    tft.fillScreen(GC9A01A_BLACK);
    tft.setTextColor(GC9A01A_WHITE);
    tft.setTextSize(2);
    tft.getTextBounds("Ready", 0, 0, &x1, &y1, &w, &h);
    tft.setCursor(120 - w/2, 110);
    tft.print("Ready");
    delay(1000);

    Serial.println("System ready\n");
}

// =============================================================================
// Loop
// =============================================================================
void loop() {
    unsigned long now = millis();

    if (now - last_display_reinit > DISPLAY_REINIT_INTERVAL) {
        reinit_display();
        // Force full redraw after reinit
        if (last_disp >= 0) {
            full_redraw(cur_class, cur_conf);
        }
    }

    if (impact_on && (now - impact_t0 > IMPACT_OVERRIDE_MS)) impact_on = false;

    if (now - last_infer >= INFERENCE_INTERVAL) {
        last_infer = now;

        unsigned long t0 = millis();
        capture_audio();
        unsigned long t1 = millis();
        compute_mfccs();
        unsigned long t2 = millis();
        float conf = 0.0f;
        int pred = run_inference(&conf);
        unsigned long t3 = millis();
        if (pred < 0) return;

        pred_hist[hist_idx] = pred;
        hist_idx = (hist_idx + 1) % SMOOTHING_WINDOW;

        int   disp_cls;
        float disp_conf;

        if (pred == IMPACT_CLASS_ID) {
            disp_cls  = IMPACT_CLASS_ID;
            disp_conf = conf;
            impact_on = true;
            impact_t0 = now;
        } else if (impact_on) {
            disp_cls  = IMPACT_CLASS_ID;
            disp_conf = cur_conf;
        } else {
            disp_cls  = smoothed_class();
            disp_conf = conf;
        }

        cur_class = disp_cls;
        cur_conf  = disp_conf;
        update_display(disp_cls, disp_conf);

        Serial.printf("Raw:%-12s(%3.0f%%) Show:%-12s Audio:%lums MFCC:%lums Infer:%lums\n",
                      CLASS_NAMES[pred], conf*100, CLASS_NAMES[disp_cls],
                      t1-t0, t2-t1, t3-t2);
    }
}
