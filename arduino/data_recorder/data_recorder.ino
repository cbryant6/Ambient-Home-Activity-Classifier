/*
 * INMP441 Data Recorder for Ambient Home Activity Classifier
 * ==========================================================
 * Streams raw audio over serial for capture by record_clips.py
 *
 * Commands (type in terminal running record_clips.py):
 *   quiet        → start recording Quiet class
 *   human        → start recording Human Activity class
 *   appliance    → start recording Appliance/Mechanical class
 *   impact       → start recording Transient Impact class
 *   ambient      → start recording Environmental/Ambient class
 *   stop         → pause recording
 */

#include <driver/i2s.h>

#define I2S_SCK     2
#define I2S_WS      3
#define I2S_SD      4

#define SAMPLE_RATE     16000
#define CLIP_SAMPLES    16000
#define BUFFER_SIZE     512

int32_t raw_buf[BUFFER_SIZE];
float   audio_buf[CLIP_SAMPLES];
int     audio_pos = 0;

bool    recording = false;
String  current_class = "";
int     clip_count = 0;

void setup() {
    Serial.begin(921600);
    delay(1000);

    Serial.println("=== INMP441 Data Recorder ===");
    Serial.println("Commands: quiet / human / appliance / impact / ambient / stop");
    Serial.println("Waiting for command...");

    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 4,
        .dma_buf_len          = BUFFER_SIZE,
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
    i2s_read(I2S_NUM_0, raw_buf, sizeof(raw_buf), &br, 100);
}

void check_command() {
    if (!Serial.available()) return;

    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toLowerCase();

    if (cmd == "quiet" || cmd == "human" || cmd == "appliance" ||
        cmd == "impact" || cmd == "ambient") {
        current_class = cmd;
        recording = true;
        clip_count = 0;
        Serial.printf("RECORDING:%s\n", current_class.c_str());
        Serial.printf("Recording %s — make sounds now!\n", current_class.c_str());
    } else if (cmd == "stop") {
        recording = false;
        Serial.printf("STOPPED — recorded %d clips of %s\n",
                      clip_count, current_class.c_str());
    } else if (cmd.length() > 0) {
        Serial.printf("Unknown command: %s\n", cmd.c_str());
        Serial.println("Commands: quiet / human / appliance / impact / ambient / stop");
    }
}

void capture_and_send() {
    // Capture 1 second of audio
    audio_pos = 0;
    while (audio_pos < CLIP_SAMPLES) {
        size_t br = 0;
        int n = min(BUFFER_SIZE, CLIP_SAMPLES - audio_pos);
        i2s_read(I2S_NUM_0, raw_buf, n * sizeof(int32_t), &br, portMAX_DELAY);
        int got = br / sizeof(int32_t);
        for (int i = 0; i < got && audio_pos < CLIP_SAMPLES; i++)
            audio_buf[audio_pos++] = (float)(raw_buf[i] >> 8) / 8388608.0f;
    }

    // Check for stop command before sending — stops within 1 second of command
    check_command();
    if (!recording) return;

    // Send clip header
    Serial.printf("CLIP:%s:%d\n", current_class.c_str(), clip_count);

    // Send samples
    for (int i = 0; i < CLIP_SAMPLES; i++) {
        Serial.printf("%.6f\n", audio_buf[i]);
    }
    Serial.println("END");

    clip_count++;
    Serial.printf("Clip %d saved — keep going!\n", clip_count);
}

void loop() {
    check_command();

    if (recording) {
        capture_and_send();
    } else {
        delay(50);
        check_command();
    }
}
