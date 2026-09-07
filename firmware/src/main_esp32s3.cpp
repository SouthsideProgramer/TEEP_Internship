/*
 * ESP32-S3 -- external I2S microphone -> causal heart/lung band split ->
 * binary frames on serial.
 *
 * Board-specific half of the firmware; the numerics live in lib/hls_filter and
 * are verified against scipy by the `native` environment. Biquads run through
 * ESP-DSP (HLS_FILTER_BACKEND_ESPDSP, set in platformio.ini) -- esp-dsp ships
 * inside the arduino-esp32 SDK, so there is nothing to add to lib_deps.
 *
 * Wiring assumes a 24-bit I2S MEMS mic (INMP441 / ICS-43434 class) on the pins
 * below; change I2S_PIN_* to match the board. The mic word is 32-bit with the
 * sample left-justified, so the top 16 bits are taken as int16 -- the same
 * scale the Python side sees after librosa loads the 16-bit dataset.
 */
#include <Arduino.h>
#include <driver/i2s.h>

#include "hls_filter.h"
#include "hls_stream.h"

static const i2s_port_t I2S_PORT = I2S_NUM_0;
static const int I2S_PIN_BCLK = 4;
static const int I2S_PIN_WS = 5;
static const int I2S_PIN_DIN = 6;

/* One DMA buffer's worth per read; 4 buffers give ~64 ms of slack. */
static const uint32_t I2S_FRAME_SAMPLES = 256;
static const int I2S_DMA_BUFFERS = 4;

static int32_t i2s_raw[I2S_FRAME_SAMPLES];
static int16_t mic_block[I2S_FRAME_SAMPLES];

static hls_pipeline_t pipeline;
static float heart_out[I2S_FRAME_SAMPLES / HLS_DECIM_FACTOR + 1];
static float lung_out[I2S_FRAME_SAMPLES / HLS_DECIM_FACTOR + 1];
static uint16_t frame_seq = 0;
static uint32_t dropped_samples = 0;

static void i2s_setup()
{
    i2s_config_t config = {};
    config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
    config.sample_rate = HLS_MIC_SR_HZ;
    config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
    config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    config.dma_buf_count = I2S_DMA_BUFFERS;
    config.dma_buf_len = I2S_FRAME_SAMPLES;
    config.use_apll = false;
    config.tx_desc_auto_clear = false;
    config.fixed_mclk = 0;

    i2s_pin_config_t pins = {};
    pins.bck_io_num = I2S_PIN_BCLK;
    pins.ws_io_num = I2S_PIN_WS;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = I2S_PIN_DIN;
    pins.mck_io_num = I2S_PIN_NO_CHANGE;

    ESP_ERROR_CHECK(i2s_driver_install(I2S_PORT, &config, 0, NULL));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_PORT, &pins));
}

/*
 * One-off timing check printed before streaming starts -- same measurement the
 * Nano 33 firmware makes, so the two boards can be compared directly in report/.
 */
static void benchmark()
{
    static int16_t buf[HLS_MIC_SR_HZ / 10];  /* 100 ms of mic data */
    const uint32_t n_buf = sizeof(buf) / sizeof(buf[0]);
    for (uint32_t i = 0; i < n_buf; ++i) {
        buf[i] = (int16_t)(8000.0f * sinf(2.0f * PI * 120.0f * (float)i / (float)HLS_MIC_SR_HZ));
    }
    static float bench_heart[HLS_MIC_SR_HZ / 10 / HLS_DECIM_FACTOR + 1];
    static float bench_lung[HLS_MIC_SR_HZ / 10 / HLS_DECIM_FACTOR + 1];

    hls_pipeline_reset(&pipeline);
    const uint32_t t0 = micros();
    for (int rep = 0; rep < 10; ++rep) {   /* 10 x 100 ms = 1 s of audio */
        hls_pipeline_process(&pipeline, buf, n_buf, bench_heart, bench_lung, NULL);
    }
    const uint32_t elapsed = micros() - t0;
    hls_pipeline_reset(&pipeline);

    Serial.printf("# benchmark: 1.000 s of audio processed in %.3f ms -> real-time factor %.1f\n",
                  elapsed / 1000.0f, 1.0e6f / (float)elapsed);
}

void setup()
{
    Serial.begin(115200);
    delay(200);

    hls_pipeline_init(&pipeline);

    Serial.println("# hls_filter -- ESP32-S3 (I2S mic, ESP-DSP)");
    Serial.printf("# mic %u Hz -> decimate /%u -> %u Hz band split\n",
                  HLS_MIC_SR_HZ, HLS_DECIM_FACTOR, HLS_BASELINE_SR_HZ);
    Serial.printf("# heart %.0f-%.0f Hz, lung %.0f-%.0f Hz, causal single-pass IIR\n",
                  HLS_HEART_LOW_HZ, HLS_HEART_HIGH_HZ, HLS_LUNG_LOW_HZ, HLS_LUNG_HIGH_HZ);
    benchmark();

    i2s_setup();
    Serial.println("# streaming binary frames");
}

void loop()
{
    size_t bytes_read = 0;
    const esp_err_t err = i2s_read(I2S_PORT, i2s_raw, sizeof(i2s_raw), &bytes_read, portMAX_DELAY);
    if (err != ESP_OK || bytes_read == 0) {
        dropped_samples += I2S_FRAME_SAMPLES;
        return;
    }

    const uint32_t n_mic = (uint32_t)(bytes_read / sizeof(int32_t));
    for (uint32_t i = 0; i < n_mic; ++i) {
        /* Left-justified 24-bit word -> int16, matching the dataset's scale. */
        mic_block[i] = (int16_t)(i2s_raw[i] >> 16);
    }

    const uint32_t n = hls_pipeline_process(&pipeline, mic_block, n_mic,
                                            heart_out, lung_out, NULL);
    if (n == 0) {
        return;
    }

    hls_frame_header_t header;
    memcpy(header.magic, HLS_STREAM_MAGIC, sizeof(header.magic));
    header.n_samples = (uint16_t)n;
    header.seq = frame_seq++;
    header.dropped = dropped_samples;

    Serial.write((const uint8_t *)&header, sizeof(header));
    Serial.write((const uint8_t *)heart_out, n * sizeof(float));
    Serial.write((const uint8_t *)lung_out, n * sizeof(float));
}
