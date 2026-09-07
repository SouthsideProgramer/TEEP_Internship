/*
 * Arduino Nano 33 BLE Sense -- onboard PDM microphone -> causal heart/lung
 * band split -> binary frames on USB serial.
 *
 * Board-specific half of the firmware: everything numerical lives in
 * lib/hls_filter (verified against scipy by the `native` environment); this
 * file only does acquisition and transport. The biquads run through CMSIS-DSP
 * (HLS_FILTER_BACKEND_CMSIS, set in platformio.ini).
 *
 * The nRF52840 has 256 kB of RAM. Nothing here scales with clip length: the
 * filters are streaming, so RAM is the ring buffer plus a few hundred bytes of
 * filter state, not the ~120 kB a buffered 15 s int16 clip would have cost.
 */
#include <Arduino.h>
#include <PDM.h>

#include "hls_filter.h"
#include "hls_stream.h"

/* PDM hands us ~256 samples per interrupt; a ring 8x that absorbs USB write
 * latency without dropping mic data. Must stay a power of two for the mask. */
static const uint32_t PDM_CHUNK_SAMPLES = 256;
static const uint32_t RING_SAMPLES = 2048;

static int16_t ring[RING_SAMPLES];
static volatile uint32_t ring_head = 0;      /* written by the PDM ISR */
static volatile uint32_t ring_tail = 0;      /* read by loop() */
static volatile uint32_t dropped_samples = 0;

static int16_t isr_scratch[PDM_CHUNK_SAMPLES];

static hls_pipeline_t pipeline;
static float heart_out[RING_SAMPLES / HLS_DECIM_FACTOR + 1];
static float lung_out[RING_SAMPLES / HLS_DECIM_FACTOR + 1];
static uint16_t frame_seq = 0;

static void onPDMdata()
{
    const int bytes = PDM.available();
    if (bytes <= 0) {
        return;
    }
    uint32_t n = (uint32_t)bytes / sizeof(int16_t);
    if (n > PDM_CHUNK_SAMPLES) {
        n = PDM_CHUNK_SAMPLES;
    }
    PDM.read(isr_scratch, n * sizeof(int16_t));

    const uint32_t head = ring_head;
    if ((head - ring_tail) + n > RING_SAMPLES) {
        /* Overrun: count what was lost instead of stalling the ISR. The count
         * rides along in every frame header so a capture can be rejected. */
        dropped_samples += n;
        return;
    }
    for (uint32_t i = 0; i < n; ++i) {
        ring[(head + i) & (RING_SAMPLES - 1)] = isr_scratch[i];
    }
    ring_head = head + n;
}

/*
 * One-off timing check printed before streaming starts: how long the band
 * split takes for one second of audio. Feeds the compute-cost table in report/
 * and shows immediately whether the MCU keeps up in real time.
 */
static void benchmark()
{
    static int16_t buf[HLS_MIC_SR_HZ / 10];  /* 100 ms of mic data */
    const uint32_t n_buf = sizeof(buf) / sizeof(buf[0]);
    for (uint32_t i = 0; i < n_buf; ++i) {
        buf[i] = (int16_t)(8000.0f * sinf(2.0f * PI * 120.0f * (float)i / (float)HLS_MIC_SR_HZ));
    }

    hls_pipeline_reset(&pipeline);
    const uint32_t t0 = micros();
    for (int rep = 0; rep < 10; ++rep) {   /* 10 x 100 ms = 1 s of audio */
        hls_pipeline_process(&pipeline, buf, n_buf, heart_out, lung_out, NULL);
    }
    const uint32_t elapsed = micros() - t0;
    hls_pipeline_reset(&pipeline);

    Serial.print("# benchmark: 1.000 s of audio processed in ");
    Serial.print(elapsed / 1000.0f, 3);
    Serial.print(" ms -> real-time factor ");
    Serial.println(1.0e6f / (float)elapsed, 1);
}

void setup()
{
    Serial.begin(115200);
    while (!Serial) {
        ;  /* native USB CDC: wait for the host to open the port */
    }

    hls_pipeline_init(&pipeline);

    Serial.println("# hls_filter -- Nano 33 BLE Sense (PDM mic, CMSIS-DSP)");
    Serial.print("# mic ");
    Serial.print(HLS_MIC_SR_HZ);
    Serial.print(" Hz -> decimate /");
    Serial.print(HLS_DECIM_FACTOR);
    Serial.print(" -> ");
    Serial.print(HLS_BASELINE_SR_HZ);
    Serial.println(" Hz band split");
    Serial.print("# heart ");
    Serial.print(HLS_HEART_LOW_HZ, 0);
    Serial.print("-");
    Serial.print(HLS_HEART_HIGH_HZ, 0);
    Serial.print(" Hz, lung ");
    Serial.print(HLS_LUNG_LOW_HZ, 0);
    Serial.print("-");
    Serial.print(HLS_LUNG_HIGH_HZ, 0);
    Serial.println(" Hz, causal single-pass IIR");
    benchmark();

    PDM.onReceive(onPDMdata);
    PDM.setBufferSize(PDM_CHUNK_SAMPLES * sizeof(int16_t));
    if (!PDM.begin(1, HLS_MIC_SR_HZ)) {
        Serial.println("# FATAL: PDM.begin failed");
        while (1) {
            ;
        }
    }
    Serial.println("# streaming binary frames");
}

void loop()
{
    const uint32_t head = ring_head;
    uint32_t available = head - ring_tail;
    if (available < PDM_CHUNK_SAMPLES) {
        return;
    }

    /* Drain to the end of the ring in one contiguous run so the pipeline sees a
     * flat buffer; the wrapped remainder is picked up on the next loop() pass. */
    const uint32_t start = ring_tail & (RING_SAMPLES - 1);
    if (start + available > RING_SAMPLES) {
        available = RING_SAMPLES - start;
    }

    const uint32_t n = hls_pipeline_process(&pipeline, &ring[start], available,
                                            heart_out, lung_out, NULL);
    ring_tail += available;
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
