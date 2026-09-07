/*
 * On-board self-test -- the board-side counterpart of the `native` environment.
 *
 * The native build proves the portable C filter matches scipy on the host, but
 * the board builds swap in CMSIS-DSP / ESP-DSP kernels, and a build that links
 * is not a build that computes the right thing. This firmware replays the
 * golden slice in lib/hls_filter/hls_golden_embedded.h through the same API the
 * real firmware uses and prints the deviation from scipy over serial, so the
 * vendor kernels can be verified the first time a board is plugged in.
 *
 * Build:  pio run -e nano33ble-selftest    (or esp32s3-selftest)
 * Expect: max|err| at the float32 quantisation floor, ~6e-8 for the heart band
 *         and ~2e-8 for the lung band (see firmware/golden/manifest.txt).
 */
#include <Arduino.h>

#include "hls_filter.h"
#include "hls_golden_embedded.h"

/* Comfortably above the float32 floor, far below any coefficient or sign error. */
static const float TOLERANCE = 1.0e-5f;

static hls_biquad_t filt;
static float out[HLS_GOLDEN_INPUT_LEN];

static bool check(const char *label, const float *sos, uint8_t sections, const float *want)
{
    hls_biquad_init(&filt, sos, sections);
    hls_biquad_process(&filt, HLS_GOLDEN_INPUT, out, HLS_GOLDEN_INPUT_LEN);

    float max_abs = 0.0f;
    uint32_t worst = 0;
    for (uint32_t i = 0; i < HLS_GOLDEN_INPUT_LEN; ++i) {
        const float e = fabsf(out[i] - want[i]);
        if (e > max_abs) {
            max_abs = e;
            worst = i;
        }
    }
    const bool ok = max_abs <= TOLERANCE;

    Serial.print(ok ? "PASS  " : "FAIL  ");
    Serial.print(label);
    Serial.print("  max|err| = ");
    Serial.print(max_abs, 9);
    Serial.print("  at i = ");
    Serial.println(worst);
    return ok;
}

/* Same input, three block sizes: the streaming state must make them identical. */
static bool check_block_invariance()
{
    static float ref[HLS_GOLDEN_INPUT_LEN];
    static const uint32_t sizes[] = {1u, 7u, 64u};
    bool ok = true;

    hls_biquad_init(&filt, HLS_HEART_SOS, (uint8_t)HLS_HEART_SECTIONS);
    hls_biquad_process(&filt, HLS_GOLDEN_INPUT, ref, HLS_GOLDEN_INPUT_LEN);

    for (uint32_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); ++k) {
        const uint32_t bs = sizes[k];
        hls_biquad_init(&filt, HLS_HEART_SOS, (uint8_t)HLS_HEART_SECTIONS);
        for (uint32_t off = 0; off < HLS_GOLDEN_INPUT_LEN; off += bs) {
            uint32_t chunk = HLS_GOLDEN_INPUT_LEN - off;
            if (chunk > bs) {
                chunk = bs;
            }
            hls_biquad_process(&filt, &HLS_GOLDEN_INPUT[off], &out[off], chunk);
        }
        bool identical = true;
        for (uint32_t i = 0; i < HLS_GOLDEN_INPUT_LEN; ++i) {
            if (out[i] != ref[i]) {
                identical = false;
                break;
            }
        }
        Serial.print(identical ? "PASS  " : "FAIL  ");
        Serial.print("block size ");
        Serial.print(bs);
        Serial.println(" == one call");
        ok = ok && identical;
    }
    return ok;
}

void setup()
{
    Serial.begin(115200);
#if defined(ARDUINO_ARCH_MBED)
    while (!Serial) {
        ;  /* native USB CDC */
    }
#else
    delay(200);
#endif

    Serial.println("# hls_filter on-board self-test vs scipy.signal.sosfilt");
    Serial.print("# backend: ");
#if defined(HLS_FILTER_BACKEND_CMSIS)
    Serial.println("CMSIS-DSP arm_biquad_cascade_df2T_f32");
#elif defined(HLS_FILTER_BACKEND_ESPDSP)
    Serial.println("ESP-DSP dsps_biquad_f32");
#else
    Serial.println("portable C");
#endif
    Serial.print("# ");
    Serial.print(HLS_GOLDEN_INPUT_LEN);
    Serial.print(" samples, tolerance ");
    Serial.println(TOLERANCE, 9);

    bool ok = true;
    ok &= check("heart 20-200 Hz  ", HLS_HEART_SOS, (uint8_t)HLS_HEART_SECTIONS, HLS_GOLDEN_HEART);
    ok &= check("lung 150-1000 Hz ", HLS_LUNG_SOS, (uint8_t)HLS_LUNG_SECTIONS, HLS_GOLDEN_LUNG);
    ok &= check_block_invariance();

    Serial.println(ok ? "ALL CHECKS PASSED" : "CHECKS FAILED");
}

void loop()
{
    delay(1000);
}
